"""
Step 3: pick the decision thresholds on the validation faces, then measure the
whole system on the test faces.

Writes into the run folder:
  thresholds.json: match + liveness thresholds (gallery.FaceID reads this)
  report.txt: everything printed below
  scores.png: genuine vs impostor score histograms with the threshold
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

N_ENROLL = 5 # scans per person used to build the template


# Scores and error rates

def make_templates(emb, people, n_enroll=N_ENROLL, seed=0):
    """
    For each person, average n_enroll random scans into a template (length 1).
    Returns (templates (P, D), template_people (P,), probe_idx): probe_idx are the
    scans NOT used for enrollment, the ones that get tested.
    """
    rng = np.random.default_rng(seed)
    templates, t_people, probes = [], [], []
    for p in np.unique(people):
        idx = rng.permutation(np.nonzero(people == p)[0])
        t = emb[idx[:n_enroll]].mean(0)
        templates.append(t / np.linalg.norm(t))
        t_people.append(p)
        probes.extend(idx[n_enroll:])
    return np.array(templates), np.array(t_people), np.array(probes, int)


def verification_scores(emb, people, n_enroll=N_ENROLL, seed=0):
    """Every probe scan against every template: (genuine scores, impostor scores)."""
    T, t_people, probes = make_templates(emb, people, n_enroll, seed)
    S = emb[probes] @ T.T # cosine similarity, (n_probes, n_people)
    same = people[probes][:, None] == t_people[None, :]
    return S[same], S[~same]


def threshold_at_far(impostor, far):
    """Lowest threshold that accepts at most a `far` share of the impostor scores."""
    s = np.sort(impostor)[::-1] # highest first
    n_allowed = int(np.floor(far * len(s))) # impostors allowed through
    return float(np.nextafter(s[min(n_allowed, len(s) - 1)], np.inf))


def tar_at_far(genuine, impostor, far):
    thr = threshold_at_far(impostor, far)
    return float((genuine >= thr).mean()), thr


def eer(genuine, impostor):
    """Equal error rate and the threshold where it happens."""
    thr = np.sort(np.concatenate([genuine, impostor]))
    imp, gen = np.sort(impostor), np.sort(genuine)
    far = 1 - np.searchsorted(imp, thr, side="left") / len(imp) # impostors >= threshold
    frr = np.searchsorted(gen, thr, side="left") / len(gen) # genuines  <  threshold
    i = int(np.argmin(np.abs(far - frr)))
    return float((far[i] + frr[i]) / 2), float(thr[i])


# The decision rule (same as gallery.FaceID.identify, for many scans at once)

def decide(emb, live_prob, templates, t_people, match_thr, live_thr):
    """
    Returns (who, best_score) per scan: who = the accepted person,
    -1 = rejected (no template close enough), -2 = rejected as a spoof.
    """
    S = emb @ templates.T
    best = S.argmax(1)
    best_score = S[np.arange(len(emb)), best]
    who = np.where(best_score >= match_thr, t_people[best], -1)
    who = np.where(live_prob >= live_thr, who, -2)
    return who, best_score


# Helpers

def pick_live_threshold(live_prob, is_live):
    """Reject every validation spoof (never go below 0.5)."""
    spoof = live_prob[~is_live]
    thr = max(0.5, float(np.nextafter(spoof.max(), np.inf))) if len(spoof) else 0.5
    return min(thr, 0.999)


def pct(x):
    return f"{100 * x:.3g}%" # 3 sig figs


def plot_scores(genuine, impostor, thr, far, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, ink2, surface = "#0b0b0b", "#52514e", "#fcfcfb"
    fig, ax = plt.subplots(figsize=(7, 3.6), dpi=150, facecolor=surface)
    ax.set_facecolor(surface)
    bins = np.linspace(min(impostor.min(), genuine.min()), 1, 80)
    n_imp, _, _ = ax.hist(impostor, bins, color="#eb6834", alpha=0.85,
                          label=f"different person ({len(impostor):,})")
    ax.hist(genuine, bins, color="#2a78d6", alpha=0.85, label=f"same person ({len(genuine):,})")
    ax.set_yscale("log")
    top = n_imp.max() * 30 # headroom above the bars for the labels
    ax.set_ylim(0.7, top)
    ax.axvline(thr, color=ink, lw=1.5, ls="--")
    ax.text(thr, top * 0.6, f" threshold {thr:.3f}\n (FAR {far:g} on val)",
            va="top", ha="left", color=ink, fontsize=8)
    ax.set_xlabel("cosine similarity to the template", color=ink2)
    ax.set_ylabel("attempts (log scale)", color=ink2)
    ax.set_title("Test people: same-person vs different-person scores", color=ink, fontsize=10, loc="left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c9c8c2")
    ax.tick_params(colors=ink2, labelsize=8)
    ax.legend(frameon=False, fontsize=8, labelcolor=ink, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, facecolor=surface)
    plt.close(fig)


# The report

def main(argv=None):
    import torch
    from .cache import load_cache
    from .model import embed_clouds, load_checkpoint, pick_device

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", help="folder written by faceid.train (has model.pt)")
    ap.add_argument("--cache", help="preprocessed cache (default: the one the model was trained on)")
    ap.add_argument("--far", type=float, default=1e-4,
                    help="target false accept rate per comparison, chosen on val people (default 1e-4 = 0.01%%)")
    ap.add_argument("--views", type=int, default=4, help="random point subsets averaged per scan")
    ap.add_argument("--device")
    a = ap.parse_args(argv)

    run = Path(a.run)
    device = pick_device(a.device)
    model, ckpt = load_checkpoint(run / "model.pt", device)
    cache = load_cache(a.cache or ckpt["cache"])
    n_points = ckpt["n_points"]
    lines = []

    def say(s=""):
        print(s)
        lines.append(s)

    say(f"Model {run / 'model.pt'}: {ckpt['config']}, {n_points} points, epoch {ckpt.get('epoch')}")
    say(f"Cache {a.cache or ckpt['cache']}, {a.views} views per scan, {N_ENROLL} enrollment scans per person")

    # embed every val and test scan once
    out = {}
    for split in ("val", "test"):
        m = cache["split"] == split
        torch.manual_seed(0)
        emb, live_prob = embed_clouds(model, cache["points"][m], n_points, a.views, device=device)
        out[split] = dict(emb=emb, live_prob=live_prob, people=cache["identity"][m],
                          is_live=cache["kind"][m] == "live", kind=cache["kind"][m])

    # 1. thresholds from the validation people
    v = out["val"]
    vg, vi = verification_scores(v["emb"][v["is_live"]], v["people"][v["is_live"]])
    match_thr = threshold_at_far(vi, a.far)
    live_thr = pick_live_threshold(v["live_prob"], v["is_live"])
    n_val_people = len(np.unique(v["people"]))
    say()
    say(f"== Thresholds, chosen on {n_val_people} validation people ==")
    say(f"match threshold {match_thr:.4f}: accepts {pct(a.far)} of {len(vi):,} val impostor attempts "
        f"(at most {int(a.far * len(vi))} of them)")
    if a.far * len(vi) < 1:
        say("  note: fewer impostor attempts than 1/FAR, so this threshold is simply 'above every val "
            "impostor'; generate more people to measure FARs this small")
    v_spoof = v["live_prob"][~v["is_live"]]
    say(f"liveness threshold {live_thr:.4f}: rejects {int((v_spoof < live_thr).sum())} of {len(v_spoof)} val "
        f"spoofs, {pct((v['live_prob'][v['is_live']] < live_thr).mean())} of val live scans")
    (run / "thresholds.json").write_text(json.dumps(
        {"match": match_thr, "live": live_thr, "far_target": a.far, "views": a.views,
         "n_enroll": N_ENROLL, "chosen_on": f"val ({n_val_people} people)"}, indent=2))

    # 2. 1:1 verification on the test people
    t = out["test"]
    live = t["is_live"]
    tg, ti = verification_scores(t["emb"][live], t["people"][live])
    e, e_thr = eer(tg, ti)
    n_test_people = len(np.unique(t["people"]))
    say()
    say(f"== 1:1 verification, {n_test_people} test people "
        f"({len(tg):,} genuine, {len(ti):,} impostor attempts) ==")
    say(f"at the val threshold:  FAR {pct((ti >= match_thr).mean())} "
        f"({int((ti >= match_thr).sum())} false accepts)   FRR {pct((tg < match_thr).mean())}")
    say(f"EER {pct(e)}  (threshold {e_thr:.3f})")
    for far in (1e-2, 1e-3, 1e-4):
        if far * len(ti) >= 1:
            tar, thr = tar_at_far(tg, ti, far)
            say(f"TAR at FAR {far:g}: {pct(tar)}  (threshold {thr:.3f}, measured on test itself)")
    say(f"lowest genuine score {tg.min():.3f}, highest impostor score {ti.max():.3f}")

    # 3. liveness on the test people
    spoof = ~live
    passed = t["live_prob"][spoof] >= live_thr
    say()
    say(f"== Liveness, test people ({int(spoof.sum())} printed-photo spoofs, {int(live.sum())} live scans) ==")
    say(f"APCER (spoofs passed as live)  {pct(passed.mean())}")
    for kind in ("print_flat", "print_curved"):
        k = t["kind"] == kind
        if k.any():
            say(f"   {kind:13s} {pct((t['live_prob'][k] >= live_thr).mean())} of {int(k.sum())}")
    say(f"BPCER (live faces rejected)    {pct((t['live_prob'][live] < live_thr).mean())}")

    # 4. 1:N open-set identification: half the test people enrolled, half strangers
    rng = np.random.default_rng(1)
    persons = rng.permutation(np.unique(t["people"]))
    enrolled = persons[: len(persons) // 2]
    is_enr = np.isin(t["people"], enrolled)
    enr_live = np.nonzero(live & is_enr)[0]
    T, t_people, probe_local = make_templates(t["emb"][enr_live], t["people"][enr_live])
    probes = enr_live[probe_local]
    who, _ = decide(t["emb"], t["live_prob"], T, t_people, match_thr, live_thr)

    truth = t["people"]
    pw = who[probes]
    stranger = np.nonzero(live & ~is_enr)[0]
    spoofs_enr = np.nonzero(spoof & is_enr)[0]
    say()
    say("== 1:N identification, the full decision (liveness, then best match >= threshold) ==")
    say(f"{len(enrolled)} test people enrolled ({N_ENROLL} scans each), {len(persons) - len(enrolled)} test people "
        f"are strangers")
    say(f"enrolled people, {len(probes)} new scans:")
    say(f"   accepted as the right person   {pct((pw == truth[probes]).mean())}")
    say(f"   accepted as the WRONG person   {pct(((pw >= 0) & (pw != truth[probes])).mean())}  "
        f"({int(((pw >= 0) & (pw != truth[probes])).sum())})")
    say(f"   rejected (no match / spoof)    {pct((pw == -1).mean())} / {pct((pw == -2).mean())}")
    say(f"strangers, {len(stranger)} scans:")
    say(f"   accepted as somebody (FPIR)    {pct((who[stranger] >= 0).mean())}  ({int((who[stranger] >= 0).sum())})")
    say(f"printed photos of enrolled people, {len(spoofs_enr)} attacks:")
    say(f"   accepted                       {pct((who[spoofs_enr] >= 0).mean())}  "
        f"({int((who[spoofs_enr] >= 0).sum())})")
    say(f"   (FPIR grows with the number of people enrolled: roughly gallery size x FAR. "
        f"With {len(enrolled)} people and FAR {a.far:g}: ~{pct(len(enrolled) * a.far)})")

    # closed-set accuracy, no threshold: is the best match the right person?
    S = t["emb"][probes] @ T.T
    say(f"rank-1 (best match is the right person, ignoring the threshold): "
        f"{pct((t_people[S.argmax(1)] == truth[probes]).mean())}")

    plot_scores(tg, ti, match_thr, a.far, run / "scores.png")
    (run / "report.txt").write_text("\n".join(lines) + "\n")
    print(f"\nWrote {run / 'thresholds.json'}, {run / 'report.txt'}, {run / 'scores.png'}")


if __name__ == "__main__":
    main()
