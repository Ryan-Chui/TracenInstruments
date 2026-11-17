"""
Step 2: train FaceNet on the training faces

What an epoch does:
    every training scan once, in random order.
    each scan = 1024 random points of its 4096 cached ones, then noise added
    (small rotation, 0.5 mm jitter, a random hole, random point dropout)

    identity loss (CosFace) on the live scans: pulls a person's scans together
                                               and pushes different people apart
    liveness loss (binary cross-entropy) on all scans: real face = 1, photo = 0

Every --eval-every epochs, on the validation faces (the 54 people never trained on):
    TAR at FAR 0.1%: set the match threshold so only 0.1% of different-person
    comparisons pass, then count how many same-person comparisons pass.
    The epoch with the best value is saved as model.pt. Picking the checkpoint by
    performance on unseen people at a low false-accept rate is what we care
    about; training accuracy on the 390 training people says little about it.

Writes: <out>/model.pt (best on val)
        <out>/last.pt
        <out>/log.csv
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from facesim.datasets import augment
from .cache import load_cache
from .evaluate import eer, tar_at_far, verification_scores
from .model import CosFaceLoss, FaceNet, embed_clouds, pick_device, save_checkpoint


class CachedScans(Dataset):
    """Scans of one split from the cache. ds[i] -> (points (n_points, 3), person label, live 1/0)."""

    def __init__(self, cache, split, n_points=1024, augment=False, seed=0):
        self.idx = np.nonzero(cache["split"] == split)[0]
        self.points = cache["points"]
        ids = cache["identity"][self.idx]
        self.people = np.unique(ids)
        self.labels = np.searchsorted(self.people, ids) # person number -> 0..n_people-1
        self.live = (cache["kind"][self.idx] == "live").astype(np.float32)
        self.n_people = len(self.people)
        self.n_points, self.augment = n_points, augment
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        p = self.points[self.idx[i]].astype(np.float32) # (4096, 3)
        p = p[self.rng.choice(len(p), self.n_points, replace=False)]
        if self.augment:
            p = augment(p, self.rng)
        return torch.from_numpy(p), int(self.labels[i]), self.live[i]


def validate(model, cache, n_points, device, split="val"):
    """Quick version of evaluate.py on one split: 1 view per scan, live scans only for matching."""
    m = cache["split"] == split
    emb, live_prob = embed_clouds(model, cache["points"][m], n_points, views=1, device=device)
    live = cache["kind"][m] == "live"
    gen, imp = verification_scores(emb[live], cache["identity"][m][live])
    tar, _ = tar_at_far(gen, imp, 1e-3)
    return {"val_tar@far1e-3": tar, "val_eer": eer(gen, imp)[0],
            "val_live_acc": float(((live_prob >= 0.5) == live).mean())}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data", help="dataset folder (has cache_4096.npz from faceid.cache)")
    ap.add_argument("--cache", help="cache file (default: <data>/cache_4096.npz)")
    ap.add_argument("--out", default="runs/edgeconv")
    ap.add_argument("--encoder", choices=["edgeconv", "pointnet"], default="edgeconv")
    ap.add_argument("--points", type=int, default=1024, help="points per scan fed to the network")
    ap.add_argument("--k", type=int, default=16, help="neighbors per point (EdgeConv)")
    ap.add_argument("--emb-dim", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--margin", type=float, default=0.35, help="CosFace margin m")
    ap.add_argument("--scale", type=float, default=30.0, help="CosFace scale s")
    ap.add_argument("--live-weight", type=float, default=1.0, help="weight of the liveness loss")
    ap.add_argument("--eval-every", type=int, default=2)
    ap.add_argument("--device", help="cuda | mps | cpu (default: best available)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    torch.manual_seed(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    device = pick_device(a.device)
    cache_path = a.cache or str(Path(a.data) / "cache_4096.npz")
    cache = load_cache(cache_path)

    train_set = CachedScans(cache, "train", a.points, augment=True, seed=a.seed)
    loader = DataLoader(train_set, batch_size=a.batch, shuffle=True, drop_last=True)
    n_live = int(train_set.live.sum())
    n_spoof = len(train_set) - n_live
    print(f"{len(train_set)} training scans ({n_live} live, {n_spoof} spoof) of {train_set.n_people} people; "
          f"device {device}")

    model = FaceNet(a.encoder, a.emb_dim, a.k).to(device)
    id_loss_fn = CosFaceLoss(a.emb_dim, train_set.n_people, a.scale, a.margin).to(device)
    params = list(model.parameters()) + list(id_loss_fn.parameters())
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=1e-4)
    # warm up over the first 5% of steps, then cosine decay to ~0
    # (min/max only matters for small test runs)
    total_steps = a.epochs * len(loader)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=total_steps,
                                                pct_start=min(0.5, max(0.05, 2 / total_steps)))
    # spoofs are ~1 in 10 scans
    # weight the classes equally in the liveness loss
    pos_weight = torch.tensor(n_spoof / max(n_live, 1), device=device)
    print(f"{a.encoder}: {sum(p.numel() for p in model.parameters()):,} weights")

    extra = {"cache": cache_path, "args": vars(a), "people_trained_on": train_set.people.tolist()}
    best = (-1.0, 0.0)
    log_f = open(out / "log.csv", "w", newline="")
    log = None
    for epoch in range(1, a.epochs + 1):
        model.train()
        id_loss_fn.train()
        t0 = time.time()
        sums = {"id_loss": 0.0, "live_loss": 0.0, "train_acc": 0.0, "n_live": 0, "n": 0}
        for pts, labels, live in loader:
            pts, labels, live = pts.to(device), labels.to(device), live.to(device)
            emb, live_logit = model(pts)
            real = live > 0.5
            id_loss, cos = id_loss_fn(emb[real], labels[real])
            live_loss = F.binary_cross_entropy_with_logits(live_logit, live, pos_weight=pos_weight)
            loss = id_loss + a.live_weight * live_loss

            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()

            sums["id_loss"] += id_loss.item() * int(real.sum())
            sums["train_acc"] += (cos.argmax(1) == labels[real]).sum().item()
            sums["n_live"] += int(real.sum())
            sums["live_loss"] += live_loss.item() * len(live)
            sums["n"] += len(live)

        row = {"epoch": epoch, "id_loss": sums["id_loss"] / sums["n_live"],
               "train_acc": sums["train_acc"] / sums["n_live"], "live_loss": sums["live_loss"] / sums["n"],
               "lr": sched.get_last_lr()[0], "seconds": time.time() - t0}
        msg = (f"epoch {epoch:3d} | id loss {row['id_loss']:.3f} | train acc {row['train_acc']:.1%} | "
               f"live loss {row['live_loss']:.4f} | {row['seconds']:.0f} s")

        if epoch % a.eval_every == 0 or epoch == a.epochs:
            row.update(validate(model, cache, a.points, device))
            msg += (f" || val TAR@FAR0.1% {row['val_tar@far1e-3']:.1%} | val EER {row['val_eer']:.2%} | "
                    f"val live acc {row['val_live_acc']:.1%}")
            key = (row["val_tar@far1e-3"], -row["val_eer"])
            if key > best:
                best = key
                save_checkpoint(out / "model.pt", model, a.points, {**extra, "epoch": epoch, "val": row})
                msg += "  * saved"
        print(msg, flush=True)

        if log is None: # columns from the first row that has validation numbers
            log = csv.DictWriter(log_f, fieldnames=list(row) + [k for k in ("val_tar@far1e-3", "val_eer",
                                 "val_live_acc") if k not in row])
            log.writeheader()
        log.writerow(row)
        log_f.flush()

    save_checkpoint(out / "last.pt", model, a.points, {**extra, "epoch": a.epochs})
    print(f"Best val TAR@FAR0.1% {best[0]:.1%} -> {out / 'model.pt'}. Next: python -m faceid.evaluate {out}")


if __name__ == "__main__":
    main()
