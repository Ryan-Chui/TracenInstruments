"""
Step 4: the piece the system and later the dashboard call.

    from faceid.gallery import FaceID
    fid = FaceID.load("runs/edgeconv")
    fid.enroll("vaughn", [cloud1, cloud2, cloud3, cloud4, cloud5])
    fid.identify(cloud) # 1:N  "who is this?" -> Result
    fid.verify("vaughn", cloud) # 1:1  "is this vaigjn?" -> Result
    fid.save_gallery("gallery.npz"); fid.load_gallery("gallery.npz")

The scanner gives a cloud: an (N, 3) array in meters, camera frame
(z away from the camera), e.g. load_scan(path)["points"] for a synthetic scan or
the triangulated cloud from calibration&scanning. FaceID runs the same
preprocess_face the training data went through.

The decision, in order:
  1. quality (enough points)
  2. liveness (in the liveness threshold)
  3. match (best cosine similarity >= match threshold?)
  4. accept, and say who

1:N: the match threshold is set so a single impostor comparison passes with
probability FAR. "Who is this?" compares against everybody enrolled, so a
stranger gets N chances (false identifications ~ N x FAR). For large galleries,
either set a smaller FAR in evaluate.py or use verify() w/ a claimed identity.
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from facesim.pointcloud import load_scan, preprocess_face
from .model import embed_clouds, load_checkpoint, pick_device

MIN_FACE_POINTS = 3000 # fewer points than this in the face crop -> ask for a rescan
ENROLL_AGREEMENT = 0.5 # enrollment scans must look like each other at least this much


@dataclass
class Result:
    decision: str # "accept" / "reject_unknown" / "reject_spoof" / "rescan"
    name: str | None = None # who, if accepted
    score: float = 0.0 # cosine similarity to the best (or claimed) template
    live: float = 0.0 # liveness probability
    closest: str | None = None # closest enrolled person even when rejected (for logs)

    def __str__(self):
        who = self.name or f"(closest: {self.closest})"
        return f"{self.decision:15s} {who:22s} score {self.score:.3f}  live {self.live:.3f}"


class FaceID:
    def __init__(self, model, n_points, match_threshold, live_threshold, views=4, device=None):
        self.model, self.n_points, self.views = model, n_points, views
        self.match_threshold, self.live_threshold = match_threshold, live_threshold
        self.device = device or next(model.parameters()).device
        self.rng = np.random.default_rng(0)
        self.names: list[str] = []
        self.templates = np.zeros((0, model.config["emb_dim"]), np.float32)
        self.counts: list[int] = []

    @classmethod
    def load(cls, run_dir, device=None):
        run_dir = Path(run_dir)
        model, ckpt = load_checkpoint(run_dir / "model.pt", pick_device(device))
        thr = json.loads((run_dir / "thresholds.json").read_text())   # written by faceid.evaluate
        return cls(model, ckpt["n_points"], thr["match"], thr["live"], thr.get("views", 4))

    # scan -> embedding

    def embed(self, cloud):
        """Raw cloud (meters) -> (embedding (D,), live probability, face points) or None if too few points."""
        try:
            pts, info = preprocess_face(cloud, self.n_points * self.views, rng=self.rng, return_info=True)
        except ValueError:
            return None
        if info["n_face_points"] < MIN_FACE_POINTS:
            return None
        emb, live = embed_clouds(self.model, pts[None], self.n_points, self.views, device=self.device)
        return emb[0], float(live[0]), info["n_face_points"]

    # enrollment

    def enroll(self, name, clouds):
        """Average several scans into the person's template. Returns how many scans were used."""
        embs, which = [], [] # usable embeddings and their position in `clouds`
        for n, c in enumerate(clouds):
            r = self.embed(c)
            if r is None:
                print(f"  enroll {name}: skipped scan {n}, too few face points")
            elif r[1] < self.live_threshold:
                print(f"  enroll {name}: skipped scan {n}, it looks like a spoof (live {r[1]:.2f})")
            else:
                embs.append(r[0])
                which.append(n)
        if not embs:
            raise ValueError(f"no usable scans to enroll {name}")
        # A bad capture would poison the template. With 3+ scans, drop any scan whose
        # median similarity to the person's other scans is below ENROLL_AGREEMENT
        # (same-person scans usually score ~0.9, strangers ~0).
        if len(embs) >= 3:
            S = np.array(embs) @ np.array(embs).T
            np.fill_diagonal(S, np.nan)
            agree = np.nanmedian(S, axis=1) >= ENROLL_AGREEMENT
            for i in np.nonzero(~agree)[0]:
                print(f"  enroll {name}: skipped scan {which[i]}, it doesn't match the person's other scans")
            if agree.sum() < 2:
                raise ValueError(f"{name}'s enrollment scans don't agree with each other; rescan")
            embs = [e for e, ok in zip(embs, agree) if ok]
        t = np.mean(embs, axis=0)
        t /= np.linalg.norm(t)
        if name in self.names: # re-enrolling replaces the old template
            i = self.names.index(name)
            self.templates[i], self.counts[i] = t, len(embs)
        else:
            self.names.append(name)
            self.templates = np.vstack([self.templates, t[None]])
            self.counts.append(len(embs))
        return len(embs)

    # decisions

    def identify(self, cloud) -> Result:
        """1:N: is this one of the enrolled people, and which one?"""
        r = self.embed(cloud)
        if r is None:
            return Result("rescan")
        emb, live, _ = r
        if not self.names:
            return Result("reject_unknown", live=live)
        scores = self.templates @ emb
        best = int(scores.argmax())
        res = Result("accept", score=float(scores[best]), live=live, closest=self.names[best])
        if live < self.live_threshold:
            res.decision = "reject_spoof"
        elif scores[best] < self.match_threshold:
            res.decision = "reject_unknown"
        else:
            res.name = self.names[best]
        return res

    def verify(self, name, cloud) -> Result:
        """1:1: is this the person they claim to be?"""
        if name not in self.names:
            raise KeyError(f"{name} is not enrolled")
        r = self.embed(cloud)
        if r is None:
            return Result("rescan")
        emb, live, _ = r
        score = float(self.templates[self.names.index(name)] @ emb)
        res = Result("accept", score=score, live=live, closest=name)
        if live < self.live_threshold:
            res.decision = "reject_spoof"
        elif score < self.match_threshold:
            res.decision = "reject_unknown"
        else:
            res.name = name
        return res

    # saving the enrolled people

    def save_gallery(self, path):
        np.savez(path, names=np.array(self.names), templates=self.templates, counts=np.array(self.counts))

    def load_gallery(self, path):
        with np.load(path, allow_pickle=False) as z:
            self.names = [str(n) for n in z["names"]]
            self.templates = z["templates"].astype(np.float32)
            self.counts = z["counts"].tolist()


def demo(run, data, n_enrolled=5, n_strangers=3):
    """Enroll a few synthetic TEST people (never trained on) and run every kind of attempt."""
    fid = FaceID.load(run)
    data = Path(data)
    with open(data / "index.csv") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == "test"]
    people = sorted({int(r["identity"]) for r in rows})
    enrolled, strangers = people[:n_enrolled], people[n_enrolled:n_enrolled + n_strangers]

    def scans(person, live=True):
        return [data / r["path"] for r in rows if int(r["identity"]) == person and (r["kind"] == "live") == live]

    print(f"Thresholds: match {fid.match_threshold:.3f}, liveness {fid.live_threshold:.3f}\n")
    for p in enrolled:
        n = fid.enroll(f"person_{p}", [load_scan(s)["points"] for s in scans(p)[:5]])
        print(f"enrolled person_{p} from {n} scans")

    print("\nidentify: a new scan of each enrolled person (expect accept, right name)")
    for p in enrolled:
        print(f"  person_{p:<4d} -> {fid.identify(load_scan(scans(p)[10])['points'])}")
    print("identify: people who were never enrolled (expect reject_unknown)")
    for p in strangers:
        print(f"  person_{p:<4d} -> {fid.identify(load_scan(scans(p)[0])['points'])}")
    print("identify: printed photos of enrolled people (expect reject_spoof or reject_unknown)")
    for p in enrolled[:3]:
        sp = scans(p, live=False)
        if sp:
            print(f"  photo of person_{p:<4d} -> {fid.identify(load_scan(sp[0])['points'])}")
    print("verify: a stranger claiming to be an enrolled person (expect reject_unknown)")
    print(f"  person_{strangers[0]} says 'I am person_{enrolled[0]}' -> "
          f"{fid.verify(f'person_{enrolled[0]}', load_scan(scans(strangers[0])[1])['points'])}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", help="folder with model.pt and thresholds.json")
    ap.add_argument("--data", default="data/synth_v1", help="synthetic dataset for the demo")
    a = ap.parse_args()
    demo(a.run, a.data)
