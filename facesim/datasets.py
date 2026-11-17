"""
PyTorch datasets for synthetic scans (needs `pip install torch`).

The model:

    sample = {"points": FloatTensor (n_points, 3),   # nose-centered, / 0.1 m
              "identity": LongTensor (),            # person index within this dataset
              "live": FloatTensor ()}               # 1 = real face, 0 = spoof

    from facesim.datasets import SynthScans, FR3DKScans
    train = SynthScans("data/synth", split="train", augment=True)
    test  = FR3DKScans("data/FR3DK")
    loader = torch.utils.data.DataLoader(train, batch_size=32, shuffle=True, num_workers=4)
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .pointcloud import load_scan, preprocess_face, read_pcd

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:  # the numpy helpers still work without torch
    torch = None

    class Dataset:  # type: ignore
        pass


def augment(points: np.ndarray, rng: np.random.Generator, max_rot_deg=10.0, jitter=0.005,
            occlusion_p=0.3, dropout_p=0.2) -> np.ndarray:
    """
    Train-time augmentation on a preprocessed cloud (units: 0.1 m).

    small rotation about the nose, per-point jitter (0.005 = 0.5 mm),
    a random spherical occlusion (hand / hair / glasses), random point dropout
    (points are resampled so the count stays fixed).
    """
    p = points.copy()
    a = np.radians(rng.uniform(-max_rot_deg, max_rot_deg, 3))
    cx, cy, cz = np.cos(a); sx, sy, sz = np.sin(a)
    R = (np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]]) @ np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
         @ np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]]))
    p = p @ R.T
    p += rng.normal(0, jitter, p.shape)
    n = len(p)
    keep = np.ones(n, bool)
    if rng.random() < occlusion_p:
        c = p[rng.integers(n)]
        keep &= np.linalg.norm(p - c, axis=1) > rng.uniform(0.15, 0.45)
    if rng.random() < dropout_p:
        keep &= rng.random(n) > rng.uniform(0.0, 0.5)
    if keep.sum() < n // 4:
        keep[:] = True
    idx = np.nonzero(keep)[0]
    return p[rng.choice(idx, n, replace=len(idx) < n)].astype(np.float32)


def _to_sample(points, identity, live):
    if torch is None:
        return {"points": points, "identity": identity, "live": float(live)}
    return {"points": torch.from_numpy(points), "identity": torch.tensor(identity, dtype=torch.long),
            "live": torch.tensor(float(live))}


class SynthScans(Dataset):
    """
    Scans written by `python -m facesim.generate`.

    split: 'train' | 'val' | 'test' | None (all); splits are by identity.
    kinds: which scan kinds to include, e.g. ('live',) or ('live','print_flat','print_curved').
    """

    def __init__(self, root, split="train", n_points=2048, augment=False, kinds=None, sampling="random", seed=0):
        self.root = Path(root)
        with open(self.root / "index.csv") as f:
            rows = list(csv.DictReader(f))
        if split:
            rows = [r for r in rows if r["split"] == split]
        if kinds:
            rows = [r for r in rows if r["kind"] in kinds]
        self.rows = rows
        ids = sorted({int(r["identity"]) for r in rows})
        self.id_to_label = {k: i for i, k in enumerate(ids)}
        self.n_identities = len(ids)
        self.n_points, self.augment, self.sampling = n_points, augment, sampling
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        d = load_scan(self.root / r["path"])
        pts = preprocess_face(d["points"], self.n_points, sampling=self.sampling, rng=self.rng)
        if self.augment:
            pts = augment(pts, self.rng)
        return _to_sample(pts, self.id_to_label[int(r["identity"])], r["kind"] == "live")
