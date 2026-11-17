"""
Step 1: preprocess every scan once and keep the results in a file.

Why: preprocess_face (outlier removal, nose tip, 9.5 cm crop) takes ~60 ms per
scan. Doing it again every epoch would make training mostly waiting on NumPy.
Therefore, run it once, keep 4096 face points per scan (float16, ~270 MB for
synth_v1), and the training loop just picks a random 1024 of those each time.

    data/synth_v1/ (11,000 .npz scans + index.csv)
        |  python -m faceid.cache data/synth_v1
        V
    data/synth_v1/cache_4096.npz
        points, identity, kind, split, path

Run from the generate_data folder:
    python -m faceid.cache data/synth_v1 --workers 8
"""
from __future__ import annotations

import argparse
import csv
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from facesim.pointcloud import load_scan, preprocess_face

CACHE_POINTS = 4096 # points kept per scan; training samples 1024 of them


def _preprocess_one(job):
    """Load one .npz and run the shared preprocessing. Runs in a worker process."""
    i, path, seed = job
    try:
        raw = load_scan(path)["points"]
        # one seed per scan, so the cache is identical however many workers you use
        rng = np.random.default_rng([seed, i])
        pts, info = preprocess_face(raw, CACHE_POINTS, rng=rng, return_info=True)
        return i, pts.astype(np.float16), info["n_face_points"], ""
    except Exception as e:  # a broken scan shouldn't kill an hour-long run
        return i, None, 0, f"{type(e).__name__}: {e}"


def build_cache(root, out=None, workers=4, seed=0):
    root = Path(root)
    out = Path(out) if out else root / f"cache_{CACHE_POINTS}.npz"
    with open(root / "index.csv") as f:
        rows = list(csv.DictReader(f))
    print(f"{len(rows)} scans in {root / 'index.csv'} -> {out}")

    points = np.zeros((len(rows), CACHE_POINTS, 3), np.float16)
    n_face = np.zeros(len(rows), np.int32)
    ok = np.zeros(len(rows), bool)
    jobs = [(i, str(root / r["path"]), seed) for i, r in enumerate(rows)]

    t0 = time.time()
    with Pool(workers) as pool:
        # imap_unordered hands results back as soon as any worker finishes one
        for done, (i, pts, nf, err) in enumerate(pool.imap_unordered(_preprocess_one, jobs, chunksize=16), 1):
            if err:
                print(f"  skipped {rows[i]['path']}: {err}")
            else:
                points[i], n_face[i], ok[i] = pts, nf, True
            if done % 500 == 0 or done == len(rows):
                rate = done / (time.time() - t0)
                print(f"  {done}/{len(rows)}  ({rate:.0f} scans/s, {(len(rows) - done) / rate:.0f} s left)")

    keep = np.nonzero(ok)[0]
    np.savez(out,
             points=points[keep],
             identity=np.array([int(rows[i]["identity"]) for i in keep], np.int32),
             kind=np.array([rows[i]["kind"] for i in keep]),
             split=np.array([rows[i]["split"] for i in keep]),
             path=np.array([rows[i]["path"] for i in keep]),
             n_face_points=n_face[keep])
    print(f"Saved {len(keep)} scans ({len(rows) - len(keep)} skipped) to {out} "
          f"in {time.time() - t0:.0f} s")
    return out


def load_cache(path) -> dict:
    """Read the cache back as a dict of NumPy arrays (strings come back as str arrays)."""
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="dataset folder written by facesim.generate (has index.csv)")
    ap.add_argument("--out", help="output file (default: <root>/cache_4096.npz)")
    ap.add_argument("--workers", type=int, default=4, help="CPU processes to use")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    build_cache(a.root, a.out, a.workers, a.seed)
