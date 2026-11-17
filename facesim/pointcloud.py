"""
Point cloud I/O and the preprocessing.

Run every source through `preprocess_face` so the network always sees the
same kind of input, whatever produced the cloud:

    raw cloud (meters, z away from the sensor)
      -> drop NaNs and stray points
      -> find the nose tip (closest well-supported point to the sensor,
         searched only where the points are face-sized crowds, so stray
         clumps of mis-decoded points in front of the face are skipped)
      -> keep a sphere around it (the face)
      -> center on the nose tip, resample to a fixed number of points
      -> divide by a FIXED scale (not per-face: face size helps identify people)
"""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

# I/O

def to_viewer_frame(points):
    """
    Camera frame (x right, y DOWN, z away from the camera) -> viewer frame
    (x right, y UP, z toward you). A 180-degree turn about x, not a mirror.

    3D viewers (MeshLab, CloudCompare, Open3D, macOS Preview) start out looking
    down -z with y up, so a face scanned head-on shows up head-on in this frame.
    In the camera frame they show the back of the face shell, upside down.
    """
    p = np.array(points, dtype=np.float32, copy=True)
    p[:, 1:] *= -1
    return p


def write_ply(path, points, colors=None, comment=None):
    """
    Binary little-endian PLY (opens in MeshLab, CloudCompare, Open3D).
    """
    pts = np.asarray(points, np.float32)
    n = len(pts)
    header = ["ply", "format binary_little_endian 1.0"]
    if comment:
        header += [f"comment {line}" for line in str(comment).splitlines()]
    header += [f"element vertex {n}", "property float x", "property float y", "property float z"]
    if colors is not None:
        header += ["property uchar red", "property uchar green", "property uchar blue"]
    header.append("end_header")
    with open(path, "wb") as f:
        f.write(("\n".join(header) + "\n").encode())
        if colors is None:
            f.write(pts.tobytes())
        else:
            c = np.asarray(colors)
            if c.ndim == 1:
                c = np.repeat(c[:, None], 3, 1)
            if c.dtype != np.uint8:
                c = np.clip(c * (255 if c.max() <= 1.0 else 1), 0, 255).astype(np.uint8)
            rec = np.empty(n, dtype=[("p", "<f4", 3), ("c", "u1", 3)])
            rec["p"], rec["c"] = pts, c
            f.write(rec.tobytes())


def read_pcd_header(path) -> dict:
    """
    Parse only the header of a PCD file (fast; gives POINTS/WIDTH/FIELDS...).
    """
    hdr = {}
    with open(path, "rb") as f:
        for raw in f:
            line = raw.decode("ascii", "replace").strip()
            if not line or line.startswith("#"):
                continue
            key, *vals = line.split()
            hdr[key.upper()] = vals
            if key.upper() == "DATA":
                break
    return hdr


def read_pcd(path):
    """
    Read a PCL .pcd file (ascii or binary).

    Returns (points (N,3) float32, colors (N,3) uint8 or None, header dict).
    Packed 'rgb'/'rgba' floats are unpacked into colors. NaN points are dropped.
    """
    path = Path(path)
    with open(path, "rb") as f:
        data = f.read()
    hdr, pos = {}, 0
    while True:
        end = data.index(b"\n", pos)
        line = data[pos:end].decode("ascii", "replace").strip()
        pos = end + 1
        if not line or line.startswith("#"):
            continue
        key, *vals = line.split()
        hdr[key.upper()] = vals
        if key.upper() == "DATA":
            break
    fields = [f.lower() for f in hdr["FIELDS"]]
    sizes = [int(s) for s in hdr.get("SIZE", ["4"] * len(fields))]
    types = hdr.get("TYPE", ["F"] * len(fields))
    counts = [int(c) for c in hdr.get("COUNT", ["1"] * len(fields))]
    npts = int(hdr.get("POINTS", hdr.get("WIDTH", ["0"]))[0])
    fmt = hdr["DATA"][0].lower()

    np_types = {("F", 4): "<f4", ("F", 8): "<f8", ("U", 1): "u1", ("U", 2): "<u2", ("U", 4): "<u4",
                ("I", 1): "i1", ("I", 2): "<i2", ("I", 4): "<i4"}
    cols = {}
    if fmt == "ascii":
        text = data[pos:].decode("ascii", "replace").split()
        width = sum(counts)
        arr = np.asarray(text[: (len(text) // width) * width], dtype=np.str_).astype(np.float64)
        arr = arr.reshape(-1, width)
        c = 0
        for fld, cnt, typ in zip(fields, counts, types):
            col = arr[:, c] if cnt == 1 else arr[:, c: c + cnt]
            if fld in ("rgb", "rgba"):
                cols[fld] = col.astype(np.float32).view(np.uint32) if typ == "F" else col.astype(np.uint32)
            else:
                cols[fld] = col
            c += cnt
    elif fmt == "binary":
        dt = np.dtype([(f"{fld}", np_types[(t, s)], (cnt,) if cnt > 1 else ())
                       for fld, s, t, cnt in zip(fields, sizes, types, counts)])
        rec = np.frombuffer(data, dtype=dt, count=npts, offset=pos)
        for fld, t in zip(fields, types):
            if fld in ("rgb", "rgba"):
                cols[fld] = rec[fld].view(np.uint32) if t == "F" else rec[fld].astype(np.uint32)
            else:
                cols[fld] = rec[fld].astype(np.float64)
    else:
        raise ValueError(f"{path}: PCD '{fmt}' data not supported (use Open3D for binary_compressed)")

    pts = np.stack([cols["x"], cols["y"], cols["z"]], 1).astype(np.float32)
    colors = None
    key = "rgb" if "rgb" in cols else ("rgba" if "rgba" in cols else None)
    if key:
        packed = cols[key].astype(np.uint32)
        colors = np.stack([(packed >> 16) & 255, (packed >> 8) & 255, packed & 255], 1).astype(np.uint8)
    good = np.isfinite(pts).all(1)
    return pts[good], (colors[good] if colors is not None else None), hdr


def write_pcd(path, points, colors=None):
    """
    ASCII PCD in the same layout FR3DK uses (x y z rgb).
    """
    pts = np.asarray(points, np.float32)
    fields = "x y z" + (" rgb" if colors is not None else "")
    n = len(pts)
    lines = ["# .PCD v0.7 - Point Cloud Data file format", "VERSION 0.7", f"FIELDS {fields}",
             "SIZE 4 4 4" + (" 4" if colors is not None else ""), "TYPE F F F" + (" F" if colors is not None else ""),
             "COUNT 1 1 1" + (" 1" if colors is not None else ""), f"WIDTH {n}", "HEIGHT 1",
             "VIEWPOINT 0 0 0 1 0 0 0", f"POINTS {n}", "DATA ascii"]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
        if colors is None:
            np.savetxt(f, pts, fmt="%.6f")
        else:
            c = np.asarray(colors, np.uint32)
            packed = ((c[:, 0] << 16) | (c[:, 1] << 8) | c[:, 2]).astype(np.uint32).view(np.float32)
            for p, rgb in zip(pts, packed):
                f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {repr(float(rgb))}\n")


def load_scan(path) -> dict:
    """
    Load a scan produced by facesim.generate (.npz) into a dict of arrays + meta.
    """
    import json
    with np.load(path, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    if "meta" in d:
        d["meta"] = json.loads(str(d["meta"]))
    return d


# Preprocessing

def remove_outliers(points, k=8, std_ratio=2.5):
    """
    Statistical outlier removal (like Open3D/PCL): drop points whose mean
    distance to their k neighbors is unusually large. Returns the kept mask.
    """
    if len(points) <= k:
        return np.ones(len(points), bool)
    d, _ = cKDTree(points).query(points, k=k + 1)
    md = d[:, 1:].mean(1)
    return md <= md.mean() + std_ratio * md.std()


def find_nose_tip(points, support_radius=0.01, min_support=20, toward="-z"):
    """
    Nose tip = the point closest to the sensor that has enough neighbors
    within `support_radius` (so isolated decoding errors are ignored).

    toward: '-z' if the sensor looks along +z (camera frame, FR3DK),
            '+z' if larger z is closer to the viewer (FLAME / mesh coordinates).
    """
    pts = np.asarray(points)
    depth = pts[:, 2] if toward == "-z" else -pts[:, 2]
    order = np.argsort(depth)
    tree = cKDTree(pts)
    for i in order[: max(2000, len(pts) // 20)]:
        if len(tree.query_ball_point(pts[i], support_radius)) >= min_support:
            # refine: average the closest few mm around the candidate
            near = tree.query_ball_point(pts[i], 0.004)
            return pts[near].mean(0)
    return pts[order[0]]


def face_sized_neighborhoods(points, radius=0.095, min_fraction=0.5, n_sample=3000, reach=0.02):
    """
    Mask of points that sit in a face-sized crowd: the sphere of `radius` around
    them holds at least `min_fraction` as many points as the most crowded sphere.

    Decoding errors can leave small clumps of points well in front of the face
    (seen in ~1.6% of synth_v1 scans, 10+ cm in front), and "closest point to the
    sensor" then picks a clump as the nose. A clump's sphere is nearly empty while
    the face's is full, so searching for the nose only in crowded places fixes it
    with no absolute point count (works at any scan density). Counted on a
    ~3000-point subsample, ~50 ms for 30k points.
    """
    pts = np.asarray(points)
    sub = pts[:: max(1, len(pts) // n_sample)]
    counts = cKDTree(sub).query_ball_point(sub, radius, return_length=True)
    crowded = sub[counts >= min_fraction * counts.max()]
    dist, _ = cKDTree(crowded).query(pts, distance_upper_bound=reach)
    return np.isfinite(dist)


def farthest_point_sample(points, n, rng=None):
    """
    Spread-out subset of n points (O(n*N); fine for N up to ~100k).
    """
    rng = rng or np.random.default_rng()
    N = len(points)
    if N <= n:
        return np.arange(N)
    idx = np.empty(n, np.int64)
    idx[0] = rng.integers(N)
    d = np.linalg.norm(points - points[idx[0]], axis=1)
    for i in range(1, n):
        idx[i] = np.argmax(d)
        d = np.minimum(d, np.linalg.norm(points - points[idx[i]], axis=1))
    return idx


def preprocess_face(points, n_points=2048, radius=0.095, scale=0.1, sampling="random",
                    toward="-z", clean=True, rng=None, return_info=False, face_check=True):
    """
    Turn any face scan into a fixed-size, nose-centered array for the model.

    points  : (N,3) meters
    radius  : face crop around the nose tip (FRGC-style papers use 80-100 mm)
    scale   : fixed divisor; 0.1 maps a 10 cm radius to ~1.0
    sampling: 'random' (fast, good with augmentation) or 'fps' (even coverage)
    face_check: look for the nose only in face-sized crowds of points
                (see face_sized_neighborhoods); False = the original closest-point rule
    Returns (n_points,3) float32 [, info dict].
    """
    rng = rng or np.random.default_rng()
    pts = np.asarray(points, np.float64)
    pts = pts[np.isfinite(pts).all(1)]
    if clean and len(pts) > 50:
        pts = pts[remove_outliers(pts)]
    search = pts[face_sized_neighborhoods(pts, radius)] if face_check and len(pts) > 50 else pts
    nose = find_nose_tip(search, toward=toward)
    pts = pts[np.linalg.norm(pts - nose, axis=1) <= radius] - nose
    n_face = len(pts)
    if n_face == 0:
        raise ValueError("no points left after cropping; check units (meters) and the 'toward' setting")
    if sampling == "fps":
        idx = farthest_point_sample(pts, n_points, rng)
    else:
        idx = rng.choice(n_face, n_points, replace=n_face < n_points)
    out = pts[idx]
    if len(out) < n_points:  # fps on a small cloud: pad by repeating
        out = out[rng.choice(len(out), n_points)]
    out = (out / scale).astype(np.float32)
    if return_info:
        return out, {"nose": nose, "n_face_points": n_face}
    return out


def depth_image(points, res=0.001, toward="-z"):
    """
    Orthographic depth image (front view) of a cloud, for quick looks.
    """
    pts = np.asarray(points)
    x, y = pts[:, 0], pts[:, 1]
    z = pts[:, 2] if toward == "-z" else -pts[:, 2]
    ix = ((x - x.min()) / res).astype(int)
    iy = ((y - y.min()) / res).astype(int)
    img = np.full((iy.max() + 1, ix.max() + 1), np.nan)
    order = np.argsort(-z)                 # far first, near overwrites
    img[iy[order], ix[order]] = z[order]
    return img
