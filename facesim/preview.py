"""Quick-look images of point clouds (synthetic .npz, FR3DK .pcd, or .ply-free arrays).

    python -m facesim.preview data/synth/id_00000/scan_000.npz data/synth/id_00001/scan_002.npz -o look.png
    python -m facesim.preview path/to/FR3DK/albert_e/0_yaw/*.pcd --max 6 -o fr3dk.png

Each panel shows the cloud from the front, colored by depth (near = bright),
and from the side, so holes, shadows and noise are easy to see.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def load_any(path):
    path = Path(path)
    if path.suffix == ".npz":
        from .pointcloud import load_scan
        d = load_scan(path)
        meta = d.get("meta", {})
        title = f"{path.parent.name}/{path.stem}  {meta.get('kind', '')}"
        if "yaw_deg" in meta:
            title += f"\nyaw {meta['yaw_deg']:.0f}°  {meta['nose_true'][2]:.2f} m  amb {meta.get('ambient', 0):.2f}"
        return d["points"], title
    if path.suffix in (".pcd", ".pcl"):
        from .pointcloud import read_pcd
        pts, _, _ = read_pcd(path)
        return pts, f"{path.parent.parent.name}/{path.parent.name}/{path.name}"
    raise ValueError(f"unsupported file: {path}")


def panel(ax_front, ax_side, pts, title, crop=0.12, toward="-z"):
    import matplotlib.pyplot as plt  # noqa: F401
    from .pointcloud import find_nose_tip
    pts = np.asarray(pts, np.float64)
    nose = find_nose_tip(pts, toward=toward)
    p = pts[np.linalg.norm(pts - nose, axis=1) < crop] - nose
    depth = p[:, 2] if toward == "-z" else -p[:, 2]
    order = np.argsort(-depth)
    ax_front.scatter(p[order, 0] * 100, -p[order, 1] * 100, c=-depth[order], s=0.3, cmap="viridis",
                     vmin=-0.08, vmax=0.0, linewidths=0)
    ax_front.set_aspect("equal"); ax_front.set_title(title + "\nas the camera sees it (bright = closer)", fontsize=7)
    ax_front.axis("off")
    ax_side.scatter(-depth * 100, -p[:, 1] * 100, s=0.2, c="k", alpha=0.3, linewidths=0)
    ax_side.set_aspect("equal"); ax_side.set_xlim(-12, 1); ax_side.axis("off")
    ax_side.set_title(f"same scan seen from the side\n(camera is to the right), {len(p)} pts", fontsize=7)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("-o", "--out", default="preview.png")
    ap.add_argument("--max", type=int, default=8)
    ap.add_argument("--crop", type=float, default=0.12, help="radius around the nose tip to show (m)")
    args = ap.parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    files = [f for f in args.files if Path(f).name != "identity.npz"][: args.max]
    n = len(files)
    fig, axes = plt.subplots(2, n, figsize=(2.6 * n, 5.6), squeeze=False,
                             gridspec_kw={"height_ratios": [1.25, 1]})
    for i, f in enumerate(files):
        pts, title = load_any(f)
        panel(axes[0, i], axes[1, i], pts, title, args.crop)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
