"""
Generate a synthetic face-scan dataset: FLAME identities scanned by the virtual rig.

Example
-------
    python -m facesim.generate --flame models/flame2023_Open.pkl --out data/synth \
        --identities 200 --scans-per-identity 10 --workers 4

Output layout
-------------
    data/synth/
      rig.json            rig used (edit + pass back with --rig to match your hardware)
      config.json         every setting, so the dataset can be regenerated exactly
      index.csv           one row per scan: path, identity, kind, split, n_points
      id_00000/
        identity.npz      FLAME shape coefficients + appearance for this person
        scan_000.npz      points, uv, intensity, label, meta (see README)
        scan_000.ply      (with --ply) for MeshLab / CloudCompare
        attack_000.npz    (with --attacks-per-identity) printed-photo spoofs of this person
      frames/             (with --save-images) raw captured pattern frames as PNG
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .flame import FlameModel
from .pointcloud import to_viewer_frame, write_ply
from .raster import PinholeCamera, rasterize
from .rig import Rig, intrinsics_from_fov
from .scanner import CameraSettings, Lighting, ProjectorSettings, VirtualScanner
from .scene import FLAME_TO_CAM, Appearance, Scene, face_mesh, photo_sheet, wall


@dataclass
class Sampling:
    # identity
    n_shape: int = 100            # shape coefficients sampled per identity (FLAME has 300)
    shape_sigma: float = 1.0      # std of each coefficient
    twin_fraction: float = 0.0    # identities made as near-copies of another (hard negatives)
    twin_sigma: float = 0.35
    # per scan
    n_expr: int = 30
    expr_sigma: float = 0.8
    p_neutral: float = 0.3
    jaw_max: float = 0.15         # radians of mouth opening
    # Head pose, uniform in +-range (degrees). The subject faces the rig, so these
    # are only the small misalignments a cooperating person still has.
    # Set all three to 0 for perfectly frontal scans.
    yaw_deg: float = 10.0         # turning left/right
    pitch_deg: float = 10.0       # nodding up/down
    roll_deg: float = 5.0         # tilting toward a shoulder
    distance: tuple = (0.40, 0.65)   # camera -> nose tip, meters
    lateral: float = 0.03            # nose offset from the optical axis, meters
    wall_gap: tuple = (0.4, 1.5)     # wall distance behind the nose, meters
    # appearance / lighting
    skin_albedo: tuple = (0.18, 0.70)  # wide on purpose: dark skin returns less light
    p_hair: float = 0.85
    hair_albedo: tuple = (0.03, 0.40)
    hairline: tuple = (0.06, 0.09)
    ambient: tuple = (0.0, 0.50)     # relative to projector white at the working distance
    p_lamp: float = 0.4
    lamp: tuple = (0.0, 0.8)
    flicker: tuple = (0.0, 0.08)     # frame-to-frame room-light variation


def _u(rng, lo_hi):
    return float(rng.uniform(*lo_hi))


def identity_params(seed: int, ident: int, smp: Sampling, n_shape_model: int, depth=0) -> dict:
    rng = np.random.default_rng([seed, ident, 1])
    shape = np.zeros(n_shape_model)
    k = min(smp.n_shape, n_shape_model)
    twin_of = None
    if depth < 5 and ident > 0 and rng.random() < smp.twin_fraction:
        twin_of = int(rng.integers(0, ident))
        shape = identity_params(seed, twin_of, smp, n_shape_model, depth + 1)["shape"].copy()
        shape[:k] += rng.normal(0, smp.twin_sigma, k)
    else:
        shape[:k] = rng.normal(0, smp.shape_sigma, k)
    return {
        "shape": shape,
        "skin_albedo": _u(rng, smp.skin_albedo),
        "has_hair": bool(rng.random() < smp.p_hair),
        "hair_albedo": _u(rng, smp.hair_albedo),
        "hairline": _u(rng, smp.hairline),
        "twin_of": -1 if twin_of is None else twin_of,
    }


def _lighting(rng, smp: Sampling) -> Lighting:
    lamp = _u(rng, smp.lamp) if rng.random() < smp.p_lamp else 0.0
    d = rng.normal(0, 1, 3)
    d[2] = -abs(d[2]) - 0.3          # lamp in front of the subject (camera side)
    return Lighting(ambient=_u(rng, smp.ambient), lamp=lamp, lamp_dir=tuple(d / np.linalg.norm(d)),
                    flicker=_u(rng, smp.flicker))


def _pose(rng, smp: Sampling):
    yaw = np.radians(rng.uniform(-smp.yaw_deg, smp.yaw_deg))
    pitch = np.radians(rng.uniform(-smp.pitch_deg, smp.pitch_deg))
    roll = np.radians(rng.uniform(-smp.roll_deg, smp.roll_deg))
    dist = _u(rng, smp.distance)
    nose = np.array([rng.uniform(-smp.lateral, smp.lateral), rng.uniform(-smp.lateral, smp.lateral), dist])
    return yaw, pitch, roll, nose


def face_photo(model: FlameModel, verts_flame: np.ndarray, albedo: float, size=(120, 160)) -> np.ndarray:
    """
    Frontal 'photo' of a face (Lambertian shading), used as a print-attack texture.
    """
    w, h = size
    v = (verts_flame - verts_flame[model.nose_tip_index]) * FLAME_TO_CAM + np.array([0, 0.02, 0.6])
    cam = PinholeCamera(intrinsics_from_fov(w, h, 22.0), w, h)
    depth, tri, bary = rasterize(cam, v, model.faces)
    img = np.full((h, w), 0.85)
    m = tri >= 0
    from .flame import vertex_normals
    vn = vertex_normals(v, model.faces)
    n = np.einsum("nk,nkc->nc", bary[m], vn[model.faces[tri[m]]])
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    light = np.array([0.2, -0.4, -1.0]); light /= np.linalg.norm(light)
    img[m] = albedo * (0.3 + 0.9 * np.clip(np.abs(n @ light), 0, 1))
    return np.clip(img, 0, 1)


# worker
_W = {}


def _init_worker(cfg: dict):
    _W["cfg"] = cfg
    _W["model"] = FlameModel(cfg["flame"], cfg.get("masks"))
    rig = Rig.from_json(cfg["rig"])
    sc = VirtualScanner(rig, phase_steps=cfg["phase_steps"], phase_period=cfg["phase_period"],
                        camera=CameraSettings(**cfg["camera"]), projector=ProjectorSettings(**cfg["projector"]))
    _W["scanner"] = sc
    _W["smp"] = Sampling(**cfg["sampling"])


def _save_scan(path: Path, res, meta: dict, cfg: dict, rng):
    pts, uv, inten, lab, gt = res.points, res.uv, res.intensity, res.label, res.gt_points
    nose = np.asarray(meta["nose_true"])
    keep = np.linalg.norm(pts - nose, axis=1) <= cfg["store_radius"] if cfg["store_radius"] > 0 else np.ones(len(pts), bool)
    idx = np.nonzero(keep)[0]
    if cfg["max_points"] and len(idx) > cfg["max_points"]:
        idx = np.sort(rng.choice(idx, cfg["max_points"], replace=False))
    meta["n_points_decoded"] = int(len(pts))
    meta["n_points_stored"] = int(len(idx))
    arrays = dict(points=pts[idx].astype(np.float32), uv=uv[idx].astype(np.int16),
                  intensity=inten[idx].astype(np.float16), label=lab[idx].astype(np.uint8),
                  meta=np.array(json.dumps(meta)))
    if cfg["with_gt"]:
        arrays["gt_points"] = gt[idx].astype(np.float32)
    np.savez_compressed(path, **arrays)
    if cfg["ply"]:
        col = np.zeros((len(idx), 3), np.uint8)
        palette = np.array([[120, 120, 120], [230, 180, 150], [40, 120, 230], [90, 60, 30], [220, 40, 40]], np.uint8)
        col[:] = palette[np.minimum(lab[idx], 4)]
        write_ply(path.with_suffix(".ply"), to_viewer_frame(pts[idx]), col,
                  comment="viewer frame: x right, y up, z toward viewer (camera frame turned 180 deg about x); "
                          "the .npz has the camera frame")
    return len(idx)


def _save_frames(folder: Path, res):
    from PIL import Image
    folder.mkdir(parents=True, exist_ok=True)
    for i, (img, name) in enumerate(zip(res.images, res.image_names)):
        Image.fromarray(img).save(folder / f"{i:02d}_{name}.png")


def run_identity(ident: int) -> list[dict]:
    cfg, model, sc, smp = _W["cfg"], _W["model"], _W["scanner"], _W["smp"]
    out = Path(cfg["out"]) / f"id_{ident:05d}"
    out.mkdir(parents=True, exist_ok=True)
    idp = identity_params(cfg["seed"], ident, smp, model.n_shape)
    np.savez(out / "identity.npz", shape=idp["shape"].astype(np.float32),
             meta=np.array(json.dumps({k: v for k, v in idp.items() if k != "shape"})))
    rows = []
    jobs = [("scan", i) for i in range(cfg["scans_per_identity"])] + \
           [("attack", i) for i in range(cfg["attacks_per_identity"])]
    for kind, i in jobs:
        path = out / f"{kind}_{i:03d}.npz"
        if path.exists() and not cfg["overwrite"]:
            try:
                with np.load(path) as z:
                    meta = json.loads(str(z["meta"]))
                rows.append(_row(path, ident, meta))
                continue
            except Exception:
                pass
        rng = np.random.default_rng([cfg["seed"], ident, 2 if kind == "scan" else 3, i])
        expr = np.zeros(model.n_expr)
        jaw = 0.0
        if rng.random() >= smp.p_neutral:
            expr[: smp.n_expr] = rng.normal(0, smp.expr_sigma, min(smp.n_expr, model.n_expr))
            jaw = float(rng.uniform(0, smp.jaw_max))
        yaw, pitch, roll, nose = _pose(rng, smp)
        light = _lighting(rng, smp)
        wall_d = nose[2] + _u(rng, smp.wall_gap)
        meta = {"identity": ident, "index": i, "yaw_deg": float(np.degrees(yaw)),
                "pitch_deg": float(np.degrees(pitch)), "roll_deg": float(np.degrees(roll)),
                "nose_true": nose.tolist(), "ambient": light.ambient, "lamp": light.lamp,
                "lamp_dir": list(light.lamp_dir), "flicker": light.flicker, "wall_distance": wall_d, "mode": cfg["mode"],
                "phase_steps": cfg["phase_steps"]}

        if kind == "scan":
            v = model.forward(shape=idp["shape"], expression=expr, jaw_pose=[jaw, 0, 0])
            app = Appearance(skin_albedo=idp["skin_albedo"], hair_albedo=idp["hair_albedo"],
                             has_hair=idp["has_hair"], hairline=idp["hairline"])
            mesh = face_mesh(model, v, app, yaw, pitch, roll, nose, rng)
            meta.update(kind="live", expression=expr[: smp.n_expr].tolist(), jaw=jaw)
        else:
            v = model.forward(shape=idp["shape"])
            photo = face_photo(model, v, idp["skin_albedo"] + 0.2)
            curved = rng.random() < 0.5
            radius = float(rng.uniform(0.10, 0.35)) if curved else None
            mesh = photo_sheet(photo, width=rng.uniform(0.13, 0.20), height=rng.uniform(0.17, 0.26),
                               bend_radius=radius, yaw=yaw, pitch=pitch, center=nose)
            meta.update(kind="print_curved" if curved else "print_flat", bend_radius=radius)

        scene = Scene([mesh], wall(wall_d, _u(rng, (0.3, 0.8))))
        save_frames = kind == "scan" and ident == cfg["first_identity"] and i < cfg["save_images"]
        res = sc.scan(scene, light, rng, mode=cfg["mode"], keep_images=save_frames)
        _save_scan(path, res, meta, cfg, rng)
        if save_frames:
            _save_frames(Path(cfg["out"]) / "frames" / f"id_{ident:05d}_scan_{i:03d}", res)
        rows.append(_row(path, ident, meta))
    return rows


def _row(path: Path, ident: int, meta: dict) -> dict:
    return {"path": str(Path(path.parent.name) / path.name), "identity": ident,
            "kind": meta.get("kind", "live"), "n_points": meta.get("n_points_stored", -1),
            "yaw_deg": round(meta.get("yaw_deg", 0), 1), "distance": round(meta["nose_true"][2], 3)}


def split_of(ident: int, seed: int, fractions=(0.8, 0.1, 0.1)) -> str:
    r = np.random.default_rng([seed, ident, 99]).random()
    return "train" if r < fractions[0] else ("val" if r < fractions[0] + fractions[1] else "test")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--flame", required=True, help="FLAME model file, e.g. flame2023_Open.pkl")
    ap.add_argument("--masks", help="optional FLAME_masks.pkl (better hair/eye regions)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--identities", type=int, default=100)
    ap.add_argument("--first-identity", type=int, default=0, help="for sharding across machines")
    ap.add_argument("--scans-per-identity", type=int, default=10)
    ap.add_argument("--attacks-per-identity", type=int, default=0, help="printed-photo spoof scans per person")
    ap.add_argument("--rig", help="rig.json (default: 1280x720 camera + 1280x720 projector, 15 cm baseline)")
    ap.add_argument("--mode", choices=["physical", "ideal"], default="physical")
    ap.add_argument("--phase-steps", type=int, default=0, help="0 = Gray code only; 4 = Gray code + 4-step phase shift")
    ap.add_argument("--phase-period", type=int, default=16)
    ap.add_argument("--sampling", help="JSON file overriding Sampling fields (pose ranges, lighting, ...)")
    ap.add_argument("--store-radius", type=float, default=0.15, help="keep points within this distance of the nose (m); 0 = keep all")
    ap.add_argument("--max-points", type=int, default=30000, help="random subsample stored clouds (0 = keep all)")
    ap.add_argument("--with-gt", action="store_true", help="also store the true surface point for each scanned point")
    ap.add_argument("--ply", action="store_true", help="also write .ply files for viewing")
    ap.add_argument("--save-images", type=int, default=0, help="save the raw pattern frames of this many scans of the first identity (PNG)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rig_path = out / "rig.json"
    rig = Rig.from_json(args.rig) if args.rig else Rig.make()
    rig.to_json(rig_path)
    smp = Sampling()
    if args.sampling:
        overrides = json.loads(Path(args.sampling).read_text())
        smp = Sampling(**{**asdict(smp), **overrides})
    cfg = {
        "flame": str(Path(args.flame).resolve()), "masks": str(Path(args.masks).resolve()) if args.masks else None,
        "out": str(out.resolve()), "rig": str(rig_path.resolve()), "mode": args.mode,
        "phase_steps": args.phase_steps, "phase_period": args.phase_period,
        "scans_per_identity": args.scans_per_identity, "attacks_per_identity": args.attacks_per_identity,
        "store_radius": args.store_radius, "max_points": args.max_points, "with_gt": args.with_gt,
        "ply": args.ply, "save_images": args.save_images, "first_identity": args.first_identity,
        "seed": args.seed, "overwrite": args.overwrite,
        "camera": asdict(CameraSettings()), "projector": asdict(ProjectorSettings()), "sampling": asdict(smp),
    }
    (out / "config.json").write_text(json.dumps({**cfg, "identities": args.identities}, indent=2))
    print(rig.describe())
    idents = list(range(args.first_identity, args.first_identity + args.identities))
    t0 = time.time()
    rows = []
    if args.workers <= 1:
        _init_worker(cfg)
        for n, k in enumerate(idents, 1):
            rows += run_identity(k)
            _progress(n, len(idents), t0)
    else:
        from multiprocessing import get_context
        with get_context("spawn").Pool(args.workers, initializer=_init_worker, initargs=(cfg,)) as pool:
            for n, r in enumerate(pool.imap_unordered(run_identity, idents), 1):
                rows += r
                _progress(n, len(idents), t0)
    write_index(out, args.seed)
    print(f"done: {len(rows)} scans in {time.time() - t0:.0f} s -> {out / 'index.csv'}")


def _progress(n, total, t0):
    el = time.time() - t0
    eta = el / n * (total - n)
    print(f"  identity {n}/{total}  elapsed {el:.0f}s  eta {eta:.0f}s", flush=True)


def write_index(out: Path, seed: int):
    """
    (Re)build index.csv from whatever scans exist under `out`.
    """
    rows = []
    for p in sorted(out.glob("id_*/*.npz")):
        if p.name == "identity.npz":
            continue
        with np.load(p) as z:
            meta = json.loads(str(z["meta"]))
        r = _row(p, meta["identity"], meta)
        r["split"] = split_of(meta["identity"], seed)
        rows.append(r)
    with open(out / "index.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "identity", "kind", "split", "n_points", "yaw_deg", "distance"])
        w.writeheader()
        w.writerows(rows)
    return rows


if __name__ == "__main__":
    main()
