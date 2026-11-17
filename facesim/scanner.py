"""
The virtual structured-light scanner.

physical mode (default) does what the real rig does:
    1. figure out what the camera sees and which of it the projector can light
       (z-buffer from the camera, shadow map from the projector)
    2. render every pattern frame the USB camera would capture: diffuse
       shading, projector falloff and black level, room light, lens blur,
       shot + read noise, 8-bit quantization, auto exposure
    3. decode the frames with patterns.PatternSet.decode (same code you run
       on the real rig)
    4. triangulate camera rays against projector column planes

So the point clouds come out with the real artifacts: holes where the
projector is shadowed (beside the nose), dropouts on dark hair and eyes,
stair-stepping from column quantization (Gray code only), noise growing
with distance, flying pixels at edges, and worse data under strong room light.

ideal mode skips the images and returns exact surface points + Gaussian
depth noise. It's fast and good for debugging.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d

from .patterns import DecodeSettings, PatternSet
from .raster import plane_depth, rasterize
from .rig import Rig
from .scene import Scene


@dataclass
class Lighting:
    ambient: float = 0.05            # uniform room light, relative to projector white at working distance
    lamp: float = 0.0                # directional room light (same units)
    lamp_dir: tuple = (0.3, -0.6, -0.75)   # from the surface toward the lamp, camera frame
    flicker: float = 0.0             # frame-to-frame room-light variation (std, fraction); LED/fluorescent
                                     # lights flicker at 100/120 Hz, so short exposures see this


@dataclass
class CameraSettings:
    target_dn: float = 200.0         # auto-exposure target for the white frame on the subject
    blur_sigma: float = 0.8          # lens / focus blur, camera pixels
    read_noise: float = 1.2          # DN
    electrons_per_dn: float = 30.0   # shot noise variance = DN / electrons_per_dn
    gamma: float = 1.0               # 2.2 mimics a webcam's sRGB curve


@dataclass
class ProjectorSettings:
    blur_sigma: float = 0.6          # projector pixels (focus + optics)
    contrast_ratio: float = 600.0    # DLP native contrast; black level = 1 / ratio
    shadow_bias: float = 0.003       # m, tolerance for the shadow-map test


# Surfaces structured light handles badly, by ground-truth label:
#   label -> (fraction of points lost, extra noise in mm along the camera ray)
# Eyes: specular cornea + refraction; hair: volumetric scattering.
DEFAULT_MATERIALS = {2: (0.6, 2.0), 3: (0.35, 2.5)}


@dataclass
class Geometry:
    hit: np.ndarray       # (H,W) bool
    P: np.ndarray         # (H,W,3) surface point seen by each camera pixel
    N: np.ndarray         # (H,W,3) normal facing the camera
    albedo: np.ndarray    # (H,W)
    label: np.ndarray     # (H,W) uint8
    u_p: np.ndarray       # (H,W) projector column of the surface point
    lit: np.ndarray       # (H,W) bool: the projector can reach it
    irradiance: np.ndarray  # (H,W) projector irradiance factor (cosine x falloff x lit)


@dataclass
class ScanResult:
    points: np.ndarray                 # (N,3) float32, camera frame, meters
    uv: np.ndarray                     # (N,2) int32 camera pixel (col,row) of each point
    intensity: np.ndarray              # (N,) float32 0..1 (white - black)
    label: np.ndarray                  # (N,) uint8 ground truth surface label at that pixel
    gt_points: np.ndarray              # (N,3) float32 true surface point at that pixel
    images: list | None = None         # captured uint8 frames (if requested)
    image_names: list | None = None
    info: dict = field(default_factory=dict)


def triangulate_columns(rig: Rig, column: np.ndarray, valid: np.ndarray | None = None):
    """Intersect camera rays with projector column planes.

    column: (H,W) projector column per camera pixel (nan = invalid)
    Returns points (N,3) and the (col,row) pixel of each point.
    """
    if valid is None:
        valid = np.isfinite(column)
    rows, cols = np.nonzero(valid)
    u = column[rows, cols]
    Kc_inv = np.linalg.inv(rig.cam_K)
    d = np.stack([cols, rows, np.ones_like(cols)], 1).astype(np.float64) @ Kc_inv.T
    Kp_inv = np.linalg.inv(rig.proj_K)
    ra = np.stack([u, np.zeros_like(u), np.ones_like(u)], 1) @ Kp_inv.T
    rb = np.stack([u, np.ones_like(u), np.ones_like(u)], 1) @ Kp_inv.T
    n_p = np.cross(ra, rb)                   # column plane normal, projector frame
    n_c = n_p @ rig.R                        # = (R^T n_p) in camera frame
    off = n_p @ np.asarray(rig.T).ravel()
    denom = (n_c * d).sum(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        lam = -off / denom
    X = d * lam[:, None]
    good = np.isfinite(lam) & (lam > 0)
    return X[good], np.stack([cols, rows], 1)[good], good


class VirtualScanner:
    def __init__(self, rig: Rig, phase_steps: int = 0, phase_period: int = 16,
                 camera: CameraSettings | None = None, projector: ProjectorSettings | None = None,
                 decode: DecodeSettings | None = None):
        self.rig = rig
        self.patterns = PatternSet(*rig.proj_size, phase_steps=phase_steps, phase_period=phase_period)
        self.cam_s = camera or CameraSettings()
        self.proj_s = projector or ProjectorSettings()
        self.dec_s = decode or DecodeSettings()
        self.materials = dict(DEFAULT_MATERIALS)
        self._prof_cache = None
        self._cam_rays = rig.camera.pixel_rays()
        self._proj_rays = rig.projector.pixel_rays()

    # ------------------------------------------------------------------
    def geometry(self, scene: Scene) -> Geometry:
        cam, proj = self.rig.camera, self.rig.projector
        H, W = cam.height, cam.width
        mesh = scene.merged()
        depth = np.full((H, W), np.inf)
        P = np.zeros((H, W, 3)); N = np.zeros((H, W, 3))
        alb = np.zeros((H, W)); lab = np.zeros((H, W), np.uint8)
        if mesh is not None:
            depth, tri, bary = rasterize(cam, mesh.verts, mesh.faces)
            m = tri >= 0
            f = mesh.faces[tri[m]]
            b = bary[m]
            P[m] = np.einsum("nk,nkc->nc", b, mesh.verts[f])
            n = np.einsum("nk,nkc->nc", b, mesh.normals[f])
            N[m] = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
            alb[m] = (b * mesh.albedo[f]).sum(1)
            lab[m] = mesh.face_label[tri[m]]
        if scene.plane is not None:
            pl = scene.plane
            pd = plane_depth(cam, pl.point, pl.normal, self._cam_rays)
            use = pd < depth
            P[use] = self._cam_rays[use] * pd[use][:, None]
            N[use] = pl.normal / np.linalg.norm(pl.normal)
            alb[use] = pl.albedo
            lab[use] = pl.label
            depth = np.minimum(depth, pd)
        hit = np.isfinite(depth)
        flip = (N * -P).sum(-1) < 0
        N[flip] *= -1

        # shadow map from the projector
        Hp, Wp = proj.height, proj.width
        sdepth = np.full((Hp, Wp), np.inf)
        if mesh is not None:
            sdepth, _, _ = rasterize(proj, mesh.verts, mesh.faces)
        if scene.plane is not None:
            sdepth = np.minimum(sdepth, plane_depth(proj, scene.plane.point, scene.plane.normal, self._proj_rays))

        u_p = np.full((H, W), np.nan)
        lit = np.zeros((H, W), bool)
        irr = np.zeros((H, W))
        Ph = P[hit]
        uvp, zp = proj.project(Ph)
        up, vp = uvp[:, 0], uvp[:, 1]
        inside = (zp > 0) & (up >= -0.5) & (up <= Wp - 0.5) & (vp >= -0.5) & (vp <= Hp - 0.5)
        x0 = np.clip(np.floor(np.nan_to_num(up)), 0, Wp - 1).astype(int); x1 = np.clip(x0 + 1, 0, Wp - 1)
        y0 = np.clip(np.floor(np.nan_to_num(vp)), 0, Hp - 1).astype(int); y1 = np.clip(y0 + 1, 0, Hp - 1)
        sd = np.minimum.reduce([sdepth[y0, x0], sdepth[y0, x1], sdepth[y1, x0], sdepth[y1, x1]])
        lit_h = inside & (zp <= sd + self.proj_s.shadow_bias)
        L = proj.center - Ph
        dist = np.linalg.norm(L, axis=1)
        cosp = np.clip((N[hit] * L).sum(1) / dist, 0, 1)
        falloff = (self.rig.working_distance / dist) ** 2
        u_p[hit] = up
        lit[hit] = lit_h
        irr[hit] = np.where(lit_h, cosp * falloff, 0.0)
        return Geometry(hit, P, N, alb, lab, u_p, lit, irr)

    # ------------------------------------------------------------------
    def _profiles(self):
        """Pattern intensity as a function of continuous projector column, with projector blur."""
        if self._prof_cache is None:
            prof = self.patterns.column_profiles()
            ss = 8
            x = (np.arange(prof.shape[1] * ss) + 0.5) / ss - 0.5
            idx = np.clip(np.floor(x + 0.5).astype(int), 0, prof.shape[1] - 1)
            p = prof[:, idx]
            if self.proj_s.blur_sigma > 0:
                p = gaussian_filter1d(p, self.proj_s.blur_sigma * ss, axis=1, mode="nearest")
            self._prof_cache = (x, p, ss)
        return self._prof_cache

    def render(self, geom: Geometry, lighting: Lighting, rng: np.random.Generator,
               roi: tuple | None = None) -> list[np.ndarray]:
        """Render the captured frames (uint8), optionally only inside roi=(r0,r1,c0,c1)."""
        r0, r1, c0, c1 = roi or (0, geom.hit.shape[0], 0, geom.hit.shape[1])
        sl = (slice(r0, r1), slice(c0, c1))
        alb, irr, N = geom.albedo[sl], geom.irradiance[sl], geom.N[sl]
        u = np.nan_to_num(geom.u_p[sl], nan=-1e6)
        ld = np.asarray(lighting.lamp_dir, float)
        ld /= np.linalg.norm(ld)
        amb = lighting.ambient + lighting.lamp * np.clip(N @ ld, 0, 1)
        base = alb * amb
        pterm = alb * irr

        x, prof, ss = self._profiles()
        pos = (u + 0.5) * ss - 0.5
        i0 = np.clip(np.floor(pos).astype(np.int64), 0, prof.shape[1] - 2)
        fr = np.clip(pos - i0, 0, 1)
        black = 1.0 / self.proj_s.contrast_ratio

        sig = []
        for k in range(prof.shape[0]):
            val = prof[k, i0] * (1 - fr) + prof[k, i0 + 1] * fr
            room = base * max(0.0, 1.0 + lighting.flicker * rng.normal()) if lighting.flicker else base
            sig.append(room + pterm * (black + (1 - black) * val))

        # auto exposure on the subject (non-background) pixels of the white frame
        subj = geom.label[sl] > 0
        ref_px = sig[0][subj] if subj.any() else sig[0][geom.hit[sl]]
        ref = np.percentile(ref_px, 99.0) if ref_px.size else 1.0
        gain = self.cam_s.target_dn / max(ref, 1e-9)

        out = []
        cs = self.cam_s
        for s in sig:
            dn = gain * s
            if cs.blur_sigma > 0:
                dn = gaussian_filter(dn, cs.blur_sigma)
            noise_sd = np.sqrt(np.maximum(dn, 0) / cs.electrons_per_dn + cs.read_noise ** 2)
            dn = dn + rng.normal(0, 1, dn.shape) * noise_sd
            if cs.gamma != 1.0:
                dn = 255.0 * (np.clip(dn, 0, None) / 255.0) ** (1.0 / cs.gamma)
            out.append(np.clip(np.round(dn), 0, 255).astype(np.uint8))
        return out

    def _apply_materials(self, pts, labels, rng):
        keep = np.ones(len(pts), bool)
        pts = pts.copy()
        for lab, (drop, noise_mm) in self.materials.items():
            m = labels == lab
            if not m.any():
                continue
            keep[m] &= rng.random(m.sum()) >= drop
            if noise_mm > 0:
                z = pts[m, 2:3]
                pts[m] *= 1 + rng.normal(0, noise_mm / 1000, (m.sum(), 1)) / z
        return pts[keep], keep

    # ------------------------------------------------------------------
    def scan(self, scene: Scene, lighting: Lighting | None = None, rng: np.random.Generator | None = None,
             mode: str = "physical", keep_images: bool = False, roi_margin: int = 48,
             ideal_noise_mm: float = 0.3) -> ScanResult:
        lighting = lighting or Lighting()
        rng = rng or np.random.default_rng()
        geom = self.geometry(scene)
        H, W = geom.hit.shape

        if mode == "ideal":
            m = geom.hit & geom.lit & (geom.irradiance > 0.05)
            rows, cols = np.nonzero(m)
            gt = geom.P[rows, cols]
            z = gt[:, 2]
            sd = ideal_noise_mm / 1000 * (z / self.rig.working_distance) ** 2
            pts = gt * (1 + rng.normal(0, 1, len(z)) * sd / z)[:, None]
            pts, keep = self._apply_materials(pts, geom.label[rows, cols], rng)
            rows, cols, gt = rows[keep], cols[keep], gt[keep]
            inten = np.clip(geom.albedo[rows, cols] * geom.irradiance[rows, cols], 0, 1)
            return ScanResult(pts.astype(np.float32), np.stack([cols, rows], 1).astype(np.int32),
                              inten.astype(np.float32), geom.label[rows, cols], gt.astype(np.float32),
                              info={"mode": "ideal"})

        if keep_images:
            roi = (0, H, 0, W)
        else:
            subj = geom.label > 0
            if subj.any():
                rr, cc = np.nonzero(subj)
                roi = (max(rr.min() - roi_margin, 0), min(rr.max() + roi_margin + 1, H),
                       max(cc.min() - roi_margin, 0), min(cc.max() + roi_margin + 1, W))
            else:
                roi = (0, H, 0, W)
        imgs = self.render(geom, lighting, rng, roi)
        column, valid, contrast = self.patterns.decode(imgs, self.dec_s)

        r0, r1, c0, c1 = roi
        col_full = np.full((H, W), np.nan)
        col_full[r0:r1, c0:c1] = column
        con_full = np.zeros((H, W), np.float32)
        con_full[r0:r1, c0:c1] = contrast
        pts, uv, good = triangulate_columns(self.rig, col_full)
        keep = (pts[:, 2] > 0.1) & (pts[:, 2] < 5.0)
        pts, uv = pts[keep], uv[keep]
        rows, cols = uv[:, 1], uv[:, 0]
        pts, keep = self._apply_materials(pts, geom.label[rows, cols], rng)
        uv = uv[keep]
        rows, cols = uv[:, 1], uv[:, 0]
        res = ScanResult(
            points=pts.astype(np.float32), uv=uv.astype(np.int32),
            intensity=np.clip(con_full[rows, cols] / 255.0, 0, 1).astype(np.float32),
            label=geom.label[rows, cols], gt_points=geom.P[rows, cols].astype(np.float32),
            info={"mode": "physical", "roi": [int(v) for v in roi]})
        if keep_images:
            res.images = imgs
            res.image_names = self.patterns.names
        return res
