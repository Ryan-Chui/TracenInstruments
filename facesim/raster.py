"""Pinhole camera model and a vectorized z-buffer rasterizer (pure NumPy).

Rasterizing a triangle mesh from a pinhole view gives exactly what ray casting
gives (the first surface hit per pixel), which is all the scanner needs: once
from the camera to see the surface, and once from the projector to find
which surface points the projector can light (shadow mapping).

Conventions follow OpenCV: x right, y down, z forward; pixel (0,0) is centered
at image coordinates (0,0); X_cam = R @ X_world + t.
"""
from __future__ import annotations

import numpy as np


class PinholeCamera:
    def __init__(self, K, width, height, R=None, t=None):
        self.K = np.asarray(K, dtype=np.float64)
        self.width = int(width)
        self.height = int(height)
        self.R = np.eye(3) if R is None else np.asarray(R, dtype=np.float64)
        self.t = np.zeros(3) if t is None else np.asarray(t, dtype=np.float64).reshape(3)

    @property
    def center(self) -> np.ndarray:
        """Optical center in world coordinates."""
        return -self.R.T @ self.t

    def to_cam(self, X: np.ndarray) -> np.ndarray:
        return X @ self.R.T + self.t

    def project(self, X: np.ndarray):
        """World points (N,3) -> pixel coords (N,2) and depth z (N,)."""
        Xc = self.to_cam(X)
        z = Xc[:, 2]
        uvw = Xc @ self.K.T
        with np.errstate(divide="ignore", invalid="ignore"):
            uv = uvw[:, :2] / uvw[:, 2:3]
        return uv, z

    def pixel_rays(self) -> np.ndarray:
        """(H,W,3) ray directions in camera coordinates with z = 1."""
        u, v = np.meshgrid(np.arange(self.width), np.arange(self.height))
        pix = np.stack([u, v, np.ones_like(u)], -1).reshape(-1, 3).astype(np.float64)
        d = pix @ np.linalg.inv(self.K).T
        return (d / d[:, 2:3]).reshape(self.height, self.width, 3)


def rasterize(cam: PinholeCamera, verts: np.ndarray, faces: np.ndarray,
              near: float = 0.02, chunk_candidates: int = 3_000_000):
    """Z-buffer render of a mesh.

    Returns
    -------
    depth : (H,W) camera-frame z of the nearest surface (inf where empty)
    tri   : (H,W) index of the triangle seen at each pixel (-1 where empty)
    bary  : (H,W,3) perspective-correct barycentric coords of the hit point
    """
    H, W = cam.height, cam.width
    depth = np.full(H * W, np.inf)
    tri_buf = np.full(H * W, -1, dtype=np.int64)
    bary_buf = np.zeros((H * W, 3))

    uv, z = cam.project(verts)
    tuv = uv[faces]                       # F,3,2
    tz = z[faces]                         # F,3
    ok = (tz > near).all(1) & np.isfinite(tuv).all((1, 2))
    xmin = np.ceil(tuv[..., 0].min(1)); xmax = np.floor(tuv[..., 0].max(1))
    ymin = np.ceil(tuv[..., 1].min(1)); ymax = np.floor(tuv[..., 1].max(1))
    xmin = np.clip(xmin, 0, W - 1); xmax = np.clip(xmax, 0, W - 1)
    ymin = np.clip(ymin, 0, H - 1); ymax = np.clip(ymax, 0, H - 1)
    x0, y0 = tuv[:, 0, 0], tuv[:, 0, 1]
    x1, y1 = tuv[:, 1, 0], tuv[:, 1, 1]
    x2, y2 = tuv[:, 2, 0], tuv[:, 2, 1]
    area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
    ok &= (xmax >= xmin) & (ymax >= ymin) & (np.abs(area) > 1e-12)
    ok &= (tuv[..., 0].max(1) >= 0) & (tuv[..., 0].min(1) <= W - 1)
    ok &= (tuv[..., 1].max(1) >= 0) & (tuv[..., 1].min(1) <= H - 1)

    idx = np.nonzero(ok)[0]
    if len(idx) == 0:
        return depth.reshape(H, W), tri_buf.reshape(H, W), bary_buf.reshape(H, W, 3)
    bw = (xmax[idx] - xmin[idx] + 1).astype(np.int64)
    bh = (ymax[idx] - ymin[idx] + 1).astype(np.int64)
    n_cand = bw * bh

    # split triangles into chunks with a bounded number of candidate pixels
    csum = np.cumsum(n_cand)
    bounds = np.searchsorted(csum, np.arange(chunk_candidates, csum[-1] + chunk_candidates, chunk_candidates))
    start = 0
    for stop in np.unique(np.append(bounds + 1, len(idx))):
        stop = min(int(stop), len(idx))
        if stop <= start:
            continue
        sel = idx[start:stop]
        cw, ch, cn = bw[start:stop], bh[start:stop], n_cand[start:stop]
        start = stop

        t_of = np.repeat(np.arange(len(sel)), cn)
        offs = np.arange(cn.sum()) - np.repeat(np.cumsum(cn) - cn, cn)
        px = xmin[sel][t_of] + offs % cw[t_of]
        py = ymin[sel][t_of] + offs // cw[t_of]
        s = sel[t_of]
        a = area[s]
        w0 = ((x1[s] - px) * (y2[s] - py) - (x2[s] - px) * (y1[s] - py)) / a
        w1 = ((x2[s] - px) * (y0[s] - py) - (x0[s] - px) * (y2[s] - py)) / a
        w2 = 1.0 - w0 - w1
        inside = (w0 >= -1e-9) & (w1 >= -1e-9) & (w2 >= -1e-9)
        if not inside.any():
            continue
        s, px, py = s[inside], px[inside], py[inside]
        w = np.stack([w0[inside], w1[inside], w2[inside]], 1)
        inv = w / tz[s]
        iz = inv.sum(1)
        zc = 1.0 / iz
        bc = inv * zc[:, None]
        pix = (py * W + px).astype(np.int64)

        order = np.lexsort((zc, pix))
        pix_o = pix[order]
        first = np.ones(len(order), bool)
        first[1:] = pix_o[1:] != pix_o[:-1]
        keep = order[first]
        pk, zk = pix[keep], zc[keep]
        better = zk < depth[pk]
        pk, keep = pk[better], keep[better]
        depth[pk] = zc[keep]
        tri_buf[pk] = s[keep]
        bary_buf[pk] = bc[keep]

    return depth.reshape(H, W), tri_buf.reshape(H, W), bary_buf.reshape(H, W, 3)


def plane_depth(cam: PinholeCamera, point, normal, rays: np.ndarray | None = None) -> np.ndarray:
    """Camera-frame depth of an infinite plane at every pixel (inf if behind)."""
    p0 = cam.to_cam(np.asarray(point, float)[None])[0]
    n = cam.R @ np.asarray(normal, float)
    rays = cam.pixel_rays() if rays is None else rays
    denom = rays @ n
    with np.errstate(divide="ignore", invalid="ignore"):
        tz = (p0 @ n) / denom
    tz[~np.isfinite(tz) | (tz <= 0)] = np.inf
    return tz
