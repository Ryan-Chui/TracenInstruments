"""Scenes in front of the rig: a posed FLAME face, a wall, or a printed-photo spoof.

All coordinates here are in the camera frame (meters, x right, y down, z forward).

Per-face labels (carried through to every scanned point as ground truth):
    0 background (wall)   1 skin   2 eye   3 hair   4 spoof medium (paper/photo)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .flame import FlameModel, euler_to_matrix, vertex_normals

LABELS = {0: "background", 1: "skin", 2: "eye", 3: "hair", 4: "spoof"}
FLAME_TO_CAM = np.array([1.0, -1.0, -1.0])   # FLAME (y up, face +z) -> camera (y down, face -z)


@dataclass
class Mesh:
    verts: np.ndarray            # (N,3) camera frame
    faces: np.ndarray            # (F,3)
    albedo: np.ndarray           # (N,) diffuse reflectance 0..1
    face_label: np.ndarray       # (F,) uint8

    @property
    def normals(self) -> np.ndarray:
        return vertex_normals(self.verts, self.faces)


@dataclass
class Plane:
    point: np.ndarray
    normal: np.ndarray
    albedo: float = 0.5
    label: int = 0


@dataclass
class Scene:
    meshes: list = field(default_factory=list)
    plane: Plane | None = None

    def merged(self) -> Mesh | None:
        if not self.meshes:
            return None
        V, F, A, L, off = [], [], [], [], 0
        for m in self.meshes:
            V.append(m.verts); F.append(m.faces + off); A.append(m.albedo); L.append(m.face_label)
            off += len(m.verts)
        return Mesh(np.concatenate(V), np.concatenate(F), np.concatenate(A), np.concatenate(L))


# --------------------------------------------------------------------------
# Faces
# --------------------------------------------------------------------------

@dataclass
class Appearance:
    skin_albedo: float = 0.45
    hair_albedo: float = 0.12
    eye_albedo: float = 0.08     # dark iris + specular cornea: structured light mostly fails here
    has_hair: bool = True
    hairline: float = 0.075      # m above the eye center (FLAME heuristic, if no masks)
    albedo_noise: float = 0.05   # low-frequency blotchiness


def face_mesh(model: FlameModel, verts_flame: np.ndarray, appearance: Appearance,
              yaw=0.0, pitch=0.0, roll=0.0, nose_position=(0.0, 0.0, 0.5),
              rng: np.random.Generator | None = None) -> Mesh:
    """Pose a FLAME head and place its nose tip at `nose_position` (camera frame)."""
    rng = rng or np.random.default_rng()
    eye_c = verts_flame[model.eye_mask].mean(0) if model.eye_mask.any() else model.eye_center
    pivot = eye_c + np.array([0.0, -0.02, -0.07])            # roughly the neck/head pivot
    R = euler_to_matrix(yaw, pitch, roll)
    v = (verts_flame - pivot) @ R.T
    v = v * FLAME_TO_CAM
    v += np.asarray(nose_position, float) - v[model.nose_tip_index]

    vlab = model.region_labels(verts_flame, appearance.hairline if appearance.has_hair else None)
    albedo = np.full(len(v), appearance.skin_albedo)
    albedo[vlab == 3] = appearance.hair_albedo
    albedo[vlab == 2] = appearance.eye_albedo
    if appearance.albedo_noise > 0:
        p = verts_flame / 0.03
        blotch = np.zeros(len(v))
        for _ in range(3):
            w = rng.normal(0, 1, 3)
            blotch += np.sin(p @ w + rng.uniform(0, 2 * np.pi))
        albedo *= 1 + appearance.albedo_noise * blotch / 3
    albedo = np.clip(albedo, 0.01, 0.95)
    # a face's label = majority label of its vertices (ties -> max)
    fl = vlab[model.faces]
    face_label = np.where(fl[:, 0] == fl[:, 1], fl[:, 0], np.max(fl, 1)).astype(np.uint8)
    return Mesh(v, model.faces.copy(), albedo, face_label)


def wall(distance: float, albedo: float = 0.6) -> Plane:
    return Plane(point=np.array([0, 0, distance]), normal=np.array([0, 0, -1.0]), albedo=albedo)


# --------------------------------------------------------------------------
# Printed-photo spoof
# --------------------------------------------------------------------------

def photo_sheet(photo: np.ndarray, width=0.15, height=0.20, bend_radius=None,
                yaw=0.0, pitch=0.0, center=(0, 0, 0.5), paper_albedo=(0.05, 0.85)) -> Mesh:
    """A printed photo (or phone screen) held in front of the rig.

    photo: (h,w) image in [0,1] that becomes the sheet's albedo.
    bend_radius: None = flat; otherwise the sheet curves around a vertical
    axis (bulging toward the camera) with this radius in meters.
    """
    ph, pw = photo.shape
    ny, nx = 61, 46
    s = np.linspace(-width / 2, width / 2, nx)
    t = np.linspace(-height / 2, height / 2, ny)
    S, T = np.meshgrid(s, t)
    if bend_radius:
        X = bend_radius * np.sin(S / bend_radius)
        Z = bend_radius * (1 - np.cos(S / bend_radius))   # center closest, edges curl away
    else:
        X, Z = S, np.zeros_like(S)
    V = np.stack([X, T, Z], -1).reshape(-1, 3)
    cy, sy, cp, sp = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    V = V @ (Rx @ Ry).T + np.asarray(center, float)

    idx = np.arange(ny * nx).reshape(ny, nx)
    a, b, c, d = idx[:-1, :-1], idx[:-1, 1:], idx[1:, :-1], idx[1:, 1:]
    F = np.concatenate([np.stack([a, b, c], -1).reshape(-1, 3), np.stack([b, d, c], -1).reshape(-1, 3)])
    # sample the photo at each vertex
    iy = np.clip(np.round(np.linspace(0, ph - 1, ny)).astype(int), 0, ph - 1)
    ix = np.clip(np.round(np.linspace(0, pw - 1, nx)).astype(int), 0, pw - 1)
    img = photo[np.ix_(iy, ix)].reshape(-1)
    lo, hi = paper_albedo
    albedo = lo + (hi - lo) * np.clip(img, 0, 1)
    return Mesh(V, F, albedo, np.full(len(F), 4, np.uint8))
