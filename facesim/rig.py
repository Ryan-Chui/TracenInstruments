"""Camera + projector rig geometry.

The rig is stored the same way OpenCV's stereoCalibrate reports a calibration,
so once you calibrate the real USB camera + DLP projector you can paste the
numbers into rig.json and the synthetic data will match your hardware:

    camera.K, projector.K : 3x3 intrinsics
    R, T                  : X_projector = R @ X_camera + T   (meters)

The world frame is the camera frame (x right, y down, z forward).
Lens distortion is not simulated; undistort real captures before triangulating.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .raster import PinholeCamera


def intrinsics_from_fov(width: int, height: int, hfov_deg: float, cx=None, cy=None) -> np.ndarray:
    f = (width / 2) / np.tan(np.radians(hfov_deg) / 2)
    cx = (width - 1) / 2 if cx is None else cx
    cy = (height - 1) / 2 if cy is None else cy
    return np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])


@dataclass
class Rig:
    cam_K: np.ndarray
    cam_size: tuple[int, int]            # (width, height)
    proj_K: np.ndarray
    proj_size: tuple[int, int]
    R: np.ndarray                        # camera -> projector rotation
    T: np.ndarray                        # camera -> projector translation (m)
    working_distance: float = 0.5        # m, used for exposure / irradiance reference
    notes: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def camera(self) -> PinholeCamera:
        return PinholeCamera(self.cam_K, *self.cam_size)

    @property
    def projector(self) -> PinholeCamera:
        return PinholeCamera(self.proj_K, *self.proj_size, R=self.R, t=self.T)

    @property
    def baseline(self) -> float:
        return float(np.linalg.norm(self.projector.center))

    # ------------------------------------------------------------------
    @staticmethod
    def make(cam_size=(1280, 720), cam_hfov=70.0, proj_size=(1280, 720), proj_hfov=40.0,
             baseline=0.15, converge_at=0.5, projector_side="right", proj_cy=None) -> "Rig":
        """Simple rig: projector beside the camera, both aimed at a point
        `converge_at` meters in front of the camera.

        proj_cy lets you model DLP offset optics (e.g. proj_cy = proj_height
        for a projector that throws everything above its lens axis).
        """
        side = 1.0 if projector_side == "right" else -1.0
        C = np.array([side * baseline, 0.0, 0.0])           # projector center
        target = np.array([0.0, 0.0, converge_at])
        z = target - C
        z /= np.linalg.norm(z)
        y = np.array([0.0, 1.0, 0.0])
        x = np.cross(y, z)
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        R = np.stack([x, y, z])
        T = -R @ C
        return Rig(
            cam_K=intrinsics_from_fov(*cam_size, cam_hfov), cam_size=tuple(cam_size),
            proj_K=intrinsics_from_fov(*proj_size, proj_hfov, cy=proj_cy), proj_size=tuple(proj_size),
            R=R, T=T, working_distance=converge_at,
            notes=(f"generated: camera {cam_size} hfov {cam_hfov}, projector {proj_size} "
                   f"hfov {proj_hfov}, baseline {baseline} m, aimed at {converge_at} m"))

    def to_json(self, path: str | Path):
        d = {
            "camera": {"width": self.cam_size[0], "height": self.cam_size[1], "K": self.cam_K.tolist()},
            "projector": {"width": self.proj_size[0], "height": self.proj_size[1], "K": self.proj_K.tolist()},
            "R": self.R.tolist(), "T": np.asarray(self.T).ravel().tolist(),
            "working_distance": self.working_distance, "notes": self.notes, **self.extra,
        }
        Path(path).write_text(json.dumps(d, indent=2))

    @staticmethod
    def from_json(path: str | Path) -> "Rig":
        d = json.loads(Path(path).read_text())
        c, p = d["camera"], d["projector"]
        known = {"camera", "projector", "R", "T", "working_distance", "notes"}
        return Rig(
            cam_K=np.array(c["K"], float), cam_size=(int(c["width"]), int(c["height"])),
            proj_K=np.array(p["K"], float), proj_size=(int(p["width"]), int(p["height"])),
            R=np.array(d["R"], float), T=np.array(d["T"], float).ravel(),
            working_distance=float(d.get("working_distance", 0.5)), notes=d.get("notes", ""),
            extra={k: v for k, v in d.items() if k not in known})

    def describe(self) -> str:
        cam_f, proj_f = self.cam_K[0, 0], self.proj_K[0, 0]
        z = self.working_distance
        dz = z * z / (proj_f * self.baseline)   # depth change per projector column
        mm_px = z / cam_f * 1000
        return (f"camera {self.cam_size[0]}x{self.cam_size[1]} (f={cam_f:.0f}px), "
                f"projector {self.proj_size[0]}x{self.proj_size[1]} (f={proj_f:.0f}px), "
                f"baseline {self.baseline * 100:.1f} cm\n"
                f"at {z:.2f} m: {mm_px:.2f} mm per camera pixel, "
                f"~{dz * 1000:.2f} mm depth step per projector column")
