"""Compare three-step phase shifting with single-shot coded-dot projection.

The decoder functions operate on grayscale camera images. For real hardware,
replace the synthetic capture functions with synchronized ELP/projector
captures. The coded-dot decoder returns camera/projector correspondences;
triangulation still requires camera-projector calibration.
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from scanner.acquisition import Camera, ProjectorWindow
from scanner.triangulation import Triangulate
from scanner.utils.visualize import write_ply
from coded_dots import (
    decode_coded_dots,
    generate_coded_dot_pattern,
    synthetic_coded_dot_capture,
)
from phase_shifting import decode_three_step, synthetic_phase_capture


def run_synthetic(args):
    rng = np.random.default_rng(args.seed)
    shape = (args.height, args.width)
    phase_images = synthetic_phase_capture(
        shape, args.period, args.disparity, args.noise, rng
    )
    start = time.perf_counter()
    phase, phase_mask = decode_three_step(phase_images)
    expected_phase = np.mod(
        2 * np.pi * (np.arange(args.width, dtype=np.float32)[None, :] + args.disparity)
        / args.period,
        2 * np.pi,
    )
    phase_error = np.abs(np.angle(np.exp(1j * (phase - expected_phase))))
    phase_error_pixels = phase_error * args.period / (2 * np.pi)
    phase_time = time.perf_counter() - start

    projector_pattern, camera_image, homography = synthetic_coded_dot_capture(
        shape, args.seed, args.noise, rng
    )
    start = time.perf_counter()
    camera_points, projector_points, dot_confidence = decode_coded_dots(
        projector_pattern, camera_image
    )
    dot_time = time.perf_counter() - start
    if len(camera_points):
        expected_projector_points = cv2.perspectiveTransform(
            camera_points.reshape(-1, 1, 2), np.linalg.inv(homography)
        ).reshape(-1, 2)
        dot_error = np.linalg.norm(
            projector_points - expected_projector_points, axis=1
        )
    else:
        dot_error = np.empty(0)

    print("Synthetic comparison (lower error is better):")
    print(
        f"  3-step phase: {phase_mask.mean():.1%} valid, "
        f"{phase_error_pixels[phase_mask].mean():.2f} px phase error, "
        f"{phase_time * 1000:.1f} ms decode"
    )
    print(
        f"  coded dots:   {len(camera_points)} matches, "
        f"{dot_error.mean() if len(dot_error) else float('nan'):.2f} px "
        f"projector-coordinate error, "
        f"{dot_time * 1000:.1f} ms decode"
    )
    print(
        "Note: phase depth requires absolute fringe-order unwrapping and "
        "camera-projector calibration. Coded dots require the same calibration "
        "to triangulate matched camera/projector points."
    )


def load_calibration(args):
    camera = Path(args.camera_calibration)
    projector = Path(args.projector_calibration)
    stereo = Path(args.stereo_calibration)
    required = [
        camera / "cam_mtx.npy",
        camera / "cam_dist.npy",
        projector / "proj_mtx.npy",
        projector / "proj_dist.npy",
        stereo / "R.npy",
        stereo / "T.npy",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing calibration files:\n  " + "\n  ".join(missing))
    return {
        "cam_mtx": np.load(camera / "cam_mtx.npy"),
        "cam_dist": np.load(camera / "cam_dist.npy"),
        "proj_mtx": np.load(projector / "proj_mtx.npy"),
        "proj_dist": np.load(projector / "proj_dist.npy"),
        "proj_R": np.load(stereo / "R.npy"),
        "proj_T": np.load(stereo / "T.npy"),
    }


def run_hardware(args):
    calibration = load_calibration(args)
    projector_pattern, _ = generate_coded_dot_pattern(
        (args.projector_height, args.projector_width),
        seed=args.seed,
        dot_spacing=args.dot_spacing,
        dot_radius=args.dot_radius,
    )
    projector = ProjectorWindow(
        "Coded dots",
        size=(args.projector_width, args.projector_height),
        fallback_x=args.projector_x,
    )
    camera = Camera(
        args.camera,
        width=args.width,
        height=args.height,
        fps=args.fps,
    )
    try:
        projector.show(projector_pattern)
        cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
        print("Pattern displayed. Press SPACE in the camera window to capture, or Q to quit.")
        while True:
            frame = camera.get_frame()
            if frame is not None:
                preview = cv2.resize(frame, (960, 540))
                cv2.imshow("Camera", preview)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                return
            if key == ord(" "):
                if frame is None:
                    raise RuntimeError("No camera frame is available.")
                camera_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                break

        camera_points, projector_points, confidence = decode_coded_dots(
            projector_pattern, camera_image
        )
        print(f"Accepted coded-dot correspondences: {len(camera_points)}")
        if len(camera_points) < 4:
            raise RuntimeError("Too few coded-dot matches for triangulation.")

        triangulator = Triangulate(
            cam_size=(args.width, args.height),
            cam_mtx=calibration["cam_mtx"],
            cam_dist=calibration["cam_dist"],
            proj_size=(args.projector_width, args.projector_height),
            proj_calib_size=(args.projector_calibration_width, args.projector_calibration_height),
            proj_mtx=calibration["proj_mtx"],
            proj_dist=calibration["proj_dist"],
            proj_R=calibration["proj_R"],
            proj_T=calibration["proj_T"],
        )
        points = triangulator.triangulate(camera_points, projector_points)
        valid = (
            np.isfinite(points[2])
            & (points[2] > args.min_depth)
            & (points[2] < args.max_depth)
        )
        if not np.any(valid):
            raise RuntimeError("Calibration produced no points in front of the camera.")
        depths = points[2, valid]
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        write_ply(str(output_path), points[:, valid])
        print(
            f"Valid 3D points: {valid.sum()}/{len(valid)}; "
            f"depth range: {depths.min():.3f}-{depths.max():.3f} m; "
            f"median: {np.median(depths):.3f} m"
        )
        print(f"Point cloud written to {output_path}")
        print(f"Mean descriptor confidence: {confidence[valid].mean():.3f}")
    finally:
        camera.stop_cam()
        projector.close()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--projector-width", type=int, default=1920)
    parser.add_argument("--projector-height", type=int, default=1080)
    parser.add_argument("--projector-x", type=int, default=1920)
    parser.add_argument("--dot-spacing", type=int, default=16)
    parser.add_argument("--dot-radius", type=int, default=3)
    parser.add_argument("--hardware", action="store_true")
    parser.add_argument("--camera-calibration", default="./data/calib_results/elp_camera")
    parser.add_argument("--projector-calibration", default="./data/calib_results/my_proj")
    parser.add_argument("--stereo-calibration", default="./data/calib_results/my_stereo")
    parser.add_argument("--projector-calibration-width", type=int, default=1920)
    parser.add_argument("--projector-calibration-height", type=int, default=1080)
    parser.add_argument("--output", default="./data/point_clouds/coded_dots.ply")
    parser.add_argument("--min-depth", type=float, default=0.2)
    parser.add_argument("--max-depth", type=float, default=3.0)
    parser.add_argument("--period", type=int, default=24)
    parser.add_argument("--disparity", type=int, default=12)
    parser.add_argument("--noise", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.hardware:
        run_hardware(arguments)
    else:
        run_synthetic(arguments)
