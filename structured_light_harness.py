"""Compare three-step phase shifting with single-shot coded-dot projection.

The decoder functions operate on grayscale camera images. For real hardware,
replace the synthetic capture functions with synchronized ELP/projector
captures. The coded-dot decoder returns camera/projector correspondences;
triangulation still requires camera-projector calibration.
"""

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from scanner.acquisition import Camera, ProjectorWindow
from scanner.triangulation import Triangulate


@dataclass(frozen=True)
class Calibration:
    """Rectified stereo/projector geometry for disparity-to-depth conversion."""

    focal_length_pixels: float
    baseline_mm: float

    def depth_mm(self, disparity_pixels):
        disparity = np.asarray(disparity_pixels, dtype=np.float32)
        safe_disparity = np.maximum(np.abs(disparity), 1e-6)
        return self.focal_length_pixels * self.baseline_mm / safe_disparity


def three_step_patterns(shape, period_pixels=24, mean=127, amplitude=100):
    """Return visible sinusoidal patterns with phase offsets 0, 120, 240 degrees."""
    height, width = shape
    x = np.arange(width, dtype=np.float32)[None, :]
    phase = 2 * np.pi * x / period_pixels
    patterns = []
    for offset in (0, 2 * np.pi / 3, 4 * np.pi / 3):
        pattern = mean + amplitude * np.cos(phase + offset)
        patterns.append(np.broadcast_to(pattern, (height, width)).astype(np.uint8))
    return patterns


def decode_three_step(images, modulation_threshold=0.08):
    """Decode wrapped phase and a confidence mask from three camera images."""
    if len(images) != 3:
        raise ValueError("Three-step decoding requires exactly three images.")
    first, second, third = (np.asarray(image, dtype=np.float32) for image in images)
    if first.shape != second.shape or first.shape != third.shape:
        raise ValueError("All phase images must have the same dimensions.")

    numerator = np.sqrt(3.0) * (third - second)
    denominator = 2.0 * first - second - third
    phase = np.mod(np.arctan2(numerator, denominator), 2 * np.pi)
    modulation = np.sqrt(numerator**2 + denominator**2) / 3.0
    brightness = (first + second + third) / 3.0
    confidence = modulation / np.maximum(brightness, 1.0)
    return phase, confidence >= modulation_threshold


def phase_to_projector_x(phase, period_pixels, fringe_order=0):
    """Convert wrapped phase to projector x coordinates for one known order."""
    return (np.asarray(phase) / (2 * np.pi) + fringe_order) * period_pixels


def generate_coded_dot_pattern(
    shape, seed=7, dot_spacing=16, dot_radius=3, background=24
):
    """Create a deterministic pseudo-random dot image and its codebook."""
    height, width = shape
    rng = np.random.default_rng(seed)
    pattern = np.full((height, width), background, dtype=np.uint8)
    points = []
    for row, y in enumerate(range(dot_spacing // 2, height, dot_spacing)):
        for column, x in enumerate(range(dot_spacing // 2, width, dot_spacing)):
            if rng.random() > 0.5:
                jitter_x = int(rng.integers(-dot_spacing // 5, dot_spacing // 5 + 1))
                jitter_y = int(rng.integers(-dot_spacing // 5, dot_spacing // 5 + 1))
                point = (x + jitter_x, y + jitter_y)
                cv2.circle(
                    pattern,
                    point,
                    dot_radius,
                    255,
                    -1,
                    lineType=cv2.LINE_AA,
                )
                points.append(point)
    return pattern, np.asarray(points, dtype=np.float32)


def decode_coded_dots(projector_pattern, camera_image, ratio_threshold=0.75):
    """Return matched camera points, projector points, and descriptor confidence.

    ORB descriptors encode each dot's local pseudo-random neighborhood. The
    ratio test rejects ambiguous matches. This is a correspondence decoder,
    not a depth decoder; triangulate accepted pairs using calibrated camera and
    projector parameters.
    """
    projector_pattern = np.asarray(projector_pattern, dtype=np.uint8)
    camera_image = np.asarray(camera_image, dtype=np.uint8)
    if projector_pattern.ndim != 2 or camera_image.ndim != 2:
        raise ValueError("Projector and camera images must be grayscale.")
    orb = cv2.ORB_create(nfeatures=4000, fastThreshold=5)
    projector_keypoints, projector_descriptors = orb.detectAndCompute(
        projector_pattern, None
    )
    camera_keypoints, camera_descriptors = orb.detectAndCompute(camera_image, None)
    if projector_descriptors is None or camera_descriptors is None:
        return np.empty((0, 2)), np.empty((0, 2)), np.empty(0)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    pairs = matcher.knnMatch(camera_descriptors, projector_descriptors, k=2)
    camera_points, projector_points, confidence = [], [], []
    for pair in pairs:
        if len(pair) < 2 or pair[0].distance >= ratio_threshold * pair[1].distance:
            continue
        camera_points.append(camera_keypoints[pair[0].queryIdx].pt)
        projector_points.append(projector_keypoints[pair[0].trainIdx].pt)
        confidence.append(1.0 - pair[0].distance / 256.0)
    camera_points = np.asarray(camera_points, dtype=np.float32).reshape(-1, 2)
    projector_points = np.asarray(projector_points, dtype=np.float32).reshape(-1, 2)
    confidence = np.asarray(confidence, dtype=np.float32)
    if len(camera_points) >= 4:
        _, inlier_mask = cv2.findHomography(
            camera_points, projector_points, cv2.RANSAC, 3.0
        )
        if inlier_mask is not None:
            keep = inlier_mask.ravel().astype(bool)
            camera_points = camera_points[keep]
            projector_points = projector_points[keep]
            confidence = confidence[keep]
    return camera_points, projector_points, confidence


def synthetic_phase_capture(shape, period, disparity, noise_std, rng):
    patterns = three_step_patterns(shape, period)
    captures = []
    for pattern in patterns:
        shifted = cv2.warpAffine(
            pattern,
            np.float32([[1, 0, disparity], [0, 1, 0]]),
            (shape[1], shape[0]),
            borderMode=cv2.BORDER_REPLICATE,
        ).astype(np.float32)
        captures.append(np.clip(shifted + rng.normal(0, noise_std, shape), 0, 255))
    return captures


def synthetic_coded_dot_capture(shape, seed, noise_std, rng):
    projector_pattern, _ = generate_coded_dot_pattern(shape, seed)
    homography = np.float32(
        [[1.02, 0.015, 8.0], [0.01, 1.03, 5.0], [0.00005, 0.00002, 1.0]]
    )
    camera_image = cv2.warpPerspective(
        projector_pattern, homography, (shape[1], shape[0]), borderValue=20
    ).astype(np.float32)
    camera_image += rng.normal(0, noise_std, shape)
    return projector_pattern, np.clip(camera_image, 0, 255).astype(np.uint8), homography


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
        valid = np.isfinite(points[2]) & (points[2] > 0)
        if not np.any(valid):
            raise RuntimeError("Calibration produced no points in front of the camera.")
        depths = points[2, valid]
        print(
            f"Valid 3D points: {valid.sum()}/{len(valid)}; "
            f"depth range: {depths.min():.3f}-{depths.max():.3f} m; "
            f"median: {np.median(depths):.3f} m"
        )
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
