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
from colored_stripes import (
    decode_color_stripes,
    generate_color_stripe_pattern,
)
from phase_shifting import (
    decode_three_step,
    phase_to_projector_x,
    synthetic_phase_capture,
    three_step_patterns,
)


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


def create_hardware_devices(args, title):
    projector = ProjectorWindow(
        title,
        size=(args.projector_width, args.projector_height),
        fallback_x=args.projector_x,
    )
    camera = Camera(
        args.camera,
        width=args.width,
        height=args.height,
        fps=args.fps,
    )
    cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline and camera.get_frame() is None:
        cv2.waitKey(10)
    return projector, camera


def select_camera_roi(camera, projector, pattern, args):
    if args.camera_roi is not None:
        return tuple(args.camera_roi)
    if args.full_frame:
        return None

    image = np.asarray(pattern, dtype=np.uint8)
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    projector.show(image)

    frame = None
    settle_until = time.monotonic() + args.projector_settle
    while time.monotonic() < settle_until:
        camera.get_frame()
        cv2.waitKey(10)

    deadline = time.monotonic() + 5.0
    while frame is None and time.monotonic() < deadline:
        frame = camera.get_frame()
        if frame is not None:
            cv2.imshow("Camera", frame)
        cv2.waitKey(10)
    if frame is None:
        raise RuntimeError("Could not obtain a camera frame for ROI selection.")

    print("Drag a rectangle around the scan area, then press ENTER or SPACE. "
          "Press C to cancel.")
    roi = cv2.selectROI("Camera", frame, fromCenter=False, showCrosshair=True)
    x, y, width, height = (int(value) for value in roi)
    if width <= 0 or height <= 0:
        raise RuntimeError("No scan area was selected.")
    print(f"Selected camera ROI: x={x}, y={y}, width={width}, height={height}")
    return x, y, width, height


def capture_frames(projector, camera, patterns, args):
    captures = []
    for index, pattern in enumerate(patterns, start=1):
        image = np.asarray(pattern, dtype=np.uint8)
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        projector.show(image)
        print(f"Pattern {index}/{len(patterns)} displayed. Press SPACE to capture, or Q to quit.")
        frame = None
        first_frame_deadline = time.monotonic() + 5.0
        while True:
            candidate = camera.get_frame()
            if candidate is not None:
                frame = candidate
                cv2.imshow("Camera", cv2.resize(frame, (960, 540)))
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                raise KeyboardInterrupt
            if key == ord(" "):
                if frame is None:
                    if time.monotonic() < first_frame_deadline:
                        print("Waiting for the first camera frame...")
                        continue
                    raise RuntimeError(
                        "Camera opened but produced no frames. Check the camera index, "
                        "resolution, and whether another application is using it."
                    )
                captures.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
                break
    return captures


def capture_color_frames(projector, camera, patterns, args):
    captures = []
    for index, pattern in enumerate(patterns, start=1):
        projector.show(pattern)
        print(
            f"Color pattern {index}/{len(patterns)} displayed. "
            "Press SPACE to capture, or Q to quit."
        )
        frame = None
        deadline = time.monotonic() + 5.0
        while True:
            candidate = camera.get_frame()
            if candidate is not None:
                frame = candidate
                cv2.imshow("Camera", cv2.resize(frame, (960, 540)))
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                raise KeyboardInterrupt
            if key == ord(" "):
                if frame is None:
                    if time.monotonic() < deadline:
                        continue
                    raise RuntimeError("Camera opened but produced no frames.")
                captures.append(frame.copy())
                break
    return captures


def run_phase_hardware(args, calibration):
    horizontal_patterns = three_step_patterns(
        (args.projector_height, args.projector_width),
        period_pixels=args.phase_period_x,
        mean=args.phase_mean,
        amplitude=args.phase_amplitude,
        axis="x",
    )
    horizontal_fine_patterns = three_step_patterns(
        (args.projector_height, args.projector_width),
        period_pixels=args.phase_fine_period_x,
        mean=args.phase_mean,
        amplitude=args.phase_amplitude,
        axis="x",
    )
    vertical_patterns = three_step_patterns(
        (args.projector_height, args.projector_width),
        period_pixels=args.phase_period_y,
        mean=args.phase_mean,
        amplitude=args.phase_amplitude,
        axis="y",
    )
    vertical_fine_patterns = three_step_patterns(
        (args.projector_height, args.projector_width),
        period_pixels=args.phase_fine_period_y,
        mean=args.phase_mean,
        amplitude=args.phase_amplitude,
        axis="y",
    )
    projector, camera = create_hardware_devices(args, "Phase shifting")
    try:
        args.camera_roi = select_camera_roi(camera, projector, horizontal_patterns[0], args)
        captures_x = capture_frames(projector, camera, horizontal_patterns, args)
        captures_x_fine = capture_frames(projector, camera, horizontal_fine_patterns, args)
        captures_y = capture_frames(projector, camera, vertical_patterns, args)
        captures_y_fine = capture_frames(projector, camera, vertical_fine_patterns, args)
        phase_x, mask_x = decode_three_step(captures_x, args.modulation_threshold)
        phase_x_fine, mask_x_fine = decode_three_step(
            captures_x_fine, args.modulation_threshold
        )
        phase_y, mask_y = decode_three_step(captures_y, args.modulation_threshold)
        phase_y_fine, mask_y_fine = decode_three_step(
            captures_y_fine, args.modulation_threshold
        )
        projector_x = unwrap_phase_coordinate(
            phase_x, phase_x_fine, args.phase_period_x, args.phase_fine_period_x
        )
        projector_y = unwrap_phase_coordinate(
            phase_y, phase_y_fine, args.phase_period_y, args.phase_fine_period_y
        )
        valid = (
            mask_x
            & mask_x_fine
            & mask_y
            & mask_y_fine
            & (projector_x >= 0)
            & (projector_x < args.projector_width)
            & (projector_y >= 0)
            & (projector_y < args.projector_height)
        )
        ys, xs = np.nonzero(valid)
        if not len(xs):
            raise RuntimeError("Phase decoding produced no valid pixels.")
        step = max(1, args.phase_sample_step)
        selected = np.arange(0, len(xs), step)
        cam_pts = np.column_stack((xs[selected], ys[selected])).astype(np.float32)
        proj_pts = np.column_stack(
            (projector_x[ys[selected], xs[selected]], projector_y[ys[selected], xs[selected]])
        ).astype(np.float32)
        roi = camera_roi_mask(cam_pts, args)
        cam_pts = cam_pts[roi]
        proj_pts = proj_pts[roi]
        if not len(cam_pts):
            raise RuntimeError("The camera ROI excluded all decoded phase points.")
        triangulator = make_triangulator(args, calibration)
        points = triangulator.triangulate(cam_pts, proj_pts)
        valid_3d = (
            np.isfinite(points[2])
            & (points[2] > args.min_depth)
            & (points[2] < args.max_depth)
        )
        if args.foreground_only and args.background_depth_tolerance is None:
            raise RuntimeError(
                "--foreground-only requires --background-depth-tolerance."
            )
        valid_3d = filter_background_plane(
            points, cam_pts, valid_3d, args, "phase"
        )
        if not np.any(valid_3d):
            raise RuntimeError("Phase decoding produced no valid 3D points.")
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        write_ply(str(output_path), points[:, valid_3d])
        np.save(output_path.with_suffix(".phase_x.npy"), phase_x)
        np.save(output_path.with_suffix(".phase_y.npy"), phase_y)
        print(
            f"Valid phase 3D points: {valid_3d.sum()}/{len(valid_3d)}; "
            f"depth range: {points[2, valid_3d].min():.3f}-"
            f"{points[2, valid_3d].max():.3f} m"
        )
        print(f"Point cloud written to {output_path}")
        print("Phase note: twelve frames were used with coarse/fine phase unwrapping.")
    finally:
        camera.stop_cam()
        projector.close()


def run_color_stripes_hardware(args, calibration):
    pattern_x = generate_color_stripe_pattern(
        (args.projector_height, args.projector_width),
        args.color_stripe_period,
        args.phase_mean,
        args.phase_amplitude,
        axis="x",
    )
    pattern_y = generate_color_stripe_pattern(
        (args.projector_height, args.projector_width),
        args.color_stripe_period,
        args.phase_mean,
        args.phase_amplitude,
        axis="y",
    )
    projector, camera = create_hardware_devices(args, "Color stripes")
    try:
        args.camera_roi = select_camera_roi(camera, projector, pattern_x, args)
        captures = capture_color_frames(
            projector, camera, [pattern_x, pattern_y], args
        )

        capture_dir = Path(args.output).parent
        capture_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(capture_dir / "color_stripes_x_camera.png"), captures[0])
        cv2.imwrite(str(capture_dir / "color_stripes_y_camera.png"), captures[1])
        cv2.imwrite(str(capture_dir / "color_stripes_x_pattern.png"), pattern_x)
        cv2.imwrite(str(capture_dir / "color_stripes_y_pattern.png"), pattern_y)
        projector_x, mask_x = decode_color_stripes(
            captures[0], args.color_stripe_period, args.modulation_threshold
        )
        projector_y, mask_y = decode_color_stripes(
            captures[1], args.color_stripe_period, args.modulation_threshold
        )
        valid = mask_x & mask_y
        ys, xs = np.nonzero(valid)
        cam_pts = np.column_stack((xs, ys)).astype(np.float32)
        proj_x = projector_x[ys, xs]
        proj_y = projector_y[ys, xs]
        proj_pts = np.column_stack((proj_x, proj_y)).astype(np.float32)
        roi = camera_roi_mask(cam_pts, args)
        cam_pts, proj_pts = cam_pts[roi], proj_pts[roi]
        if not len(cam_pts):
            raise RuntimeError("Color stripe decoding produced no valid ROI pixels.")
        triangulator = make_triangulator(args, calibration)
        points = triangulator.triangulate(cam_pts, proj_pts)
        valid = (
            np.isfinite(points[2])
            & (points[2] > args.min_depth)
            & (points[2] < args.max_depth)
        )
        if args.foreground_only and args.background_depth_tolerance is None:
            raise RuntimeError(
                "--foreground-only requires --background-depth-tolerance."
            )
        valid = filter_background_plane(
            points, cam_pts, valid, args, "color-stripe"
        )
        valid = filter_local_depth_outliers(points, cam_pts, valid, args)
        valid = filter_depth_band(points, valid, args, "color-stripe")
        if not np.any(valid):
            raise RuntimeError("Calibration produced no valid color-stripe points.")
        output_path = Path(args.output)
        write_ply(str(output_path), points[:, valid])
        print(
            f"Valid color-stripe points: {valid.sum()}/{len(valid)}; "
            f"depth range: {points[2, valid].min():.3f}-"
            f"{points[2, valid].max():.3f} m"
        )
        print(f"Point cloud written to {output_path}")
        print("Two-shot RGB color-stripe reconstruction completed.")
    finally:
        camera.stop_cam()
        projector.close()


def make_triangulator(args, calibration):
    return Triangulate(
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


def unwrap_phase_coordinate(coarse_phase, fine_phase, coarse_period, fine_period):
    coarse = phase_to_projector_x(coarse_phase, coarse_period)
    fine = phase_to_projector_x(fine_phase, fine_period)
    order = np.rint((coarse - fine) / fine_period)
    return fine + order * fine_period


def camera_roi_mask(points, args):
    if args.camera_roi is None:
        return np.ones(len(points), dtype=bool)
    x, y, width, height = args.camera_roi
    return (
        (points[:, 0] >= x)
        & (points[:, 0] < x + width)
        & (points[:, 1] >= y)
        & (points[:, 1] < y + height)
    )


def build_camera_roi_mask(args):
    mask = np.ones((args.height, args.width), dtype=np.uint8) * 255
    if args.camera_roi is None:
        return mask
    x, y, width, height = args.camera_roi
    mask.fill(0)
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(args.width, x + width)
    y1 = min(args.height, y + height)
    if x1 > x0 and y1 > y0:
        mask[y0:y1, x0:x1] = 255
    return mask


def filter_background_plane(points, cam_pts, valid, args, label):
    if args.foreground_only and args.background_depth_tolerance is None:
        raise RuntimeError(
            "--foreground-only requires --background-depth-tolerance."
        )
    if args.background_depth_tolerance is None:
        return valid
    if args.camera_roi is None:
        x, y, width, height = 0, 0, args.width, args.height
    else:
        x, y, width, height = args.camera_roi
    border = (
        (cam_pts[:, 0] < x + width * 0.12)
        | (cam_pts[:, 0] >= x + width * 0.88)
        | (cam_pts[:, 1] < y + height * 0.12)
        | (cam_pts[:, 1] >= y + height * 0.88)
    )
    border_indices = np.flatnonzero(valid & border)
    if len(border_indices) < 20:
        raise RuntimeError(
            f"Not enough valid {label} ROI-border points to estimate "
            "the background plane."
        )
    background_points = points[:, border_indices].T
    design = np.column_stack(
        (background_points[:, 0], background_points[:, 1], np.ones(len(background_points)))
    )
    coefficients, _, _, _ = np.linalg.lstsq(
        design, background_points[:, 2], rcond=None
    )
    residuals = background_points[:, 2] - design @ coefficients
    scale = max(float(np.median(np.abs(residuals))) * 2.5, 0.01)
    inliers = np.abs(residuals) <= scale
    if inliers.sum() < 10:
        raise RuntimeError(f"Could not fit a stable {label} background plane.")
    coefficients, _, _, _ = np.linalg.lstsq(
        design[inliers], background_points[inliers, 2], rcond=None
    )
    predicted = (
        coefficients[0] * points[0]
        + coefficients[1] * points[1]
        + coefficients[2]
    )
    difference = points[2] - predicted
    band = args.background_depth_tolerance
    if args.foreground_only:
        valid &= difference < -band
    else:
        valid &= np.abs(difference) > band
    print(
        f"Removed {label} points using a fitted background plane "
        f"(border depth {background_points[:, 2].min():.3f}-"
        f"{background_points[:, 2].max():.3f} m, ±{band:.3f} m band)."
    )
    return valid


def filter_local_depth_outliers(points, cam_pts, valid, args):
    """Reject isolated depth spikes while preserving broad surface changes."""
    if args.depth_outlier_tolerance <= 0:
        return valid
    depth = np.full((args.height, args.width), np.nan, dtype=np.float32)
    present = np.zeros((args.height, args.width), dtype=np.float32)
    rounded = np.rint(cam_pts).astype(int)
    inside = (
        (rounded[:, 0] >= 0)
        & (rounded[:, 0] < args.width)
        & (rounded[:, 1] >= 0)
        & (rounded[:, 1] < args.height)
    )
    indices = np.flatnonzero(inside)
    depth[rounded[indices, 1], rounded[indices, 0]] = points[2, indices]
    present[rounded[indices, 1], rounded[indices, 0]] = valid[indices].astype(
        np.float32
    )
    finite_depths = depth[np.isfinite(depth)]
    if not len(finite_depths):
        return valid
    fill_value = float(np.median(finite_depths))
    filled = np.nan_to_num(depth, nan=fill_value).astype(np.float32)
    local_median = cv2.medianBlur(filled, 5)
    neighborhood = cv2.boxFilter(present, -1, (5, 5), normalize=False)
    keep = np.abs(points[2] - local_median[rounded[:, 1], rounded[:, 0]]) <= (
        args.depth_outlier_tolerance
    )
    keep &= neighborhood[rounded[:, 1], rounded[:, 0]] >= 5
    return valid & keep


def filter_depth_band(points, valid, args, label):
    if args.depth_band_tolerance <= 0 or valid.sum() < 20:
        return valid
    depths = points[2, valid]
    center = float(np.median(depths))
    spread = float(np.median(np.abs(depths - center)))
    allowed = max(args.depth_band_tolerance, spread * 4.0)
    keep = np.abs(points[2] - center) <= allowed
    print(
        f"Limited {label} Z depth around {center:.3f} m "
        f"to ±{allowed:.3f} m."
    )
    return valid & keep


def run_dots_hardware(args, calibration):
    projector_pattern, _ = generate_coded_dot_pattern(
        (args.projector_height, args.projector_width),
        seed=args.seed,
        dot_spacing=args.dot_spacing,
        dot_radius=args.dot_radius,
        density=args.dot_density,
    )
    projector, camera = create_hardware_devices(args, "Coded dots")
    try:
        args.camera_roi = select_camera_roi(camera, projector, projector_pattern, args)
        projector.show(projector_pattern)
        cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
        print("Pattern displayed. Press SPACE in the camera window to capture, or Q to quit.")
        frame = None
        first_frame_deadline = time.monotonic() + 5.0
        while True:
            candidate = camera.get_frame()
            if candidate is not None:
                frame = candidate
                preview = cv2.resize(frame, (960, 540))
                cv2.imshow("Camera", preview)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                return
            if key == ord(" "):
                if frame is None:
                    if time.monotonic() < first_frame_deadline:
                        print("Waiting for the first camera frame...")
                        continue
                    raise RuntimeError(
                        "Camera opened but produced no frames. Check the camera index, "
                        "resolution, and whether another application is using it."
                    )
                camera_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                break

        capture_dir = Path(args.output).parent
        capture_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(capture_dir / "dots_camera_capture.png"), camera_image)
        cv2.imwrite(str(capture_dir / "dots_projector_pattern.png"), projector_pattern)
        camera_points, projector_points, confidence, diagnostics = decode_coded_dots(
            projector_pattern,
            camera_image,
            camera_mask=build_camera_roi_mask(args),
            geometric_filter=False,
            return_diagnostics=True,
        )
        roi = camera_roi_mask(camera_points, args)
        camera_points = camera_points[roi]
        projector_points = projector_points[roi]
        confidence = confidence[roi]
        print(
            "Coded-dot features: "
            f"detector={diagnostics['detector']}; "
            f"projector={diagnostics['projector_keypoints']}, "
            f"camera={diagnostics['camera_keypoints']}; "
            f"ratio={diagnostics['ratio_matches']}, "
            f"mutual={diagnostics['mutual_matches']}"
        )
        print(f"Accepted coded-dot correspondences: {len(camera_points)}")
        if len(camera_points) < 4:
            raise RuntimeError("Too few coded-dot matches for triangulation.")

        triangulator = make_triangulator(args, calibration)
        points = triangulator.triangulate(camera_points, projector_points)
        valid = (
            np.isfinite(points[2])
            & (points[2] > args.min_depth)
            & (points[2] < args.max_depth)
        )
        valid = filter_background_plane(
            points, camera_points, valid, args, "coded-dot"
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


def run_hardware(args):
    calibration = load_calibration(args)
    if args.algorithm == "phase":
        run_phase_hardware(args, calibration)
    elif args.algorithm == "color_stripes":
        run_color_stripes_hardware(args, calibration)
    else:
        run_dots_hardware(args, calibration)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--camera", type=int, default=1)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument(
        "--projector-settle",
        type=float,
        default=1.0,
        help="Seconds to wait after showing a pattern before ROI selection.",
    )
    parser.add_argument("--projector-width", type=int, default=1280)
    parser.add_argument("--projector-height", type=int, default=720)
    parser.add_argument("--projector-x", type=int, default=1920)
    parser.add_argument(
        "--dot-spacing",
        type=int,
        default=20,
        help="Projector-space spacing between coded dots (default: 20).",
    )
    parser.add_argument(
        "--dot-radius",
        type=int,
        default=6,
        help="Projector-space coded-dot radius (default: 6).",
    )
    parser.add_argument(
        "--dot-density",
        type=float,
        default=0.65,
        help="Fraction of dot-grid cells that contain a dot (default: 0.65).",
    )
    parser.add_argument("--hardware", action="store_true")
    parser.add_argument(
        "--algorithm",
        choices=("dots", "phase", "color_stripes"),
        default="dots",
    )
    parser.add_argument("--camera-calibration", default="./data/calib_results/elp_camera")
    parser.add_argument("--projector-calibration", default="./data/calib_results/my_proj")
    parser.add_argument("--stereo-calibration", default="./data/calib_results/my_stereo")
    parser.add_argument("--projector-calibration-width", type=int, default=1280)
    parser.add_argument("--projector-calibration-height", type=int, default=720)
    parser.add_argument("--output", default="./data/point_clouds/coded_dots.ply")
    parser.add_argument("--min-depth", type=float, default=0.2)
    parser.add_argument("--max-depth", type=float, default=3.0)
    parser.add_argument(
        "--depth-outlier-tolerance",
        type=float,
        default=0.03,
        help="Reject isolated depth spikes beyond this local tolerance in meters.",
    )
    parser.add_argument(
        "--depth-band-tolerance",
        type=float,
        default=0.20,
        help="Keep points within this robust Z-depth band around the median (m).",
    )
    parser.add_argument("--period", type=int, default=24)
    parser.add_argument("--fringe-order", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--phase-period-x", type=int, default=1280)
    parser.add_argument("--phase-period-y", type=int, default=720)
    parser.add_argument("--phase-fine-period-x", type=int, default=64)
    parser.add_argument("--phase-fine-period-y", type=int, default=64)
    parser.add_argument("--fringe-order-x", type=int, default=0)
    parser.add_argument("--fringe-order-y", type=int, default=0)
    parser.add_argument("--phase-mean", type=int, default=127)
    parser.add_argument("--phase-amplitude", type=int, default=100)
    parser.add_argument(
        "--color-stripe-period",
        type=int,
        default=1280,
        help="Projector pixels per RGB stripe cycle for both axes (default: 1280).",
    )
    parser.add_argument("--modulation-threshold", type=float, default=0.15)
    parser.add_argument(
        "--phase-sample-step",
        type=int,
        default=1,
        help="Keep every Nth valid phase pixel (default: 1, keep all).",
    )
    parser.add_argument(
        "--camera-roi",
        type=int,
        nargs=4,
        metavar=("X", "Y", "WIDTH", "HEIGHT"),
        help="Restrict output to a camera rectangle before triangulation.",
    )
    parser.add_argument(
        "--full-frame",
        action="store_true",
        help="Skip manual ROI selection and scan the complete camera image.",
    )
    parser.add_argument(
        "--background-depth-tolerance",
        type=float,
        default=None,
        help="Remove points within this depth (m) of the ROI-border background.",
    )
    parser.add_argument(
        "--foreground-only",
        action="store_true",
        help="Keep only points in front of the estimated ROI-border background.",
    )
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
