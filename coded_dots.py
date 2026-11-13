"""Single-shot coded-dot camera/projector algorithm."""

import cv2
import numpy as np


def generate_coded_dot_pattern(
    shape, seed=7, dot_spacing=20, dot_radius=6, density=0.65, background=24
):
    """Create a deterministic pseudo-random dot image and its codebook."""
    height, width = shape
    rng = np.random.default_rng(seed)
    pattern = np.full((height, width), background, dtype=np.uint8)
    points = []
    for y in range(dot_spacing // 2, height, dot_spacing):
        for x in range(dot_spacing // 2, width, dot_spacing):
            if rng.random() < density:
                jitter_x = int(rng.integers(-dot_spacing // 5, dot_spacing // 5 + 1))
                jitter_y = int(rng.integers(-dot_spacing // 5, dot_spacing // 5 + 1))
                point = (x + jitter_x, y + jitter_y)
                cv2.circle(pattern, point, dot_radius, 255, -1, lineType=cv2.LINE_AA)
                points.append(point)
    return pattern, np.asarray(points, dtype=np.float32)


def _normalize_for_features(image):
    """Improve dot contrast without changing the image geometry."""
    image = np.asarray(image, dtype=np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)


def decode_coded_dots(
    projector_pattern,
    camera_image,
    ratio_threshold=0.8,
    ransac_threshold=3.0,
    camera_mask=None,
    geometric_filter=True,
    return_diagnostics=False,
):
    """Return geometrically consistent camera/projector dot matches.

    The projector pattern and camera image may have different perspective
    distortion. Matches are accepted only when they pass a mutual ratio test
    and agree with a single planar homography.
    """
    projector_pattern = np.asarray(projector_pattern, dtype=np.uint8)
    camera_image = np.asarray(camera_image, dtype=np.uint8)
    if projector_pattern.ndim != 2 or camera_image.ndim != 2:
        raise ValueError("Projector and camera images must be grayscale.")
    if camera_mask is not None:
        camera_mask = np.asarray(camera_mask, dtype=np.uint8)
        if camera_mask.shape != camera_image.shape:
            raise ValueError("Camera mask must match the camera image dimensions.")
    projector_features = _normalize_for_features(projector_pattern)
    camera_features = _normalize_for_features(camera_image)
    if hasattr(cv2, "SIFT_create"):
        detector = cv2.SIFT_create(
            nfeatures=10000,
            contrastThreshold=0.01,
            edgeThreshold=10,
        )
        descriptor_norm = cv2.NORM_L2
        detector_name = "SIFT"
    else:
        detector = cv2.ORB_create(
            nfeatures=10000,
            scaleFactor=1.1,
            nlevels=8,
            edgeThreshold=5,
            patchSize=31,
            fastThreshold=3,
        )
        descriptor_norm = cv2.NORM_HAMMING
        detector_name = "ORB"
    projector_keypoints, projector_descriptors = detector.detectAndCompute(
        projector_features, None
    )
    camera_keypoints, camera_descriptors = detector.detectAndCompute(
        camera_features, camera_mask
    )
    diagnostics = {
        "projector_keypoints": len(projector_keypoints),
        "camera_keypoints": len(camera_keypoints),
        "ratio_matches": 0,
        "mutual_matches": 0,
        "geometric_matches": 0,
        "detector": detector_name,
    }
    if projector_descriptors is None or camera_descriptors is None:
        result = (np.empty((0, 2)), np.empty((0, 2)), np.empty(0))
        return (*result, diagnostics) if return_diagnostics else result
    matcher = cv2.BFMatcher(descriptor_norm)
    camera_to_projector = matcher.knnMatch(
        camera_descriptors, projector_descriptors, k=2
    )
    projector_to_camera = matcher.knnMatch(
        projector_descriptors, camera_descriptors, k=2
    )
    reverse_best = {}
    for pair in projector_to_camera:
        if len(pair) == 2 and pair[0].distance < ratio_threshold * pair[1].distance:
            reverse_best[pair[0].queryIdx] = pair[0].trainIdx

    camera_points, projector_points, confidence = [], [], []
    for pair in camera_to_projector:
        if len(pair) < 2 or pair[0].distance >= ratio_threshold * pair[1].distance:
            continue
        diagnostics["ratio_matches"] += 1
        camera_match = pair[0]
        if reverse_best.get(camera_match.trainIdx) != camera_match.queryIdx:
            continue
        diagnostics["mutual_matches"] += 1
        camera_points.append(camera_keypoints[camera_match.queryIdx].pt)
        projector_points.append(projector_keypoints[camera_match.trainIdx].pt)
        confidence.append(
            1.0 - camera_match.distance / max(pair[1].distance, 1.0)
        )
    camera_points = np.asarray(camera_points, dtype=np.float32).reshape(-1, 2)
    projector_points = np.asarray(projector_points, dtype=np.float32).reshape(-1, 2)
    confidence = np.asarray(confidence, dtype=np.float32)
    diagnostics["geometric_matches"] = len(camera_points)
    if geometric_filter and len(camera_points) >= 4:
        homography, inlier_mask = cv2.findHomography(
            camera_points, projector_points, cv2.RANSAC, ransac_threshold
        )
        if homography is not None and inlier_mask is not None:
            keep = inlier_mask.ravel().astype(bool)
            projected = cv2.perspectiveTransform(
                camera_points.reshape(-1, 1, 2), homography
            ).reshape(-1, 2)
            reprojection_error = np.linalg.norm(
                projected - projector_points, axis=1
            )
            keep &= reprojection_error <= ransac_threshold
            camera_points = camera_points[keep]
            projector_points = projector_points[keep]
            confidence = confidence[keep] * np.exp(
                -reprojection_error[keep] / max(ransac_threshold, 1e-6)
            )
    if geometric_filter:
        diagnostics["geometric_matches"] = len(camera_points)
    result = (camera_points, projector_points, confidence)
    return (*result, diagnostics) if return_diagnostics else result


def synthetic_coded_dot_capture(shape, seed, noise_std, rng):
    """Simulate one camera capture of a perspective-distorted dot pattern."""
    projector_pattern, _ = generate_coded_dot_pattern(shape, seed)
    homography = np.float32(
        [[1.02, 0.015, 8.0], [0.01, 1.03, 5.0], [0.00005, 0.00002, 1.0]]
    )
    camera_image = cv2.warpPerspective(
        projector_pattern, homography, (shape[1], shape[0]), borderValue=20
    ).astype(np.float32)
    camera_image += rng.normal(0, noise_std, shape)
    return projector_pattern, np.clip(camera_image, 0, 255).astype(np.uint8), homography
