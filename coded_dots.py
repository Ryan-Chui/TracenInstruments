"""Single-shot coded-dot camera/projector algorithm."""

import cv2
import numpy as np


def generate_coded_dot_pattern(
    shape, seed=7, dot_spacing=16, dot_radius=3, background=24
):
    """Create a deterministic pseudo-random dot image and its codebook."""
    height, width = shape
    rng = np.random.default_rng(seed)
    pattern = np.full((height, width), background, dtype=np.uint8)
    points = []
    for y in range(dot_spacing // 2, height, dot_spacing):
        for x in range(dot_spacing // 2, width, dot_spacing):
            if rng.random() > 0.5:
                jitter_x = int(rng.integers(-dot_spacing // 5, dot_spacing // 5 + 1))
                jitter_y = int(rng.integers(-dot_spacing // 5, dot_spacing // 5 + 1))
                point = (x + jitter_x, y + jitter_y)
                cv2.circle(pattern, point, dot_radius, 255, -1, lineType=cv2.LINE_AA)
                points.append(point)
    return pattern, np.asarray(points, dtype=np.float32)


def decode_coded_dots(projector_pattern, camera_image, ratio_threshold=0.75):
    """Return matched camera/projector points and descriptor confidence."""
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
