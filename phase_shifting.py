"""Three-step phase-shifting structured-light algorithm."""

import cv2
import numpy as np


def three_step_patterns(shape, period_pixels=24, mean=127, amplitude=100):
    """Return sinusoidal patterns with phase offsets 0, 120, and 240 degrees."""
    height, width = shape
    x = np.arange(width, dtype=np.float32)[None, :]
    phase = 2 * np.pi * x / period_pixels
    return [
        np.broadcast_to(
            mean + amplitude * np.cos(phase + offset), (height, width)
        ).astype(np.uint8)
        for offset in (0, 2 * np.pi / 3, 4 * np.pi / 3)
    ]


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


def synthetic_phase_capture(shape, period, disparity, noise_std, rng):
    """Simulate three camera captures with a horizontal projector displacement."""
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
