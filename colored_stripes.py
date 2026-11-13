"""Single-shot RGB sinusoidal stripe projection and decoding."""

import cv2
import numpy as np


def generate_color_stripe_pattern(
    shape, period_pixels, mean=127, amplitude=100, axis="x"
):
    """Create one BGR image whose three channels carry phase-shifted stripes."""
    height, width = shape
    if axis == "x":
        coordinate = np.arange(width, dtype=np.float32)[None, :]
    elif axis == "y":
        coordinate = np.arange(height, dtype=np.float32)[:, None]
    else:
        raise ValueError("axis must be 'x' or 'y'")
    phase = 2 * np.pi * coordinate / period_pixels
    channels = [
        mean + amplitude * np.cos(phase + offset)
        for offset in (0, 2 * np.pi / 3, 4 * np.pi / 3)
    ]
    return np.stack(
        [np.broadcast_to(channel, (height, width)) for channel in channels],
        axis=2,
    ).clip(0, 255).astype(np.uint8)


def decode_color_stripes(
    image, period_pixels, modulation_threshold=0.15, normalize_channels=True
):
    """Decode projector X and a confidence mask from one BGR camera image."""
    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Color stripe decoding requires a BGR image.")
    if normalize_channels:
        normalized = np.empty_like(image)
        reference = np.median(
            np.percentile(image, [5, 95], axis=(0, 1)), axis=1
        )
        for index in range(3):
            low, high = np.percentile(image[:, :, index], [5, 95])
            scale = reference[1] - reference[0]
            channel_scale = max(high - low, 1.0)
            normalized[:, :, index] = (
                (image[:, :, index] - low) * scale / channel_scale
                + reference[0]
            )
        image = normalized
    first, second, third = (image[:, :, index] for index in range(3))
    numerator = np.sqrt(3.0) * (third - second)
    denominator = 2.0 * first - second - third
    phase = np.mod(np.arctan2(numerator, denominator), 2 * np.pi)
    modulation = np.sqrt(numerator**2 + denominator**2) / 3.0
    brightness = (first + second + third) / 3.0
    confidence = modulation / np.maximum(brightness, 1.0)
    projector_x = phase / (2 * np.pi) * period_pixels
    return projector_x, confidence >= modulation_threshold
