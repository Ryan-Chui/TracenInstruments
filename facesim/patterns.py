"""
Structured-light patterns and decoding.

This module is hardware-agnostic: the same `PatternSet` makes the images you
send to the real DLP projector and decodes the frames your USB camera
captures. The virtual scanner uses it too, so code you test on synthetic
captures runs unchanged on the real rig.

Sequence (column coding, vertical stripes):
    white, black,
    Gray-code bit 0 (MSB), its inverse, bit 1, inverse, ..., bit n-1, inverse,
    [optional] N phase-shifted sinusoids for sub-pixel precision.

Decoding gives, for every camera pixel, the projector column it sees
(float). Triangulation then intersects the camera ray with that column's
light plane (see scanner.triangulate_columns).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DecodeSettings:
    min_contrast: float = 10.0     # white - black (DN) below this = shadow / no signal
    bit_abs: float = 3.0           # |bit - inverse| below this (DN) = unsure bit ...
    bit_rel: float = 0.06          # ... or below this fraction of (white - black)
    min_reliable_stripe: int = 4   # stripes narrower than this (projector px) may be unsure
    saturation: float = 254.0      # white >= this = clipped, reject
    min_phase_modulation: float = 0.1  # phase amplitude / (white - black)


class PatternSet:
    def __init__(self, proj_width: int, proj_height: int, phase_steps: int = 0, phase_period: int = 16):
        self.w, self.h = int(proj_width), int(proj_height)
        self.n_bits = int(np.ceil(np.log2(self.w)))
        self.phase_steps = int(phase_steps)
        self.phase_period = int(phase_period)
        cols = np.arange(self.w)
        gray = cols ^ (cols >> 1)
        self._bits = [((gray >> (self.n_bits - 1 - k)) & 1).astype(np.float64) for k in range(self.n_bits)]

    @property
    def names(self) -> list[str]:
        n = ["white", "black"]
        for k in range(self.n_bits):
            n += [f"gray{k:02d}", f"gray{k:02d}_inv"]
        n += [f"phase{j}" for j in range(self.phase_steps)]
        return n

    def column_profiles(self) -> np.ndarray:
        """
        (n_patterns, proj_width) intensity in [0,1] for each projector column.
        """
        rows = [np.ones(self.w), np.zeros(self.w)]
        for b in self._bits:
            rows += [b, 1.0 - b]
        cols = np.arange(self.w)
        for j in range(self.phase_steps):
            rows.append(0.5 + 0.5 * np.cos(2 * np.pi * cols / self.phase_period - 2 * np.pi * j / self.phase_steps))
        return np.stack(rows)

    def projector_images(self) -> list[np.ndarray]:
        """
        uint8 images (proj_height, proj_width) to display on the real projector, in order.
        """
        prof = np.round(self.column_profiles() * 255).astype(np.uint8)
        return [np.repeat(p[None, :], self.h, axis=0) for p in prof]

    def stripe_width(self, k: int) -> int:
        return 2 ** (self.n_bits - 1) if k == 0 else 2 ** (self.n_bits - k)

    def decode(self, images, settings: DecodeSettings | None = None):
        """
        Decode captured frames (same order as `names`).

        Returns
        -------
        column : (H,W) float projector column seen by each camera pixel (nan = invalid)
        valid  : (H,W) bool
        contrast : (H,W) float, white - black (useful as an intensity / confidence map)
        """
        s = settings or DecodeSettings()
        imgs = [np.asarray(im, dtype=np.float32) for im in images]
        if imgs[0].ndim == 3:  # color capture -> gray
            imgs = [im.mean(-1) for im in imgs]
        white, black = imgs[0], imgs[1]
        contrast = white - black
        valid = (contrast > s.min_contrast) & (white < s.saturation)

        thresh = np.maximum(s.bit_abs, s.bit_rel * contrast)
        binary = np.zeros(white.shape, np.int64)
        prev = np.zeros(white.shape, np.int64)
        for k in range(self.n_bits):
            d = imgs[2 + 2 * k] - imgs[3 + 2 * k]
            g = (d > 0).astype(np.int64)
            if self.stripe_width(k) >= s.min_reliable_stripe:
                valid &= np.abs(d) >= thresh
            prev = prev ^ g                 # Gray -> binary, MSB first
            binary = (binary << 1) | prev
        valid &= binary < self.w
        column = binary.astype(np.float64)

        if self.phase_steps >= 3:
            base = 2 + 2 * self.n_bits
            ph = np.stack(imgs[base: base + self.phase_steps])
            j = np.arange(self.phase_steps).reshape(-1, 1, 1)
            S = (ph * np.sin(2 * np.pi * j / self.phase_steps)).sum(0)
            C = (ph * np.cos(2 * np.pi * j / self.phase_steps)).sum(0)
            phi = np.arctan2(S, C) % (2 * np.pi)
            amp = 2.0 / self.phase_steps * np.hypot(S, C)
            valid &= amp > s.min_phase_modulation * np.maximum(contrast, 1e-6)
            P = self.phase_period
            u_phase = phi / (2 * np.pi) * P              # position within the period
            # unwrap with the Gray-code column (robust near period boundaries)
            delta = (u_phase - column) % P
            delta = np.where(delta >= P / 2, delta - P, delta)
            fine = column + delta
            valid &= np.abs(delta) <= 1.5               # phase and gray must agree
            column = fine

        column = np.where(valid, column, np.nan)
        return column, valid, contrast
