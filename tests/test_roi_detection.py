import unittest
from types import SimpleNamespace

import cv2
import numpy as np

import structured_light_harness as harness
from colored_stripes import generate_color_stripe_pattern


class RoiDetectionTests(unittest.TestCase):
    def test_add_tags_to_non_contiguous_color_stripe_pattern(self):
        args = SimpleNamespace(roi_tag_size=160, roi_tag_margin=40)
        pattern = generate_color_stripe_pattern(
            (720, 1280), 1280, mean=127, amplitude=100, axis="x"
        )
        self.assertFalse(pattern.flags.c_contiguous)

        tagged, centers = harness.add_roi_tags(pattern, args)

        self.assertTrue(tagged.flags.c_contiguous)
        self.assertEqual(tagged.shape, (720, 1280, 3))
        self.assertEqual(set(centers), {0, 1, 2, 3})

    def test_detects_roi_from_projected_tags(self):
        args = SimpleNamespace(roi_tag_size=160, roi_tag_margin=40)
        pattern = np.zeros((720, 1280), dtype=np.uint8)
        tagged, centers = harness.add_roi_tags(pattern, args)
        transform = np.array(
            [[0.7, 0.0, 120.0], [0.0, 0.75, 60.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        camera_frame = cv2.warpPerspective(
            tagged, transform, (1000, 600), flags=cv2.INTER_NEAREST
        )

        roi = harness.detect_camera_roi(camera_frame, (1280, 720), centers)

        self.assertLessEqual(roi[0], 120)
        self.assertLessEqual(roi[1], 60)
        self.assertGreaterEqual(roi[2], 875)
        self.assertGreaterEqual(roi[3], 535)

    def test_crop_preserves_camera_roi_offset(self):
        frame = np.zeros((100, 160, 3), dtype=np.uint8)
        frame[20:60, 30:90] = (1, 2, 3)

        cropped = harness.crop_camera_frame(frame, (30, 20, 60, 40))

        self.assertEqual(cropped.shape, (40, 60, 3))
        np.testing.assert_array_equal(cropped[0, 0], (1, 2, 3))


if __name__ == "__main__":
    unittest.main()
