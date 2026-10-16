"""Known-shape checks for measurements, connectivity, and threshold behavior."""
import unittest

import numpy as np

from blob_detection import Parameters, detect_blobs, heat_score, summarize


class BlobDetectionTests(unittest.TestCase):
    def test_known_rectangles_measurements_and_area_filter(self):
        image = np.ones((40, 60, 3))
        image[0:4, 0:5] = (1, 0, 0)       # 20 pixels, touches the border
        image[10:20, 30:50] = (1, 0, 0)   # 200 pixels, centered at x=39.5, y=14.5
        image[30, 10] = (1, 0, 0)        # isolated noise
        result = detect_blobs(image, Parameters(smooth_sigma=0, min_area=10))
        self.assertEqual([b['area_px'] for b in result['blobs']], [20, 200])
        self.assertTrue(result['blobs'][0]['touches_border'])
        blob = result['blobs'][1]
        self.assertEqual((blob['center_x_px'], blob['center_y_px']), (39.5, 14.5))
        self.assertFalse(blob['touches_border'])
        self.assertEqual((blob['bbox_x0_px'], blob['bbox_y0_px'],
                          blob['bbox_x1_px'], blob['bbox_y1_px']), (30, 10, 50, 20))
        self.assertAlmostEqual(np.pi * blob['equivalent_radius_px'] ** 2, 200)
        summary = summarize('rectangles', image, result)
        self.assertEqual(summary['total_area_px'], 220)
        self.assertAlmostEqual(summary['coverage_percent'], 100 * 220 / (40 * 60))

    def test_threshold_breaks_a_weak_bridge(self):
        image = np.ones((30, 50, 3))
        image[10:20, 5:15] = (1, 0, 0)
        image[10:20, 30:40] = (1, 0, 0)
        image[14:16, 15:30] = (1, 1, 0)  # yellow bridge has score 0.5
        low = detect_blobs(image, Parameters(.35, 0, 10))
        high = detect_blobs(image, Parameters(.60, 0, 10))
        self.assertEqual(len(low['blobs']), 1)
        self.assertEqual(len(high['blobs']), 2)

    def test_corner_neighbors_are_connected(self):
        image = np.ones((5, 5, 3))
        image[1, 1] = image[2, 2] = (1, 0, 0)
        result = detect_blobs(image, Parameters(smooth_sigma=0, min_area=1))
        self.assertEqual([b['area_px'] for b in result['blobs']], [2])

    def test_no_hot_regions_and_transparency(self):
        image = np.array([[(1., 1., 1.), (0., 0., 0.), (0., 1., 0.), (0., 0., 1.)]])
        self.assertFalse(heat_score(image).any())
        result = detect_blobs(image, Parameters(smooth_sigma=0, min_area=1))
        self.assertEqual(result['blobs'], [])
        self.assertEqual(summarize('empty', image, result)['mean_area_px'], 0)
        transparent_red = np.array([[[1., 0., 0., 0.]]])
        self.assertEqual(heat_score(transparent_red)[0, 0], 0)

    def test_smoothing_suppresses_a_single_pixel(self):
        image = np.ones((30, 30, 3))
        image[15, 15] = (1, 0, 0)
        unsmoothed = detect_blobs(image, Parameters(smooth_sigma=0, min_area=1))
        smoothed = detect_blobs(image, Parameters(smooth_sigma=1.5, min_area=1))
        self.assertEqual(len(unsmoothed['blobs']), 1)
        self.assertEqual(len(smoothed['blobs']), 0)


if __name__ == '__main__':
    unittest.main()
