"""Single-subband mask regressions and compatibility with multiband voltages."""

import contextlib
import io
import pathlib
import sys
import unittest

import numpy as np

PIPELINE_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPELINE_DIR))
import integrated_pipeline as pipeline


class SubbandMaskTests(unittest.TestCase):
    def flag(self, data, mask):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = pipeline.flag_xband_data(data, mask=mask)
        return result, output.getvalue()

    def test_single_band_keep_preserves_voltage_and_precision(self):
        for dtype in (np.float32, np.float64):
            with self.subTest(dtype=dtype):
                data = np.array([-3.3359, -1, 0, 1, 3.3359], dtype=dtype)
                original = data.copy()
                result, warning = self.flag(data, [1])
                self.assertEqual(warning, '')
                self.assertEqual(result.shape, data.shape)
                self.assertEqual(result.dtype, data.dtype)
                np.testing.assert_array_equal(result, original)
                np.testing.assert_array_equal(data, original)

    def test_single_band_zero_masks_finite_voltage_without_mutation(self):
        data = np.array([-3, -1, 0, 1, 3], dtype=np.float32)
        original = data.copy()
        result, warning = self.flag(data, [0])
        self.assertEqual(warning, '')
        self.assertEqual(result.dtype, data.dtype)
        np.testing.assert_array_equal(result, np.zeros(5, dtype=np.float32))
        np.testing.assert_array_equal(data, original)

    def test_single_band_nan_mask_propagates_nan(self):
        data = np.array([-1, 0, 1], dtype=np.float32)
        result, warning = self.flag(data, [np.nan])
        self.assertEqual(warning, '')
        self.assertEqual(result.dtype, data.dtype)
        self.assertTrue(np.isnan(result).all())

    def test_empty_single_band_preserves_shape_and_precision(self):
        result, warning = self.flag(np.array([], dtype=np.float32), [1])
        self.assertEqual(warning, '')
        self.assertEqual(result.shape, (0,))
        self.assertEqual(result.dtype, np.dtype(np.float32))

    def test_single_sample_and_strided_voltage_keep_one_dimension(self):
        arrays = [np.array([2], dtype=np.float32),
                  np.arange(15, dtype=np.float32)[::3]]
        for data in arrays:
            for weight in (0, 1):
                with self.subTest(shape=data.shape, weight=weight):
                    result, warning = self.flag(data, [weight])
                    self.assertEqual(warning, '')
                    self.assertEqual(result.shape, data.shape)
                    self.assertEqual(result.dtype, data.dtype)
                    np.testing.assert_array_equal(
                        result, data if weight else np.zeros_like(data))

    def test_two_dimensional_masks_preserve_existing_values_and_precision(self):
        for count in (1, 2, 4, 8, 16):
            for dtype in (np.float32, np.float64):
                data = np.arange(7 * count, dtype=dtype).reshape(7, count) + 1
                original = data.copy()
                masks = ([1] * count, [0] + [1] * (count - 1),
                         [np.nan] + [1] * (count - 1))
                for mask in masks:
                    with self.subTest(count=count, dtype=dtype, mask=mask):
                        result, warning = self.flag(data, mask)
                        self.assertEqual(warning, '')
                        self.assertEqual(result.shape, data.shape)
                        self.assertEqual(result.dtype, np.dtype(np.float64))
                        for column, weight in enumerate(mask):
                            if np.isnan(weight):
                                self.assertTrue(np.isnan(result[:, column]).all())
                            elif weight == 0:
                                self.assertTrue((result[:, column] == 0).all())
                            else:
                                np.testing.assert_array_equal(
                                    result[:, column], original[:, column])
                        np.testing.assert_array_equal(data, original)

    def test_invalid_mask_lengths_keep_warning_and_passthrough(self):
        cases = [(np.arange(7, dtype=np.float32), [1, 1]),
                 (np.ones((7, 4), dtype=np.float32), [1, 1]),
                 (np.ones((7, 4), dtype=np.float32), [])]
        for data, mask in cases:
            with self.subTest(shape=data.shape, mask=mask):
                result, warning = self.flag(data, mask)
                self.assertIs(result, data)
                self.assertIn('mask length != subband number', warning)

    def test_single_band_keep_preserves_channelized_output(self):
        data = np.random.default_rng(811).normal(size=4096).astype(np.float32)
        params = dict(reduction_factor=512, nchans=256, usb='L',
                      subband_centers=[2419.8], subband_width=512,
                      flag_edge_channels=0, calibrate_bandpass=False)
        # Before the fix, the warning branch returned the voltage unchanged.
        expected = pipeline.Process_vdif_data_multiband(data, **params)
        flagged, warning = self.flag(data, [1])
        actual = pipeline.Process_vdif_data_multiband(flagged, **params)
        self.assertEqual(warning, '')
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.tobytes(), expected.tobytes())


if __name__ == '__main__':
    unittest.main(verbosity=2)
