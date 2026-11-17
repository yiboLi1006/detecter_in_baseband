"""Display regression tests using synthetic PSRFITS, never observation data.

Run with the authorized NTSC pulsar interpreter. These tests inspect rendered
artists to distinguish display averaging from changes to science samples.
"""
import contextlib
import copy
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import warnings

import numpy as np
from astropy.io import fits

PIPELINE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPELINE_DIR))
import pulse_plotter as plotter


def make_hdulist(data, frequencies, tbin=1e-6, dm=0.0, ref=1416.0):
    """One subintegration with a legal non-TDIM17 DATA column."""
    data = np.asarray(data, dtype=np.float32)
    nsamp, nchan = data.shape
    columns = [
        fits.Column(name='TSUBINT', format='D', array=[nsamp * tbin]),
        fits.Column(name='OFFS_SUB', format='D', array=[nsamp * tbin / 2]),
        fits.Column(name='DAT_FREQ', format=f'{nchan}E', array=[frequencies]),
        fits.Column(name='DAT_SCL', format=f'{nchan}E', array=[np.ones(nchan)]),
        fits.Column(name='DAT_OFFS', format=f'{nchan}E', array=[np.zeros(nchan)]),
        fits.Column(name='DATA', format=f'{nsamp * nchan}E',
                    dim=f'(1,{nchan},1,{nsamp})', array=[data.T.reshape(-1)]),
    ]
    table = fits.BinTableHDU.from_columns(columns, name='SUBINT')
    table.header.update(NCHAN=nchan, NPOL=1, NSBLK=nsamp, TBIN=tbin)
    if dm is not None:
        table.header['DM'] = dm
    if ref is not None:
        table.header['REFFREQ'] = ref
    primary = fits.PrimaryHDU()
    primary.header['OBSFREQ'] = 1416.0
    primary.header['DATE-OBS'] = '2026-08-07T03:25:00'
    return fits.HDUList([primary, table])


def pulse(center, fwhm_us=2.355):
    return dict(Precise_Center_Index=center, Coarse_Index=int(center),
                FWHM_ms=fwhm_us / 1000, Amplitude=4.0,
                Background_Fit=1.0, Background_Level=1.0, Noise_Sigma=0.1,
                R_2=0.99, SNR_Amplitude_Fit=40.0, SNR_Flux=50.0)


class PlotterTests(unittest.TestCase):
    def setUp(self):
        root = PIPELINE_DIR / '_tmp'
        root.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix='plotter-', dir=root)
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.addCleanup(plotter.plt.close, 'all')

    def render(self, raw, corrected, frequencies, candidate, *, dm=None,
               ref_freq=None, header_dm=0.0, header_ref=1416.0,
               tbin=1e-6, disk=False, actual_save=False):
        raw_hdu = make_hdulist(raw, frequencies, tbin, dm=0.0, ref=None)
        corrected_hdu = make_hdulist(corrected, frequencies, tbin, header_dm, header_ref)
        raw_before = raw_hdu['SUBINT'].data['DATA'].copy()
        corrected_before = corrected_hdu['SUBINT'].data['DATA'].copy()
        candidate_before = copy.deepcopy(candidate)
        captured = []
        savefig = plotter.plt.savefig

        def capture(*args, **kwargs):
            figure = plotter.plt.gcf()
            captured.append(figure)
            if actual_save:
                savefig(*args, **kwargs)

        raw_path = self.directory / 'raw.fits'
        corrected_path = self.directory / 'corrected.fits'
        if disk:
            raw_hdu.writeto(raw_path, overwrite=True)
            corrected_hdu.writeto(corrected_path, overwrite=True)
        log = io.StringIO()
        with contextlib.redirect_stdout(log), warnings.catch_warnings(), \
                patch.object(plotter.plt, 'savefig', side_effect=capture), \
                patch.object(plotter.plt, 'close'):
            warnings.simplefilter('ignore', RuntimeWarning)
            plotter.plot_pulses_for_hdulist(
                str(raw_path), str(corrected_path) if disk else None,
                [candidate], str(self.directory), 'SYNTHETIC', 'TEST', 0, 0,
                raw_hdulist=None if disk else raw_hdu,
                corrected_hdulist=None if disk else corrected_hdu,
                dm=dm, ref_freq=ref_freq)
        self.assertEqual(len(captured), 1, log.getvalue())
        np.testing.assert_array_equal(raw_hdu['SUBINT'].data['DATA'], raw_before)
        np.testing.assert_array_equal(corrected_hdu['SUBINT'].data['DATA'], corrected_before)
        self.assertEqual(candidate, candidate_before)
        raw_hdu.close()
        corrected_hdu.close()
        figure = captured[0]
        images = [axis.images[0] for axis in figure.axes if axis.images]
        self.assertEqual(len(images), 2)
        return figure, images

    @staticmethod
    def deterministic_background(nsamp, nchan):
        rng = np.random.default_rng(812)
        return 1.0 + rng.normal(0, 0.1, (nsamp, nchan))

    @staticmethod
    def line(figure, label):
        return next(line for axis in figure.axes for line in axis.lines
                    if line.get_label() == label)

    def test_raw_window_contains_both_ends_of_a_dispersed_sweep(self):
        frequencies = np.array([1368., 1400., 1432., 1464.])
        center, nsamp, dm, ref = 16384, 32768, 56.72, 1416.0
        delays_us = 4.1488064239e3 * dm * (frequencies ** -2 - ref ** -2) * 1e6
        raw = self.deterministic_background(nsamp, len(frequencies))
        corrected = raw.copy()
        for channel, delay in enumerate(delays_us):
            raw[center + int(round(delay)), channel] += 100.0
        corrected[center] += 10.0
        _, images = self.render(raw, corrected, frequencies, pulse(center), dm=dm, ref_freq=ref)
        display = np.asarray(images[0].get_array())
        bottom, top = images[0].get_extent()[2:]
        self.assertLess(bottom * 1000, delays_us.min())
        self.assertGreater(top * 1000, delays_us.max())
        self.assertGreater(display[:, 0].max(), 4.0)
        self.assertGreater(display[:, -1].max(), 4.0)

    def test_raw_rebin_does_not_average_corrected_peak_or_lightcurve(self):
        frequencies = np.array([1400., 1432.])
        center, nsamp = 2048.25, 4096
        raw = self.deterministic_background(nsamp, 2)
        corrected = raw.copy()
        corrected[2048] = [20.0, 18.0]
        figure, images = self.render(raw, corrected, frequencies, pulse(center), dm=0, ref_freq=1416)
        raw_image, corrected_image = images
        raw_dt_us = np.diff(raw_image.get_extent()[2:])[0] * 1000 / raw_image.get_array().shape[0]
        self.assertAlmostEqual(raw_dt_us, 16.0, places=6)
        self.assertAlmostEqual(float(np.max(corrected_image.get_array())), 20.0)
        self.assertGreaterEqual(corrected_image.get_array().shape[0], 49)
        self.assertLessEqual(corrected_image.get_array().shape[0], 51)
        black = self.line(figure, 'Light curve')
        # The approved change intentionally preserves the old LC, including its
        # integer-center window. No raw display binning may leak into this data.
        expected = np.mean(corrected[1968:2128], axis=1)
        np.testing.assert_allclose(black.get_ydata(), expected, rtol=1e-7)
        np.testing.assert_allclose(black.get_xdata(), np.arange(1968, 2128) * .001)
        self.assertAlmostEqual(float(np.max(black.get_ydata())), 19.0)
        fit = next(line for axis in figure.axes for line in axis.lines
                   if line.get_label().startswith('Gaussian fit'))
        x_samples = np.asarray(fit.get_xdata()) / .001
        expected_fit = 4 * np.exp(-.5 * (x_samples - 2048) ** 2) + 1
        np.testing.assert_allclose(fit.get_ydata(), expected_fit, rtol=1e-10, atol=1e-10)

    def test_color_limits_are_independent_and_do_not_clip_the_data(self):
        frequencies = np.array([1400., 1432.])
        raw = self.deterministic_background(4096, 2)
        corrected = raw.copy()
        candidate = pulse(2048)
        _, initial = self.render(raw, corrected, frequencies, candidate, dm=0, ref_freq=1416)
        raw_limits = initial[0].get_clim()
        corrected_limits = initial[1].get_clim()
        spiked = corrected.copy()
        spiked[2048, 0] = 1e8
        _, changed = self.render(raw, spiked, frequencies, candidate, dm=0, ref_freq=1416)
        np.testing.assert_allclose(changed[0].get_clim(), raw_limits, rtol=0, atol=0)
        # An on-pulse spike must not set the robust off-pulse colour range.
        np.testing.assert_allclose(changed[1].get_clim(), corrected_limits, rtol=1e-7, atol=1e-7)
        self.assertGreater(float(np.max(changed[1].get_array())), changed[1].norm.vmax)
        for image in changed:
            self.assertTrue(image.norm.clip)
            self.assertTrue(np.isfinite(image.norm.vmin))
            self.assertGreater(image.norm.vmax, image.norm.vmin)

    def test_header_dm_and_reference_match_explicit_arguments(self):
        frequencies = np.array([1368., 1464.])
        data = self.deterministic_background(32768, 2)
        candidate = pulse(16384)
        _, explicit = self.render(data, data, frequencies, candidate,
                                  dm=56.72, ref_freq=1416, header_dm=0, header_ref=1400)
        _, fallback = self.render(data, data, frequencies, candidate,
                                  header_dm=56.72, header_ref=1416)
        np.testing.assert_allclose(explicit[0].get_extent(), fallback[0].get_extent())
        np.testing.assert_array_equal(explicit[0].get_array(), fallback[0].get_array())
        # Missing REFFREQ falls back to primary OBSFREQ=1416, not the band edge.
        _, primary = self.render(data, data, frequencies, candidate,
                                 header_dm=56.72, header_ref=None)
        np.testing.assert_allclose(explicit[0].get_extent(), primary[0].get_extent())
        np.testing.assert_array_equal(explicit[0].get_array(), primary[0].get_array())

    def test_explicit_zero_dm_overrides_nonzero_header(self):
        frequencies = np.array([1368., 1464.])
        data = self.deterministic_background(32768, 2)
        candidate = pulse(16384)
        _, zero_override = self.render(data, data, frequencies, candidate,
                                       dm=0.0, ref_freq=1416, header_dm=56.72)
        _, zero_header = self.render(data, data, frequencies, candidate,
                                     header_dm=0.0, header_ref=1416)
        _, dispersed_header = self.render(data, data, frequencies, candidate,
                                          header_dm=56.72, header_ref=1416)
        np.testing.assert_allclose(zero_override[0].get_extent(), zero_header[0].get_extent())
        np.testing.assert_array_equal(zero_override[0].get_array(), zero_header[0].get_array())
        self.assertLess(zero_override[0].get_array().shape[0],
                        dispersed_header[0].get_array().shape[0])

    def test_single_channel_and_nan_background_have_valid_display_ranges(self):
        for kind in ('constant', 'nan_background', 'all_nan'):
            with self.subTest(kind=kind):
                values = np.ones((256, 1))
                if kind != 'constant':
                    values[:] = np.nan
                if kind == 'nan_background':
                    values[127:130] = 5.0
                _, images = self.render(values, values, [1416.], pulse(128),
                                        dm=0, ref_freq=1416)
                for image in images:
                    x0, x1, y0, y1 = image.get_extent()
                    self.assertTrue(np.all(np.isfinite([x0, x1, y0, y1])))
                    self.assertGreater(x1, x0)
                    self.assertGreater(y1, y0)
                    self.assertTrue(np.isfinite(image.norm.vmin))
                    self.assertGreater(image.norm.vmax, image.norm.vmin)

    def test_short_boundary_window_does_not_discard_finite_samples(self):
        values = np.arange(1, 9, dtype=float).reshape(-1, 1)
        for center in (0.2, 6.8):
            with self.subTest(center=center):
                _, images = self.render(values, values, [1416.], pulse(center),
                                        dm=0, ref_freq=1416)
                self.assertGreater(images[0].get_array().size, 0)
                self.assertGreater(np.isfinite(images[0].get_array()).sum(), 0)
                finite_corrected = np.asarray(images[1].get_array()).ravel()
                finite_corrected = finite_corrected[np.isfinite(finite_corrected)]
                np.testing.assert_array_equal(finite_corrected, values.ravel())

    def test_coarse_sampling_keeps_native_samples_in_short_corrected_window(self):
        values = self.deterministic_background(256, 1)
        values[128, 0] = 20.0
        for tbin_us in (4., 32.):
            with self.subTest(tbin_us=tbin_us):
                figure, images = self.render(values, values, [1416.], pulse(128),
                                             dm=0, ref_freq=1416, tbin=tbin_us * 1e-6)
                for image in images:
                    self.assertGreater(image.get_array().size, 0)
                    self.assertGreater(np.isfinite(image.get_array()).sum(), 0)
                    self.assertTrue(np.all(np.isfinite(image.get_extent())))
                    self.assertGreater(image.get_extent()[3], image.get_extent()[2])
                self.assertAlmostEqual(float(np.max(images[1].get_array())), 20.0)
                black = self.line(figure, 'Light curve')
                np.testing.assert_allclose(np.diff(black.get_xdata()), tbin_us / 1000,
                                           rtol=1e-10, atol=1e-10)

    def test_disk_data_column_dimension_and_header_fallback_render(self):
        values = self.deterministic_background(256, 1)
        _, images = self.render(values, values, [1416.], pulse(128),
                                header_dm=0, header_ref=None, disk=True, actual_save=True)
        self.assertEqual(len(images), 2)
        png = self.directory / 'PSR_SYNTHETIC_TEST_000000_v0_pulse00.png'
        self.assertTrue(png.is_file())
        self.assertGreater(png.stat().st_size, 1000)

    def test_reader_applies_calibration_without_modifying_input_hdu(self):
        for nchan in (1, 2):
            with self.subTest(nchan=nchan):
                values = np.arange(16 * nchan, dtype=np.float32).reshape(16, nchan)
                hdulist = make_hdulist(values, [1400., 1432.][:nchan])
                self.addCleanup(hdulist.close)
                scale, offset = np.array([2., 3.][:nchan]), np.array([5., -7.][:nchan])
                hdulist['SUBINT'].data['DAT_SCL'][0] = scale if nchan > 1 else scale[0]
                hdulist['SUBINT'].data['DAT_OFFS'][0] = offset if nchan > 1 else offset[0]
                original = hdulist['SUBINT'].data['DATA'].copy()
                first, _ = plotter._read_psrfits_data_from_hdulist(hdulist)
                second, _ = plotter._read_psrfits_data_from_hdulist(hdulist)
                np.testing.assert_allclose(first, values * scale + offset)
                np.testing.assert_array_equal(second, first)
                np.testing.assert_array_equal(hdulist['SUBINT'].data['DATA'], original)


if __name__ == '__main__':
    unittest.main(verbosity=2)

