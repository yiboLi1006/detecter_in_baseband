"""v7.13 回归测试：仅使用临时文件和小规模合成数据，不读取观测数据。"""

import contextlib
import csv
import io
import math
import pathlib
import re
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from astropy import units as u
from astropy.io import fits
from astropy.time import Time

PIPELINE_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPELINE_DIR))
import integrated_pipeline as pipeline
import pulse_detection_module as detection


def make_hdulist(ref_freq=1400.0, dm=50.0):
    """构造已对齐至参考频率的强单脉冲，形状遵循现有读取器约定。"""
    nsamp, nchan, tbin = 2048, 8, 0.00025
    x = np.arange(nsamp)
    rng = np.random.default_rng(713)
    data = 100.0 + rng.normal(0, 1, (nchan, nsamp))
    data += 50.0 * np.exp(-0.5 * ((x - 1000.25) / 4.0) ** 2)
    columns = [
        fits.Column(name='TSUBINT', format='D', array=[nsamp * tbin]),
        fits.Column(name='OFFS_SUB', format='D', array=[nsamp * tbin / 2]),
        fits.Column(name='DAT_FREQ', format=f'{nchan}E',
                    array=[np.linspace(1360, 1440, nchan)]),
        fits.Column(name='DAT_SCL', format=f'{nchan}E', array=[np.ones(nchan)]),
        fits.Column(name='DAT_OFFS', format=f'{nchan}E', array=[np.zeros(nchan)]),
        fits.Column(name='DATA', format=f'{nsamp * nchan}E',
                    dim=f'(1,{nchan},1,{nsamp})', array=[data.reshape(-1)]),
    ]
    table = fits.BinTableHDU.from_columns(columns, name='SUBINT')
    table.header.update(NCHAN=nchan, NPOL=1, NSBLK=nsamp, TBIN=tbin,
                        DM=dm, REFFREQ=ref_freq)
    primary = fits.PrimaryHDU()
    primary.header['DATE-OBS'] = '2020-01-02T00:00:00.000000000'
    primary.header['OBSFREQ'] = 1400.0
    return fits.HDUList([primary, table])


DETECTION_PARAMS = {
    'amp_snr_threshold': 4.0,
    'flux_snr_threshold': 3.0,
    'fit_quality_snr_threshold': 3.0,
    'peak_distance': 100,
    'sigma_remove_rfi_frequency': 1e6,
    'sigma_remove_rfi_time_frequency': 1e6,
}


class TemporaryFilesTest(unittest.TestCase):
    def setUp(self):
        temporary_root = PIPELINE_DIR / '_tmp'
        temporary_root.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix='v713-', dir=temporary_root)
        self.addCleanup(self.temp.cleanup)
        self.directory = pathlib.Path(self.temp.name)

    def config(self, value='10', ref_freq='', center_freq='1400', dm='50'):
        text = '[output]\n'
        if value is not None:
            text += f'max_files = {value}\n'
        text += (f'\n[frequency]\ncenter_freq = {center_freq}\n'
                 f'\n[dm_correction]\nref_freq = {ref_freq}\n'
                 f'\n[observation]\ndm_source = {dm}\n')
        path = self.directory / 'input.ini'
        path.write_text(text, encoding='utf-8')
        return path


class ConfigTests(TemporaryFilesTest):
    def test_false_is_unbounded(self):
        for text in ('False', ' False '):
            with self.subTest(value=text):
                try:
                    result = pipeline.read_config(str(self.config(text)))['max_files']
                except ValueError as exc:
                    self.fail(f'False 必须解析为无上限，而不是报错：{exc}')
                self.assertTrue(math.isinf(result))

    def test_nonnegative_integers_are_exact_limits(self):
        for text, expected in [('0', 0), ('1', 1), ('10', 10), ('0010', 10)]:
            with self.subTest(value=text):
                result = pipeline.read_config(str(self.config(text)))['max_files']
                self.assertEqual(result, expected)
                self.assertIs(type(result), int)

    def test_empty_missing_and_other_values_are_rejected(self):
        for text in ('', '   ', None, 'None', 'True', 'false', 'FALSE',
                     '-1', '1.5', '1.0', '1e3', 'inf', 'NaN', '1_000'):
            with self.subTest(value=text):
                with self.assertRaisesRegex(ValueError, 'max_files'):
                    pipeline.read_config(str(self.config(text)))

    def test_ref_freq_empty_keeps_existing_fallback_contract(self):
        result = pipeline.read_config(str(self.config(ref_freq='')))
        self.assertIsNone(result['dm_ref_freq'])
        self.assertEqual(result['center_freq'], 1400.0)

    def test_explicit_ref_freq_is_not_replaced_by_center(self):
        result = pipeline.read_config(str(self.config(ref_freq='1200')))
        self.assertEqual(result['dm_ref_freq'], 1200.0)
        self.assertEqual(result['center_freq'], 1400.0)

    def test_invalid_dispersion_parameters_are_rejected(self):
        for kwargs in ({'ref_freq': '0'}, {'ref_freq': '-1'}, {'ref_freq': 'nan'},
                       {'ref_freq': 'inf'}, {'center_freq': '0'},
                       {'dm': '-1'}, {'dm': 'nan'}, {'dm': 'inf'}):
            with self.subTest(parameters=kwargs):
                with self.assertRaises(ValueError):
                    pipeline.read_config(str(self.config(**kwargs)))


class ProcessingLimitTests(unittest.TestCase):
    def plan(self, max_files, start_file=0, t_start=0, total_samples=105):
        self.assertTrue(hasattr(pipeline, '_plan_hdulist_ranges'),
                        '缺少单进程／多进程一致的有界任务划分')
        params = dict(chunk_size=10, max_subints_per_file=2,
                      max_files=max_files, start_file=start_file, t_start=t_start)
        return pipeline._plan_hdulist_ranges(total_samples, 10.0, params, 8)

    def test_integer_limit_never_schedules_excess_workers(self):
        self.assertEqual(self.plan(2), [(0, 1), (1, 2)])

    def test_unbounded_includes_final_partial_hdulist(self):
        self.assertEqual(self.plan(math.inf), [(i, i + 1) for i in range(6)])

    def test_resume_respects_global_exclusive_limit(self):
        self.assertEqual(self.plan(4, start_file=2), [(2, 3), (3, 4)])
        self.assertEqual(self.plan(2, start_file=2), [])

    def test_start_time_and_zero_limits_produce_no_extra_work(self):
        self.assertEqual(self.plan(math.inf, t_start=2), [(i, i + 1) for i in range(5)])
        self.assertEqual(self.plan(0), [])
        self.assertEqual(self.plan(math.inf, total_samples=0), [])
        self.assertEqual(self.plan(math.inf, t_start=20), [])

    def test_zero_single_process_limit_does_not_open_input(self):
        with patch.object(pipeline, 'open_data_file',
                          side_effect=AssertionError('零上限不得打开原始数据')):
            result = pipeline.vdif_to_psrfits('unused.vdif', max_files=0,
                                               return_pulses=True)
        self.assertEqual(result, [])

    def test_zero_multiprocess_limit_does_not_open_input(self):
        params = dict(max_files=0, start_file=0, vdif_file='unused.vdif',
                      data_format='vdif', withsubband=False, subbands=[0])
        with patch.object(pipeline, 'open_data_file',
                          side_effect=AssertionError('零上限不得打开原始数据')):
            pipeline.run_multiprocess(params, {}, 1400.0, 4)


class ElapsedTimeTests(unittest.TestCase):
    def test_elapsed_hours_minutes_and_rounding(self):
        self.assertTrue(hasattr(pipeline, '_format_elapsed_time'),
                        '缺少小时加分钟的总耗时格式化')
        for seconds, expected in [(0, '0 h 00.0 min'), (90, '0 h 01.5 min'),
                                  (7650, '2 h 07.5 min'), (3599.9, '1 h 00.0 min'),
                                  (90000, '25 h 00.0 min')]:
            with self.subTest(seconds=seconds):
                self.assertEqual(pipeline._format_elapsed_time(seconds), expected)


class ToaTests(TemporaryFilesTest):
    # MJD 小数点后 15 位的量化步长为 0.0864 ns；差分容差取 0.2 ns。
    def columns(self, time_ref, dm, ref_freq):
        self.assertTrue(hasattr(detection, '_toa_columns'),
                        '缺少参考频率／无穷大频率 TOA 输出')
        return detection._toa_columns(time_ref, dm, ref_freq)

    def test_mhz_delay_and_ref_toa_preservation(self):
        original = Time('2020-01-01T12:00:00.123456789', scale='utc')
        columns = self.columns(original, 50.0, 1000.0)
        time_ref = Time(columns['TOA_Ref_Freq_MJD'], format='mjd', scale='utc')
        time_inf = Time(columns['TOA_Inf_Freq_MJD'], format='mjd', scale='utc')
        self.assertLess(abs((time_ref - original).to_value(u.ns)), 1)
        self.assertAlmostEqual((time_ref - time_inf).to_value(u.s),
                               0.207440321195, delta=2e-10)
        self.assertEqual(columns['TOA_Ref_Freq_UTC'], '2020-01-01T12:00:00.123456789')

    def test_both_user_frequency_examples(self):
        original = Time('2020-01-01T12:00:00', scale='utc')
        for ref_freq, delay in [(1400.0, 0.10583689856887756),
                                (1200.0, 0.14405577860763889)]:
            with self.subTest(ref_freq=ref_freq):
                columns = self.columns(original, 50.0, ref_freq)
                time_inf = Time(columns['TOA_Inf_Freq_MJD'], format='mjd', scale='utc')
                self.assertAlmostEqual((original - time_inf).to_value(u.s), delay, delta=2e-10)
                self.assertEqual(columns['TOA_Ref_Freq_UTC'], '2020-01-01T12:00:00.000000000')

    def test_zero_dm_and_midnight(self):
        original = Time('2020-01-02T00:00:00.100000000', scale='utc')
        zero = self.columns(original, 0.0, 1200.0)
        self.assertEqual(zero['TOA_Ref_Freq_MJD'], zero['TOA_Inf_Freq_MJD'])
        self.assertEqual(zero['TOA_Ref_Freq_UTC'], zero['TOA_Inf_Freq_UTC'])
        shifted = self.columns(original, 50.0, 1000.0)
        self.assertEqual(shifted['TOA_Inf_Freq_UTC'], '2020-01-01T23:59:59.892559679')

    def test_invalid_toa_parameters_fail_instead_of_emitting_nan(self):
        self.assertTrue(hasattr(detection, '_toa_columns'))
        original = Time('2020-01-01T00:00:00', scale='utc')
        for dm, freq in [(-1, 1400), (math.nan, 1400), (math.inf, 1400),
                         (50, 0), (50, -1200), (50, math.nan), (50, math.inf)]:
            with self.subTest(dm=dm, ref_freq=freq):
                with self.assertRaises(ValueError):
                    detection._toa_columns(original, dm, freq)

    def test_detection_uses_hdulist_reference_not_center(self):
        for freq, delay in [(1400.0, 0.10583689856887756),
                            (1200.0, 0.14405577860763889)]:
            with self.subTest(ref_freq=freq), make_hdulist(freq) as hdulist:
                rows = detection.detect_pulses_in_hdulist(hdulist, DETECTION_PARAMS)
                self.assertEqual(len(rows), 1)
                row = rows[0]
                self.assertIn('TOA_Ref_Freq_MJD', row)
                ref = Time(row['TOA_Ref_Freq_MJD'], format='mjd', scale='utc')
                inf = Time(row['TOA_Inf_Freq_MJD'], format='mjd', scale='utc')
                original = Time(row['Precise_JD1'], row['Precise_JD2'],
                                format='jd', scale='utc')
                self.assertLess(abs((ref - original).to_value(u.ns)), 1)
                self.assertAlmostEqual((ref - inf).to_value(u.s), delay, delta=2e-10)

    def test_full_dedispersion_reference_change_preserves_infinite_toa(self):
        from dm_correction_module import dm_correct_hdulist
        infinite_times = []
        reference_times = []
        for freq in (1400.0, 1200.0):
            with self.subTest(ref_freq=freq), make_hdulist(freq) as hdulist:
                channels = hdulist['SUBINT'].data['DAT_FREQ'][0].astype(float)
                times = np.arange(2048) * 0.00025
                # 同一份原始合成色散脉冲，真实无穷大频率到达时刻为 0.2 s。
                arrivals = 0.2 + 4.1488064239e3 * 50.0 / channels**2
                raw = 100.0 + np.random.default_rng(713).normal(0, 0.01, (8, 2048))
                raw += 50.0 * np.exp(-0.5 * ((times[None, :] - arrivals[:, None]) / 0.001) ** 2)
                hdulist['SUBINT'].data['DATA'][0] = raw.reshape(
                    hdulist['SUBINT'].data['DATA'][0].shape)
                dm_correct_hdulist(hdulist, dm=50, ref_freq=freq,
                                   method='freq_domain', normalize=True)
                rows = detection.detect_pulses_in_hdulist(hdulist, DETECTION_PARAMS)
                self.assertEqual(len(rows), 1)
                reference_times.append(Time(rows[0]['TOA_Ref_Freq_MJD'], format='mjd', scale='utc'))
                infinite_times.append(Time(rows[0]['TOA_Inf_Freq_MJD'], format='mjd', scale='utc'))
        expected = Time('2020-01-02T00:00:00.200000000', scale='utc')
        for result in infinite_times:
            self.assertLess(abs((result - expected).to_value(u.s)), 1e-5)
        self.assertLess(abs((infinite_times[0] - infinite_times[1]).to_value(u.s)), 1e-5)
        self.assertAlmostEqual((reference_times[1] - reference_times[0]).to_value(u.s),
                               0.03821888003876133, delta=1e-5)

    def test_csv_writer_keeps_both_toa_groups_and_precision(self):
        columns = self.columns(Time('2020-01-01T00:00:00.123456789', scale='utc'),
                               50, 1400)
        path = self.directory / 'pulses.csv'
        with contextlib.redirect_stdout(io.StringIO()):
            pipeline._save_pulse_collector_csv([dict(Coarse_Index=1, **columns)], str(path))
        with path.open(encoding='utf-8', newline='') as stream:
            result = next(csv.DictReader(stream))
        for name, value in columns.items():
            self.assertIn(name, result)
            self.assertEqual(result[name], value)


class SyntheticVdifTests(TemporaryFilesTest):
    def setUp(self):
        super().setUp()
        from baseband import vdif
        self.vdif_path = self.directory / 'synthetic.vdif'
        with vdif.open(str(self.vdif_path), 'ws', sample_rate=8000 * u.Hz,
                       nchan=1, nthread=2, complex_data=False, bps=2,
                       samples_per_frame=64, edv=0,
                       time=Time('2020-01-01T00:00:00'), station='XX') as writer:
            data = np.random.default_rng(713).normal(
                size=(8192,) + writer.sample_shape).astype('float32')
            writer.write(data)

    def test_real_single_process_respects_limit_and_eof(self):
        original_writer = pipeline.write_psrfits_file_multiple_subints
        for limit, expected in [(0, []), (2, [0, 1]), (math.inf, [0, 1, 2, 3])]:
            processed = []

            def record_and_write(*args, **kwargs):
                processed.append(kwargs['file_counter'])
                return original_writer(*args, **kwargs)

            with self.subTest(limit=limit), contextlib.redirect_stdout(io.StringIO()) as output:
                with patch.object(pipeline, 'write_psrfits_file_multiple_subints',
                                  side_effect=record_and_write):
                    pipeline.vdif_to_psrfits(
                        str(self.vdif_path), reduction_factor=16, subset=[0, 1], nchans=8,
                        chunk_size=1024, center_freq=1400, nband=2, bandwidth=80,
                        max_subints_per_file=2, max_files=limit, mask=[1, 1],
                        withsubband=False, sample_rate_str='0.008*u.MHz',
                        flag_edge_channels=0, calibrate_bandpass=False,
                        dm_value=0, dm_ref_freq=1400,
                        detection_params=dict(DETECTION_PARAMS, amp_snr_threshold=1e6),
                        cleanup_every_n_hdulists=0, return_pulses=True)
            self.assertEqual(processed, expected)
            self.assertNotIn('Error processing chunk', output.getvalue())

    def test_real_multiprocess_cli_uses_limits_and_final_elapsed_format(self):
        for limit, count, stops in [('0', 0, []), ('2', 2, [1, 2]), ('False', 4, [2, 4])]:
            text = f'''[paths]
vdif_file = {self.vdif_path}
data_format = vdif
[observation]
sample_rate = 0.008*u.MHz
dm_source = 0
[output]
max_subints_per_file = 2
max_files = {limit}
[frequency]
center_freq = 1400
subbands = 0,1
mask_sband = 1,1
withsubband = False
subband_width = 80
[processing]
chunk_size = 1024
reduction_factor = 16
nchans = 8
calib_bandpass = False
flag_band_edge = 0
[dm_correction]
ref_freq =
[detection]
amp_snr_threshold = 1000000
[integrated_output]
output_path = {self.directory / 'output'}
[performance]
n_processes = 2
cleanup_every_n_hdulists = 0
'''
            config = self.directory / 'multiprocess.ini'
            config.write_text(text, encoding='utf-8')
            with self.subTest(limit=limit):
                result = subprocess.run(
                    [sys.executable, '-B', str(PIPELINE_DIR / 'integrated_pipeline.py'), str(config)],
                    cwd=PIPELINE_DIR, capture_output=True, text=True, timeout=60)
                output = result.stdout + result.stderr
                self.assertEqual(result.returncode, 0, output)
                self.assertNotIn('Error processing chunk', output)
                self.assertNotIn('FATAL', output)
                self.assertNotIn('Traceback', output)
                self.assertNotIn('RuntimeWarning', output)
                self.assertRegex(output, r'### USED time: \d+ h \d{2}\.\d min')
                actual_stops = sorted(int(x) for x in re.findall(r'Reached hdulist limit \((\d+)\)', output))
                self.assertEqual(actual_stops, stops, output)
                if count:
                    self.assertIn(f'Processing {count} hdulists with 2 workers', output)
                else:
                    self.assertNotIn('Processing ', output)


if __name__ == '__main__':
    unittest.main(verbosity=2)
