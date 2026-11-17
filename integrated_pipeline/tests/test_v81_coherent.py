"""v8.1 相干消色散回归测试：仅使用临时文件和小规模合成数据，不读取观测数据。"""

import csv
import pathlib
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from astropy import units as u
from astropy.io import fits
from astropy.time import Time

PIPELINE_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPELINE_DIR))
import coherent_dedisp_module as coherent
import integrated_pipeline as pipeline
import pulse_detection_module as detection

K_DM = 4.1488064239e3   # s·MHz²·pc⁻¹·cm³
T_START = Time('2020-01-01T00:00:00', scale='utc')


def dispersed_sweep(fs, nsamp, t0, center, width, sideband, dm, ref_freq):
    """只由色散延迟定律构造的实采样扫频（不调用被测代码）：天空频率 f 于 t0 + τ(f) − τ(ref) 到达。"""
    t = np.arange(nsamp) / fs
    inv2 = (t - t0) / (K_DM * dm) + ref_freq ** -2
    with np.errstate(invalid='ignore', divide='ignore'):
        f = np.where(inv2 > 0, inv2 ** -0.5, np.nan)
    lo, hi = center - width / 2.0, center + width / 2.0
    inside = np.isfinite(f) & (f >= lo) & (f <= hi)
    fbb = np.where(inside, (f - lo) if sideband == 'U' else (hi - f), 0.0)
    return np.where(inside, np.cos(2 * np.pi * np.cumsum(fbb * 1e6 / fs)), 0.0)


class ArrayReader:
    """内存版 baseband 流读取器：只实现管线用到的接口，越过文件末尾读取时与 baseband 一样报错。"""

    def __init__(self, data, sample_rate_hz, unreadable=None):
        self.data = data
        self.shape = data.shape
        self.sample_rate = sample_rate_hz * u.Hz
        self.unreadable = unreadable
        self.position = 0

    def seek(self, offset, whence=0):
        self.position = int(offset)
        return self.position

    def tell(self, unit=None):
        if unit is None:
            return self.position
        return T_START + self.position / self.sample_rate.to_value(u.Hz) * u.s

    def read(self, count):
        start, stop = self.position, self.position + int(count)
        if stop > self.shape[0]:
            raise EOFError('cannot read from beyond end of input')
        if self.unreadable and start < self.unreadable[1] and self.unreadable[0] < stop:
            raise OSError('corrupt frame')
        self.position = stop
        return self.data[start:stop].copy()

    def close(self):
        pass


class TemporaryFilesTest(unittest.TestCase):
    def setUp(self):
        temporary_root = PIPELINE_DIR / '_tmp'
        temporary_root.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix='v81-', dir=temporary_root)
        self.addCleanup(self.temp.cleanup)
        self.directory = pathlib.Path(self.temp.name)


class CoherentKernelTests(unittest.TestCase):
    FS, WIDTH, DM, REF = 64e6, 32.0, 56.73, 2264.0
    NSAMP, T0 = 2 ** 21, 0.012
    CASES = (('U', 2216.0), ('U', 2312.0), ('L', 2216.0), ('L', 2312.0))
    CHUNK, BLOCK = 2 ** 17, 2 ** 19

    @classmethod
    def setUpClass(cls):
        cls.n0 = int(round(cls.T0 * cls.FS))
        cls.voltage = np.stack(
            [dispersed_sweep(cls.FS, cls.NSAMP, cls.T0, center, cls.WIDTH, sideband, cls.DM, cls.REF)
             for sideband, center in cls.CASES], axis=1).astype(np.float32)
        cls.centers = [center for _, center in cls.CASES]
        cls.sidebands = [sideband for sideband, _ in cls.CASES]

    def make_reader(self, data, centers, sidebands, active=None, block_len=None, unreadable=None):
        active = [True] * len(centers) if active is None else active
        plan = coherent.plan_coherent(self.FS, centers, self.WIDTH, active, self.DM, self.REF,
                                      block_len=block_len or self.BLOCK, chunk_size=self.CHUNK)
        reader = coherent.CoherentDedispReader(
            ArrayReader(data, self.FS, unreadable), plan, centers, self.WIDTH,
            sidebands, active, self.DM, self.REF)
        return reader, plan

    def read_chunks(self, reader, chunk=None):
        chunk = chunk or self.CHUNK
        parts, failed = [], []
        for start in range(0, self.NSAMP, chunk):
            reader.seek(start)
            try:
                parts.append(reader.read(min(chunk, self.NSAMP - start)))
            except OSError:
                failed.append(start)
                parts.append(None)
        return parts, failed

    def dedispersed(self, sidebands=None, active=None, block_len=None):
        reader, _ = self.make_reader(self.voltage, self.centers, sidebands or self.sidebands,
                                     active=active, block_len=block_len)
        parts, failed = self.read_chunks(reader)
        self.assertEqual(failed, [])
        return np.concatenate(parts)

    def energy_near_t0(self, series, half_width=32):
        power = series.astype(np.float64) ** 2
        return power[self.n0 - half_width:self.n0 + half_width + 1].sum() / power.sum()

    def test_sweep_is_compressed_at_reference_time_for_both_sidebands(self):
        out = self.dedispersed()
        for column, (sideband, center) in enumerate(self.CASES):
            with self.subTest(sideband=sideband, center=center):
                self.assertLess(self.energy_near_t0(self.voltage[:, column]), 0.05)
                power = out[:, column].astype(np.float64) ** 2
                self.assertLessEqual(abs(int(np.argmax(power)) - self.n0), 1)
                self.assertGreater(self.energy_near_t0(out[:, column]), 0.8)

    def test_wrong_sideband_does_not_compress(self):
        flipped = ['L' if sideband == 'U' else 'U' for sideband in self.sidebands]
        out = self.dedispersed(sidebands=flipped)
        for column in range(len(self.CASES)):
            with self.subTest(column=column):
                self.assertLess(self.energy_near_t0(out[:, column]), 0.05)

    def test_block_boundaries_do_not_change_output(self):
        blocked = self.dedispersed(block_len=self.BLOCK)
        whole = self.dedispersed(block_len=self.NSAMP)
        self.assertLess(np.max(np.abs(blocked - whole)) / np.max(np.abs(whole)), 1e-4)

    def test_plan_covers_band_edge_delays(self):
        plan = coherent.plan_coherent(64e6, [2216.0, 2248.0, 2280.0, 2312.0], 32.0, [True] * 4,
                                      56.73, 2264.0, block_len=32 * 2 ** 20, chunk_size=2 ** 20)
        # 手算：K·DM·(2200⁻² − 2264⁻²) = 2.71045 ms，K·DM·(2264⁻² − 2328⁻²) = 2.49000 ms
        self.assertAlmostEqual(plan['look_ms'], 2.71045, delta=1e-4)
        self.assertAlmostEqual(plan['hist_ms'], 2.49000, delta=1e-4)
        self.assertGreaterEqual(plan['n_look'], 2.71045e-3 * 64e6)
        self.assertGreaterEqual(plan['n_hist'], 2.49000e-3 * 64e6)
        overlap = plan['n_hist'] + plan['n_look']
        self.assertGreaterEqual(plan['fft_len'], 2 * overlap)
        self.assertEqual(plan['step'], plan['fft_len'] - overlap)

    def test_reference_above_band_needs_no_history(self):
        plan = coherent.plan_coherent(64e6, [2216.0, 2312.0], 32.0, [True, True], 56.73, 2400.0,
                                      block_len=2 ** 22, chunk_size=2 ** 20)
        # 手算：K·DM·(2200⁻² − 2400⁻²) = 7.76705 ms；全部频率晚于参考频率到达
        self.assertEqual(plan['hist_ms'], 0.0)
        self.assertAlmostEqual(plan['look_ms'], 7.76705, delta=1e-4)

    def test_zero_dm_or_fully_masked_band_needs_no_plan(self):
        sizes = dict(block_len=2 ** 20, chunk_size=2 ** 17)
        self.assertIsNone(coherent.plan_coherent(64e6, [2216.0], 32.0, [True], 0.0, 2264.0, **sizes))
        self.assertIsNone(coherent.plan_coherent(64e6, [2216.0, 2312.0], 32.0, [False, False],
                                                 56.73, 2264.0, **sizes))

    def test_masked_subband_is_returned_unchanged(self):
        out = self.dedispersed(active=[True, False, True, True])
        np.testing.assert_array_equal(out[:, 1], self.voltage[:, 1])
        self.assertGreater(self.energy_near_t0(out[:, 0]), 0.8)

    def test_single_subband_reads_up_to_file_end(self):
        data = np.ascontiguousarray(self.voltage[:, 0])
        reader, _ = self.make_reader(data, [2216.0], ['U'])
        chunk = 300_000
        parts, failed = self.read_chunks(reader, chunk=chunk)
        self.assertEqual(failed, [])
        self.assertTrue(all(part.ndim == 1 for part in parts))
        self.assertEqual([len(part) for part in parts], [chunk] * 6 + [self.NSAMP - 6 * chunk])
        self.assertGreater(self.energy_near_t0(np.concatenate(parts)), 0.8)

    def test_unreadable_input_only_fails_chunks_that_depend_on_it(self):
        unreadable = (700_000, 710_000)
        clean_reader, plan = self.make_reader(self.voltage, self.centers, self.sidebands)
        clean, _ = self.read_chunks(clean_reader)
        faulty_reader, _ = self.make_reader(self.voltage, self.centers, self.sidebands,
                                            unreadable=unreadable)
        parts, failed = self.read_chunks(faulty_reader)
        bad_lo = unreadable[0] // self.CHUNK * self.CHUNK
        bad_hi = bad_lo + self.CHUNK
        affected = (bad_lo - plan['n_look'], bad_hi + plan['n_hist'])
        expected = [start for start in range(0, self.NSAMP, self.CHUNK)
                    if start < affected[1] and affected[0] < start + self.CHUNK]
        self.assertIn(bad_lo, expected)
        self.assertEqual(failed, expected)
        scale = max(np.max(np.abs(part)) for part in clean)
        for start, part, reference in zip(range(0, self.NSAMP, self.CHUNK), parts, clean):
            if part is not None:
                self.assertLess(np.max(np.abs(part - reference)) / scale, 1e-3, start)

    def test_time_comes_from_underlying_reader(self):
        reader, _ = self.make_reader(self.voltage, self.centers, self.sidebands)
        reader.seek(123_456)
        expected = T_START + 123_456 / self.FS * u.s
        self.assertLess(abs((reader.tell(unit='time') - expected).to_value(u.ns)), 1e-3)
        self.assertEqual(reader.tell(), 123_456)

    def test_complex_or_mismatched_input_is_rejected(self):
        plan = coherent.plan_coherent(self.FS, self.centers, self.WIDTH, [True] * 4, self.DM,
                                      self.REF, block_len=self.BLOCK, chunk_size=self.CHUNK)
        complex_raw = ArrayReader(self.voltage, self.FS)
        complex_raw.complex_data = True
        with self.assertRaises(ValueError):
            coherent.CoherentDedispReader(complex_raw, plan, self.centers, self.WIDTH,
                                          self.sidebands, [True] * 4, self.DM, self.REF)
        with self.assertRaises(ValueError):
            coherent.CoherentDedispReader(ArrayReader(self.voltage, self.FS), plan,
                                          self.centers[:2], self.WIDTH, self.sidebands[:2],
                                          [True] * 2, self.DM, self.REF)


class CoherentConfigTests(TemporaryFilesTest):
    def config(self, method=None, max_fwhm=None, sample_rate='2*u.MHz', width='1.0',
               subbands='0,1', usb='U,U', centers='99.5, 100.5'):
        lines = ['[observation]', f'sample_rate = {sample_rate}', 'dm_source = 0.3',
                 '[output]', 'max_files = False',
                 '[frequency]', 'center_freq = 100.0', f'subbands = {subbands}',
                 f'USB = {usb}', f'subband_width = {width}']
        if centers is not None:
            lines.append(f'subband_centers = {centers}')
        lines.append('[dm_correction]')
        if method is not None:
            lines.append(f'method = {method}')
        lines.append('[detection]')
        if max_fwhm is not None:
            lines.append(f'max_fwhm_ms = {max_fwhm}')
        path = self.directory / 'v81.ini'
        path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
        return pipeline.read_config(str(path))

    def test_method_defaults_to_incoherent(self):
        for method in (None, '', '   '):
            with self.subTest(method=method):
                self.assertEqual(self.config(method=method)['dm_method'], 'incoherent')

    def test_method_accepts_both_names_in_any_case(self):
        for text, expected in [('coherent', 'coherent'), (' Coherent ', 'coherent'),
                               ('INCOHERENT', 'incoherent')]:
            with self.subTest(text=text):
                self.assertEqual(self.config(method=text)['dm_method'], expected)

    def test_unknown_method_is_rejected(self):
        for text in ('fft', 'coherent2', 'True', '1'):
            with self.subTest(text=text):
                with self.assertRaisesRegex(ValueError, 'method'):
                    self.config(method=text)

    def test_max_fwhm_blank_or_missing_disables_limit(self):
        for value in (None, '', '  '):
            with self.subTest(value=value):
                self.assertIsNone(self.config(max_fwhm=value)['max_fwhm_ms'])

    def test_max_fwhm_accepts_positive_milliseconds(self):
        for text, expected in [('1', 1.0), ('0.25', 0.25), (' 2.5 ', 2.5)]:
            with self.subTest(text=text):
                self.assertEqual(self.config(max_fwhm=text)['max_fwhm_ms'], expected)

    def test_invalid_max_fwhm_is_rejected(self):
        for text in ('0', '-1', 'abc', 'nan', 'inf', '1ms'):
            with self.subTest(text=text):
                with self.assertRaisesRegex(ValueError, 'max_fwhm_ms'):
                    self.config(max_fwhm=text)

    def test_coherent_mode_requires_centers_for_multiple_subbands(self):
        with self.assertRaisesRegex(ValueError, 'subband_centers'):
            self.config(method='coherent', centers=None)
        single = self.config(method='coherent', subbands='0', usb='U', centers=None)
        self.assertEqual(single['dm_method'], 'coherent')

    def test_coherent_mode_requires_real_sampling_at_twice_the_width(self):
        with self.assertRaisesRegex(ValueError, 'sample_rate'):
            self.config(method='coherent', width='2.0')
        self.assertEqual(self.config(method='coherent')['dm_method'], 'coherent')

    def test_coherent_mode_requires_positive_band_edges(self):
        with self.assertRaisesRegex(ValueError, 'subband'):
            self.config(method='coherent', centers='0.4, 1.4')

    def test_incoherent_mode_keeps_v713_tolerance(self):
        params = self.config(method='incoherent', width='80', centers=None)
        self.assertEqual(params['dm_method'], 'incoherent')


def make_pulse_hdulist(sigma_samples=4.0):
    """已对齐的强单脉冲：tbin = 0.25 ms，FWHM = 2.3548·σ·tbin ≈ 2.355 ms。"""
    nsamp, nchan, tbin = 2048, 8, 0.00025
    x = np.arange(nsamp)
    rng = np.random.default_rng(81)
    data = 100.0 + rng.normal(0, 1, (nchan, nsamp))
    data += 50.0 * np.exp(-0.5 * ((x - 1000.25) / sigma_samples) ** 2)
    columns = [
        fits.Column(name='TSUBINT', format='D', array=[nsamp * tbin]),
        fits.Column(name='OFFS_SUB', format='D', array=[nsamp * tbin / 2]),
        fits.Column(name='DAT_FREQ', format=f'{nchan}E', array=[np.linspace(1360, 1440, nchan)]),
        fits.Column(name='DAT_SCL', format=f'{nchan}E', array=[np.ones(nchan)]),
        fits.Column(name='DAT_OFFS', format=f'{nchan}E', array=[np.zeros(nchan)]),
        fits.Column(name='DATA', format=f'{nsamp * nchan}E', dim=f'(1,{nchan},1,{nsamp})',
                    array=[data.reshape(-1)]),
    ]
    table = fits.BinTableHDU.from_columns(columns, name='SUBINT')
    table.header.update(NCHAN=nchan, NPOL=1, NSBLK=nsamp, TBIN=tbin, DM=50.0, REFFREQ=1400.0)
    primary = fits.PrimaryHDU()
    primary.header['DATE-OBS'] = '2020-01-02T00:00:00.000000000'
    primary.header['OBSFREQ'] = 1400.0
    return fits.HDUList([primary, table])


class MaxFwhmFilterTests(unittest.TestCase):
    PARAMS = {
        'amp_snr_threshold': 4.0,
        'flux_snr_threshold': 3.0,
        'fit_quality_snr_threshold': 3.0,
        'peak_distance': 100,
        'sigma_remove_rfi_frequency': 1e6,
        'sigma_remove_rfi_time_frequency': 1e6,
    }

    def rows(self, **extra):
        with make_pulse_hdulist() as hdulist:
            return detection.detect_pulses_in_hdulist(hdulist, dict(self.PARAMS, **extra))

    def test_disabled_limit_keeps_v713_result(self):
        baseline = self.rows()
        self.assertEqual(len(baseline), 1)
        self.assertAlmostEqual(baseline[0]['FWHM_ms'], 2.355, delta=0.1)
        self.assertEqual(self.rows(max_fwhm_ms=None), baseline)

    def test_pulse_wider_than_limit_is_rejected(self):
        self.assertEqual(self.rows(max_fwhm_ms=1.0), [])

    def test_pulse_narrower_than_limit_is_kept_unchanged(self):
        self.assertEqual(self.rows(max_fwhm_ms=5.0), self.rows())


class CoherentPipelineTests(TemporaryFilesTest):
    """合成色散脉冲的 2 线程 VDIF：tbin = 25 μs，每个 hdulist 62.5 ms，脉冲位于编号 3 的 hdulist。

    脉冲由 64 个随机符号的子扫频叠加而成，到达时刻按 σ = 40 μs 的高斯分布展开，
    使消色散后的脉冲跨约 4 个 bin（与实测相干脉冲约 2 个 bin 的情形一致）。
    理想冲激在 2 MHz 带宽下只有约 1 μs 宽，落在单个 bin 内时高斯宽度无法约束，
    会被现有 Fit-quality 判据拒绝，那不是本测试要考察的内容。
    """
    FS, DM, REF, WIDTH = 2e6, 0.3, 100.0, 1.0
    CENTERS = (99.5, 100.5)
    NCHANS, REDUCTION, CHUNK, SUBINTS = 25, 50, 31250, 4
    NSAMP = 4 * 4 * 31250
    N0 = 437_513
    AMPLITUDE = 0.5
    N_SUBPULSES, SPREAD_S = 64, 40e-6
    HALF_WINDOW = 8000   # 扫频只在注入时刻 ±4 ms 内非零

    def setUp(self):
        super().setUp()
        from baseband import vdif
        from baseband.base.encoding import TWO_BIT_1_SIGMA
        rng = np.random.default_rng(81)
        offsets = rng.normal(0.0, self.SPREAD_S, self.N_SUBPULSES)
        signs = rng.choice([-1.0, 1.0], self.N_SUBPULSES)
        window = slice(self.N0 - self.HALF_WINDOW, self.N0 + self.HALF_WINDOW)
        pulse = np.zeros((self.NSAMP, len(self.CENTERS)))
        for offset, sign in zip(offsets, signs):
            for column, center in enumerate(self.CENTERS):
                pulse[window, column] += sign * dispersed_sweep(
                    self.FS, 2 * self.HALF_WINDOW, self.HALF_WINDOW / self.FS + offset,
                    center, self.WIDTH, 'U', self.DM, self.REF)
        pulse *= self.AMPLITUDE / np.sqrt(self.N_SUBPULSES)
        noise = np.random.default_rng(81).normal(size=pulse.shape)
        self.vdif_path = self.directory / 'dispersed.vdif'
        with vdif.open(str(self.vdif_path), 'ws', sample_rate=self.FS * u.Hz, nchan=1, nthread=2,
                       complex_data=False, bps=2, samples_per_frame=800, edv=0,
                       time=T_START, station='XX') as writer:
            writer.write((TWO_BIT_1_SIGMA * (noise + pulse)).astype('float32'))
        self.truth = T_START + (self.N0 / self.FS + offsets.mean()) * u.s
        self.tbin = self.REDUCTION / self.FS

    def run_cli(self, method, n_processes=1):
        output_dir = self.directory / f'out_{method}_{n_processes}'
        config = self.directory / f'{method}_{n_processes}.ini'
        config.write_text(f'''[paths]
vdif_file = {self.vdif_path}
data_format = vdif
[observation]
sample_rate = 2*u.MHz
telescope = SYN
source_name = B0531+21
dm_source = {self.DM}
[output]
max_subints_per_file = {self.SUBINTS}
max_files = False
version = 0
[frequency]
center_freq = {self.REF}
subbands = 0,1
USB = U,U
mask_sband = 1,1
withsubband = False
subband_width = {self.WIDTH}
subband_centers = {self.CENTERS[0]}, {self.CENTERS[1]}
[processing]
chunk_size = {self.CHUNK}
reduction_factor = {self.REDUCTION}
nchans = {self.NCHANS}
calib_bandpass = False
flag_band_edge = 0
[dm_correction]
method = {method}
ref_freq =
[detection]
amp_snr_threshold = 6
flux_snr_threshold = 5
Fit-quality_SNR = 3
peak_distance = 5000
max_fwhm_ms = 1
[integrated_output]
output_path = {output_dir}
[performance]
n_processes = {n_processes}
cleanup_every_n_hdulists = 0
''', encoding='utf-8')
        result = subprocess.run(
            [sys.executable, '-B', str(PIPELINE_DIR / 'integrated_pipeline.py'), str(config)],
            cwd=PIPELINE_DIR, capture_output=True, text=True, timeout=300)
        output = result.stdout + result.stderr
        self.assertEqual(result.returncode, 0, output)
        for marker in ('Traceback', 'FATAL', 'Error processing chunk'):
            self.assertNotIn(marker, output)
        csv_files = sorted(output_dir.glob('pulse_detection_*.csv'))
        self.assertEqual(len(csv_files), 1, output)
        with csv_files[0].open(encoding='utf-8', newline='') as stream:
            rows = list(csv.DictReader(stream))
        return rows, output_dir, output

    @staticmethod
    def toa(row):
        return Time(row['TOA_Ref_Freq_MJD'], format='mjd', scale='utc')

    @staticmethod
    def raw_psrfits(output_dir):
        files = sorted(output_dir.glob('pulse_psrfits_vdif_*/PSR_B0531+21_SYN_000003_v0.fits'))
        assert len(files) == 1, files
        with fits.open(files[0]) as hdul:
            return hdul['SUBINT'].header['DM'], np.array(hdul['SUBINT'].data['DATA'])

    def test_coherent_mode_recovers_injected_toa_without_half_channel_bias(self):
        coherent_rows, coherent_dir, _ = self.run_cli('coherent')
        incoherent_rows, incoherent_dir, _ = self.run_cli('incoherent')
        self.assertEqual(len(coherent_rows), 1)
        self.assertEqual(len(incoherent_rows), 1)
        coh, inc = coherent_rows[0], incoherent_rows[0]
        self.assertEqual(coh['Fits'], 'PSR_B0531+21_SYN_000003_v0.fits')
        # 相干：时间戳取 FFT 块起点，可分辨脉冲的 TOA 比真实时刻早 tbin/2；
        # 若仍有半通道偏差会再偏晚约 50 μs，故容差取 0.4 tbin = 10 μs
        expected = self.truth - self.tbin / 2 * u.s
        self.assertLess(abs((self.toa(coh) - expected).to_value(u.s)), 0.4 * self.tbin)
        # 非相干：频率标签高半个通道，偏晚 K·DM·Δν/ν³ = 49.786 μs
        lag = (self.toa(inc) - self.toa(coh)).to_value(u.s)
        self.assertAlmostEqual(lag, 49.786e-6, delta=0.25 * self.tbin)
        self.assertLess(float(coh['FWHM_ms']), float(inc['FWHM_ms']))
        # 原始 PSRFITS 两种模式一致，均为未消色散数据
        coherent_dm, coherent_data = self.raw_psrfits(coherent_dir)
        incoherent_dm, incoherent_data = self.raw_psrfits(incoherent_dir)
        self.assertEqual((coherent_dm, incoherent_dm), (0, 0))
        np.testing.assert_array_equal(coherent_data, incoherent_data)

    def test_coherent_multiprocess_matches_single_process(self):
        start_sample, _, _ = pipeline.calculate_start_sample(
            0, 2, self.SUBINTS, self.CHUNK, self.FS)
        self.assertEqual(start_sample, 2 * self.SUBINTS * self.CHUNK)
        single, _, _ = self.run_cli('coherent', n_processes=1)
        multi, _, output = self.run_cli('coherent', n_processes=2)
        self.assertIn('Processing 4 hdulists with 2 workers', output)
        self.assertEqual((len(single), len(multi)), (1, 1))
        for name in ('Fits', 'Coarse_Index', 'Precise_Center_Index', 'Center_Err', 'Amplitude',
                     'FWHM_ms', 'SNR_Amplitude_Detection', 'SNR_Amplitude_Fit', 'SNR_Flux',
                     'Fit-quality_SNR', 'Noise_Sigma'):
            self.assertEqual(single[0][name], multi[0][name], name)
        self.assertLess(abs((self.toa(single[0]) - self.toa(multi[0])).to_value(u.ns)), 1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
