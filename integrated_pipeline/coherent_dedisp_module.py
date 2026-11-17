"""
Coherent dedispersion module (v8.1) -- voltage-domain, per-subband overlap-save.

Each real-sampled subband is multiplied in the frequency domain by the inverse
ISM transfer function ("chirp") before the pipeline's unchanged channelisation.
Phases are evaluated in float64; the zero-delay reference is ref_freq, so the
output is aligned at ref_freq exactly like the incoherent channel shifts of
dm_correction_module.

CoherentDedispReader stands in for the baseband stream reader inside
vdif_to_psrfits: the first read inside a block loads [block - history,
block + look-ahead] of raw voltage, dedisperses it and caches the block; later
chunk reads are served from the cache. Timestamps still come from the
underlying reader, so chunk boundaries, OFFS_SUB and TOAs are unchanged.
"""

import math

import numpy as np
import scipy.fft as sfft
import astropy.units as u

# 与 pulse_detection_module.DM_DELAY_CONSTANT_MHZ 一致；MHz 输入，秒输出。
DM_DELAY_CONSTANT_MHZ = 4.1488064239e3

_PHASE_SLICE = 1 << 22   # 分段计算 chirp，限制 complex128 临时数组大小
_CHIRP_CACHE = {}


def dispersion_delay(freq_mhz, dm, ref_freq):
    """相对 ref_freq 的色散延迟（秒），低于参考频率为正。"""
    freq = np.asarray(freq_mhz, dtype=np.float64)
    return DM_DELAY_CONSTANT_MHZ * dm * (freq ** -2 - float(ref_freq) ** -2)


def _choose_fft_len(overlap, block_len):
    """取总运算量 ceil(block/(L-overlap))*L*log2(L) 最小的 FFT 长度，要求 L >= 2*overlap。"""
    k_min = max(10, math.ceil(math.log2(2 * overlap)))
    candidates = [2 ** k for k in range(k_min, max(k_min, 28) + 1)]
    one_shot = sfft.next_fast_len(block_len + overlap, real=True)
    if one_shot >= 2 * overlap:
        candidates.append(one_shot)

    def cost(length):
        return math.ceil(block_len / (length - overlap)) * length * math.log2(length)

    return min(candidates, key=lambda length: (cost(length), length))


def plan_coherent(sample_rate_hz, centers, width, active, dm, ref_freq, block_len, chunk_size):
    """计算历史/前瞻重叠（样本）与 FFT 长度；DM=0 或没有有效子带时返回 None。"""
    edges = [(center - width / 2.0, center + width / 2.0)
             for center, on in zip(centers, active) if on]
    if dm == 0 or not edges:
        return None
    # 低频晚到需要前瞻，高频早到需要历史；参考频率在带外时其一为 0
    look = max(0.0, float(np.max(dispersion_delay([lo for lo, _ in edges], dm, ref_freq))))
    hist = max(0.0, float(np.max(-dispersion_delay([hi for _, hi in edges], dm, ref_freq))))
    margin = int(2048 + 0.02 * (look + hist) * sample_rate_hz)
    n_look = int(math.ceil(look * sample_rate_hz)) + margin
    n_hist = int(math.ceil(hist * sample_rate_hz)) + margin
    fft_len = _choose_fft_len(n_hist + n_look, int(block_len))
    return {
        'n_hist': n_hist, 'n_look': n_look, 'fft_len': fft_len,
        'step': fft_len - n_hist - n_look, 'block_len': int(block_len),
        'chunk_size': int(chunk_size), 'hist_ms': hist * 1e3, 'look_ms': look * 1e3,
    }


def estimate_extra_bytes(plan, n_columns, n_active):
    """相干模式每个进程额外占用的内存估计（字节）：电压块、缓存、工作缓冲、频谱与 chirp。"""
    length, step, block = plan['fft_len'], plan['step'], plan['block_len']
    n_in = (math.ceil(block / step) - 1) * step + length
    nbin = length // 2 + 1
    return 4 * (n_in * n_columns + block * n_columns + n_in + length) + 8 * nbin * (1 + n_active)


def make_chirp(fft_len, sample_rate_hz, center, width, sideband, dm, ref_freq):
    """单个实采样子带在 rfft 网格上的消色散传递函数（complex64，按参数缓存）。

    天空频率 f（USB: fc - bw/2 + f_bb；LSB: fc + bw/2 - f_bb）处的相位
    phi = 2*pi*1e6*K*DM*[(f-fc)*(fc^-2 - f_ref^-2) - (f-fc)^2/(fc^2*f)]，
    其导数正比于 f^-2 - f_ref^-2，因此输出在 f_ref 对齐；LSB 频谱反转，相位取共轭。
    """
    key = (int(fft_len), float(sample_rate_hz), float(center), float(width),
           sideband, float(dm), float(ref_freq))
    chirp = _CHIRP_CACHE.get(key)
    if chirp is not None:
        return chirp
    sign = 1.0 if sideband == 'U' else -1.0
    edge = center - sign * width / 2.0
    bin_mhz = sample_rate_hz / fft_len / 1e6
    scale = 2.0 * np.pi * 1e6 * DM_DELAY_CONSTANT_MHZ * dm
    nbin = fft_len // 2 + 1
    chirp = np.empty(nbin, dtype=np.complex64)
    for start in range(0, nbin, _PHASE_SLICE):
        freq = edge + sign * np.arange(start, min(nbin, start + _PHASE_SLICE)) * bin_mhz
        offset = freq - center
        phase = scale * (offset * (center ** -2 - ref_freq ** -2)
                         - offset * offset / (center * center * freq))
        chirp[start:start + len(freq)] = np.exp(1j * sign * phase)
    _CHIRP_CACHE[key] = chirp
    return chirp


class CoherentDedispReader:
    """替代 baseband 流读取器：按块缓存相干消色散后的电压，接口与原读取器一致。"""

    def __init__(self, raw, plan, centers, width, sidebands, active, dm, ref_freq):
        if getattr(raw, 'complex_data', False):
            raise ValueError("相干消色散仅支持实采样数据，当前输入为复采样")
        n_columns = 1 if len(raw.shape) == 1 else raw.shape[1]
        if len(raw.shape) > 2 or n_columns != len(centers):
            raise ValueError(f"相干消色散：数据形状 {raw.shape} 与 subband_centers 个数 "
                             f"{len(centers)} 不一致")
        self.raw = raw
        self.plan = plan
        self.shape = raw.shape
        self.sample_rate = raw.sample_rate
        self._nsamp = int(raw.shape[0])
        self._squeeze = len(raw.shape) == 1
        fs = raw.sample_rate.to_value(u.Hz)
        self._chirps = [make_chirp(plan['fft_len'], fs, center, width, sideband, dm, ref_freq)
                        if on else None
                        for center, sideband, on in zip(centers, sidebands, active)]
        self._position = 0
        self._block_start = self._block_stop = -1
        self._cache = None
        self._unreadable = []

    def seek(self, offset, whence=0):
        base = {0: 0, 1: self._position, 2: self._nsamp}[whence]
        self._position = base + int(offset)
        return self._position

    def tell(self, unit=None):
        if unit is None:
            return self._position
        self.raw.seek(self._position)
        return self.raw.tell(unit=unit)

    def read(self, count=None):
        start = self._position
        stop = self._nsamp if count is None else min(self._nsamp, start + int(count))
        if not (self._block_start <= start and stop <= self._block_stop):
            self._fill(start)
        self._position = stop
        for bad_lo, bad_hi in self._unreadable:
            if start < bad_hi and bad_lo < stop:
                raise OSError(f"相干消色散所需的原始电压在样本 [{bad_lo}, {bad_hi}) "
                              "范围内不可读，跳过该 chunk")
        out = self._cache[start - self._block_start:stop - self._block_start].copy()
        return out[:, 0] if self._squeeze else out

    def close(self):
        self._cache = None
        self.raw.close()

    def _fill(self, start):
        plan = self.plan
        n_hist, length, step = plan['n_hist'], plan['fft_len'], plan['step']
        self._cache, self._block_start, self._block_stop = None, -1, -1
        block_start = start
        block_stop = min(self._nsamp, start + plan['block_len'])
        n_out = block_stop - block_start
        n_blocks = max(1, math.ceil(n_out / step))
        input_start = block_start - n_hist
        n_in = (n_blocks - 1) * step + length
        read_start = max(0, input_start)
        read_stop = min(self._nsamp, input_start + n_in)
        voltage, bad_pieces = self._read_input(read_start, read_stop, block_start)

        out = np.empty((n_out, voltage.shape[1]), dtype=np.float32)
        buffer = np.empty(n_in, dtype=np.float32)
        for column, chirp in enumerate(self._chirps):
            if chirp is None:   # 掩码子带原样返回，由下游 flag_xband_data 置零
                out[:, column] = voltage[block_start - read_start:block_stop - read_start, column]
                continue
            buffer[:] = 0.0     # 文件首尾以外补零
            buffer[read_start - input_start:read_stop - input_start] = voltage[:, column]
            for j in range(n_blocks):
                spectrum = sfft.rfft(buffer[j * step:j * step + length])
                spectrum *= chirp
                segment = sfft.irfft(spectrum, n=length)
                valid = min(step, n_out - j * step)
                out[j * step:j * step + valid, column] = segment[n_hist:n_hist + valid]
        del voltage, buffer
        # 输出样本 t 依赖输入 [t - n_hist, t + n_look]；坏段影响的输出范围据此外推
        self._unreadable = [(lo - plan['n_look'], hi + n_hist) for lo, hi in bad_pieces]
        self._cache, self._block_start, self._block_stop = out, block_start, block_stop

    def _read_input(self, read_start, read_stop, grid_origin):
        """读取原始电压；整段失败时按 chunk 网格分段重读，坏段补零并返回其样本区间。"""
        n_columns = 1 if self._squeeze else self.shape[1]
        try:
            self.raw.seek(read_start)
            voltage = np.asarray(self.raw.read(read_stop - read_start), dtype=np.float32)
            return voltage.reshape(read_stop - read_start, n_columns), []
        except Exception:
            pass
        chunk = self.plan['chunk_size']
        voltage = np.zeros((read_stop - read_start, n_columns), dtype=np.float32)
        bad_pieces = []
        first = grid_origin + math.floor((read_start - grid_origin) / chunk) * chunk
        for grid in range(first, read_stop, chunk):
            lo, hi = max(grid, read_start), min(grid + chunk, read_stop)
            try:
                self.raw.seek(lo)
                piece = np.asarray(self.raw.read(hi - lo), dtype=np.float32)
                voltage[lo - read_start:hi - read_start] = piece.reshape(hi - lo, n_columns)
            except Exception:
                bad_pieces.append((lo, hi))
        return voltage, bad_pieces
