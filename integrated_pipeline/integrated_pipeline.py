#!/usr/bin/env python
"""
Integrated VDIF/Mark5B -> DM correction -> Pulse detection pipeline (v5b.1).

v5b.1 change vs v5.1: output only raw PSRFITS files; DM-corrected PSRFITS removed
to avoid NaN/0-value ambiguity in the corrected output.

v5.1 change vs v5:  uint8 PSRFITS output (PRESTO / DSPSR compatibility).
All v5 time-system and TOA fixes are unchanged.

v5.1 uint8 output:
  - Raw PSRFITS DATA columns are written as uint8
    (format='B', NBITS=8) instead of float32.  Quantisation (float32→uint8,
    clip to [0,255]) happens ONLY at the final writeto step, AFTER
    DM correction and pulse detection have completed in float32/float64.
    Detection results are therefore identical to v5.
  - This makes the output files directly readable by PRESTO prepsubband,
    DSPSR, and other C-based pulsar toolchains that expect integer data.

---
v5 change vs v3:  PSRFITS time-system reconstruction.
Memory cleanup layers inherited from v2/v3 are unchanged.

v5 time-system changes:
  1.  Fixed global observation start time (T0_obs_start). The first chunk's
      VDIF timestamp is captured once and reused for ALL hdulists, so every
      output PSRFITS file shares the same DATE-OBS.  This is the PSRFITS
      standard behaviour.
  2.  Removed the global_offset parameter that was threaded through
      create_primary_header -> create_table_columns ->
      write_psrfits_file_multiple_subints.  Time alignment is now handled
      entirely via the fixed T0_obs_start + OFFS_SUB column.
  3.  Subint offset now uses a half-integer shift:
        subint_offset = (current_subint + 0.5) * tsamp * n_time_bins
      so that OFFS_SUB records the subint *centre* rather than its start,
      matching the PSRFITS convention.
  4.  Synchronised with pulse_detection_module.py (v5) which now reads
      OFFS_SUB and TSUBINT from the SUBINT table to compute accurate
      absolute TOAs:
        TOA = DATE-OBS + (OFFS_SUB - TSUBINT/2) + mu_fit * TBIN

Net effect: all output files have a unified DATE-OBS, OFFS_SUB increases
monotonically across file boundaries, and standard tools (PSRCHIVE, PRESTO,
tempo2) can process the files correctly.

v5 log-simplification (compared to v3):
  - Removed full configuration dump; only key params printed on startup.
  - Removed per-hdulist DM-correction / pulse-detection banners.
  - Removed per-file PSRFITS write confirmations.
  - Removed per-hdulist gc.collect object count + malloc_trim status.
  - Condensed memory-startup announcement (silent unless tracemalloc on).
  - Condensed per-hdulist progress to one compact line.
  - Removed verbose output from dm_correction_module and
    pulse_detection_module (see their respective changelogs).

---
v2 change vs v1: aggressive memory cleanup. The detection / DM algorithms
and float64 precision are unchanged.

Cleanup layers added in v2:
  L1 (per chunk):   del raw_data, chunk_data, processed_data right after
                    they have been handed off to subint_data_list.
  L2 (per hdulist): write_psrfits_file_multiple_subints does `del` of the
                    in-memory hdulist + corrected_hdulist + pulse_data_list
                    before returning; the main loop also `del`s the four
                    subint_*_list / subint_chunk_ranges lists before
                    re-creating empty replacements.
  L3 (every N):     after every cleanup_every_n_hdulists writes (configured
                    in [performance] cleanup_every_n_hdulists, default 50),
                    run gc.collect() + libc.malloc_trim(0). The latter is a
                    no-op on non-glibc platforms.

v2 Round 3 additions (R2 measurements still showed linear growth; R3
strengthens cleanup cadence and adds an opt-in tracemalloc diagnostic):
  R3a: gc.collect() + libc.malloc_trim(0) now run after EVERY hdulist
       (not every cleanup_every_n_hdulists). Per-hdulist cost is sub-second
       and makes any sawtooth pattern visible immediately in mem_log.
  R3b: per-hdulist psutil RSS log (was: only on cleanup boundary).
  R3c: tracemalloc diagnostic, opt-in via env TRACEMALLOC_ENABLED=1.
       When enabled, every hdulist prints the top-10 allocation-growth
       call sites since the previous hdulist. Use this to pinpoint which
       file:line is accumulating bytes between hdulists.
  R3d: cleanup_every_n_hdulists now controls ONLY the baseband reader
       close+reopen cycle (no longer also gates the gc/trim/log).

Recommended runtime knob (no code change required):
  Long-running numpy/scipy workloads on Linux often appear to "leak"
  memory due to glibc malloc multi-arena fragmentation. Two well-known
  cures, in increasing order of effectiveness:
    1) Run the script with `MALLOC_ARENA_MAX=2 python ...` -- caps arenas.
    2) LD_PRELOAD jemalloc:
       `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2 python ...`
  These don't fix Python-side reference leaks but eliminate most of the
  "RSS keeps growing forever" symptom for numerics-heavy long-jobs.

Set cleanup_every_n_hdulists = 0 in the ini to disable R2a (reader recycle).
"""

import os
import sys
import time
import math
import datetime
import warnings
import configparser
import gc
import ctypes

import numpy as np
import pandas as pd
import baseband
from astropy.time import Time
from astropy.io import fits
import astropy.units as u
from astropy.coordinates import SkyCoord
from scipy.interpolate import UnivariateSpline
from scipy.signal import savgol_filter

from dm_correction_module import dm_correct_hdulist
from pulse_detection_module import detect_pulses_in_hdulist
from vdif_segment_writer import save_baseband_segment

warnings.filterwarnings("ignore", category=UserWarning)


# =========================================================================
# v2: heap cleanup helper
# =========================================================================

_LIBC = None

try:
    import psutil as _psutil
    _HAS_PSUTIL = True
except ImportError:
    _psutil = None
    _HAS_PSUTIL = False

# v2 R3: optional tracemalloc diagnostic.
# Enable by setting environment variable TRACEMALLOC_ENABLED=1 before launch.
_TRACEMALLOC_ENABLED = os.environ.get('TRACEMALLOC_ENABLED', '0') == '1'
if _TRACEMALLOC_ENABLED:
    try:
        import tracemalloc as _tracemalloc
    except ImportError:
        _tracemalloc = None
        _TRACEMALLOC_ENABLED = False
else:
    _tracemalloc = None


def _trim_malloc():
    """Release fragmented glibc heap back to the OS (Linux only; no-op elsewhere)."""
    global _LIBC
    try:
        if _LIBC is None:
            _LIBC = ctypes.CDLL("libc.so.6")
        _LIBC.malloc_trim(0)
        return True
    except (OSError, AttributeError):
        return False


def _log_rss(tag=''):
    """Print this process RSS in GiB if psutil is available; otherwise silent."""
    if not _HAS_PSUTIL:
        return
    try:
        rss_gib = _psutil.Process().memory_info().rss / (1 << 30)
        print(f"### [memory] {tag} RSS = {rss_gib:.3f} GiB")
    except Exception:
        pass


def _dump_tracemalloc_top(cur_snap, ref_snap, top_n, tag, label):
    """Print top-N (file:lineno) blocks by size_diff vs ref_snap."""
    if cur_snap is None or ref_snap is None:
        return
    try:
        stats = cur_snap.compare_to(ref_snap, 'lineno')
    except Exception as e:
        print(f"### [tracemalloc] {tag}: compare_to failed: {e}")
        return
    print(f"### [tracemalloc] {tag}: top {top_n} growth {label}:")
    for i, stat in enumerate(stats[:top_n]):
        diff_kib = stat.size_diff / 1024.0
        cur_kib = stat.size / 1024.0
        try:
            frame = stat.traceback[0]
            site = f"{frame.filename}:{frame.lineno}"
        except Exception:
            site = "<unknown>"
        print(f"  #{i+1:2d} {diff_kib:+10.1f} KiB net  "
              f"({stat.count_diff:+d} blocks, cur={cur_kib:.1f} KiB)  {site}")
        try:
            for line in stat.traceback.format():
                print(f"        {line}")
        except Exception:
            pass


def _tracemalloc_diff(prev_snapshot, tag=''):
    """If tracemalloc is enabled, print top-N allocation growth since prev_snapshot.

    Delegates to _dump_tracemalloc_top for richer output (file:lineno + full
    traceback frames per top stat).

    Returns the new snapshot (or None if tracemalloc disabled).
    """
    if not _TRACEMALLOC_ENABLED or _tracemalloc is None:
        return None
    try:
        snap = _tracemalloc.take_snapshot()
        if prev_snapshot is not None:
            _dump_tracemalloc_top(snap, prev_snapshot, 10, tag,
                                  "since last snapshot")
        else:
            print(f"### [tracemalloc] baseline snapshot taken ({tag})")
        return snap
    except Exception as e:
        print(f"### [tracemalloc] error: {e}")
        return prev_snapshot


# =========================================================================
# Configuration
# =========================================================================

def _parse_freq_mask_ranges(raw_str):
    ranges = []
    if not raw_str:
        return ranges
    for line in raw_str.splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        parts = [p.strip() for p in line.split(',')]
        if len(parts) != 2:
            print(f"Warning: Skipping malformed manual mask line: {line!r}")
            continue
        try:
            ranges.append((float(parts[0]), float(parts[1])))
        except ValueError:
            print(f"Warning: Skipping non-numeric manual mask line: {line!r}")
    return ranges


def read_config(config_file):
    """Read all configuration parameters (extended for integrated pipeline v2)."""
    config = configparser.ConfigParser()
    config.read(config_file)

    params = {}

    # -- paths --
    params['output_dir'] = config.get('paths', 'output_dir', fallback='./presto_fits')
    params['vdif_file'] = config.get('paths', 'vdif_file',
                                     fallback='./ywwu_data/p51015_km_no0009.vdif')
    params['data_format'] = config.get('paths', 'data_format',
                                       fallback='vdif').lower()

    # -- observation --
    params['telescope'] = config.get('observation', 'telescope', fallback='KM40m')
    params['source_name'] = config.get('observation', 'source_name', fallback='B0531+21')
    params['dm_source'] = config.getfloat('observation', 'dm_source', fallback=57.6)
    params['t_start'] = config.getfloat('observation', 't_start', fallback=0.0)
    params['ref_time'] = config.get('observation', 'ref_time', fallback='')
    params['start_file'] = config.getint('observation', 'start_file', fallback=0)
    params['sample_rate'] = config.get('observation', 'sample_rate',
                                       fallback='32*u.MHz').strip()

    # -- output --
    params['max_subints_per_file'] = config.getint('output', 'max_subints_per_file', fallback=2)
    params['max_files'] = config.getint('output', 'max_files', fallback=1)
    params['version'] = config.getint('output', 'version', fallback=4)

    # -- frequency --
    params['center_freq'] = config.getfloat('frequency', 'center_freq', fallback=2232.9)
    params['subband_width'] = config.getfloat('frequency', 'subband_width', fallback=16.0)
    params['withsubband'] = config.getboolean('frequency', 'withsubband', fallback=True)
    params['nchan'] = config.getint('frequency', 'nchan', fallback=1)

    usb_str = config.get('frequency', 'USB', fallback='U').upper()
    usb_list = [x.strip() for x in usb_str.split(',')]
    if len(usb_list) == 1:
        params['usb'] = usb_list[0]
        params['usb_list'] = None
    else:
        valid_usb = []
        for i, val in enumerate(usb_list):
            if val in ['U', 'L']:
                valid_usb.append(val)
            else:
                print(f"Warning: Invalid usb value '{val}' at pos {i}. Defaulting to 'U'.")
                valid_usb.append('U')
        params['usb'] = None
        params['usb_list'] = valid_usb

    subbands_str = config.get('frequency', 'subbands', fallback='8,9,10,11,12,13,14,15')
    params['subbands'] = [int(x.strip()) for x in subbands_str.split(',')]

    if params['usb_list'] is not None and len(params['usb_list']) != len(params['subbands']):
        print("Warning: usb list length != subbands length. Defaulting to single USB value.")
        params['usb'] = params['usb_list'][0]
        params['usb_list'] = None

    mask_str = config.get('frequency', 'mask_sband', fallback='0,0,1,1,1,1,1,1')
    params['mask_sband'] = [int(x.strip()) for x in mask_str.split(',')]

    subband_centers_str = config.get('frequency', 'subband_centers', fallback='')
    if subband_centers_str and subband_centers_str.strip():
        params['subband_centers'] = [float(x.strip())
                                     for x in subband_centers_str.split(',')]
        if len(params['subband_centers']) != len(params['subbands']):
            print("Warning: subband_centers length mismatch; disabling.")
            params['subband_centers'] = None
    else:
        params['subband_centers'] = None

    # -- processing --
    params['chunk_size'] = config.getint('processing', 'chunk_size', fallback=2 ** 23)
    params['reduction_factor'] = config.getint('processing', 'reduction_factor',
                                               fallback=512 * 32)
    params['nchans'] = config.getint('processing', 'nchans', fallback=512)
    params['calib_bandpass'] = config.getboolean('processing', 'calib_bandpass', fallback=True)
    flag_band_edge_str = config.get('processing', 'flag_band_edge', fallback='2')
    try:
        params['flag_band_edge'] = int(flag_band_edge_str.strip())
    except ValueError:
        params['flag_band_edge'] = 2

    params['n_tbin'] = int(params['chunk_size'] / params['nchans'] / 2)

    # -- dm_correction --
    ref_freq_str = ''
    if config.has_section('dm_correction'):
        ref_freq_str = config.get('dm_correction', 'ref_freq', fallback='').strip()
    if ref_freq_str == '':
        params['dm_ref_freq'] = None
    else:
        params['dm_ref_freq'] = float(ref_freq_str)

    # -- detection --
    if config.has_section('detection'):
        params['amp_snr_threshold'] = config.getfloat('detection', 'amp_snr_threshold', fallback=4.0)
        params['flux_snr_threshold'] = config.getfloat('detection', 'flux_snr_threshold', fallback=4.0)
        params['peak_distance'] = config.getint('detection', 'peak_distance', fallback=5000)
        params['sigma_remove_rfi_frequency'] = config.getfloat(
            'detection', 'sigma_remove_rfi_frequency', fallback=5.0)
        params['sigma_remove_rfi_time_frequency'] = config.getfloat(
            'detection', 'sigma_remove_rfi_time_frequency', fallback=9.0)
    else:
        params['amp_snr_threshold'] = 4.0
        params['flux_snr_threshold'] = 4.0
        params['peak_distance'] = 5000
        params['sigma_remove_rfi_frequency'] = 5.0
        params['sigma_remove_rfi_time_frequency'] = 9.0

    # -- detection_manual_mask --
    if config.has_section('detection_manual_mask'):
        raw_str = config.get('detection_manual_mask', 'freq_mask_ranges', fallback='')
    else:
        raw_str = ''
    params['detection_freq_mask_ranges'] = _parse_freq_mask_ranges(raw_str)

    # -- integrated_output --
    if config.has_section('integrated_output'):
        params['csv_output_path'] = config.get('integrated_output', 'csv_output_path',
                                               fallback='./pulse_csv_results')
        params['output_raw_psrfits_dir'] = config.get('integrated_output',
                                                      'output_raw_psrfits_dir',
                                                      fallback='./pulse_psrfits_raw')
        params['output_corrected_psrfits_dir'] = config.get('integrated_output',
                                                            'output_corrected_psrfits_dir',
                                                            fallback='./pulse_psrfits_corrected')
        params['plot_output_dir'] = config.get('integrated_output', 'plot_output_dir',
                                               fallback='')
    else:
        params['csv_output_path'] = './pulse_csv_results'
        params['output_raw_psrfits_dir'] = './pulse_psrfits_raw'
        params['output_corrected_psrfits_dir'] = './pulse_psrfits_corrected'
        params['plot_output_dir'] = ''

    # -- performance (v2) --
    if config.has_section('performance'):
        params['cleanup_every_n_hdulists'] = config.getint(
            'performance', 'cleanup_every_n_hdulists', fallback=50)
    else:
        params['cleanup_every_n_hdulists'] = 50

    return params


# =========================================================================
# Parameter consistency check  (verbatim from v16)
# =========================================================================

def check_parameters(chunk_size, reduction_factor, nchans):
    n_tbin = int(chunk_size / nchans / 2)
    check1 = chunk_size >= (reduction_factor * n_tbin)
    check2 = nchans == (reduction_factor / 2)
    check3 = n_tbin == (chunk_size / reduction_factor)
    if not (check1 and check2 and check3):
        print(f"### PARAMETER ERROR: chunk_size={chunk_size}, "
              f"reduction_factor={reduction_factor}, nchans={nchans}")
        print(f"    check1 (chunk >= redfac * n_tbin): {check1}")
        print(f"    check2 (nchans == redfac/2):       {check2}")
        print(f"    check3 (n_tbin == chunk/redfac):    {check3}")
        return False
    return True


# =========================================================================
# Subband flagging / bandpass calibration  (verbatim from v16)
# =========================================================================

def flag_xband_data(data, mask=[np.nan, np.nan, 1, 1, 1, 1, 1, 1]):
    if data.T.shape[0] != len(mask):
        print("mask length != subband number")
        return data
    mask_array = np.array(mask).reshape(-1, 1)
    return (data.T * mask_array).T


def flag_bandpass_edges(data, edge_channels=2):
    n_subbands, nchans, n_time_bins = data.shape
    if edge_channels * 2 >= nchans:
        edge_channels = max(0, (nchans - 1) // 2)
    flagged_data = data.copy()
    flagged_data[:, :edge_channels, :] = np.nan
    flagged_data[:, -edge_channels:, :] = np.nan
    return flagged_data


def fit_bandpass_with_bspline(spectrum, degree=5, smooth_factor=1.0,
                              use_savgol_prefilter=True):
    nchans = len(spectrum)
    channel_indices = np.arange(nchans)
    valid_mask = ~np.isnan(spectrum)
    valid_indices = channel_indices[valid_mask]
    valid_spectrum = spectrum[valid_mask]

    if len(valid_spectrum) < degree + 1:
        spline = UnivariateSpline(valid_indices, valid_spectrum, k=1,
                                  s=len(valid_spectrum) * smooth_factor)
        fitted_values = np.full(nchans, np.nanmean(valid_spectrum))
        fitted_values[valid_mask] = spline(valid_indices)
        return spline, fitted_values

    if use_savgol_prefilter and len(valid_spectrum) > 7:
        try:
            window_length = min(21, len(valid_spectrum) // 2 * 2 + 1)
            if window_length >= 5:
                valid_spectrum = savgol_filter(valid_spectrum, window_length, 2)
        except Exception:
            pass

    try:
        spline = UnivariateSpline(valid_indices, valid_spectrum, k=degree,
                                  s=len(valid_spectrum) * smooth_factor)
        fitted_values = np.full(nchans, np.nan)
        fitted_values[valid_mask] = spline(valid_indices)
        fitted_values = np.maximum(fitted_values, 0.1 * np.nanmax(fitted_values))
    except Exception:
        spline = UnivariateSpline(valid_indices, valid_spectrum, k=1,
                                  s=len(valid_spectrum) * smooth_factor)
        fitted_values = np.full(nchans, np.nanmean(valid_spectrum))
        fitted_values[valid_mask] = spline(valid_indices)

    return spline, fitted_values


def calibrate_bandpass_with_bspline(data, degree=5, smooth_factor=1.0, target_level=1.0):
    n_subbands, nchans, n_time_bins = data.shape
    if n_subbands == 0:
        return data, []
    calibrated_data = data.copy()
    bandpass_models = []

    for subband_idx in range(n_subbands):
        subband_spectrum = data[subband_idx]
        time_avg_spectrum = np.zeros(nchans)
        for i in range(nchans):
            channel_data = subband_spectrum[i, :]
            if np.any(~np.isnan(channel_data)):
                time_avg_spectrum[i] = np.nanmedian(channel_data)
            else:
                time_avg_spectrum[i] = np.nan

        valid_mask = ~np.isnan(time_avg_spectrum)
        n_valid = np.sum(valid_mask)
        if n_valid < degree + 1:
            bandpass_models.append({
                'spline': None, 'fitted_values': np.ones(nchans) * target_level,
                'original_mean': np.nanmean(time_avg_spectrum) if n_valid > 0 else np.nan,
                'fitted_mean': target_level,
            })
            continue

        spline, fitted_bandpass = fit_bandpass_with_bspline(
            time_avg_spectrum, degree=degree, smooth_factor=smooth_factor)
        bandpass_models.append({
            'spline': spline, 'fitted_values': fitted_bandpass,
            'original_mean': np.nanmean(time_avg_spectrum),
            'fitted_mean': np.nanmean(fitted_bandpass),
        })

        bandpass_mean = np.nanmean(fitted_bandpass)
        if bandpass_mean > 0:
            normalized_bandpass = fitted_bandpass / bandpass_mean * target_level
        else:
            normalized_bandpass = np.ones_like(fitted_bandpass) * target_level

        for t in range(n_time_bins):
            time_slice = calibrated_data[subband_idx, :, t]
            valid = ~np.isnan(time_slice) & ~np.isnan(normalized_bandpass)
            if np.any(valid):
                calibrated_data[subband_idx, valid, t] = time_slice[valid] / normalized_bandpass[valid]

    return calibrated_data, bandpass_models


def calibrate_subband_flux(data, use_median=True, reference_level=1.0):
    n_subbands, nchans, n_time_bins = data.shape
    if n_subbands == 1:
        return data
    calibrated_data = data.copy()

    subband_spectra = np.zeros((n_subbands, nchans))
    for subband_idx in range(n_subbands):
        for chan_idx in range(nchans):
            channel_data = data[subband_idx, chan_idx, :]
            if np.any(~np.isnan(channel_data)):
                subband_spectra[subband_idx, chan_idx] = (
                    np.nanmedian(channel_data) if use_median else np.nanmean(channel_data))
            else:
                subband_spectra[subband_idx, chan_idx] = np.nan

    subband_norms = np.zeros(n_subbands)
    for i in range(n_subbands):
        valid_vals = subband_spectra[i, ~np.isnan(subband_spectra[i])]
        subband_norms[i] = np.mean(valid_vals) if len(valid_vals) > 0 else np.nan

    valid_norms = ~np.isnan(subband_norms) & (subband_norms > 0)
    if not np.all(valid_norms):
        valid_median = np.median(subband_norms[valid_norms])
        subband_norms[~valid_norms] = valid_median

    scaling_factors = reference_level / subband_norms
    for subband_idx in range(n_subbands):
        calibrated_data[subband_idx] *= scaling_factors[subband_idx]
    return calibrated_data


def channelize_ts_batch(ts, freq_num=4096, usb='U', flag_edge_channels=2,
                        calibrate_flux=True, calibrate_bandpass=True,
                        bspline_degree=3, bspline_smooth=1.0):
    ts_transposed = ts.T
    n_subbands, total_samples = ts_transposed.shape
    n_time_bins = total_samples // (freq_num * 2)
    if n_time_bins == 0:
        return np.array([])
    reshape_ts = ts_transposed[:, :n_time_bins * freq_num * 2].reshape(
        n_subbands, n_time_bins, freq_num * 2)
    spectra = np.fft.fft(reshape_ts, n=freq_num * 2, axis=-1)
    if usb == 'U':
        positive_spectra = np.abs(spectra[:, :, :freq_num])
    else:
        positive_spectra = np.abs(spectra[:, :, freq_num:])
    del spectra, reshape_ts  # v2: free FFT temporaries promptly
    positive_spectra = positive_spectra.transpose(0, 2, 1)
    if flag_edge_channels > 0:
        positive_spectra = flag_bandpass_edges(positive_spectra,
                                               edge_channels=flag_edge_channels)
    if calibrate_bandpass and n_subbands > 0:
        positive_spectra, _ = calibrate_bandpass_with_bspline(
            positive_spectra, degree=bspline_degree, smooth_factor=bspline_smooth)
    if calibrate_flux and n_subbands > 1:
        positive_spectra = calibrate_subband_flux(positive_spectra, use_median=True,
                                                  reference_level=1.0)
    return positive_spectra


def channelize_ts_batch_per_subband(ts, freq_num=4096, usb_list=None,
                                    flag_edge_channels=2, calibrate_flux=True,
                                    calibrate_bandpass=True, bspline_degree=3,
                                    bspline_smooth=1.0):
    if usb_list is None:
        usb_list = ['U'] * ts.shape[1]
    ts_transposed = ts.T
    n_subbands, total_samples = ts_transposed.shape
    n_time_bins = total_samples // (freq_num * 2)
    if n_time_bins == 0:
        return np.array([])
    reshape_ts = ts_transposed[:, :n_time_bins * freq_num * 2].reshape(
        n_subbands, n_time_bins, freq_num * 2)
    spectra = np.fft.fft(reshape_ts, n=freq_num * 2, axis=-1)
    positive_spectra = np.zeros((n_subbands, freq_num, n_time_bins), dtype=np.float32)
    for subband_idx in range(n_subbands):
        usb = usb_list[subband_idx]
        if usb == 'U':
            positive_spectra[subband_idx] = np.abs(spectra[subband_idx, :, :freq_num]).T
        else:
            positive_spectra[subband_idx] = np.abs(spectra[subband_idx, :, freq_num:]).T
    del spectra, reshape_ts  # v2: free FFT temporaries promptly
    if flag_edge_channels > 0:
        positive_spectra = flag_bandpass_edges(positive_spectra,
                                               edge_channels=flag_edge_channels)
    if calibrate_bandpass and n_subbands > 0:
        positive_spectra, _ = calibrate_bandpass_with_bspline(
            positive_spectra, degree=bspline_degree, smooth_factor=bspline_smooth)
    if calibrate_flux and n_subbands > 1:
        positive_spectra = calibrate_subband_flux(positive_spectra, use_median=True,
                                                  reference_level=1.0)
    return positive_spectra


def create_continuous_frequency_array(subband_centers, subband_width,
                                      nchans_per_subband, channel_bandwidth,
                                      overall_center_freq=None):
    n_subbands = len(subband_centers)
    sorted_indices = np.argsort(subband_centers)
    sorted_centers = np.array(subband_centers)[sorted_indices]
    subband_starts = sorted_centers - subband_width / 2
    subband_ends = sorted_centers + subband_width / 2
    min_freq = subband_starts[0]
    max_freq = subband_ends[-1]
    total_bandwidth = max_freq - min_freq
    total_channels_needed = int(math.ceil(total_bandwidth / channel_bandwidth))
    continuous_freqs = min_freq + channel_bandwidth * (
        np.arange(total_channels_needed) + 0.5)

    data_indices_dict = {}
    for i, idx in enumerate(sorted_indices):
        sub_center = sorted_centers[i]
        sub_start = subband_starts[i]
        sub_end = subband_ends[i]
        start_idx = int(round((sub_start - min_freq) / channel_bandwidth))
        end_idx = start_idx + nchans_per_subband
        start_idx = max(0, start_idx)
        end_idx = min(total_channels_needed, end_idx)
        data_indices_dict[idx] = {
            'start_idx': start_idx, 'end_idx': end_idx,
            'nchans': end_idx - start_idx, 'center_freq': sub_center,
        }
    return continuous_freqs, data_indices_dict


def merge_subbands_with_gaps(filterbank_data, data_indices_dict, total_channels):
    n_subbands, nchans_per_subband, n_time_bins = filterbank_data.shape
    merged_data = np.zeros((total_channels, n_time_bins),
                           dtype=filterbank_data.dtype) + np.nan
    for subband_idx in range(n_subbands):
        if subband_idx in data_indices_dict:
            info = data_indices_dict[subband_idx]
            start_idx, end_idx = info['start_idx'], info['end_idx']
            subband_data = filterbank_data[subband_idx]
            nchans_to_copy = min(subband_data.shape[0], end_idx - start_idx)
            if nchans_to_copy > 0:
                merged_data[start_idx:start_idx + nchans_to_copy, :] = subband_data[:nchans_to_copy, :]
    return merged_data


def Process_vdif_data_multiband(data, reduction_factor, nchans, usb='U', usb_list=None,
                                subband_centers=None, subband_width=None,
                                create_continuous=True, flag_edge_channels=2,
                                calibrate_flux=True, calibrate_bandpass=True,
                                bspline_degree=3, bspline_smooth=1.0):
    if len(data.shape) == 1:
        data = data.reshape(-1, 1)
    if data.size == 0:
        return np.array([])
    n_subbands = data.shape[1]

    if usb_list is not None:
        filterbank_data = channelize_ts_batch_per_subband(
            data, freq_num=nchans, usb_list=usb_list,
            flag_edge_channels=flag_edge_channels, calibrate_flux=calibrate_flux,
            calibrate_bandpass=calibrate_bandpass, bspline_degree=bspline_degree,
            bspline_smooth=bspline_smooth)
    else:
        filterbank_data = channelize_ts_batch(
            data, freq_num=nchans, usb=usb,
            flag_edge_channels=flag_edge_channels, calibrate_flux=calibrate_flux,
            calibrate_bandpass=calibrate_bandpass, bspline_degree=bspline_degree,
            bspline_smooth=bspline_smooth)

    if filterbank_data.size == 0:
        return np.array([])

    if create_continuous and subband_centers is not None and len(subband_centers) == n_subbands:
        channel_bandwidth = subband_width / nchans
        continuous_freqs, data_indices_dict = create_continuous_frequency_array(
            subband_centers, subband_width, nchans, channel_bandwidth)
        merged_data = merge_subbands_with_gaps(
            filterbank_data, data_indices_dict, len(continuous_freqs))
        return merged_data
    n_subbands_, n_freq, n_time_bins = filterbank_data.shape
    return filterbank_data.reshape(n_subbands_ * n_freq, n_time_bins)


# =========================================================================
# Primary HDU / table column construction  (verbatim from v16)
# =========================================================================

def create_primary_header(obs_start_time, tsamp, nsamples_per_subint, n_subints,
                          center_freq, chan_bw, nchans, dm, source_name,
                          telescope, coord, file_counter):
    primary_hdu = fits.PrimaryHDU()
    header = primary_hdu.header
    actual_start_time = obs_start_time  # v5: fixed global observation start, no per-file offset

    ra_str = coord.ra.to_string(unit=u.hourangle, sep=':', precision=2)
    dec_str = coord.dec.to_string(unit=u.degree, sep=':', precision=1)
    stt_imjd = int(actual_start_time.mjd)
    stt_smjd = int((actual_start_time.mjd - stt_imjd) * 86400)
    stt_offs = (actual_start_time.mjd - stt_imjd) * 86400 - stt_smjd

    header['SIMPLE'] = True
    header['BITPIX'] = 8
    header['NAXIS'] = 0
    header['EXTEND'] = True
    header['HDRVER'] = '3.4'
    header['FITSTYPE'] = 'PSRFITS'
    header['DATE'] = Time.now().isot
    header['OBSERVER'] = 'VLBI_Observer'
    header['PROJID'] = 'VDIF_CONV'
    header['TELESCOP'] = telescope
    header['ANT_X'] = 0.0
    header['ANT_Y'] = 0.0
    header['ANT_Z'] = 0.0
    header['FRONTEND'] = 'S_BAND'
    header['NRCVR'] = 1
    header['FD_POLN'] = 'LIN'
    header['FD_HAND'] = 1
    header['FD_SANG'] = 0.0
    header['FD_XYPH'] = 0.0
    header['BACKEND'] = 'VDIF_CONVERTER'
    header['BECONFIG'] = 'N/A'
    header['BE_PHASE'] = 1
    header['BE_DCC'] = 0
    header['BE_DELAY'] = 0.0
    header['TCYCLE'] = 0.0
    header['OBS_MODE'] = 'SEARCH'
    header['DATE-OBS'] = actual_start_time.isot
    header['OBSFREQ'] = center_freq
    header['OBSBW'] = chan_bw * nchans
    header['OBSNCHAN'] = nchans
    header['CHAN_DM'] = dm
    header['SRC_NAME'] = source_name
    header['COORD_MD'] = 'J2000'
    header['EQUINOX'] = 2000.0
    header['RA'] = ra_str
    header['DEC'] = dec_str
    header['BMAJ'] = 0.0
    header['BMIN'] = 0.0
    header['BPA'] = 0.0
    header['STT_CRD1'] = ra_str
    header['STT_CRD2'] = dec_str
    header['TRK_MODE'] = 'TRACK'
    header['STP_CRD1'] = ra_str
    header['STP_CRD2'] = dec_str
    header['SCANLEN'] = tsamp * nsamples_per_subint * n_subints
    header['FD_MODE'] = 'FA'
    header['FA_REQ'] = 0.0
    header['CAL_MODE'] = 'OFF'
    header['CAL_FREQ'] = 0.0
    header['CAL_DCYC'] = 0.0
    header['CAL_PHS'] = 0.0
    header['STT_IMJD'] = stt_imjd
    header['STT_SMJD'] = stt_smjd
    header['STT_OFFS'] = stt_offs
    header['STT_LST'] = 0.0
    header['NSUBOFFS'] = file_counter * n_subints
    return primary_hdu


def create_table_columns(subint_data_list, subint_offsets_list, tsamp,
                         center_freq, nchans, chan_bw, coord, file_counter,
                         continuous_freqs=None, data_format='E'):
    n_subints = len(subint_data_list)
    first_subint_data = subint_data_list[0]
    nsamples_per_subint = first_subint_data.shape[1]
    actual_nchans = first_subint_data.shape[0]

    # v5: OFFS_SUB already carries the correct offset (w.r.t. T0_obs_start);
    # no extra global_offset needed here.
    adjusted_offs_sub_arr = np.array(subint_offsets_list, dtype=np.float64)

    tsubint_arr = np.full(n_subints, tsamp * nsamples_per_subint, dtype=np.float64)
    offs_sub_arr = adjusted_offs_sub_arr
    lst_sub_arr = np.zeros(n_subints, dtype=np.float64)
    ra_sub_arr = np.full(n_subints, coord.ra.deg, dtype=np.float64)
    dec_sub_arr = np.full(n_subints, coord.dec.deg, dtype=np.float64)
    glon_sub_arr = np.full(n_subints, coord.galactic.l.deg, dtype=np.float64)
    glat_sub_arr = np.full(n_subints, coord.galactic.b.deg, dtype=np.float64)
    fd_ang_arr = np.zeros(n_subints, dtype=np.float32)
    pos_ang_arr = np.zeros(n_subints, dtype=np.float32)
    par_ang_arr = np.zeros(n_subints, dtype=np.float32)
    tel_az_arr = np.zeros(n_subints, dtype=np.float32)
    tel_zen_arr = np.zeros(n_subints, dtype=np.float32)
    indexval_arr = np.arange(n_subints, dtype=np.float64) + file_counter * n_subints

    if continuous_freqs is not None:
        freqs = continuous_freqs
    else:
        freqs = center_freq + chan_bw * (
            np.arange(actual_nchans) - actual_nchans / 2 + 0.5)

    dat_wts = np.ones(actual_nchans, dtype=np.float32)
    dat_offs = np.zeros(actual_nchans, dtype=np.float32)
    dat_scl = np.ones(actual_nchans, dtype=np.float32)

    total_elements_per_subint = 1 * actual_nchans * 1 * nsamples_per_subint
    # v5.1: uint8 output for PRESTO compatibility
    if data_format == 'B':
        data_arrays = [
            np.clip(np.round(np.nan_to_num(sd, nan=0.0)), 0, 255).astype(np.uint8)
            .reshape(1, actual_nchans, 1, nsamples_per_subint).flatten()
            for sd in subint_data_list
        ]
    else:
        data_arrays = [
            sd.reshape(1, actual_nchans, 1, nsamples_per_subint).flatten()
            for sd in subint_data_list
        ]
    data_column = np.array(data_arrays)

    cols = [
        fits.Column(name='TSUBINT', format='1D', unit='s', array=tsubint_arr),
        fits.Column(name='OFFS_SUB', format='1D', unit='s', array=offs_sub_arr),
        fits.Column(name='LST_SUB', format='1D', unit='s', array=lst_sub_arr),
        fits.Column(name='RA_SUB', format='1D', unit='deg', array=ra_sub_arr),
        fits.Column(name='DEC_SUB', format='1D', unit='deg', array=dec_sub_arr),
        fits.Column(name='GLON_SUB', format='1D', unit='deg', array=glon_sub_arr),
        fits.Column(name='GLAT_SUB', format='1D', unit='deg', array=glat_sub_arr),
        fits.Column(name='FD_ANG', format='1E', unit='deg', array=fd_ang_arr),
        fits.Column(name='POS_ANG', format='1E', unit='deg', array=pos_ang_arr),
        fits.Column(name='PAR_ANG', format='1E', unit='deg', array=par_ang_arr),
        fits.Column(name='TEL_AZ', format='1E', unit='deg', array=tel_az_arr),
        fits.Column(name='TEL_ZEN', format='1E', unit='deg', array=tel_zen_arr),
        fits.Column(name='DAT_FREQ', format=f'{actual_nchans}E', unit='MHz',
                    dim=f'({actual_nchans})', array=[freqs] * n_subints),
        fits.Column(name='DAT_WTS', format=f'{actual_nchans}E',
                    dim=f'({actual_nchans})', array=[dat_wts] * n_subints),
        fits.Column(name='DAT_OFFS', format=f'{actual_nchans}E',
                    dim=f'({actual_nchans})', array=[dat_offs] * n_subints),
        fits.Column(name='DAT_SCL', format=f'{actual_nchans}E',
                    dim=f'({actual_nchans})', array=[dat_scl] * n_subints),
        fits.Column(name='DATA', format=f'{total_elements_per_subint}{data_format}',
                    dim=f'(1, {actual_nchans}, 1, {nsamples_per_subint})',
                    unit='UNCALIB', array=data_column),
        fits.Column(name='INDEXVAL', format='1D', array=indexval_arr),
    ]

    return cols, nsamples_per_subint, n_subints, actual_nchans


# =========================================================================
# *** MODIFIED in v2 ***  hdulist write with conditional disk output AND
# explicit del of all heavy locals before returning.
# =========================================================================

def write_psrfits_file_multiple_subints(subint_data_list, subint_times_list,
                                        subint_offsets_list,
                                        tsamp, center_freq, nband, chan_bw,
                                        nchans, dm, source_name, telescope,
                                        coord, output_file, file_counter,
                                        continuous_freqs=None,
                                        obs_start_time=None,
                                        # ---------- v1 params ----------
                                        dm_value=None, dm_ref_freq=None,
                                        detection_params=None,
                                        output_raw_psrfits_dir=None,
                                        pulse_collector=None,
                                        vdif_input_file=None,
                                        data_format='vdif',
                                        actual_sample_rate=None,
                                        ref_time_str='',
                                        vdif_nchan=1,
                                        hdulist_first_sample=0,
                                        hdulist_total_samples=0,
                                        source_name_for_file=None,
                                        telescope_for_file=None,
                                        version_for_file=0,
                                        plot_output_dir=None):
    n_subints = len(subint_data_list)
    if n_subints == 0:
        print("\nNo subint data to write")
        return

    start_time = subint_times_list[0]

    # Build with float32 for DM correction + pulse detection accuracy;
    # uint8 conversion deferred to final writeto below (PRESTO/DSPSR compat).
    cols, nsamples_per_subint, n_subints, actual_nchans = create_table_columns(
        subint_data_list, subint_offsets_list, tsamp, center_freq,
        nchans, chan_bw, coord, file_counter, continuous_freqs,
    )

    primary_hdu = create_primary_header(
        obs_start_time, tsamp, nsamples_per_subint, n_subints,
        center_freq, chan_bw, actual_nchans, dm, source_name, telescope,
        coord, file_counter,
    )

    table_hdu = fits.BinTableHDU.from_columns(cols, name='SUBINT')
    th = table_hdu.header
    th['INT_TYPE'] = 'TIME'
    th['INT_UNIT'] = 'SEC'
    th['SCALE'] = 'FluxDen'
    th['NPOL'] = 1
    th['POL_TYPE'] = 'AA+BB'
    th['TBIN'] = tsamp
    th['NBIN'] = 1
    th['NBIN_PRD'] = 0
    th['PHS_OFFS'] = 0.0
    th['NBITS'] = -32  # float32 during processing; updated to 8 at writeto
    th['NSUBOFFS'] = file_counter * n_subints
    th['NCHAN'] = actual_nchans
    th['CHAN_BW'] = chan_bw
    th['NCHNOFFS'] = 0
    th['NSBLK'] = nsamples_per_subint
    th['EXTVER'] = 1
    th['TDIM16'] = f'(1,{actual_nchans},1,{nsamples_per_subint})'
    th['DM'] = dm_value if dm_value is not None else dm
    th['REFFREQ'] = dm_ref_freq if dm_ref_freq is not None else center_freq

    hdulist = fits.HDUList([primary_hdu, table_hdu])
    del cols  # v2: column list no longer needed after BinTable is built

    # DM correction is IN-PLACE on `hdulist` (v3 memory fix)
    dm_correct_hdulist(
        hdulist, dm=dm_value,
        ref_freq=dm_ref_freq if dm_ref_freq is not None else center_freq,
        method='freq_domain', normalize=True,
    )

    pulse_data_list = detect_pulses_in_hdulist(hdulist, detection_params)

    if not pulse_data_list:
        try:
            hdulist.close()
        except Exception:
            pass
        del hdulist, pulse_data_list
        return 0

    n_pulses = len(pulse_data_list)

    src = source_name_for_file if source_name_for_file is not None else source_name
    tel = telescope_for_file if telescope_for_file is not None else telescope
    version = version_for_file

    os.makedirs(output_raw_psrfits_dir, exist_ok=True)

    raw_out = os.path.join(output_raw_psrfits_dir,
                           f"PSR_{src}_{tel}_{file_counter:06d}_v{version}.fits")

    fits_basename = os.path.basename(raw_out)
    for pd in pulse_data_list:
        pd['Fits'] = fits_basename

    # v5b.1: only raw PSRFITS output; DM-corrected FITS removed (NaN/0 ambiguity)
    # v5.1: rebuild raw hdulist from the still-available subint_data_list,
    # converting to uint8 for PRESTO/DSPSR compatibility.
    # This path is rare (only when pulses are detected) so the rebuild cost
    # is acceptable; the steady-state non-detection path now never copies.
    raw_subint_data_list = [
        np.clip(np.round(np.nan_to_num(sd, nan=0.0)), 0, 255).astype(np.uint8)
        for sd in subint_data_list
    ]
    raw_cols, raw_nsamples, raw_nsubints, raw_nchans = create_table_columns(
        raw_subint_data_list, subint_offsets_list, tsamp, center_freq,
        nchans, chan_bw, coord, file_counter, continuous_freqs,
        data_format='B',
    )
    raw_primary = create_primary_header(
        obs_start_time, tsamp, raw_nsamples, raw_nsubints,
        center_freq, chan_bw, raw_nchans, dm, source_name, telescope,
        coord, file_counter,
    )
    raw_table = fits.BinTableHDU.from_columns(raw_cols, name='SUBINT')
    rth = raw_table.header
    rth['INT_TYPE'] = 'TIME'
    rth['INT_UNIT'] = 'SEC'
    rth['SCALE'] = 'FluxDen'
    rth['NPOL'] = 1
    rth['POL_TYPE'] = 'AA+BB'
    rth['TBIN'] = tsamp
    rth['NBIN'] = 1
    rth['NBIN_PRD'] = 0
    rth['PHS_OFFS'] = 0.0
    rth['NBITS'] = 8
    rth['NSUBOFFS'] = file_counter * raw_nsubints
    rth['NCHAN'] = raw_nchans
    rth['CHAN_BW'] = chan_bw
    rth['NCHNOFFS'] = 0
    rth['NSBLK'] = raw_nsamples
    rth['EXTVER'] = 1
    rth['TDIM16'] = f'(1,{raw_nchans},1,{raw_nsamples})'
    rth['DM'] = dm_value if dm_value is not None else dm
    rth['REFFREQ'] = dm_ref_freq if dm_ref_freq is not None else center_freq
    raw_hdulist = fits.HDUList([raw_primary, raw_table])
    del raw_cols, raw_subint_data_list
    raw_hdulist.writeto(raw_out, overwrite=True, checksum=True)

    if vdif_input_file is not None and hdulist_total_samples > 0:
        seg_out = os.path.join(
            output_raw_psrfits_dir,
            f"PSR_{src}_{tel}_{file_counter:06d}_v{version}_segment.vdif"
            if data_format == 'vdif'
            else f"PSR_{src}_{tel}_{file_counter:06d}_v{version}_segment.m5b"
        )
        try:
            save_baseband_segment(
                input_file=vdif_input_file, output_file=seg_out,
                sample_start=hdulist_first_sample,
                sample_count=hdulist_total_samples,
                data_format=data_format,
                sample_rate_hz=actual_sample_rate,
                ref_time=ref_time_str if ref_time_str else None,
                nchan=vdif_nchan,
            )
        except Exception as e:
            print(f"  WARNING: failed to save baseband segment: {e}")

    if plot_output_dir and pulse_data_list:
        try:
            from pulse_plotter import plot_pulses_for_hdulist
            plot_pulses_for_hdulist(
                raw_out, None, pulse_data_list, plot_output_dir,
                src, tel, file_counter, version,
                raw_hdulist=raw_hdulist, corrected_hdulist=hdulist,
            )
        except Exception as e:
            print(f"  WARNING: failed to plot pulse waterfalls: {e}")

    if pulse_collector is not None:
        pulse_collector.extend(pulse_data_list)

    # v2 R2: close hdulists to drop astropy internal caches before del
    try:
        raw_hdulist.close()
    except Exception:
        pass
    try:
        hdulist.close()
    except Exception:
        pass
    del hdulist, raw_hdulist, pulse_data_list
    return n_pulses


# =========================================================================
# Data-file opening + start-sample math  (verbatim from v16)
# =========================================================================

def open_data_file(data_file, data_format, withsubband, subset,
                   sample_rate_value, ref_time_str=None, nchan=1):
    if data_format == 'mark5b':
        ref_time = None
        if ref_time_str and ref_time_str.strip():
            try:
                ref_time = Time(ref_time_str)
            except Exception as e:
                print(f"Could not parse ref_time '{ref_time_str}': {e}")
                return None
        else:
            print("Mark5B format requires a reference time (ref_time parameter)")
            return None
        if withsubband:
            return baseband.open(
                data_file, mode='rs', format='mark5b', nchan=nchan,
                sample_rate=sample_rate_value * u.Hz, subset=subset,
                ref_time=ref_time, verify=True)
        return baseband.open(
            data_file, mode='rs', format='mark5b', nchan=nchan,
            sample_rate=sample_rate_value * u.Hz, ref_time=ref_time,
            verify=True)
    if withsubband:
        return baseband.open(
            data_file, mode='rs', format='vdif', subset=subset,
            verify=True, sample_rate=sample_rate_value * u.Hz)
    return baseband.open(
        data_file, mode='rs', format='vdif',
        verify=True, sample_rate=sample_rate_value * u.Hz)


def calculate_start_sample(file_counter, start_file, max_subints_per_file,
                           chunk_size, sample_rate_value, t_start=0.0):
    if start_file > 0:
        total_subints_processed = start_file * max_subints_per_file
        total_samples_processed = total_subints_processed * chunk_size
        time_offset = total_samples_processed / sample_rate_value
        global_offset = t_start + time_offset
        start_sample = int(global_offset * sample_rate_value)
        file_counter = start_file
        return start_sample, file_counter, global_offset
    start_sample = int(t_start * sample_rate_value)
    return start_sample, file_counter, t_start


# =========================================================================
# *** MODIFIED in v2 ***  main loop with chunk-level del + periodic
# gc.collect + malloc_trim every N hdulists.
# =========================================================================

def vdif_to_psrfits(vdif_file, output_dir, reduction_factor=32, subset=[0],
                    nchans=512, dm=0.0, source_name="B0531+21", telescope='Badary',
                    chunk_size=2 ** 25, center_freq=2248.9, nband=1, bandwidth=96.0,
                    max_subints_per_file=128, max_files=100,
                    mask=[0, 0, 1, 1, 1, 1, 1, 1], version=0, withsubband=True,
                    subband_centers=None, t_start=0.0,
                    sample_rate_str='32*u.MHz', usb='U', usb_list=None,
                    flag_edge_channels=2, calibrate_flux=True,
                    calibrate_bandpass=True, bspline_degree=3, bspline_smooth=1.0,
                    data_format='vdif', ref_time_str='', nchan=1, start_file=0,
                    # ------------- v1 params ----------------
                    dm_value=None, dm_ref_freq=None,
                    detection_params=None,
                    csv_output_path=None,
                    output_raw_psrfits_dir=None,
                    plot_output_dir=None,
                    # ------------- v2 params ----------------
                    cleanup_every_n_hdulists=50):
    """Integrated VDIF -> DM correction -> pulse detection -> CSV pipeline (v2)."""
    pulsar_coords = {
        "B0531+21": SkyCoord('05h34m31.97s', '+22d00m52.1s', frame='icrs'),
        "J0332+5434": SkyCoord('03h32m59.37s', '+54d34m43.6s', frame='icrs'),
        "J1713+0747": SkyCoord('17h13m49.53s', '+07d47m37.5s', frame='icrs'),
    }
    coord = pulsar_coords.get(source_name, pulsar_coords["B0531+21"])

    mask_array = np.array(mask)

    try:
        if '*u.MHz' in sample_rate_str:
            sample_rate_value = float(sample_rate_str.split('*')[0].strip()) * 1e6
        else:
            expr = sample_rate_str.replace('u.', '')
            sample_rate_value = float(expr.split('*')[0].strip()) * 1e6
    except Exception:
        sample_rate_value = 32e6

    continuous_freqs = None
    if subband_centers is not None and len(subband_centers) == len(subset):
        channel_bandwidth = bandwidth / nchans
        continuous_freqs, _ = create_continuous_frequency_array(
            subband_centers, bandwidth, nchans, channel_bandwidth)

    data_reader = open_data_file(
        vdif_file, data_format, withsubband, subset,
        sample_rate_value, ref_time_str, nchan)
    if data_reader is None:
        print(f"Error opening data file: {vdif_file}")
        return

    pulse_collector = []

    # memory cleanup setup
    _mem_reader_recycle = cleanup_every_n_hdulists > 0
    if _TRACEMALLOC_ENABLED and _tracemalloc is not None:
        try:
            _tracemalloc.start(25)
        except Exception as e:
            print(f"### [memory] tracemalloc.start() failed: {e}")
    _tm_prev_snapshot = None

    file_obj = data_reader
    data_reader = None  # avoid keeping a second ref to the original reader
    try:
        actual_sample_rate = file_obj.sample_rate.value
        reduced_sample_rate = actual_sample_rate / reduction_factor
        tsamp = 1.0 / reduced_sample_rate
        chan_bw = bandwidth / nchans

        print(f"Sample rate: {actual_sample_rate/1e6:.1f} MHz  "
              f"tsamp: {tsamp:.6f} s  channels: {nchans}  "
              f"file shape: {file_obj.shape}")
        _log_rss(tag='startup')

        start_sample, file_counter, global_offset = calculate_start_sample(
            0, start_file, max_subints_per_file, chunk_size,
            actual_sample_rate, t_start,
        )

        total_samples = file_obj.shape[0] - start_sample
        current_subint = 0

        total_chunks_planned = max(1, int(math.ceil(total_samples / chunk_size)))
        chunks_processed = 0

        subint_data_list = []
        subint_times_list = []
        subint_offsets_list = []
        subint_chunk_ranges = []

        T0_obs_start = None  # v5: capture observation start time (fixed for all files)

        for chunk_start in range(start_sample, start_sample + total_samples, chunk_size):
            end_sample = min(chunk_start + chunk_size, start_sample + total_samples)
            samples_to_read = end_sample - chunk_start
            chunks_processed += 1

            try:
                file_obj.seek(chunk_start)
                raw_data = file_obj.read(samples_to_read)
                chunk_data = flag_xband_data(raw_data, mask=mask_array)
                if chunk_data.size == 0:
                    print(f"\nEmpty chunk at {chunk_start}, skipping")
                    del raw_data, chunk_data
                    break

                processed_data = Process_vdif_data_multiband(
                    chunk_data, reduction_factor, nchans, usb=usb, usb_list=usb_list,
                    subband_centers=subband_centers, subband_width=bandwidth,
                    create_continuous=True, flag_edge_channels=flag_edge_channels,
                    calibrate_flux=calibrate_flux, calibrate_bandpass=calibrate_bandpass,
                    bspline_degree=bspline_degree, bspline_smooth=bspline_smooth,
                )
                if processed_data.size == 0:
                    print(f"\nNo data after processing chunk {current_subint}, skipping")
                    del raw_data, chunk_data, processed_data
                    continue

                file_obj.seek(chunk_start)
                start_time_chunk = file_obj.tell(unit='time')
                if T0_obs_start is None:
                    T0_obs_start = start_time_chunk  # v5: fixed observation start time
                subint_offset = (current_subint + 0.5) * tsamp * processed_data.shape[1]

                subint_data_list.append(processed_data)
                subint_times_list.append(start_time_chunk)
                subint_offsets_list.append(subint_offset)
                subint_chunk_ranges.append((chunk_start, samples_to_read))

                # v2 L1: release transient per-chunk buffers (list still holds
                # processed_data; this only drops the local name binding).
                del raw_data, chunk_data, processed_data

                current_subint += 1

                if (current_subint % max_subints_per_file == 0
                        or chunk_start + chunk_size >= start_sample + total_samples):
                    if subint_data_list:
                        hdulist_first_sample = subint_chunk_ranges[0][0]
                        last_start, last_count = subint_chunk_ranges[-1]
                        hdulist_total_samples = (last_start + last_count
                                                 - hdulist_first_sample)

                        n_pulses = write_psrfits_file_multiple_subints(
                            subint_data_list, subint_times_list, subint_offsets_list,
                            tsamp, center_freq, nband, chan_bw, nchans * nband, dm,
                            source_name, telescope, coord,
                            output_file=None, file_counter=file_counter,
                            continuous_freqs=continuous_freqs,
                            obs_start_time=T0_obs_start,
                            dm_value=dm_value, dm_ref_freq=dm_ref_freq,
                            detection_params=detection_params,
                            output_raw_psrfits_dir=output_raw_psrfits_dir,
                            pulse_collector=pulse_collector,
                            vdif_input_file=vdif_file,
                            data_format=data_format,
                            actual_sample_rate=actual_sample_rate,
                            ref_time_str=ref_time_str,
                            vdif_nchan=nchan,
                            hdulist_first_sample=hdulist_first_sample,
                            hdulist_total_samples=hdulist_total_samples,
                            source_name_for_file=source_name,
                            telescope_for_file=telescope,
                            version_for_file=version,
                            plot_output_dir=plot_output_dir,
                        )

                        # v2 L2: del the big lists, then re-create empty ones
                        del subint_data_list, subint_times_list
                        del subint_offsets_list, subint_chunk_ranges
                        subint_data_list = []
                        subint_times_list = []
                        subint_offsets_list = []
                        subint_chunk_ranges = []

                        file_counter += 1

                        # progress at hdulist boundary (inline refresh)
                        pct_h = 100.0 * chunks_processed / total_chunks_planned
                        rss_str = ''
                        if _HAS_PSUTIL:
                            try:
                                rss_str = f'  RSS={_psutil.Process().memory_info().rss / (1 << 30):.2f}GiB'
                            except Exception:
                                pass
                        print(f"\r### progress: {pct_h:.1f}%  "
                              f"hdulist #{file_counter - 1}  "
                              f"chunk {chunks_processed}/{total_chunks_planned}"
                              f"  pulses: {n_pulses}{rss_str}  ",
                              end='', flush=True)

                        # per-hdulist memory cleanup (gc + malloc_trim)
                        gc.collect()
                        _trim_malloc()
                        _tm_prev_snapshot = _tracemalloc_diff(
                            _tm_prev_snapshot,
                            tag=f'after-hdulist-{file_counter - 1}')

                        # periodic baseband reader recycle
                        if _mem_reader_recycle:
                            written_count = file_counter - start_file
                            if written_count > 0 and written_count % cleanup_every_n_hdulists == 0:

                                try:
                                    file_obj.close()
                                except Exception:
                                    pass

                                file_obj = open_data_file(
                                    vdif_file, data_format, withsubband, subset,
                                    sample_rate_value, ref_time_str, nchan)
                                if file_obj is None:
                                    print("\n### WARN: failed to reopen baseband reader; aborting.")
                                    break
                                try:
                                    actual_sample_rate = file_obj.sample_rate.value
                                except Exception:
                                    pass
                                gc.collect()
                                _trim_malloc()
                                print(f"\n### baseband reader recycled (after hdulist #{file_counter - 1})")

                if file_counter >= max_files:
                    print(f"\n### Reached max_files limit of {max_files}")
                    break

            except Exception as e:
                print(f"\nError processing chunk {current_subint}: {e}")
                import traceback
                traceback.print_exc()
                continue
    finally:
        try:
            file_obj.close()
        except Exception:
            pass

    print()  # finalise inline progress line
    if pulse_collector:
        _save_pulse_collector_csv(pulse_collector, csv_output_path)
    else:
        print("\nNo pulses detected. CSV not generated.")


def _save_pulse_collector_csv(pulse_data_list, csv_base_path):
    if not pulse_data_list:
        return None
    df = pd.DataFrame(pulse_data_list)

    column_order = [
        'Coarse_Index', 'Fits',
        'Precise_Rel_Time_ms',
        'Precise_Center_Index', 'Center_Err',
        'Amplitude', 'Amp_Err', 'FWHM_ms', 'FWHM_Err', 'R_2',
        'Background_Level', 'Background_Fit', 'Noise_Sigma',
        'SNR_Amplitude_Fit', 'SNR_Amplitude_Detection',
        'Flux_From_Fit', 'Flux_Err', 'SNR_Flux_From_Fit',
    ]

    if 'Precise_JD1' in df.columns and 'Precise_JD2' in df.columns:
        df['Precise_JD1'] = df['Precise_JD1'].apply(
            lambda x: repr(x) if not pd.isna(x) else '')
        df['Precise_JD2'] = df['Precise_JD2'].apply(
            lambda x: repr(x) if not pd.isna(x) else '')
    if 'Precise_Abs_MJD_Str' in df.columns:
        df['Precise_Abs_MJD_Str'] = df['Precise_Abs_MJD_Str'].astype(str)

    for extra in ['Precise_JD1', 'Precise_JD2', 'Precise_Abs_MJD_Str']:
        if extra in df.columns and extra not in column_order:
            column_order.append(extra)

    existing_columns = [c for c in column_order if c in df.columns]
    df = df[existing_columns]

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_dir = os.path.dirname(csv_base_path)
    if csv_dir:
        os.makedirs(csv_dir, exist_ok=True)
    csv_filename = f"{os.path.splitext(csv_base_path)[0]}_{timestamp}.csv"
    df.to_csv(csv_filename, index=False)

    print("\n" + "=" * 60)
    print(f"CSV OUTPUT: {csv_filename}")
    print(f"Total pulses recorded: {len(df)}")
    print("=" * 60)
    return csv_filename


# =========================================================================
# main
# =========================================================================

if __name__ == "__main__":
    print("=" * 50)
    print("  detecter_in_baseband  v5b.1")
    print("=" * 50)
    print(f"### Start time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
    t0 = time.time()

    if len(sys.argv) < 2:
        print("Usage: python integrated_pipeline.py <config.ini>")
        sys.exit(1)
    config_file = sys.argv[1]
    if not os.path.exists(config_file):
        print(f"Configuration file not found: {config_file}")
        sys.exit(1)

    params = read_config(config_file)

    print(f"VDIF file : {params['vdif_file']}")
    print(f"Source    : {params['source_name']}  DM={params['dm_source']}  "
          f"Telescope={params['telescope']}")
    print(f"Chunks    : chunk_size={params['chunk_size']}  "
          f"nchans={params['nchans']}  reduction_factor={params['reduction_factor']}")
    print(f"Max files : {params['max_files']}  "
          f"subints_per_file={params['max_subints_per_file']}")

    detection_params = {
        'amp_snr_threshold': params['amp_snr_threshold'],
        'flux_snr_threshold': params['flux_snr_threshold'],
        'peak_distance': params['peak_distance'],
        'sigma_remove_rfi_frequency': params['sigma_remove_rfi_frequency'],
        'sigma_remove_rfi_time_frequency': params['sigma_remove_rfi_time_frequency'],
        'manual_mask_freq_ranges': params['detection_freq_mask_ranges'],
    }

    dm_ref_freq = params['dm_ref_freq']
    if dm_ref_freq is None:
        dm_ref_freq = params['center_freq']
        print(f"DM ref_freq blank in ini; defaulting to center_freq = {dm_ref_freq} MHz")

    if not check_parameters(params['chunk_size'], params['reduction_factor'],
                            params['nchans']):
        sys.exit(1)

    vdif_to_psrfits(
        vdif_file=params['vdif_file'],
        output_dir=params['output_dir'],
        reduction_factor=params['reduction_factor'],
        subset=params['subbands'],
        nchans=params['nchans'],
        dm=params['dm_source'],
        source_name=params['source_name'],
        telescope=params['telescope'],
        chunk_size=params['chunk_size'],
        center_freq=params['center_freq'],
        nband=len(params['subbands']),
        bandwidth=params['subband_width'],
        max_subints_per_file=params['max_subints_per_file'],
        max_files=params['max_files'],
        mask=params['mask_sband'],
        version=params['version'],
        withsubband=params['withsubband'],
        subband_centers=params.get('subband_centers'),
        t_start=params['t_start'],
        sample_rate_str=params.get('sample_rate', '32*u.MHz'),
        usb=params.get('usb', 'U'),
        usb_list=params.get('usb_list'),
        flag_edge_channels=params['flag_band_edge'],
        calibrate_flux=True,
        calibrate_bandpass=params['calib_bandpass'],
        bspline_degree=3, bspline_smooth=1.0,
        data_format=params['data_format'],
        ref_time_str=params.get('ref_time', ''),
        nchan=params.get('nchan', 1),
        start_file=params['start_file'],
        dm_value=params['dm_source'],
        dm_ref_freq=dm_ref_freq,
        detection_params=detection_params,
        csv_output_path=params['csv_output_path'],
        output_raw_psrfits_dir=params['output_raw_psrfits_dir'],
        plot_output_dir=params.get('plot_output_dir') or None,
        cleanup_every_n_hdulists=params['cleanup_every_n_hdulists'],
    )

    t1 = time.time()
    print(f"### USED time: {t1 - t0:.3f} sec")
    print(f"### Stop time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
    print("#" * 60)
