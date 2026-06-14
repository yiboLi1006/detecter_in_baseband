"""
Pulse detection module (v2) — operates on an in-memory astropy.io.fits HDUList.

v2 change vs v1: detect_pulses_in_hdulist drops each large intermediate
array (data_2d, data_freq_cleaned, data_tf_cleaned, fit_results) as soon
as the next stage no longer needs it, so peak resident memory during
detection is one full-size array instead of three. Algorithm and float64
precision are unchanged.

Adapted from former_script/Pulse_detection_v4.1_ur.py.
"""

import math
import numpy as np
from astropy.time import Time
import astropy.units as u
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from decimal import Decimal, getcontext


# ---------------------------------------------------------------------------
# Detection-flow helpers (copied verbatim from Pulse_detection_v4.1_ur.py)
# ---------------------------------------------------------------------------

def robust_sigma_clip(data, sigma, max_iter=100):
    valid_indices = ~np.isnan(data)
    if np.sum(valid_indices) == 0:
        return np.zeros_like(data, dtype=bool)
    valid_data = data[valid_indices]
    if len(valid_data) < 3:
        return valid_indices
    mask = np.zeros_like(data, dtype=bool)
    mask[valid_indices] = True
    for _ in range(max_iter):
        current_valid = data[mask]
        if len(current_valid) == 0:
            break
        median_val = np.median(current_valid)
        mad_val = np.median(np.abs(current_valid - median_val))
        sigma_equiv = 1.4826 * mad_val if mad_val > 0 else 0
        if sigma_equiv == 0:
            break
        new_mask = np.zeros_like(data, dtype=bool)
        deviation = np.abs(data - median_val)
        valid_mask = valid_indices & (deviation <= sigma * sigma_equiv)
        new_mask[valid_mask] = True
        if np.array_equal(mask, new_mask):
            break
        mask = new_mask
    return mask


def remove_rfi_frequency_domain(data_2d, freqs=None, method='sigma_clip',
                                sigma=2.0, percentile=70, manual_mask=None):
    print("Performing frequency domain RFI removal...")
    if method == 'percentile':
        _ = np.percentile(data_2d, percentile)
    mean_spectrum = np.nanmean(data_2d, axis=0)
    freq_mask = np.ones(mean_spectrum.shape, dtype=bool)

    if method == 'sigma_clip':
        freq_mask = robust_sigma_clip(mean_spectrum, sigma=sigma)
    elif method == 'percentile':
        threshold = np.percentile(mean_spectrum, percentile)
        freq_mask = mean_spectrum <= threshold

    cleaned_data = data_2d.copy()
    cleaned_data[:, ~freq_mask] = np.nan

    auto_retained = np.sum(freq_mask)
    total_channels = len(freq_mask)
    print(f"Retained {auto_retained}/{total_channels} "
          f"({auto_retained / total_channels * 100:.2f}%) channels after auto RFI removal")

    if manual_mask is not None and freqs is not None:
        print("=" * 50)
        print("Applying manual frequency mask...")
        freq_ranges = []
        if 'freq_ranges' in manual_mask and manual_mask['freq_ranges']:
            freq_ranges = manual_mask['freq_ranges']
            print(f"Using absolute frequency ranges: {freq_ranges}")
        elif 'fractional_ranges' in manual_mask and manual_mask['fractional_ranges']:
            fractional_ranges = manual_mask['fractional_ranges']
            freq_min, freq_max = freqs[0], freqs[-1]
            for start_frac, end_frac in fractional_ranges:
                freq_start = freq_min + (freq_max - freq_min) * start_frac
                freq_end = freq_min + (freq_max - freq_min) * end_frac
                freq_ranges.append((freq_start, freq_end))
            print(f"Using fractional ranges {fractional_ranges} -> {freq_ranges}")

        if freq_ranges:
            for i, (freq_start, freq_end) in enumerate(freq_ranges):
                if freq_start > freq_end:
                    freq_start, freq_end = freq_end, freq_start
                print(f"Manual mask {i + 1}: {freq_start:.2f} - {freq_end:.2f} MHz")
                mask_indices = np.where((freqs >= freq_start) & (freqs <= freq_end))[0]
                if len(mask_indices) == 0:
                    print("  Warning: No frequency channels found in this range")
                    continue
                freq_mask[mask_indices] = False
                cleaned_data[:, mask_indices] = np.nan
                print(f"  Masked {len(mask_indices)} frequency channels")
        print("=" * 50)

    return cleaned_data, freq_mask


def remove_rfi_time_frequency(data_2d, window_size, sigma, overlap=0.5):
    print("Performing time-frequency domain RFI removal...")
    cleaned_data = data_2d.copy()
    tf_mask = np.ones(data_2d.shape, dtype=bool)
    ntimes = data_2d.shape[0]
    for start_idx in range(0, ntimes, int(window_size * overlap)):
        end_idx = min(start_idx + window_size, ntimes)
        window_data = data_2d[start_idx:end_idx, :]
        window_mean = np.nanmean(window_data)
        window_std = np.nanstd(window_data)
        outliers = np.abs(window_data - window_mean) > sigma * window_std
        tf_mask[start_idx:end_idx, :] &= ~outliers
        cleaned_data[start_idx:end_idx, :][outliers] = np.nan
    print(f"Retained {np.sum(tf_mask)} / {np.prod(tf_mask.shape)} time-frequency points")
    return cleaned_data, tf_mask


def create_lightcurve(data_2d):
    return np.nanmean(data_2d, axis=1)


def initial_noise_estimation(lightcurve):
    M0 = np.nanmedian(lightcurve)
    MAD0 = np.nanmedian(np.abs(lightcurve - M0))
    sigma0 = 1.4826 * MAD0
    return M0, MAD0, sigma0


def iterative_sigma_clipping(lightcurve, initial_K=3, final_K=5,
                             tolerance=1e-7, max_iterations=100):
    M_i, _, sigma_i = initial_noise_estimation(lightcurve)
    data = lightcurve.copy()
    for i in range(max_iterations):
        K = initial_K if i < max_iterations - 5 else final_K
        threshold = M_i + K * sigma_i
        mask = data < threshold
        if np.sum(mask) < 10:
            print(f"Warning: Too few points remaining ({np.sum(mask)}) after iter {i + 1}.")
            break
        M_next = np.nanmedian(data[mask])
        MAD_next = np.nanmedian(np.abs(data[mask] - M_next))
        sigma_next = 1.4826 * MAD_next
        if abs(M_next - M_i) < tolerance and abs(sigma_next - sigma_i) < tolerance:
            return M_next, sigma_next, mask
        M_i, sigma_i = M_next, sigma_next
    return M_i, sigma_i, mask


def calculate_amplitude_snr(lightcurve, peak_indices, background_level, noise_sigma):
    amplitude_snrs = []
    for peak_idx in peak_indices:
        peak_value = lightcurve[peak_idx]
        snr = (peak_value - background_level) / noise_sigma if noise_sigma > 0 else 0
        amplitude_snrs.append(snr)
    return np.array(amplitude_snrs)


def filter_peaks_by_amplitude_snr(lightcurve, peak_indices, peak_properties,
                                  background_level, noise_sigma,
                                  n_sigma_amplitude,
                                  min_prominence_sigma_factor=0.5):
    print(f"\nApplying Amplitude SNR filtering (>= {n_sigma_amplitude:.2f} sigma)...")
    amplitude_snrs = calculate_amplitude_snr(lightcurve, peak_indices,
                                             background_level, noise_sigma)
    selected_mask = (amplitude_snrs >= n_sigma_amplitude)
    selected_indices = np.where(selected_mask)[0]

    filtered_peaks = peak_indices[selected_indices]
    filtered_properties = {}
    for key in peak_properties.keys():
        if isinstance(peak_properties[key], np.ndarray):
            filtered_properties[key] = peak_properties[key][selected_indices]

    filtered_amplitude_snrs = amplitude_snrs[selected_indices]
    print(f"  Peaks passing amplitude SNR threshold: {len(filtered_peaks)}")
    return filtered_peaks, filtered_properties, filtered_amplitude_snrs


def detect_pulses_v3(lightcurve, times, background_level, noise_sigma,
                     n_sigma_amplitude=4.0, n_sigma_flux=3.0,
                     min_prominence_sigma_factor=0.5, peak_distance=None):
    detection_threshold = background_level + n_sigma_amplitude * noise_sigma
    print(f"\nSTEP 1: Initial peak detection (threshold={detection_threshold:.6f})")
    prominence_threshold = max(noise_sigma * min_prominence_sigma_factor,
                               noise_sigma * 0.3)

    peaks, properties = find_peaks(
        lightcurve, height=detection_threshold, prominence=prominence_threshold,
    )
    print(f"  Initial peaks: {len(peaks)}")
    if len(peaks) == 0:
        return np.array([]), {}, np.array([])

    print("\nSTEP 2: Amplitude SNR filtering")
    filtered_peaks, filtered_properties, filtered_amplitude_snrs = filter_peaks_by_amplitude_snr(
        lightcurve, peaks, properties,
        background_level, noise_sigma,
        n_sigma_amplitude,
        min_prominence_sigma_factor=min_prominence_sigma_factor,
    )

    if peak_distance is not None and len(filtered_peaks) > 1:
        print(f"\nSTEP 3: Peak distance filtering (min distance={peak_distance})")
        peak_heights = lightcurve[filtered_peaks]
        sorted_indices = np.argsort(peak_heights)[::-1]
        selected_peaks = []
        selected_properties = {k: [] for k in filtered_properties.keys()}
        selected_snrs = []
        for idx in sorted_indices:
            current_peak = filtered_peaks[idx]
            if all(abs(current_peak - sp) >= peak_distance for sp in selected_peaks):
                selected_peaks.append(current_peak)
                for key in filtered_properties.keys():
                    selected_properties[key].append(filtered_properties[key][idx])
                selected_snrs.append(filtered_amplitude_snrs[idx])
        sorted_selected = np.argsort(selected_peaks)
        filtered_peaks = np.array(selected_peaks)[sorted_selected]
        for key in selected_properties.keys():
            selected_properties[key] = np.array(selected_properties[key])[sorted_selected]
        filtered_amplitude_snrs = np.array(selected_snrs)[sorted_selected]
        filtered_properties = selected_properties
        print(f"  After distance filter: {len(filtered_peaks)} peaks retained")

    return filtered_peaks, filtered_properties, filtered_amplitude_snrs


def calculate_pulse_widths(lightcurve, times, peaks_index, threshold,
                           method='fwhm', smooth_window=5):
    if len(peaks_index) == 0:
        return np.array([]), {}

    pulse_widths = []
    width_details = []
    time_resolution = times[1] - times[0] if len(times) > 1 else 1.0

    def robust_background_estimation(data, max_iter=50, tolerance=1e-6):
        mask = robust_sigma_clip(data, sigma=1.5, max_iter=50)
        if np.sum(mask) < len(data) * 0.3:
            mask = np.ones(len(data), dtype=bool)
        background_mean = np.mean(data[mask])
        background_std = np.std(data[mask])
        for _ in range(max_iter):
            current_threshold = background_mean + 3 * background_std
            new_mask = data <= current_threshold
            if np.sum(new_mask) < len(data) * 0.3:
                break
            new_mean = np.mean(data[new_mask])
            new_std = np.std(data[new_mask])
            if abs(new_mean - background_mean) < tolerance and abs(new_std - background_std) < tolerance:
                break
            background_mean = new_mean
            background_std = new_std
            mask = new_mask
        return background_mean, background_std

    for i, peak_idx in enumerate(peaks_index):
        window_size = max(4000, smooth_window * 10)
        start_idx = max(0, peak_idx - window_size // 2)
        end_idx = min(len(lightcurve), peak_idx + window_size // 2)
        local_lightcurve = lightcurve[start_idx:end_idx].copy()
        local_peak_idx = peak_idx - start_idx

        if len(local_lightcurve) >= smooth_window:
            smoothed_lc = np.convolve(local_lightcurve,
                                      np.ones(smooth_window) / smooth_window, mode='same')
            pad = smooth_window // 2
            if pad > 0:
                smoothed_lc[:pad] = local_lightcurve[:pad]
                smoothed_lc[-pad:] = local_lightcurve[-pad:]
        else:
            smoothed_lc = local_lightcurve

        local_mean, local_std = robust_background_estimation(smoothed_lc)
        local_threshold = local_mean + 2 * local_std

        if method == 'fwhm':
            half_max = (smoothed_lc[local_peak_idx] + local_threshold) / 2
            left_idx = local_peak_idx
            while left_idx > 0 and smoothed_lc[left_idx] >= half_max:
                left_idx -= 1
            right_idx = local_peak_idx
            while right_idx < len(smoothed_lc) - 1 and smoothed_lc[right_idx] >= half_max:
                right_idx += 1
            width_samples = right_idx - left_idx
            pulse_width = width_samples * time_resolution
            width_details.append({
                'method': 'fwhm_smoothed_local',
                'left_idx': left_idx + start_idx, 'right_idx': right_idx + start_idx,
                'width_samples': width_samples, 'local_mean': local_mean, 'local_std': local_std,
            })
        elif method == 'base':
            left_idx = local_peak_idx
            while left_idx > 0 and smoothed_lc[left_idx] >= local_threshold:
                left_idx -= 1
            right_idx = local_peak_idx
            while right_idx < len(smoothed_lc) - 1 and smoothed_lc[right_idx] >= local_threshold:
                right_idx += 1
            width_samples = right_idx - left_idx
            pulse_width = width_samples * time_resolution
            width_details.append({
                'method': 'base_smoothed_local',
                'left_idx': left_idx + start_idx, 'right_idx': right_idx + start_idx,
                'width_samples': width_samples, 'local_mean': local_mean, 'local_std': local_std,
            })
        else:
            raise ValueError("Method must be 'fwhm' or 'base'")
        pulse_widths.append(pulse_width)
        del local_lightcurve, smoothed_lc  # release per-pulse working buffers

    pulse_widths = np.array(pulse_widths)
    pulse_widths_ms = pulse_widths * 1000

    width_info = {
        'method': method, 'time_resolution': time_resolution,
        'smooth_window': smooth_window,
        'pulse_widths_sec': pulse_widths, 'pulse_widths_ms': pulse_widths_ms,
        'details': width_details,
    }
    return pulse_widths_ms, width_info


def gaussian_pulse(x, A, mu, sigma, background):
    if sigma <= 0:
        sigma = 1e-10
    result = A * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) + background
    return np.asarray(result, dtype=float)


def calculate_background_init(lightcurve, start_idx, end_idx, window_size_samples,
                              window_multiplier=3):
    extension = window_size_samples * window_multiplier // 2
    extended_start = max(0, start_idx - extension)
    extended_end = min(len(lightcurve), end_idx + extension)
    if extended_start < start_idx and extended_end > end_idx:
        background_data = np.concatenate([lightcurve[extended_start:start_idx],
                                          lightcurve[end_idx:extended_end]])
    elif extended_start < start_idx:
        background_data = lightcurve[extended_start:start_idx]
    elif extended_end > end_idx:
        background_data = lightcurve[end_idx:extended_end]
    else:
        background_data = lightcurve[extended_start:extended_end]
    if len(background_data) > 0:
        return np.mean(background_data), np.std(background_data)
    return np.median(lightcurve), np.std(lightcurve)


def calculate_flux_snr_from_fit(A_fit, A_err, sigma_fit, sigma_err, noise_sigma):
    flux = A_fit * sigma_fit * np.sqrt(2 * np.pi)
    if A_fit > 0 and sigma_fit > 0 and A_err > 0 and sigma_err > 0:
        rel_A = A_err / A_fit
        rel_sigma = sigma_err / sigma_fit
        flux_err = flux * np.sqrt(rel_A ** 2 + rel_sigma ** 2)
    else:
        flux_err = max(flux * 0.05, noise_sigma * sigma_fit * np.sqrt(2 * np.pi))
    flux_snr = flux / flux_err if flux_err > 0 else 0
    return flux, flux_err, flux_snr


# ---------------------------------------------------------------------------
# v4.0 TOA helpers (verbatim)
# ---------------------------------------------------------------------------

def calculate_pulse_absolute_time(mu_fit, tbin, date_obs_iso, offs_sub=0.0, tsubint=0.0):
    relative_time_sec = mu_fit * tbin
    relative_time_ms = relative_time_sec * 1000.0
    absolute_offset_sec = offs_sub - tsubint / 2.0 + mu_fit * tbin
    t0 = Time(date_obs_iso, format='isot', scale='utc')
    absolute_time = t0 + absolute_offset_sec * u.s
    try:
        absolute_time.precision = 9
    except Exception:
        pass
    absolute_utc = absolute_time.isot
    jd1 = absolute_time.jd1
    jd2 = absolute_time.jd2
    return relative_time_sec, relative_time_ms, absolute_time, jd1, jd2, absolute_utc


def _compose_mjd_string_from_jd_parts(jd1, jd2, digits=15):
    getcontext().prec = max(28, digits + 5)
    dec_jd1 = Decimal(repr(jd1))
    dec_jd2 = Decimal(repr(jd2))
    dec_mjd = dec_jd1 + dec_jd2 - Decimal('2400000.5')
    fmt = f"%.{digits}f"
    return fmt % (dec_mjd,)


def calculate_mjd_from_offset(t0_iso, relative_time_ms):
    t0 = Time(t0_iso, format='isot', scale='utc')
    offset = relative_time_ms * u.ms
    new_time = t0 + offset
    return new_time.mjd


def mjd_to_utc_string(mjd_times, subsecond_digits=6):
    def format_single_mjd(mjd, digits):
        t = Time(mjd, format='mjd', scale='utc')
        day_fraction = mjd - math.floor(mjd)
        seconds_of_day = day_fraction * 86400.0
        hours = int(seconds_of_day // 3600)
        minutes = int((seconds_of_day % 3600) // 60)
        seconds_int = int(seconds_of_day % 60)
        seconds_frac = seconds_of_day - int(seconds_of_day)
        if digits <= 0:
            return f"{t.iso[:10]}T{hours:02d}:{minutes:02d}:{seconds_int:02d}"
        seconds_frac_rounded = round(seconds_frac, digits)
        frac_str = f"{seconds_frac_rounded:.{digits}f}".split('.')[1]
        return f"{t.iso[:10]}T{hours:02d}:{minutes:02d}:{seconds_int:02d}.{frac_str}"

    if np.isscalar(mjd_times):
        return format_single_mjd(mjd_times, subsecond_digits)
    return [format_single_mjd(mjd, subsecond_digits) for mjd in mjd_times]


def get_coarse_time_precision(time_resolution):
    if time_resolution <= 1e-6:
        return 3, 12, 6
    if time_resolution <= 1e-3:
        return 3, 9, 3
    return 1, 6, 0


# ---------------------------------------------------------------------------
# Precise pulse timing with Gaussian fitting (verbatim)
# ---------------------------------------------------------------------------

def precise_pulse_timing(lightcurve, times, peaks_index, pulse_widths_ms, width_info,
                         time_resolution, method='base', fit_window_factor=3.0,
                         global_noise_sigma=None, amplitude_snrs=None,
                         n_sigma_flux=3.0, tbin=None, date_obs_iso=None,
                         offs_sub_arr=None, tsubint_arr=None):
    precise_peaks = []
    precise_times = []
    fit_results = []

    for i, (peak_idx, pulse_width_ms) in enumerate(zip(peaks_index, pulse_widths_ms)):
        window_size_samples = int((pulse_width_ms * fit_window_factor) /
                                  (time_resolution * 1000))
        window_size_samples = max(window_size_samples, 301)
        if window_size_samples % 2 == 0:
            window_size_samples += 1

        half_window = window_size_samples // 2
        start_idx = max(0, peak_idx - half_window)
        end_idx = min(len(lightcurve), peak_idx + half_window + 1)

        x_data = np.arange(start_idx, end_idx)
        y_data = lightcurve[start_idx:end_idx]

        if len(y_data) < 10:
            fit_results.append({
                'pulse_index': i, 'coarse_peak_idx': peak_idx,
                'precise_peak_idx': float(peak_idx),
                'coarse_time': times[peak_idx], 'precise_time': times[peak_idx],
                'fit_success': False, 'error': 'Insufficient data points',
            })
            continue

        background_init, _ = calculate_background_init(
            lightcurve, start_idx, end_idx, window_size_samples, window_multiplier=5
        )
        amplitude_init = max(y_data.max() - background_init, 1e-6)
        mu_init = float(peak_idx)
        if method == 'fwhm':
            sigma_init = (pulse_width_ms * 1e-3) / 2.355 / time_resolution
        else:
            pulse_width_samples = int(pulse_width_ms / (time_resolution * 1000))
            sigma_init = max(pulse_width_samples / 4, 1.0)

        if not (start_idx <= mu_init <= end_idx - 1):
            mu_init = float(peak_idx)

        p0 = [amplitude_init, mu_init, sigma_init, background_init]
        tolerance = max(1, (end_idx - start_idx) * 0.15)
        bounds = (
            [0, start_idx - tolerance, time_resolution / 10, 0],
            [np.inf, end_idx - 1 + tolerance,
             (end_idx - start_idx) * time_resolution, np.inf],
        )

        try:
            result = curve_fit(gaussian_pulse, x_data, y_data, p0=p0, bounds=bounds,
                               maxfev=10000, ftol=1e-8, xtol=1e-8)
            popt, pcov = result[0], result[1]
        except Exception as e:
            try:
                result_u = curve_fit(gaussian_pulse, x_data, y_data, p0=p0, maxfev=8000)
                popt, pcov = result_u[0], result_u[1]
                A_fit, mu_fit, _, _ = popt
                if not (start_idx <= mu_fit <= end_idx - 1):
                    raise ValueError("Invalid center position from unconstrained fit")
            except Exception as e2:
                fit_results.append({
                    'pulse_index': i, 'coarse_peak_idx': peak_idx,
                    'precise_peak_idx': float(peak_idx),
                    'coarse_time': times[peak_idx], 'precise_time': times[peak_idx],
                    'fit_success': False,
                    'error': f"Constrained: {e}; Unconstrained: {e2}",
                })
                continue

        A_fit, mu_fit, sigma_fit, background_fit = popt
        try:
            param_errors = np.sqrt(np.diag(pcov))
            A_err, mu_err, sigma_err, background_err = param_errors
        except Exception:
            A_err = mu_err = sigma_err = background_err = 0

        fwhm_fit = 2.355 * sigma_fit * time_resolution
        y_fit = gaussian_pulse(x_data, *popt)
        ss_res = np.sum((y_data - y_fit) ** 2)
        ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        if global_noise_sigma is not None and global_noise_sigma > 0:
            snr_amp = A_fit / global_noise_sigma
        else:
            residuals = y_data - y_fit
            noise_std_local = np.std(residuals) if len(residuals) > 10 else 1.0
            snr_amp = A_fit / noise_std_local if noise_std_local > 0 else 0

        detection_amplitude_snr = (amplitude_snrs[i]
                                   if amplitude_snrs is not None and i < len(amplitude_snrs)
                                   else None)

        flux, flux_err, flux_snr_from_fit = calculate_flux_snr_from_fit(
            A_fit, A_err, sigma_fit, sigma_err, global_noise_sigma
        )
        flux_snr_pass = flux_snr_from_fit >= n_sigma_flux

        absolute_time_obj = None
        jd1 = jd2 = None
        precise_mjd_str = None
        if tbin is not None and date_obs_iso is not None:
            # v4: use OFFS_SUB[0] as baseline + mu_fit * tbin for absolute time
            _offs_sub = 0.0
            _tsubint = 0.0
            if offs_sub_arr is not None and tsubint_arr is not None and len(offs_sub_arr) > 0:
                _offs_sub = float(offs_sub_arr[0])
                _tsubint = float(tsubint_arr[0])
            (relative_time_sec, relative_time_ms, absolute_time_obj,
             jd1, jd2, absolute_utc) = calculate_pulse_absolute_time(
                mu_fit, tbin, date_obs_iso, offs_sub=_offs_sub, tsubint=_tsubint)
            absolute_mjd = absolute_time_obj.mjd
            precise_mjd_str = _compose_mjd_string_from_jd_parts(jd1, jd2, digits=15)
        else:
            relative_time_sec = times[start_idx] + (mu_fit - start_idx) * time_resolution
            relative_time_ms = relative_time_sec * 1000
            absolute_mjd = None
            absolute_utc = None

        fit_result = {
            'pulse_index': i, 'coarse_peak_idx': peak_idx,
            'precise_peak_idx': mu_fit,
            'coarse_time': times[peak_idx], 'precise_time': relative_time_sec,
            'precise_time_ms': relative_time_ms,
            'precise_time_mjd': absolute_mjd,
            'precise_time_utc': absolute_utc,
            'precise_time_timeobj': absolute_time_obj,
            'precise_time_jd1': jd1, 'precise_time_jd2': jd2,
            'precise_time_mjd_str': precise_mjd_str,
            'amplitude': A_fit, 'amplitude_err': A_err,
            'sigma': sigma_fit, 'sigma_err': sigma_err,
            'fwhm_sec': fwhm_fit, 'fwhm_ms': fwhm_fit * 1000,
            'background': background_fit, 'background_err': background_err,
            'r_squared': r_squared,
            'snr_amplitude': snr_amp,
            'snr_amplitude_detection': detection_amplitude_snr,
            'flux': flux, 'flux_err': flux_err,
            'flux_snr': flux_snr_from_fit, 'flux_snr_pass': flux_snr_pass,
            'global_noise_sigma_used': global_noise_sigma or 0,
            'param_covariance': pcov,
            'fit_success': True,
            'window_size_samples': window_size_samples,
            'fit_data_range': (start_idx, end_idx),
        }

        if flux_snr_pass:
            precise_peaks.append(mu_fit)
            precise_times.append(relative_time_sec)
            fit_results.append(fit_result)

    return np.array(precise_peaks), np.array(precise_times), fit_results


# ---------------------------------------------------------------------------
# HDUList adapters
# ---------------------------------------------------------------------------

def _read_psrfits_data_from_hdulist(hdulist, subint_range=None, channel_range=None,
                                    normalize=True, pol_index=0):
    """In-memory equivalent of read_psrfits_data."""
    primary_hdr = hdulist[0].header
    subint_hdu = hdulist['SUBINT']
    subint_hdr = subint_hdu.header
    subint_data = subint_hdu.data

    nsubint_total = subint_hdr['NAXIS2']
    nchan = subint_hdr['NCHAN']
    npol = subint_hdr['NPOL']
    nsblk = subint_hdr['NSBLK']

    if subint_range is None:
        subint_range = (0, nsubint_total)
    start_sub, end_sub = subint_range
    start_sub = max(0, start_sub)
    end_sub = min(nsubint_total, end_sub)
    nsubint_used = end_sub - start_sub

    if channel_range is None:
        channel_range = (0, nchan)
    start_chan, end_chan = channel_range
    start_chan = max(0, start_chan)
    end_chan = min(nchan, end_chan)
    nchan_used = end_chan - start_chan

    tdim_str = (subint_hdr.get('TDIM17') or subint_hdr.get('TDIM16')
                or subint_hdr.get('TDIM18'))
    if tdim_str is None:
        for key in subint_hdr.keys():
            if key.startswith('TDIM'):
                tdim_str = subint_hdr[key]
                break
    tdim = tdim_str.strip('()')
    dims = [int(x) for x in tdim.split(',')]

    data_2d = np.zeros((nsubint_used * nsblk, nchan_used), dtype=np.float64)
    times = np.zeros(nsubint_used * nsblk, dtype=np.float64)
    freqs_all = subint_data['DAT_FREQ'][0]
    freqs = freqs_all[start_chan:end_chan]
    tbin = subint_hdr['TBIN']

    for i in range(start_sub, end_sub):
        subint_idx = i - start_sub
        subint_raw = subint_data['DATA'][i]
        subint_reshaped = subint_raw.reshape(dims)

        if len(dims) == 4 and dims[0] == 1:
            subint_pol = subint_reshaped[0, start_chan:end_chan, pol_index, :]
            subint_2d = subint_pol.T
        else:
            nsblk_idx = np.argwhere(np.array(dims) == nsblk)[0][0]
            nchan_idx = np.argwhere(np.array(dims) == nchan)[0][0]
            npol_idx = np.argwhere(np.array(dims) == npol)[0][0]
            perm = [nsblk_idx, nchan_idx, npol_idx]
            subint_reordered = subint_reshaped.transpose(perm)
            subint_2d = subint_reordered[:, start_chan:end_chan, pol_index]

        subint_2d = subint_2d.astype(np.float64, copy=False)

        if normalize:
            dat_scl = subint_data['DAT_SCL'][i]
            dat_offs = subint_data['DAT_OFFS'][i]
            dat_scl_resh = dat_scl.reshape(nchan, npol)[start_chan:end_chan, pol_index]
            dat_offs_resh = dat_offs.reshape(nchan, npol)[start_chan:end_chan, pol_index]
            for chan in range(nchan_used):
                subint_2d[:, chan] = subint_2d[:, chan] * dat_scl_resh[chan] + dat_offs_resh[chan]

        start_idx = subint_idx * nsblk
        end_idx = (subint_idx + 1) * nsblk
        data_2d[start_idx:end_idx, :] = subint_2d

        tsubint = subint_data['TSUBINT'][i]
        offs_sub = subint_data['OFFS_SUB'][i]
        start_time = offs_sub - tsubint / 2
        times[start_idx:end_idx] = start_time + np.arange(nsblk) * tbin

        del subint_raw, subint_reshaped, subint_2d  # release per-subint temporaries

    header_info = {
        'src_name': primary_hdr.get('SRC_NAME', 'Unknown'),
        'freq_center': primary_hdr.get('OBSFREQ', 0),
        'bw': primary_hdr.get('OBSBW', 0),
        'telescope': primary_hdr.get('TELESCOP', 'Unknown'),
        'npol': npol,
        'pol_type': subint_hdr.get('POL_TYPE', 'Unknown'),
        'tbin': tbin,
        'nsubint_total': nsubint_total,
    }
    return data_2d, freqs, times, header_info


def _get_observation_start_info_from_hdulist(hdulist):
    primary_hdr = hdulist[0].header
    subint_hdr = hdulist['SUBINT'].header
    date_obs_iso = primary_hdr.get('DATE-OBS')
    if not date_obs_iso:
        raise ValueError("hdulist primary header is missing DATE-OBS keyword.")
    tbin = subint_hdr.get('TBIN', None)
    if tbin is None:
        raise ValueError("hdulist SUBINT header is missing TBIN keyword.")
    obs_info = {
        'mjd_int': primary_hdr.get('STT_IMJD', 0),
        'mjd_sec': primary_hdr.get('STT_SMJD', 0),
        'mjd_offs': primary_hdr.get('STT_OFFS', 0.0),
        'date_obs': date_obs_iso,
        'tbin': tbin,
        'telescope': primary_hdr.get('TELESCOP', 'Unknown'),
        'observer': primary_hdr.get('OBSERVER', 'Unknown'),
    }
    return date_obs_iso, obs_info


# ---------------------------------------------------------------------------
# CSV extraction (drops FITS_File column)
# ---------------------------------------------------------------------------

def _extract_pulse_data_for_csv(peaks_detected, final_times, date_obs_iso, fit_results,
                                background_level, noise_sigma,
                                coarse_relative_times_rounded, coarse_absolute_mjd,
                                coarse_utc_times,
                                n_sigma_amplitude, n_sigma_flux):
    pulse_data_list = []
    if len(peaks_detected) == 0:
        return pulse_data_list

    time_resolution = (final_times[1] - final_times[0]
                       if len(final_times) > 1 else 1e-6)
    _, coarse_mjd_digits, _ = get_coarse_time_precision(time_resolution)

    for i, peak_idx in enumerate(peaks_detected):
        if fit_results is None or i >= len(fit_results):
            continue
        result = fit_results[i]
        snr_amp_det = result.get('snr_amplitude_detection', None)
        snr_flux_fit = result.get('flux_snr', None)
        fit_success = result.get('fit_success', False)
        if not (fit_success and snr_amp_det is not None and snr_flux_fit is not None
                and snr_amp_det >= n_sigma_amplitude and snr_flux_fit >= n_sigma_flux):
            continue

        rel_time_ms_txt = coarse_relative_times_rounded[i] * 1000
        abs_mjd_txt = coarse_absolute_mjd[i]
        utc_time_txt = (coarse_utc_times[i] if isinstance(coarse_utc_times, list)
                        else coarse_utc_times)

        precise_rel_time_ms = result.get('precise_time_ms', result['precise_time'] * 1000)
        precise_abs_mjd = result.get('precise_time_mjd')
        if precise_abs_mjd is None:
            precise_abs_mjd = calculate_mjd_from_offset(date_obs_iso, precise_rel_time_ms)
        precise_utc = result.get('precise_time_utc')
        if precise_utc is None:
            precise_utc = mjd_to_utc_string(precise_abs_mjd, subsecond_digits=9)

        pulse_data = {
            'Coarse_Index': int(peak_idx),
            'Precise_Rel_Time_ms': float(precise_rel_time_ms),
            'Precise_Abs_MJD': float(precise_abs_mjd) if precise_abs_mjd is not None else np.nan,
            'Precise_UTC_Time': precise_utc,
            'Background_Level': float(background_level),
            'Noise_Sigma': float(noise_sigma),
        }

        jd1_val = result.get('precise_time_jd1')
        jd2_val = result.get('precise_time_jd2')
        if jd1_val is not None and jd2_val is not None:
            precise_mjd_string = (result.get('precise_time_mjd_str')
                                  or _compose_mjd_string_from_jd_parts(jd1_val, jd2_val,
                                                                       digits=coarse_mjd_digits))
            pulse_data.update({
                'Precise_JD1': float(jd1_val),
                'Precise_JD2': float(jd2_val),
                'Precise_Abs_MJD_Str': precise_mjd_string,
            })

        time_error_ms = 0.001
        center_error = 0.0
        if result.get('param_covariance') is not None:
            try:
                center_error = np.sqrt(result['param_covariance'][1, 1])
                tres = final_times[1] - final_times[0] if len(final_times) > 1 else time_resolution
                time_error_ms = center_error * tres * 1000
            except Exception:
                pass

        fwhm_error_ms = 2.355 * result.get('sigma_err', 0) if result.get('sigma_err', 0) > 0 else 0
        flux_val = result.get('flux', None)
        flux_err_val = result.get('flux_err', None)

        pulse_data.update({
            'Precise_Time_Err_ms': float(time_error_ms),
            'Precise_Center_Index': float(result.get('precise_peak_idx', 0)),
            'Center_Err': float(center_error),
            'Amplitude': float(result.get('amplitude', 0)),
            'Amp_Err': float(result.get('amplitude_err', 0)),
            'FWHM_ms': float(result.get('fwhm_ms', 0)),
            'FWHM_Err': float(fwhm_error_ms),
            'R_2': float(result.get('r_squared', 0)),
            'Background_Fit': float(result.get('background', 0)),
            'SNR_Amplitude_Fit': float(result.get('snr_amplitude', 0)),
            'SNR_Amplitude_Detection': float(snr_amp_det) if snr_amp_det is not None else np.nan,
            'Flux_From_Fit': float(flux_val) if flux_val is not None else np.nan,
            'Flux_Err': float(flux_err_val) if flux_err_val is not None else np.nan,
            'SNR_Flux_From_Fit': float(snr_flux_fit) if snr_flux_fit is not None else np.nan,
            'Fit_Success': True,
        })
        pulse_data_list.append(pulse_data)

    return pulse_data_list


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_pulses_in_hdulist(hdulist, params):
    """
    Run the full v4.1 pulse detection pipeline on an in-memory PSRFITS HDUList.

    v2: explicit `del` of each huge intermediate array between detection
    stages so that, at any moment, only one full-size data buffer is
    resident instead of (data_2d + data_freq_cleaned + data_tf_cleaned).

    Returns
    -------
    list[dict]
        Pulse rows (no FITS_File column). Empty if nothing passes both SNRs.
    """
    n_sigma_amplitude = params['amp_snr_threshold']
    n_sigma_flux = params['flux_snr_threshold']
    peak_distance = params.get('peak_distance')
    sigma_rfi_freq = params['sigma_remove_rfi_frequency']
    sigma_rfi_tf = params['sigma_remove_rfi_time_frequency']
    manual_mask_ranges = params.get('manual_mask_freq_ranges') or []
    min_prom_factor = params.get('min_prominence_sigma_factor', 0.3)
    tf_window = params.get('rfi_time_freq_window', 200)

    print("=" * 60)
    print("Pulse detection (in-memory, v2)")
    print("=" * 60)

    # ----- Step 1: Read data from hdulist -----
    data_2d, freqs, times, header_info = _read_psrfits_data_from_hdulist(
        hdulist, normalize=True, pol_index=0
    )

    # ----- Step 2: Frequency-domain RFI removal -----
    manual_mask = ({'freq_ranges': list(manual_mask_ranges)}
                   if manual_mask_ranges else None)
    data_freq_cleaned, _ = remove_rfi_frequency_domain(
        data_2d, freqs=freqs, method='sigma_clip',
        sigma=sigma_rfi_freq, percentile=80, manual_mask=manual_mask
    )
    # data_2d is no longer needed; freq_cleaned has its own buffer
    del data_2d

    # ----- Step 3: Time-frequency RFI removal -----
    data_tf_cleaned, _ = remove_rfi_time_frequency(
        data_freq_cleaned, window_size=tf_window, sigma=sigma_rfi_tf
    )
    del data_freq_cleaned

    # ----- Step 4: Light curve + background -----
    lightcurve = create_lightcurve(data_tf_cleaned)
    del data_tf_cleaned  # the small lightcurve replaces the big 2D buffer
    final_times = times

    M_final, sigma_final, _ = iterative_sigma_clipping(
        lightcurve, initial_K=3, final_K=5, max_iterations=50
    )
    print(f"Background={M_final:.6f}, noise sigma={sigma_final:.6f}")

    if not np.isfinite(M_final) or not np.isfinite(sigma_final) or sigma_final <= 0:
        print("Invalid background/noise estimate; skipping.")
        return []

    # ----- Step 5: Peak detection -----
    peaks_detected, _, amplitude_snrs_detected = detect_pulses_v3(
        lightcurve=lightcurve, times=final_times,
        background_level=M_final, noise_sigma=sigma_final,
        n_sigma_amplitude=n_sigma_amplitude,
        n_sigma_flux=n_sigma_flux,
        min_prominence_sigma_factor=min_prom_factor,
        peak_distance=peak_distance,
    )
    if len(peaks_detected) == 0:
        print("No initial peaks passed amplitude SNR.")
        return []

    # ----- Step 6: Pulse widths -----
    pulse_widths_ms, width_info = calculate_pulse_widths(
        lightcurve, final_times, peaks_detected,
        M_final + n_sigma_amplitude * sigma_final, method='base'
    )

    # ----- Step 7: DATE-OBS + TBIN + OFFS_SUB/TSUBINT -----
    date_obs_iso, obs_info = _get_observation_start_info_from_hdulist(hdulist)
    # v4: read OFFS_SUB and TSUBINT arrays for correct absolute TOA calculation
    subint_data = hdulist['SUBINT'].data
    offs_sub_arr = subint_data['OFFS_SUB']
    tsubint_arr = subint_data['TSUBINT']

    # ----- Step 8: Precise pulse timing -----
    time_resolution = (final_times[1] - final_times[0]
                       if len(final_times) > 1 else header_info.get('tbin', 1e-6))
    _precise_peaks, _precise_times, fit_results = precise_pulse_timing(
        lightcurve, final_times, peaks_detected, pulse_widths_ms, width_info,
        time_resolution, method=width_info['method'], fit_window_factor=3.0,
        global_noise_sigma=sigma_final,
        amplitude_snrs=amplitude_snrs_detected,
        n_sigma_flux=n_sigma_flux,
        tbin=obs_info['tbin'],
        date_obs_iso=date_obs_iso,
        offs_sub_arr=offs_sub_arr,
        tsubint_arr=tsubint_arr,
    )
    if len(fit_results) == 0:
        print("No pulses passed the flux SNR threshold after Gaussian fitting.")
        del lightcurve, final_times, times, freqs, _precise_peaks, _precise_times
        return []

    # ----- Step 9: Coarse-time arrays -----
    coarse_relative_times = final_times[peaks_detected]
    coarse_relative_times_rounded = np.round(coarse_relative_times, 6)
    coarse_relative_times_ms = coarse_relative_times_rounded * 1000
    coarse_absolute_mjd = np.array([
        calculate_mjd_from_offset(date_obs_iso, t_ms)
        for t_ms in coarse_relative_times_ms
    ])
    _, _, coarse_utc_digits = get_coarse_time_precision(time_resolution)
    coarse_utc_times = mjd_to_utc_string(coarse_absolute_mjd,
                                         subsecond_digits=coarse_utc_digits)

    # ----- Step 10: CSV rows -----
    pulse_data_list = _extract_pulse_data_for_csv(
        peaks_detected, final_times, date_obs_iso, fit_results,
        M_final, sigma_final,
        coarse_relative_times_rounded, coarse_absolute_mjd, coarse_utc_times,
        n_sigma_amplitude=n_sigma_amplitude, n_sigma_flux=n_sigma_flux,
    )
    print(f"Detected {len(pulse_data_list)} pulses passing both SNR thresholds.")

    # Drop heavy intermediates before returning (param_covariance arrays inside
    # fit_results were the largest accumulation per call)
    del fit_results, lightcurve, final_times, times, freqs
    return pulse_data_list
