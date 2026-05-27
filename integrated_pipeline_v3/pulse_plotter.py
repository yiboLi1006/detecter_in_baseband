"""
Pulse waterfall plotter -- generates comparison images (raw + DM-corrected)
for detected pulses. Adapted from pulse_viewer_v3.py for inline pipeline use.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.io import fits


def _read_psrfits_data(fits_file, pol_index=0):
    """Read PSRFITS data and return data_2d, header_info."""
    with fits.open(fits_file) as hdul:
        subint_hdu = hdul['SUBINT']
        subint_hdr = subint_hdu.header
        subint_data = subint_hdu.data

        nsubint_total = subint_hdr['NAXIS2']
        nchan = subint_hdr['NCHAN']
        npol = subint_hdr['NPOL']
        nsblk = subint_hdr['NSBLK']
        tbin = subint_hdr['TBIN']

        tdim = subint_hdr['TDIM17'].strip('()')
        dims = [int(x) for x in tdim.split(',')]

        data_2d = np.zeros((nsubint_total * nsblk, nchan))

        for i in range(nsubint_total):
            subint_raw = subint_data['DATA'][i]
            subint_reshaped = subint_raw.reshape(dims)

            if len(dims) == 4 and dims[0] == 1:
                subint_pol = subint_reshaped[0, :, pol_index, :]
                subint_2d = subint_pol.T
            else:
                nsblk_idx = np.argwhere(np.array(dims) == nsblk)[0][0]
                nchan_idx = np.argwhere(np.array(dims) == nchan)[0][0]
                npol_idx = np.argwhere(np.array(dims) == npol)[0][0]
                perm = [nsblk_idx, nchan_idx, npol_idx]
                subint_reordered = subint_reshaped.transpose(perm)
                subint_2d = subint_reordered[:, :, pol_index]

            dat_scl = subint_data['DAT_SCL'][i]
            dat_offs = subint_data['DAT_OFFS'][i]
            dat_scl_reshaped = dat_scl.reshape(nchan, npol)[:, pol_index]
            dat_offs_reshaped = dat_offs.reshape(nchan, npol)[:, pol_index]
            for chan in range(nchan):
                subint_2d[:, chan] = (subint_2d[:, chan] * dat_scl_reshaped[chan]
                                      + dat_offs_reshaped[chan])

            start_idx = i * nsblk
            end_idx = (i + 1) * nsblk
            data_2d[start_idx:end_idx, :] = subint_2d

        freqs = subint_data['DAT_FREQ'][0]  # MHz

        n_total_samples = nsubint_total * nsblk
        times = np.arange(n_total_samples) * tbin  # seconds

        header_info = {
            'tbin': tbin,
            'nchan': nchan,
            'nsblk': nsblk,
            'nsubint_total': nsubint_total,
            'freqs': freqs,
            'times': times,
        }
        return data_2d, header_info


def _safe_slice(data, center_idx, half_width, num_widths=5):
    """Slice data around center_idx with padding for boundary cases."""
    data_height, data_width = data.shape
    desired_start = center_idx - num_widths * half_width
    desired_end = center_idx + num_widths * half_width
    crop_start = max(0, desired_start)
    crop_end = min(data_height, desired_end)
    pad_top = max(0, -desired_start)
    pad_bottom = max(0, desired_end - data_height)

    if crop_start >= crop_end:
        sliced_data = np.zeros((desired_end - desired_start, data_width))
        return sliced_data, crop_start, crop_end, pad_top, pad_bottom, desired_start, desired_end

    sliced_data = data[crop_start:crop_end, :]
    if pad_top > 0 or pad_bottom > 0:
        final_height = desired_end - desired_start
        final_data = np.zeros((final_height, data_width))
        actual_start = pad_top
        actual_end = pad_top + (crop_end - crop_start)
        final_data[actual_start:actual_end, :] = sliced_data
        return final_data, crop_start, crop_end, pad_top, pad_bottom, desired_start, desired_end

    return sliced_data, crop_start, crop_end, pad_top, pad_bottom, desired_start, desired_end


def _gaussian_pulse(x, A, mu, sigma, background):
    return A * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) + background


def _create_lightcurve(data_2d):
    return np.nanmean(data_2d, axis=1)


def _plot_single_pulse(original_data, corrected_data, center_idx, half_width,
                       fwhm_ms, raw_fits_name, corrected_fits_name, save_path,
                       tbin_us, snr, pulse_index, freqs,
                       corrected_lightcurve, pulse_dict):
    """Generate a three-panel figure: raw waterfall, DM-corrected waterfall,
    and light curve with Gaussian fit overlay."""
    plt.rcParams['font.family'] = 'Times New Roman'

    fig = plt.figure(figsize=(12, 11))
    gs = fig.add_gridspec(3, 1, hspace=0.35)
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1], sharex=ax0)
    ax2 = fig.add_subplot(gs[2])

    original_num_widths = 80
    corrected_num_widths = 20
    tbin_ms = tbin_us / 1000.0
    tbin_s = tbin_us / 1e6

    # --- waterfall slices ---
    (original_sliced, orig_crop_start, orig_crop_end,
     orig_pad_top, orig_pad_bottom, orig_desired_start, orig_desired_end) = _safe_slice(
        original_data, center_idx, half_width, num_widths=original_num_widths)
    (corrected_sliced, corr_crop_start, corr_crop_end,
     corr_pad_top, corr_pad_bottom, corr_desired_start, corr_desired_end) = _safe_slice(
        corrected_data, center_idx, half_width, num_widths=corrected_num_widths)

    vmin = min(np.nanmin(original_sliced) if np.any(~np.isnan(original_sliced)) else 0,
               np.nanmin(corrected_sliced) if np.any(~np.isnan(corrected_sliced)) else 0)
    vmax = max(np.nanmax(original_sliced) if np.any(~np.isnan(original_sliced)) else 1,
               np.nanmax(corrected_sliced) if np.any(~np.isnan(corrected_sliced)) else 1)
    if vmin == vmax:
        vmin -= 0.1
        vmax += 0.1

    ref_center_in_original_slice = center_idx - orig_desired_start
    ref_center_in_corrected_slice = center_idx - corr_desired_start
    ref_line_upper_corrected = ref_center_in_corrected_slice + half_width
    ref_line_lower_corrected = ref_center_in_corrected_slice - half_width
    ref_center_in_original_slice = max(0, min(ref_center_in_original_slice, original_sliced.shape[0] - 1))
    ref_center_in_corrected_slice = max(0, min(ref_center_in_corrected_slice, corrected_sliced.shape[0] - 1))
    ref_line_upper_corrected = max(0, min(ref_line_upper_corrected, corrected_sliced.shape[0] - 1))
    ref_line_lower_corrected = max(0, min(ref_line_lower_corrected, corrected_sliced.shape[0] - 1))

    freq_start, freq_end = freqs[0], freqs[-1]

    # --- panel 1: raw waterfall ---
    im1 = ax0.imshow(
        original_sliced, aspect='auto', cmap='inferno', origin='lower', vmin=vmin, vmax=vmax,
        extent=[freq_start, freq_end, 0, original_sliced.shape[0] * tbin_ms])
    ax0.axhline(y=ref_center_in_original_slice * tbin_ms, color='cyan',
                linestyle='--', linewidth=1.6, alpha=0.5)
    if orig_pad_top > 0:
        ax0.axhline(y=orig_pad_top * tbin_ms, color='white', linestyle=':', linewidth=1, alpha=0.5)
    if orig_pad_bottom > 0 and orig_pad_bottom < original_sliced.shape[0]:
        ax0.axhline(y=(original_sliced.shape[0] - orig_pad_bottom - 1) * tbin_ms,
                    color='white', linestyle=':', linewidth=1, alpha=0.5)
    ax0.set_title(f"Raw: {raw_fits_name}", fontsize=12, fontweight='bold')
    ax0.set_ylabel('Time (ms)', fontsize=11)

    # --- panel 2: DM-corrected waterfall ---
    # --- panel 2: DM-corrected waterfall ---
    im2 = ax1.imshow(
        corrected_sliced, aspect='auto', cmap='inferno', origin='lower', vmin=vmin, vmax=vmax,
        extent=[freq_start, freq_end, 0, corrected_sliced.shape[0] * tbin_ms])
    ax1.axhline(y=ref_line_upper_corrected * tbin_ms, color='red', linestyle='--',
                linewidth=2, alpha=0.8, label=f'Pulse bounds (±{fwhm_ms:.3f} ms)')
    ax1.axhline(y=ref_line_lower_corrected * tbin_ms, color='red', linestyle='--',
                linewidth=2, alpha=0.8)
    ax1.axhline(y=ref_center_in_corrected_slice * tbin_ms, color='cyan',
                linestyle='--', linewidth=0.7, alpha=0.3, label='Pulse center')
    if corr_pad_top > 0:
        ax1.axhline(y=corr_pad_top * tbin_ms, color='white', linestyle=':', linewidth=1, alpha=0.5)
    if corr_pad_bottom > 0 and corr_pad_bottom < corrected_sliced.shape[0]:
        ax1.axhline(y=(corrected_sliced.shape[0] - corr_pad_bottom - 1) * tbin_ms,
                    color='white', linestyle=':', linewidth=1, alpha=0.5)
    ax1.set_title(f"DM Corrected: {corrected_fits_name}", fontsize=12, fontweight='bold')
    ax1.set_ylabel('Time (ms)', fontsize=11)
    ax1.set_xlabel('Frequency (MHz)', fontsize=11)
    ax1.legend(loc='upper right', fontsize=9, framealpha=0.8)

    # --- panel 3: light curve + Gaussian fit ---
    amplitude = pulse_dict.get('Amplitude', 0)
    background_fit = pulse_dict.get('Background_Fit', 0)
    background_level = pulse_dict.get('Background_Level', 0)
    noise_sigma = pulse_dict.get('Noise_Sigma', 0)
    r_squared = pulse_dict.get('R_2', np.nan)

    sigma_sec = fwhm_ms / 1000.0 / 2.355
    sigma_samples = sigma_sec / tbin_s if tbin_s > 0 else fwhm_ms * 1000 / 2.355

    lc_num_widths = 8
    n_total = len(corrected_lightcurve)
    lc_desired_start = center_idx - lc_num_widths * half_width
    lc_desired_end = center_idx + lc_num_widths * half_width
    lc_crop_start = max(0, lc_desired_start)
    lc_crop_end = min(n_total, lc_desired_end)

    lc_values = corrected_lightcurve[lc_crop_start:lc_crop_end].astype(float)
    lc_indices = np.arange(lc_crop_start, lc_crop_end, dtype=float)

    pad_left = max(0, -lc_desired_start)
    pad_right = max(0, lc_desired_end - n_total)
    if pad_left > 0 or pad_right > 0:
        padded = np.full(int(lc_desired_end - lc_desired_start), np.nan)
        padded[pad_left:pad_left + len(lc_values)] = lc_values
        lc_values = padded
        lc_indices = np.arange(lc_desired_start, lc_desired_end, dtype=float)

    lc_times_ms = lc_indices * tbin_ms

    ax2.plot(lc_times_ms, lc_values, color='black', linewidth=0.8, label='Light curve')

    # Gaussian fit (20x oversampled)
    x_smooth = np.linspace(lc_indices[0], lc_indices[-1], int(len(lc_indices) * 20))
    y_fit = _gaussian_pulse(x_smooth, amplitude, center_idx, sigma_samples, background_fit)
    ax2.plot(x_smooth * tbin_ms, y_fit, color='red', linestyle='--', linewidth=1.5,
             label=f'Gaussian fit (R²={r_squared:.3f})' if not np.isnan(r_squared) else 'Gaussian fit')

    # Center, bounds, threshold
    center_time_ms = center_idx * tbin_ms
    ax2.axvline(x=center_time_ms, color='green', linestyle='--', linewidth=1.2,
                alpha=0.7, label=f'Center ({center_time_ms:.3f} ms)')

    upper_bound_ms = (center_idx + half_width) * tbin_ms
    lower_bound_ms = (center_idx - half_width) * tbin_ms
    ax2.axvline(x=upper_bound_ms, color='red', linestyle=':', linewidth=1.0, alpha=0.5)
    ax2.axvline(x=lower_bound_ms, color='red', linestyle=':', linewidth=1.0, alpha=0.5)

    threshold = background_level + 4 * noise_sigma
    if threshold > 0 and np.any(np.isfinite(lc_values)):
        ax2.axhline(y=threshold, color='blue', linestyle=':', linewidth=1.0, alpha=0.6,
                    label=f'Threshold ({threshold:.3f})')

    # Padding markers
    if pad_left > 0:
        ax2.axvline(x=lc_indices[pad_left] * tbin_ms, color='gray', linestyle=':', linewidth=0.5, alpha=0.5)
    if pad_right > 0:
        ax2.axvline(x=lc_indices[-pad_right - 1] * tbin_ms, color='gray', linestyle=':', linewidth=0.5, alpha=0.5)

    ax2.set_xlabel('Time (ms)', fontsize=11)
    ax2.set_ylabel('Flux', fontsize=11)
    ax2.set_title('Light Curve + Gaussian Fit', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=8, framealpha=0.8)
    ax2.grid(False, alpha=0.3)

    # --- colorbar for waterfall panels ---
    fig.subplots_adjust(top=0.93, right=0.88)
    cbar_ax = fig.add_axes([0.90, 0.52, 0.012, 0.34])
    fig.colorbar(im1, cax=cbar_ax, orientation='vertical').set_label('Intensity', fontsize=11)

    fig.suptitle(f"Pulse #{pulse_index}  |  "
                 f"FWHM: {fwhm_ms:.4f} ms  |  SNR: {snr:.2f}",
                 fontsize=14, fontweight='bold', y=0.98)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  -> pulse plot: {save_path}")


def plot_pulses_for_hdulist(raw_fits_path, corrected_fits_path,
                            pulse_data_list, output_dir,
                            src, tel, file_counter, version):
    """Plot waterfall images for all pulses detected in one hdulist.

    Parameters
    ----------
    raw_fits_path : str
        Path to the raw (un-corrected) PSRFITS file.
    corrected_fits_path : str
        Path to the DM-corrected PSRFITS file.
    pulse_data_list : list[dict]
        List of pulse detection result dicts (from detect_pulses_in_hdulist).
    output_dir : str
        Directory to save PNG images.
    src, tel : str
        Source name and telescope for file naming.
    file_counter : int
        FITS file counter, embedded in the image filename.
    version : int
        Version number for the filename.
    """
    if not pulse_data_list:
        return

    os.makedirs(output_dir, exist_ok=True)

    raw_fits_name = os.path.basename(raw_fits_path)
    corrected_fits_name = os.path.basename(corrected_fits_path)

    try:
        original_data, orig_header = _read_psrfits_data(raw_fits_path)
    except Exception as e:
        print(f"  WARNING: failed to read raw FITS {raw_fits_path}: {e}")
        return
    try:
        corrected_data, corr_header = _read_psrfits_data(corrected_fits_path)
    except Exception as e:
        print(f"  WARNING: failed to read corrected FITS {corrected_fits_path}: {e}")
        return

    tbin_us = corr_header.get('tbin', orig_header['tbin']) * 1e6
    freqs = corr_header.get('freqs', orig_header.get('freqs'))
    corrected_lightcurve = _create_lightcurve(corrected_data)

    for idx, pulse in enumerate(pulse_data_list):
        center_idx = int(pulse.get('Precise_Center_Index', pulse.get('Coarse_Index', 0)))
        fwhm_ms = pulse.get('FWHM_ms', 0)
        snr = pulse.get('SNR_Amplitude_Detection',
                        pulse.get('SNR_Flux_From_Fit', 0))

        if fwhm_ms <= 0:
            print(f"  pulse #{idx}: FWHM <= 0, skipping plot")
            continue

        tbin_ms = tbin_us / 1000.0
        fwhm_samples = fwhm_ms / tbin_ms if tbin_ms > 0 else fwhm_ms * 1000
        half_width_px = int(fwhm_samples)
        if half_width_px < 10:
            half_width_px = 10

        image_filename = (f"PSR_{src}_{tel}_{file_counter:06d}"
                          f"_v{version}_pulse{idx:02d}.png")
        image_path = os.path.join(output_dir, image_filename)

        try:
            _plot_single_pulse(
                original_data, corrected_data,
                center_idx, half_width_px, fwhm_ms,
                raw_fits_name, corrected_fits_name,
                image_path, tbin_us, snr, idx, freqs,
                corrected_lightcurve, pulse)
        except Exception as e:
            print(f"  WARNING: failed to plot pulse #{idx}: {e}")
            continue
