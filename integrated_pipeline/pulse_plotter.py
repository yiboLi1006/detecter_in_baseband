"""
Pulse waterfall plotter -- generates comparison images (raw + DM-corrected)
for detected pulses. Adapted from pulse_viewer_v3.py for inline pipeline use.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from astropy.io import fits


# Display parameters only; detection and timing do not use these values.
RAW_DISPLAY_BIN_US = 16.0
RAW_MARGIN_MS = 1.0
DEDISPERSED_HALF_WINDOW_US = 25.0
BACKGROUND_HALF_WINDOW_US = 1000.0
BACKGROUND_EXCLUDE_US = 100.0
RAW_TRACK_EXCLUDE_MS = 0.4
PNG_DPI = 300
PNG_SIZE_TARGET = 1_000_000


def _read_psrfits_data(fits_file, pol_index=0):
    """Decode a disk file through the same reader used for in-memory HDUs."""
    with fits.open(fits_file) as hdul:
        return _read_psrfits_data_from_hdulist(hdul, pol_index)


def _read_psrfits_data_from_hdulist(hdulist, pol_index=0):
    """Read PSRFITS data from an in-memory HDUList.

    Returns (data_2d, header_info) matching _read_psrfits_data format,
    with 'freqs' and 'times' inside header_info.

    Parameters
    ----------
    hdulist : astropy.io.fits.HDUList
        In-memory PSRFITS HDUList (must have 'SUBINT' extension).
    pol_index : int
        Polarization index to extract (default 0).

    Returns
    -------
    data_2d : np.ndarray, shape (total_samples, nchan), dtype float64
    header_info : dict
        Keys: tbin, nchan, nsblk, nsubint_total, freqs, times
    """
    subint_hdu = hdulist['SUBINT']
    subint_hdr = subint_hdu.header
    subint_data = subint_hdu.data

    nsubint_total = subint_hdr['NAXIS2']
    nchan = subint_hdr['NCHAN']
    npol = subint_hdr['NPOL']
    nsblk = subint_hdr['NSBLK']
    tbin = subint_hdr['TBIN']

    # DATA need not be column17; other columns can also have TDIM entries.
    data_column = list(subint_hdu.columns.names).index('DATA') + 1
    tdim_str = subint_hdr.get(f'TDIM{data_column}')
    if tdim_str is None:
        raise ValueError('PSRFITS DATA column has no TDIM')
    tdim = tdim_str.strip('()')
    dims = [int(x) for x in tdim.split(',')]

    data_2d = np.zeros((nsubint_total * nsblk, nchan), dtype=np.float64)

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

        subint_2d = subint_2d.astype(np.float64, copy=True)

        dat_scl = subint_data['DAT_SCL'][i]
        dat_offs = subint_data['DAT_OFFS'][i]
        dat_scl_resh = dat_scl.reshape(nchan, npol)[:, pol_index]
        dat_offs_resh = dat_offs.reshape(nchan, npol)[:, pol_index]
        for chan in range(nchan):
            subint_2d[:, chan] = (subint_2d[:, chan] * dat_scl_resh[chan]
                                  + dat_offs_resh[chan])

        start_idx = i * nsblk
        end_idx = (i + 1) * nsblk
        data_2d[start_idx:end_idx, :] = subint_2d

    freqs = np.atleast_1d(subint_data['DAT_FREQ'][0]).copy()  # MHz
    n_total_samples = nsubint_total * nsblk
    times = np.arange(n_total_samples) * tbin  # seconds

    header_info = {
        'tbin': tbin,
        'nchan': nchan,
        'nsblk': nsblk,
        'nsubint_total': nsubint_total,
        'freqs': freqs,
        'times': times,
        'dm': subint_hdr.get('DM'),
        'ref_freq': subint_hdr.get('REFFREQ'),
        'obsfreq': hdulist[0].header.get('OBSFREQ'),
        'chan_bw': subint_hdr.get('CHAN_BW'),
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


def _background_limits(background, fallback, lower_sigma, upper_sigma):
    """Robust limits; empty or constant backgrounds still yield a valid scale."""
    values = np.asarray(background)
    values = values[np.isfinite(values)]
    if values.size == 0:
        values = np.asarray(fallback)
        values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0, 1.0
    median = float(np.median(values))
    sigma = float(1.4826 * np.median(np.abs(values - median)))
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = max(abs(median) * 1e-6, 1e-6)
    return median + lower_sigma * sigma, median + upper_sigma * sigma


def _display_parameters(header, freqs, dm, ref_freq):
    """Resolve metadata without treating the raw HDU's DM=0 as the applied DM."""
    if dm is None:
        dm = header.get('dm')
        if dm is None:
            print('  WARNING: plot DM unavailable; displaying a zero-DM raw window')
            dm = 0.0
    if ref_freq is None:
        ref_freq = header.get('ref_freq')
    if ref_freq is None:
        ref_freq = header.get('obsfreq')
    if ref_freq is None:
        ref_freq = float(np.median(freqs))
        print('  WARNING: plot reference frequency unavailable; using channel median')
    dm, ref_freq = float(dm), float(ref_freq)
    if not np.isfinite(dm) or dm < 0:
        raise ValueError('plot DM must be finite and non-negative')
    if not np.isfinite(ref_freq) or ref_freq <= 0:
        raise ValueError('plot reference frequency must be positive (MHz)')
    return dm, ref_freq


def _plot_single_pulse(original_data, corrected_data, center_idx, half_width,
                       fwhm_ms, raw_fits_name, corrected_fits_name, save_path,
                       tbin_us, snr_amp, snr_flux, pulse_index, freqs,
                       corrected_lightcurve, pulse_dict, dm=0.0,
                       ref_freq=None, chan_bw=None):
    """Generate a three-panel figure: raw waterfall, DM-corrected waterfall,
    and light curve with Gaussian fit overlay."""
    plt.rcParams['font.family'] = 'Times New Roman'

    fig = plt.figure(figsize=(12, 11))
    gs = fig.add_gridspec(3, 1, hspace=0.35)
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1], sharex=ax0)
    ax2 = fig.add_subplot(gs[2])

    tbin_ms = tbin_us / 1000.0
    tbin_s = tbin_us / 1e6
    # Only waterfall arrays, windows and colours are changed. The LC below
    # still receives the original full-resolution data and integer fit center.
    wf_freqs = np.atleast_1d(np.asarray(freqs, dtype=float))
    if (wf_freqs.ndim != 1 or not len(wf_freqs)
            or not np.all(np.isfinite(wf_freqs) & (wf_freqs > 0))):
        raise ValueError('waterfall frequencies must be positive finite MHz values')
    if not np.isfinite(tbin_us) or tbin_us <= 0:
        raise ValueError('waterfall sample interval must be finite and positive')
    wf_dm, wf_ref = _display_parameters({}, wf_freqs, dm, ref_freq)
    # Dispersion constant has units seconds MHz^2; convert seconds to ms.
    wf_delay_ms = 4148.8064239 * wf_dm * (wf_freqs**-2 - wf_ref**-2) * 1000.0
    wf_center = float(pulse_dict.get('Precise_Center_Index', center_idx))
    wf_start = max(0, int(np.floor(wf_center + (wf_delay_ms.min()-RAW_MARGIN_MS)/tbin_ms)))
    wf_end = min(len(original_data), int(np.ceil(wf_center + (wf_delay_ms.max()+RAW_MARGIN_MS)/tbin_ms)))
    if wf_end <= wf_start:
        raise ValueError('raw dispersion window lies outside the available data')
    wf_raw = original_data[wf_start:wf_end]
    wf_t = (np.arange(wf_start, wf_end)-wf_center) * tbin_ms
    wf_rebin = min(len(wf_t), max(1, int(round(RAW_DISPLAY_BIN_US/tbin_us))))
    wf_n = len(wf_t)//wf_rebin*wf_rebin
    wf_blocks = wf_raw[:wf_n].reshape(-1, wf_rebin, wf_raw.shape[1])
    wf_count = np.isfinite(wf_blocks).sum(axis=1)
    wf_raw_plot = np.nansum(wf_blocks, axis=1)/np.where(wf_count, wf_count, np.nan)
    wf_raw_t = wf_t[:wf_n].reshape(-1, wf_rebin).mean(axis=1)
    wf_off = np.abs(wf_raw_t[:, None]-wf_delay_ms[None, :]) > RAW_TRACK_EXCLUDE_MS
    wf_raw_limits = _background_limits(wf_raw_plot[wf_off], wf_raw_plot, 0.5, 3.0)

    wf_cs = max(0, int(np.floor(wf_center-DEDISPERSED_HALF_WINDOW_US/tbin_us)))
    wf_ce = min(len(corrected_data), int(np.ceil(wf_center+DEDISPERSED_HALF_WINDOW_US/tbin_us)))
    wf_corrected = corrected_data[wf_cs:wf_ce]
    wf_corr_t = (np.arange(wf_cs, wf_ce)-wf_center)*tbin_us
    wf_bs = max(0, int(wf_center-BACKGROUND_HALF_WINDOW_US/tbin_us))
    wf_be = min(len(corrected_data), int(wf_center+BACKGROUND_HALF_WINDOW_US/tbin_us))
    wf_bt = (np.arange(wf_bs, wf_be)-wf_center)*tbin_us
    wf_background = corrected_data[wf_bs:wf_be][np.abs(wf_bt) > BACKGROUND_EXCLUDE_US]
    wf_corr_limits = _background_limits(wf_background, wf_corrected, 0.0, 6.0)
    if len(wf_freqs) > 1:
        wf_df = float(np.median(np.diff(wf_freqs)))
    else:
        wf_df = float(chan_bw) if chan_bw is not None else 1.0
        if not np.isfinite(wf_df) or wf_df == 0:
            wf_df = 1.0
    wf_freq_edges = [wf_freqs[0]-wf_df/2, wf_freqs[-1]+wf_df/2]
    im1 = ax0.imshow(wf_raw_plot, aspect='auto', cmap='inferno', origin='lower',
        interpolation='nearest', norm=Normalize(*wf_raw_limits, clip=True),
        extent=wf_freq_edges+[wf_raw_t[0]-wf_rebin*tbin_ms/2,
                             wf_raw_t[-1]+wf_rebin*tbin_ms/2])
    ax0.set_title(f"Raw: {raw_fits_name}", fontsize=12, fontweight='bold')
    ax0.set_ylabel('Time from reference TOA (ms)', fontsize=11)
    im2 = ax1.imshow(wf_corrected, aspect='auto', cmap='inferno', origin='lower',
        interpolation='nearest', norm=Normalize(*wf_corr_limits, clip=True),
        extent=wf_freq_edges+[wf_corr_t[0]-tbin_us/2, wf_corr_t[-1]+tbin_us/2])
    ax1.set_title(f"Dedispersed: {corrected_fits_name}", fontsize=12, fontweight='bold')
    ax1.set_ylabel('Time from fitted center (us)', fontsize=11)
    ax1.set_xlabel('Frequency (MHz)', fontsize=11)

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
    for wf_ax,wf_image in [(ax0,im1),(ax1,im2)]:
        wf_box=wf_ax.get_position()
        wf_cax=fig.add_axes([0.90,wf_box.y0,0.012,wf_box.height])
        fig.colorbar(wf_image,cax=wf_cax,orientation='vertical',extend='both').set_label('Intensity',fontsize=11)

    fig.suptitle(f"Pulse #{pulse_index}  |  "
                 f"FWHM: {fwhm_ms:.4f} ms  |  "
                 f"SNR_Amp={snr_amp:.2f}, SNR_Flux={snr_flux:.2f}",
                 fontsize=14, fontweight='bold', y=0.98)
    plt.savefig(save_path, dpi=PNG_DPI, bbox_inches='tight',
                pil_kwargs={'optimize': True})
    if os.path.isfile(save_path) and os.path.getsize(save_path) > PNG_SIZE_TARGET:
        print(f'  NOTE: lossless pulse PNG exceeds 1 MB: {save_path}')
    plt.close()


def plot_pulses_for_hdulist(raw_fits_path, corrected_fits_path,
                            pulse_data_list, output_dir,
                            src, tel, file_counter, version,
                            raw_hdulist=None, corrected_hdulist=None,
                            dm=None, ref_freq=None):
    """Plot waterfall images for all pulses detected in one hdulist.

    Parameters
    ----------
    raw_fits_path : str
        Path to the raw (un-corrected) PSRFITS file (disk).
    corrected_fits_path : str or None
        Path to the DM-corrected PSRFITS file (disk); None if not saved.
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
    raw_hdulist : astropy.io.fits.HDUList, optional
        In-memory raw (un-corrected) HDUList. Takes priority over raw_fits_path.
    corrected_hdulist : astropy.io.fits.HDUList, optional
        In-memory DM-corrected HDUList. Takes priority over corrected_fits_path.
    dm, ref_freq : float, optional
        Applied DM and zero-delay reference frequency in MHz. Missing values
        are read from corrected SUBINT DM/REFFREQ, then primary OBSFREQ for
        the reference. Raw-DM metadata is never used for this purpose.
    """
    if not pulse_data_list:
        return

    os.makedirs(output_dir, exist_ok=True)

    raw_fits_name = os.path.basename(raw_fits_path) if raw_fits_path else "raw (in-memory)"
    if corrected_fits_path:
        corrected_fits_name = os.path.basename(corrected_fits_path)
    else:
        corrected_fits_name = (f"PSR_{src}_{tel}_{file_counter:06d}"
                               f"_v{version}_dmcorr.fits")

    try:
        if raw_hdulist is not None:
            original_data, orig_header = _read_psrfits_data_from_hdulist(raw_hdulist)
        else:
            original_data, orig_header = _read_psrfits_data(raw_fits_path)
    except Exception as e:
        print(f"  WARNING: failed to read raw data: {e}")
        return
    try:
        if corrected_hdulist is not None:
            corrected_data, corr_header = _read_psrfits_data_from_hdulist(corrected_hdulist)
        else:
            corrected_data, corr_header = _read_psrfits_data(corrected_fits_path)
    except Exception as e:
        print(f"  WARNING: failed to read corrected data: {e}")
        return

    tbin_us = corr_header.get('tbin', orig_header['tbin']) * 1e6
    if not np.isfinite(tbin_us) or tbin_us <= 0:
        print('  WARNING: invalid waterfall sample interval')
        return
    freqs = corr_header.get('freqs', orig_header.get('freqs'))
    try:
        display_dm, display_ref = _display_parameters(corr_header, freqs, dm, ref_freq)
    except (TypeError, ValueError) as exc:
        print(f'  WARNING: invalid waterfall metadata: {exc}')
        return
    corrected_lightcurve = _create_lightcurve(corrected_data)

    for idx, pulse in enumerate(pulse_data_list):
        try:
            fitted_center = float(pulse.get('Precise_Center_Index', pulse.get('Coarse_Index', 0)))
            if not np.isfinite(fitted_center):
                raise ValueError('non-finite center')
            center_idx = int(fitted_center)
        except (TypeError, ValueError, OverflowError):
            print(f'  WARNING: pulse #{idx}: invalid center, skipping plot')
            continue
        fwhm_ms = pulse.get('FWHM_ms', 0)
        snr_amp = pulse.get('SNR_Amplitude_Fit', 0)
        snr_flux = pulse.get('SNR_Flux', 0)

        if not np.isfinite(fwhm_ms) or fwhm_ms <= 0:
            print(f"  pulse #{idx}: FWHM <= 0, skipping plot")
            continue

        if not (0 <= center_idx < min(len(original_data), len(corrected_data))):
            print(f'  WARNING: pulse #{idx}: center outside available data')
            continue
        tbin_ms = tbin_us / 1000.0
        fwhm_samples = fwhm_ms / tbin_ms if tbin_ms > 0 else fwhm_ms * 1000
        half_width_px = int(fwhm_samples)
        if half_width_px < 10:
            half_width_px = 10

        image_filename = (f"PSR_{src}_{tel}_{file_counter:06d}"
                          f"_v{version}_pulse{idx:02d}.png")
        image_path = os.path.join(output_dir, image_filename)

        figures_before = set(plt.get_fignums())
        try:
            _plot_single_pulse(
                original_data, corrected_data,
                center_idx, half_width_px, fwhm_ms,
                raw_fits_name, corrected_fits_name,
                image_path, tbin_us, snr_amp, snr_flux, idx, freqs,
                corrected_lightcurve, pulse, dm=display_dm,
                ref_freq=display_ref, chan_bw=corr_header.get('chan_bw'))
        except Exception as e:
            print(f"  WARNING: failed to plot pulse #{idx}: {e}")
        finally:
            for figure_number in set(plt.get_fignums()) - figures_before:
                plt.close(figure_number)
