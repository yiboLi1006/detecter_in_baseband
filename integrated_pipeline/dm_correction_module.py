"""
DM correction module that operates on an in-memory astropy.io.fits HDUList.

Adapted from former_script/dm_cor/dm_cor_psrfits_latest_25-12-12.py.
The DM algorithm itself (fractional_shift_freq_domain / dm_delay / apply_dm_correction_full)
is copied verbatim; only the file IO layer is replaced so the function can be
called inline inside the VDIF -> PSRFITS pipeline without touching disk.
"""

import os
import numpy as np
from astropy.io import fits
from scipy.interpolate import interp1d


def fractional_shift_freq_domain(x, shift_delta):
    """
    Shifts a real-valued time series 'x' by 'shift_delta' samples
    using a frequency-domain method.
    """
    N = len(x)
    X = np.fft.fft(x)
    k = np.fft.fftfreq(N) * N
    H = np.exp(-1j * 2 * np.pi * k * shift_delta / N)
    Y = X * H
    y_shifted = np.fft.ifft(Y)
    return np.real(y_shifted)


def dm_delay(freq_mhz, dm, ref_freq=None):
    """
    Calculate dispersive delay in seconds due to DM.
    Frequencies are in MHz, delay in seconds relative to ref_freq.
    """
    if ref_freq is None:
        ref_freq = np.max(freq_mhz)

    k_dm = 4.1488064239e-3  # MHz^2 * s * pc^-1 * cm^3
    delay = k_dm * dm * (freq_mhz ** -2 - ref_freq ** -2)
    return delay


def _extract_from_hdulist(hdulist, normalize=True):
    """
    In-memory equivalent of read_psrfits_full_observation: parse an existing
    HDUList and return (data_4d, freqs, times, header_info).
    """
    primary_hdr = hdulist[0].header
    subint_hdu = hdulist['SUBINT']
    subint_hdr = subint_hdu.header
    subint_data = subint_hdu.data

    nsubint_total = subint_hdr['NAXIS2']
    nchan = subint_hdr['NCHAN']
    npol = subint_hdr['NPOL']
    nsblk = subint_hdr['NSBLK']

    # Parse TDIM for data layout. Original DM script uses TDIM17; v16-produced
    # tables may use TDIM16. Try both, fall back to scanning the header.
    tdim_str = (subint_hdr.get('TDIM17') or subint_hdr.get('TDIM16')
                or subint_hdr.get('TDIM18'))
    if tdim_str is None:
        # Final fallback: find any TDIMnn that contains nsblk
        for key in subint_hdr.keys():
            if key.startswith('TDIM'):
                tdim_str = subint_hdr[key]
                break
    if tdim_str is None:
        raise ValueError("Could not find TDIM keyword for DATA column in SUBINT header")

    tdim = tdim_str.strip('()')
    dims = [int(x) for x in tdim.split(',')]

    total_samples = nsubint_total * nsblk
    data_4d = np.zeros((total_samples, nchan, npol), dtype=np.float64)
    times = np.zeros(total_samples, dtype=np.float64)

    freqs = subint_data['DAT_FREQ'][0]  # MHz
    tbin = subint_hdr['TBIN']

    for i in range(nsubint_total):
        subint_raw = subint_data['DATA'][i]
        subint_reshaped = subint_raw.reshape(dims)

        if len(dims) == 4 and dims[0] == 1:  # (1, nchan, npol, nsblk)
            subint_3d = subint_reshaped[0, :, :, :].transpose(2, 0, 1)
        elif len(dims) == 3:  # (nsblk, npol, nchan) or similar
            subint_3d = subint_reshaped
        else:
            nsblk_idx = np.argwhere(np.array(dims) == nsblk)[0][0]
            nchan_idx = np.argwhere(np.array(dims) == nchan)[0][0]
            npol_idx = np.argwhere(np.array(dims) == npol)[0][0]
            perm = [nsblk_idx, nchan_idx, npol_idx]
            subint_3d = subint_reshaped.transpose(perm)

        # subint_3d may still be integer; convert to float for scaling math
        subint_3d = subint_3d.astype(np.float64, copy=False)

        if normalize:
            dat_scl = subint_data['DAT_SCL'][i]
            dat_offs = subint_data['DAT_OFFS'][i]
            dat_scl_reshaped = dat_scl.reshape(nchan, npol)
            dat_offs_reshaped = dat_offs.reshape(nchan, npol)

            for chan in range(nchan):
                for pol in range(npol):
                    subint_3d[:, chan, pol] = (
                        subint_3d[:, chan, pol] * dat_scl_reshaped[chan, pol]
                        + dat_offs_reshaped[chan, pol]
                    )

        start_idx = i * nsblk
        end_idx = (i + 1) * nsblk
        data_4d[start_idx:end_idx, :, :] = subint_3d

        tsubint = subint_data['TSUBINT'][i]
        offs_sub = subint_data['OFFS_SUB'][i]
        start_time = offs_sub - tsubint / 2
        times[start_idx:end_idx] = start_time + np.arange(nsblk) * tbin

    header_info = {
        'src_name': primary_hdr.get('SRC_NAME', 'Unknown'),
        'freq_center': primary_hdr.get('OBSFREQ', 0),
        'bw': primary_hdr.get('OBSBW', 0),
        'telescope': primary_hdr.get('TELESCOP', 'Unknown'),
        'npol': npol,
        'pol_type': subint_hdr.get('POL_TYPE', 'Unknown'),
        'nsubint_total': nsubint_total,
        'nchan': nchan,
        'nsblk': nsblk,
        'tbin': tbin,
        'tdim_dims': dims,
        'primary_hdr': primary_hdr,
        'subint_hdr': subint_hdr,
    }
    return data_4d, freqs, times, header_info


def apply_dm_correction_full(data_4d, freqs, times, dm, ref_freq=None, method='freq_domain'):
    """
    Apply DM correction to full data array for all polarizations.
    Verbatim from dm_cor_psrfits_latest_25-12-12.py.
    """
    if ref_freq is None:
        ref_freq = np.max(freqs)

    print(f"Applying DM correction: DM = {dm} pc/cm^3, ref_freq = {ref_freq} MHz")
    print(f"Using method: {method}")

    # NOTE: original DM script calls dm_delay(freqs/1E3, dm, ref_freq/1E3),
    # i.e. it converts MHz to GHz before passing into dm_delay. Keep that
    # convention so numerical behavior matches the proven script.
    delays = dm_delay(freqs / 1e3, dm, ref_freq / 1e3)
    print(f"Freqs range: {np.min(freqs):.6f} to {np.max(freqs):.6f} MHz")
    print(f"Delay range: {np.min(delays):.6f} to {np.max(delays):.6f} seconds")

    dt = times[1] - times[0] if len(times) > 1 else 1.0
    print(f"Time resolution: {dt:.6e} seconds")

    corrected_data = np.zeros_like(data_4d)
    nchan = len(freqs)
    npol = data_4d.shape[2]

    if method == 'freq_domain':
        for chan in range(nchan):
            delay_samples = -1.0 * delays[chan] / dt
            for pol in range(npol):
                corrected_data[:, chan, pol] = fractional_shift_freq_domain(
                    data_4d[:, chan, pol], delay_samples
                )
    else:
        for chan in range(nchan):
            shifted_times = times - delays[chan]
            for pol in range(npol):
                interp_func = interp1d(
                    times, data_4d[:, chan, pol],
                    kind='linear', bounds_error=False,
                    fill_value=np.nan, assume_sorted=True,
                )
                corrected_data[:, chan, pol] = interp_func(times)

        nan_mask = np.isnan(corrected_data)
        if np.any(nan_mask):
            print(f"Warning: {np.sum(nan_mask)} NaN values after DM correction (edge effects)")
            for pol in range(npol):
                for chan in range(nchan):
                    chan_data = corrected_data[:, chan, pol]
                    nan_indices = np.where(np.isnan(chan_data))[0]
                    if len(nan_indices) > 0:
                        valid_indices = np.where(~np.isnan(chan_data))[0]
                        if len(valid_indices) > 0:
                            for idx in nan_indices:
                                nearest_valid = valid_indices[np.argmin(np.abs(valid_indices - idx))]
                                corrected_data[idx, chan, pol] = corrected_data[nearest_valid, chan, pol]

    return corrected_data


def _rebuild_corrected_hdulist(original_hdulist, corrected_data, header_info,
                               dm, ref_freq, method):
    """
    Build a new HDUList that mirrors original_hdulist but with DM-corrected DATA.
    No disk IO is performed.
    """
    nsubint_total = header_info['nsubint_total']
    nchan = header_info['nchan']
    npol = header_info['npol']
    nsblk = header_info['nsblk']
    dims = header_info['tdim_dims']

    # Copy primary HDU
    new_primary = original_hdulist[0].copy()

    # Copy SUBINT HDU's data and header
    subint_hdu = original_hdulist['SUBINT']
    new_subint_data = subint_hdu.data.copy()
    new_subint_hdr = subint_hdu.header.copy()

    # Record DM correction in HISTORY
    new_subint_hdr['HISTORY'] = f'DM correction applied DM = {dm}'
    new_subint_hdr['HISTORY'] = f'DM correction method: {method}'
    if ref_freq is not None:
        new_subint_hdr['HISTORY'] = f'DM reference frequency: {ref_freq} MHz'
    new_subint_hdr['HISTORY'] = 'DM correction applied in-memory (integrated pipeline)'

    # Write corrected data back into the SUBINT table's DATA column
    for i in range(nsubint_total):
        start_idx = i * nsblk
        end_idx = (i + 1) * nsblk
        subint_corrected = corrected_data[start_idx:end_idx, :, :]

        subint_raw = subint_hdu.data['DATA'][i]
        target_shape = subint_raw.shape  # flattened shape stored in column

        if len(dims) == 4 and dims[0] == 1:  # (1, nchan, npol, nsblk)
            # We hold (nsblk, nchan, npol) -> reshape back to (1, nchan, npol, nsblk)
            tmp = subint_corrected.transpose(1, 2, 0)
            reshaped_buf = np.empty(dims, dtype=subint_raw.dtype)
            reshaped_buf[0, :, :, :] = tmp.astype(subint_raw.dtype, copy=False)
        elif len(dims) == 3:
            tmp = subint_corrected.transpose(0, 2, 1)  # (nsblk, npol, nchan)
            reshaped_buf = tmp.astype(subint_raw.dtype, copy=False)
        else:
            # Reverse the permutation chosen at read time
            nsblk_idx = np.argwhere(np.array(dims) == nsblk)[0][0]
            nchan_idx = np.argwhere(np.array(dims) == nchan)[0][0]
            npol_idx = np.argwhere(np.array(dims) == npol)[0][0]
            perm = [nsblk_idx, nchan_idx, npol_idx]
            reverse_perm = [perm.index(j) for j in range(3)]
            tmp = subint_corrected.transpose(reverse_perm)
            reshaped_buf = tmp.astype(subint_raw.dtype, copy=False)

        new_subint_data['DATA'][i] = reshaped_buf.reshape(target_shape)

    new_subint_hdu = fits.BinTableHDU(data=new_subint_data, header=new_subint_hdr, name='SUBINT')

    new_hdul = fits.HDUList()
    new_hdul.append(new_primary)
    new_hdul.append(new_subint_hdu)

    # Preserve any extra HDUs
    for i in range(1, len(original_hdulist)):
        if original_hdulist[i].name.upper() == 'SUBINT':
            continue
        new_hdul.append(original_hdulist[i].copy())

    return new_hdul


def dm_correct_hdulist(hdulist, dm, ref_freq=None, method='freq_domain', normalize=True):
    """
    Apply DM correction directly on an in-memory PSRFITS HDUList.

    Parameters
    ----------
    hdulist : astropy.io.fits.HDUList
        PSRFITS hdulist with a PRIMARY HDU and a SUBINT BinTable.
    dm : float
        Dispersion measure in pc/cm^3.
    ref_freq : float or None
        Reference frequency in MHz. None -> use highest channel frequency.
    method : {'freq_domain', 'interp'}
        DM correction method.
    normalize : bool
        Whether to apply DAT_SCL / DAT_OFFS when reading. v16-produced files
        store unscaled float data with DAT_SCL=1, DAT_OFFS=0, so this is safe.

    Returns
    -------
    astropy.io.fits.HDUList
        New hdulist with DM-corrected DATA column. The input hdulist is not modified.
    """
    print("=" * 60)
    print("DM correction (in-memory)")
    print("=" * 60)

    data_4d, freqs, times, header_info = _extract_from_hdulist(hdulist, normalize=normalize)

    corrected_data = apply_dm_correction_full(
        data_4d, freqs, times, dm, ref_freq=ref_freq, method=method
    )

    new_hdul = _rebuild_corrected_hdulist(
        hdulist, corrected_data, header_info, dm, ref_freq, method
    )
    print("DM correction done.")
    print("=" * 60)
    return new_hdul
