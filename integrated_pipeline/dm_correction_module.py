"""
DM correction module (v5.1) — operates on an in-memory astropy.io.fits HDUList.

v5.1 fix vs v4: interp1d fill_value restored to np.nan. Using 0.0 for
out-of-bounds interpolation introduced a systematic downward bias in the
per-channel mean spectrum, causing frequency-edge channels to be incorrectly
flagged as RFI -> fewer channels in the lightcurve -> lower S/N -> missed pulses.
PRESTO uint8 compatibility is handled exclusively at PSRFITS write time.

v5 change vs v3: log simplification only — verbose prints removed.
Algorithm, float64 precision, and in-place HDUList mutation unchanged.

v2 change vs v1: explicit `del` of large intermediate arrays (data_4d,
corrected_data, header_info) before returning, so glibc / numpy's allocator
can reclaim them as soon as possible. Algorithm and float64 precision
unchanged.

Adapted from former_script/dm_cor/dm_cor_psrfits_latest_25-12-12.py.
"""

import numpy as np

def fractional_shift_freq_domain(x, shift_delta):
    """Shift a real-valued time series 'x' by 'shift_delta' samples (frequency-domain)."""
    N = len(x)
    X = np.fft.fft(x)
    k = np.fft.fftfreq(N) * N
    H = np.exp(-1j * 2 * np.pi * k * shift_delta / N)
    Y = X * H
    y_shifted = np.fft.ifft(Y)
    # Release intermediates before returning
    del X, k, H, Y
    return np.real(y_shifted)


def dm_delay(freq_mhz, dm, ref_freq=None):
    """Dispersive delay (seconds) due to DM. Frequencies in MHz."""
    if ref_freq is None:
        ref_freq = np.max(freq_mhz)
    k_dm = 4.1488064239e-3  # MHz^2 * s * pc^-1 * cm^3
    return k_dm * dm * (freq_mhz ** -2 - ref_freq ** -2)


def _extract_from_hdulist(hdulist, normalize=True):
    """
    In-memory equivalent of read_psrfits_full_observation.
    Returns (data_4d, freqs, times, header_info).
    """
    primary_hdr = hdulist[0].header
    subint_hdu = hdulist['SUBINT']
    subint_hdr = subint_hdu.header
    subint_data = subint_hdu.data

    nsubint_total = subint_hdr['NAXIS2']
    nchan = subint_hdr['NCHAN']
    npol = subint_hdr['NPOL']
    nsblk = subint_hdr['NSBLK']

    tdim_str = (subint_hdr.get('TDIM17') or subint_hdr.get('TDIM16')
                or subint_hdr.get('TDIM18'))
    if tdim_str is None:
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

        if len(dims) == 4 and dims[0] == 1:
            subint_3d = subint_reshaped[0, :, :, :].transpose(2, 0, 1)
        elif len(dims) == 3:
            subint_3d = subint_reshaped
        else:
            nsblk_idx = np.argwhere(np.array(dims) == nsblk)[0][0]
            nchan_idx = np.argwhere(np.array(dims) == nchan)[0][0]
            npol_idx = np.argwhere(np.array(dims) == npol)[0][0]
            perm = [nsblk_idx, nchan_idx, npol_idx]
            subint_3d = subint_reshaped.transpose(perm)

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

        del subint_raw, subint_reshaped, subint_3d  # release per-subint temporaries

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
    """Apply DM correction to (n_samples, nchan, npol) data. Verbatim float64 algorithm."""
    if ref_freq is None:
        ref_freq = np.max(freqs)

    delays = dm_delay(freqs / 1e3, dm, ref_freq / 1e3)
    dt = times[1] - times[0] if len(times) > 1 else 1.0

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
        from scipy.interpolate import interp1d
        for chan in range(nchan):
            for pol in range(npol):
                interp_func = interp1d(
                    times, data_4d[:, chan, pol],
                    kind='linear', bounds_error=False,
                    fill_value=np.nan, assume_sorted=True,
                )  # NaN for out-of-bounds: unknown ≠ zero; PRESTO uint8 compat handled at FITS write
                corrected_data[:, chan, pol] = interp_func(times - delays[chan])
                del interp_func

    del delays
    return corrected_data


def _apply_corrected_data_inplace(hdulist, corrected_data, header_info,
                                   dm, ref_freq, method):
    """Write corrected_data into hdulist['SUBINT'].data['DATA'] IN-PLACE.

    v3 (post-tracemalloc fix): replaces the old _rebuild_corrected_hdulist
    which did `new_subint_data = subint_hdu.data.copy()` + BinTableHDU(...).
    tracemalloc localized a ~213 MiB / hdulist leak to that .copy() call
    (astropy/io/fits/fitsrec.py:597, FITS_rec.copy()). astropy retains the
    copied FITS_rec across our `del corrected_hdulist`, so the only safe fix
    is to NOT copy in the first place: mutate the SUBINT.DATA column of the
    input hdulist directly.
    """
    nsubint_total = header_info['nsubint_total']
    nchan = header_info['nchan']
    npol = header_info['npol']
    nsblk = header_info['nsblk']
    dims = header_info['tdim_dims']

    subint_hdu = hdulist['SUBINT']
    subint_data = subint_hdu.data
    hdr = subint_hdu.header

    hdr['HISTORY'] = f'DM correction applied DM = {dm}'
    hdr['HISTORY'] = f'DM correction method: {method}'
    if ref_freq is not None:
        hdr['HISTORY'] = f'DM reference frequency: {ref_freq} MHz'
    hdr['HISTORY'] = 'DM correction applied in-place (integrated pipeline v3)'
    hdr['DM'] = dm
    if ref_freq is not None:
        hdr['REFFREQ'] = ref_freq

    for i in range(nsubint_total):
        start_idx = i * nsblk
        end_idx = (i + 1) * nsblk
        subint_corrected = corrected_data[start_idx:end_idx, :, :]

        subint_raw = subint_data['DATA'][i]
        target_shape = subint_raw.shape

        if len(dims) == 4 and dims[0] == 1:
            tmp = subint_corrected.transpose(1, 2, 0)
            reshaped_buf = np.empty(dims, dtype=subint_raw.dtype)
            reshaped_buf[0, :, :, :] = tmp.astype(subint_raw.dtype, copy=False)
        elif len(dims) == 3:
            tmp = subint_corrected.transpose(0, 2, 1)
            reshaped_buf = tmp.astype(subint_raw.dtype, copy=False)
        else:
            nsblk_idx = np.argwhere(np.array(dims) == nsblk)[0][0]
            nchan_idx = np.argwhere(np.array(dims) == nchan)[0][0]
            npol_idx = np.argwhere(np.array(dims) == npol)[0][0]
            perm = [nsblk_idx, nchan_idx, npol_idx]
            reverse_perm = [perm.index(j) for j in range(3)]
            tmp = subint_corrected.transpose(reverse_perm)
            reshaped_buf = tmp.astype(subint_raw.dtype, copy=False)

        subint_data['DATA'][i] = reshaped_buf.reshape(target_shape)
        del tmp, reshaped_buf, subint_corrected


def dm_correct_hdulist(hdulist, dm, ref_freq=None, method='freq_domain', normalize=True):
    """
    Apply DM correction IN-PLACE on the provided PSRFITS HDUList.

    v3 change: returns the SAME hdulist that was passed in (now with corrected
    DATA column) instead of building a new one. Eliminates the per-hdulist
    ~213 MiB FITS_rec.copy() leak diagnosed via tracemalloc.

    The caller MUST treat the input hdulist as "consumed" — if the caller
    still needs the pre-correction raw data, it must rebuild that hdulist
    from the original subint_data_list it controlled before this call.

    Returns
    -------
    astropy.io.fits.HDUList
        The same hdulist passed in, with DATA column replaced by DM-corrected
        values and HISTORY entries appended.
    """
    data_4d, freqs, times, header_info = _extract_from_hdulist(hdulist, normalize=normalize)

    corrected_data = apply_dm_correction_full(
        data_4d, freqs, times, dm, ref_freq=ref_freq, method=method
    )
    del data_4d

    _apply_corrected_data_inplace(hdulist, corrected_data, header_info, dm, ref_freq, method)

    del corrected_data, header_info, freqs, times
    return hdulist
