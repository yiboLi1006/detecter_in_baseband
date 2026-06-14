"""
Frame-aligned byte copy of VDIF / Mark5B segments.

When a PSRFITS hdulist is found to contain pulses, we save the original VDIF
(or Mark5B) bytes covering that hdulist's sample range to a new file. Frames
are the smallest readable unit for both formats, so the saved range is
always aligned to whole frames -- this means the segment may extend slightly
past the requested sample range, but is guaranteed to be a valid file that
baseband can re-open.
"""

import math
import os
import baseband
import astropy.units as u
from astropy.time import Time


def _open_reader(input_file, data_format, sample_rate_hz=None, ref_time=None, nchan=1):
    """Open a baseband reader just long enough to inspect framing."""
    if data_format == 'mark5b':
        return baseband.open(
            input_file, mode='rs', format='mark5b',
            nchan=nchan,
            sample_rate=(sample_rate_hz * u.Hz) if sample_rate_hz else None,
            ref_time=ref_time, verify=False,
        )
    return baseband.open(
        input_file, mode='rs', format='vdif',
        sample_rate=(sample_rate_hz * u.Hz) if sample_rate_hz else None,
        verify=False,
    )


def _get_frame_info(reader):
    """Return (framesize_bytes, samples_per_frame) from the first header."""
    header0 = reader.header0
    framesize = int(getattr(header0, 'frame_nbytes',
                            getattr(header0, 'framesize', None)))
    samples_per_frame = int(getattr(header0, 'samples_per_frame', None))
    if framesize is None or samples_per_frame is None or samples_per_frame <= 0:
        raise ValueError("Could not determine framesize/samples_per_frame from header0.")
    return framesize, samples_per_frame


def save_baseband_segment(input_file, output_file, sample_start, sample_count,
                          data_format='vdif', sample_rate_hz=None, ref_time=None,
                          nchan=1):
    """
    Copy the [sample_start, sample_start + sample_count) sample range from
    `input_file` to `output_file`, rounded out to whole frames.

    For multi-thread VDIF (where frames at the same time index repeat across
    threads), we treat frames-per-time-tick * threads as a single block by
    using the on-disk file size as upper bound only; otherwise sample-to-byte
    math assumes a single-thread layout.
    """
    parsed_ref_time = None
    if data_format == 'mark5b':
        if not ref_time:
            raise ValueError("Mark5B segment saving requires ref_time")
        parsed_ref_time = Time(ref_time)

    reader = _open_reader(
        input_file, data_format,
        sample_rate_hz=sample_rate_hz,
        ref_time=parsed_ref_time,
        nchan=nchan,
    )
    try:
        framesize, samples_per_frame = _get_frame_info(reader)
    finally:
        try:
            reader.close()
        except Exception:
            pass

    sample_end = int(sample_start) + int(sample_count)
    start_frame = int(sample_start) // samples_per_frame
    end_frame = int(math.ceil(sample_end / samples_per_frame))

    byte_start = start_frame * framesize
    byte_end = end_frame * framesize

    file_size = os.path.getsize(input_file)
    if byte_end > file_size:
        byte_end = file_size
    if byte_start >= file_size:
        raise ValueError(
            f"Requested sample_start={sample_start} maps to byte {byte_start} "
            f"beyond file size {file_size}."
        )

    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    chunk_bytes = 1 << 20  # 1 MiB
    remaining = byte_end - byte_start
    with open(input_file, 'rb') as src, open(output_file, 'wb') as dst:
        src.seek(byte_start)
        while remaining > 0:
            read = src.read(min(chunk_bytes, remaining))
            if not read:
                break
            dst.write(read)
            remaining -= len(read)

    print(f"Saved {data_format.upper()} segment: frames [{start_frame}, {end_frame}), "
          f"bytes [{byte_start}, {byte_end}) -> {output_file}")
    return output_file
