import spikeinterface.preprocessing as spre

# Converts unsigned integer recording to signed
def unsigned_to_signed(rec):
    return spre.unsigned_to_signed(rec)

# Applies a bandpass filter specified as a fraction of the Nyquist frequency, returns filtered recording object
def bandpass_to_frac_nyq(rec, fmin: float, frac_nyq: float): # Should rec be typed here?
    fs_hz = rec.get_sampling_frequency()
    nyq = fs_hz / 2.0
    fmax = frac_nyq * nyq
    # enforce fmax < nyq
    if fmax >= nyq:
        fmax = 0.99 * nyq # Double check fmax computation
    if not (0 < fmin < fmax < nyq):
        raise ValueError(f"Bad bandpass: fmin={fmin}, fmax={fmax}, nyq={nyq}")
    return spre.bandpass_filter(rec, freq_min=fmin, freq_max=fmax)

# Slices the recording by time in seconds, if requested, and returns the sliced recording object
def slice_seconds(rec, start_s: float, dur_s: float | None):
    if dur_s is None:
        return rec
    fs_hz = rec.get_sampling_frequency()
    start_f = int(round(start_s * fs_hz))
    end_f = int(round((start_s + dur_s) * fs_hz))
    return rec.frame_slice(start_f, end_f)
