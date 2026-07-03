import sounddevice as sd
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.fft import rfft, irfft, next_fast_len
from collections import deque

sample_rate = 44100
buffer_size = 2048
fft_size = 8192


CALIBRATION_MODE = False
MIN_FREQ = 150.0   # placeholder — replace after calibrating
MAX_FREQ = 1500.0  # placeholder — replace after calibrating


ENERGY_THRESHOLD = 0.0005


SA_NOTE = 'B'   # Change this if your bansuri is a different scale

# Chromatic note -> index (C=0 ... B=11)
_NOTE_TO_SEMITONE = {
    'C': 0, 'C#': 1, 'D': 2, 'D#': 3, 'E': 4, 'F': 5,
    'F#': 6, 'G': 7, 'G#': 8, 'A': 9, 'A#': 10, 'B': 11
}
SA_SEMITONE = _NOTE_TO_SEMITONE[SA_NOTE]

_INTERVAL_TO_SARGAM = {
    0:  'Sa',
    1:  'Re♭',      # Komal Re
    2:  'Re',       # Shuddha Re
    3:  'Ga♭',      # Komal Ga
    4:  'Ga',       # Shuddha Ga
    5:  'Ma',       # Shuddha Ma
    6:  'Ma\'',     # Tivra Ma
    7:  'Pa',
    8:  'Dha♭',     # Komal Dha
    9:  'Dha',      # Shuddha Dha
    10: 'Ni♭',      # Komal Ni
    11: 'Ni',       # Shuddha Ni
}


CONFIDENCE_THRESHOLD = 0.5


SMOOTHING_WINDOW = 3
recent_notes = deque(maxlen=SMOOTHING_WINDOW)


def freq_to_note(freq):
    """Return Western note name (e.g. 'B4') — kept for reference/debug."""
    if freq < 20.0:
        return None
    A4 = 440.0
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    n = int(round(12 * np.log2(freq / A4) + 69))
    if n < 0 or n > 127:
        return None
    note_name = notes[n % 12]
    octave = n // 12 - 1
    return f"{note_name}{octave}"


def freq_to_sargam(freq):
    """Map a frequency to its sargam swara relative to SA_NOTE.
    Returns e.g. 'Sa', 'Re', 'Ga', 'Pa', etc.  Adds an octave marker
    (·  for mandra, '  for taar) when the note is outside the middle
    octave that starts at SA_NOTE."""
    if freq < 20.0:
        return None
    A4 = 440.0
    midi = int(round(12 * np.log2(freq / A4) + 69))
    if midi < 0 or midi > 127:
        return None
    # Semitone interval from Sa (mod 12) gives the swara
    interval = (midi % 12 - SA_SEMITONE) % 12
    sargam = _INTERVAL_TO_SARGAM.get(interval, '?')
    # Octave indicator relative to Sa's home octave
    sa_octave = 4  # default middle octave for Sa
    note_octave = midi // 12 - 1
    # Which "sargam octave" are we in?
    # If the note's chromatic pitch < Sa in the same octave, it belongs
    # to the next sargam octave up.
    effective_oct = note_octave
    if (midi % 12) < SA_SEMITONE:
        effective_oct -= 1
    relative_oct = effective_oct - sa_octave
    if relative_oct < 0:
        sargam = sargam + '॰'   # mandra saptak (lower)
    elif relative_oct > 0:
        sargam = sargam + "'"   # taar saptak (upper)
    return sargam


def frame_energy(x):
    """E = (1/N) * sum(x[n]^2) — used to gate silence, independent of
    whatever the FFT/autocorrelation are doing."""
    return float(np.mean(x.astype(np.float64) ** 2))


def parabolic_interpolation(array, x):
    if x < 1 or x > len(array) - 2:
        return x
    alpha = array[x - 1]
    beta = array[x]
    gamma = array[x + 1]
    denominator = alpha - 2 * beta + gamma
    if denominator == 0:
        return x
    p = 0.5 * (alpha - gamma) / denominator
    return x + p


def autocorrelate_normalized(x):
    """
    R[k] normalized by the number of overlapping samples at each lag.

    Uses FFT-based autocorrelation: R = IFFT(|FFT(x)|^2), which is
    O(N log N) instead of np.correlate's O(N^2). For buffer_size=2048
    this is ~100x faster.
    """
    n = len(x)
    x = x.astype(np.float64)
    # Pad to at least 2N-1 for linear (non-circular) correlation,
    # then round up to a FFT-friendly size
    fft_len = next_fast_len(2 * n - 1)
    X = rfft(x, fft_len)
    corr = irfft(X * np.conj(X), fft_len)[:n]
    overlap_counts = np.arange(n, 0, -1, dtype=np.float64)
    return corr / overlap_counts


def find_fundamental_lag(corr, min_lag, max_lag, threshold_ratio=0.8):
    
    max_lag = min(max_lag, len(corr) - 1)
    if min_lag >= max_lag:
        return None, 0.0

    search = corr[min_lag:max_lag]
    global_max = np.max(search)

    if global_max <= 0:
        return None, 0.0  # no real periodicity anywhere in range

    threshold = threshold_ratio * global_max

    for i in range(1, len(search) - 1):
        if search[i] >= threshold and search[i] >= search[i - 1] and search[i] >= search[i + 1]:
            return i + min_lag, search[i]

    idx = int(np.argmax(search))
    return idx + min_lag, search[idx]


def pitch_from_autocorrelation(samples, sample_rate, min_freq=MIN_FREQ, max_freq=MAX_FREQ):

    corr = autocorrelate_normalized(samples)

    min_lag = int(sample_rate / max_freq)
    max_lag = int(sample_rate / min_freq)

    peak_lag, peak_val = find_fundamental_lag(corr, min_lag, max_lag)
    if peak_lag is None or peak_lag == 0:
        return None, 0.0

    confidence = peak_val / corr[0] if corr[0] != 0 else 0.0

    refined_lag = parabolic_interpolation(corr, peak_lag)
    if refined_lag <= 0:
        return None, 0.0

    return sample_rate / refined_lag, confidence


def smoothed_note(note, history):

    history.append(note)
    if len(history) < history.maxlen:
        return None
    if all(n == history[0] for n in history):
        return history[0]
    return None


plt.ion()

fig, (ax1, ax2) = plt.subplots(2, 1)
line1, = ax1.plot(np.zeros(buffer_size))
ax1.set_ylim(-1, 1)
ax1.set_xlim(0, buffer_size)
ax1.set_title("Time Domain")

freqs = np.fft.rfftfreq(fft_size, 1 / sample_rate)
line2, = ax2.plot(freqs, np.zeros(len(freqs)))
ax2.set_xlim(0, sample_rate / 2)
ax2.set_ylim(0, 0.1)
ax2.set_title("Frequency Domain")
plt.tight_layout()


fig.canvas.draw()
bg1 = fig.canvas.copy_from_bbox(ax1.bbox)
bg2 = fig.canvas.copy_from_bbox(ax2.bbox)

stream = sd.InputStream(
    samplerate=sample_rate,
    channels=1,
    blocksize=buffer_size
)

stream.start()

hann_window = np.hanning(buffer_size)


lowcut = MIN_FREQ
highcut = MAX_FREQ
nyq = 0.5 * sample_rate
low = lowcut / nyq
high = highcut / nyq
sos = signal.butter(4, [low, high], btype='band', output='sos')


zi = signal.sosfilt_zi(sos)
zi = np.tile(zi[:, :, None], (1, 1, 1)).reshape(sos.shape[0], 2) * 0.0

while plt.fignum_exists(fig.number):

    data, overflowed = stream.read(buffer_size)
    if overflowed:

        print("[warning] input overflow — buffer underrun, samples dropped")

    samples = data[:, 0]

    samples = samples - np.mean(samples)

    samples, zi = signal.sosfilt(sos, samples, zi=zi)

    line1.set_ydata(samples)

    windowed_samples = samples * hann_window

    fft_result = np.fft.rfft(windowed_samples, n=fft_size)
    magnitude = np.abs(fft_result) / (buffer_size / 2)

    energy = frame_energy(samples)

    if CALIBRATION_MODE:
        if energy > ENERGY_THRESHOLD:
            peak_freq, confidence = pitch_from_autocorrelation(
                windowed_samples, sample_rate, MIN_FREQ, MAX_FREQ
            )
            if peak_freq is not None:
                print(f"[calibrate] {peak_freq:7.1f} Hz  confidence={confidence:.2f}")
        ax2.set_title("CALIBRATION MODE — play lowest/highest notes, read console")
    elif energy > ENERGY_THRESHOLD:
        peak_freq, confidence = pitch_from_autocorrelation(
            windowed_samples, sample_rate, MIN_FREQ, MAX_FREQ
        )
        if peak_freq is not None and confidence >= CONFIDENCE_THRESHOLD:
            sargam = freq_to_sargam(peak_freq)
            western = freq_to_note(peak_freq)
            stable_note = smoothed_note(sargam, recent_notes)
            if stable_note:
                ax2.set_title(f"Frequency Domain | {stable_note}  [{western}] ({peak_freq:.1f} Hz)")
                print(f"Detected: {stable_note}  [{western}] at {peak_freq:.1f} Hz (confidence={confidence:.2f})")
            else:
                ax2.set_title("Frequency Domain | (stabilizing...)")
        else:
            recent_notes.clear()  # reset smoothing on a low-confidence/no-pitch frame
            ax2.set_title("Frequency Domain | (low confidence / breath noise)")
    else:
        recent_notes.clear()
        ax2.set_title("Frequency Domain")

    # Update frequency domain
    line2.set_ydata(magnitude)
    ax2.set_ylim(0, max(0.01, np.max(magnitude) * 1.2))

    # Blit: restore backgrounds, redraw only the artists that changed
    fig.canvas.restore_region(bg1)
    fig.canvas.restore_region(bg2)
    ax1.draw_artist(line1)
    ax2.draw_artist(line2)
    fig.canvas.blit(ax1.bbox)
    fig.canvas.blit(ax2.bbox)
    fig.canvas.flush_events()

stream.stop()
stream.close()