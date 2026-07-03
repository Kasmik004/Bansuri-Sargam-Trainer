import sounddevice as sd
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

sample_rate = 44100
buffer_size = 2048
fft_size = 8192

# Silence gate — tune this against your own mic/room by printing
# frame_energy() values during silence vs playing (same as dsp.py's
# ENERGY_THRESHOLD). This is a *separate* concern from pitch detection.
ENERGY_THRESHOLD = 0.0005


def freq_to_note(freq):
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

    Fix: raw np.correlate(x, x, 'full') sums over N-k terms at lag k,
    so R[k] shrinks with lag even for a perfectly periodic signal —
    biasing peak-picking toward shorter lags (higher frequency,
    i.e. harmonics/octave errors) regardless of actual periodicity.
    Dividing by the overlap count removes that systematic bias so
    peak height reflects periodicity strength, not lag length.
    """
    n = len(x)
    x = x.astype(np.float64)
    corr = np.correlate(x, x, mode='full')[n - 1:]
    overlap_counts = np.arange(n, 0, -1)  # n, n-1, ..., 1
    return corr / overlap_counts


def find_fundamental_lag(corr, min_lag, max_lag):
    """
    First local peak after the initial downslope within [min_lag, max_lag],
    not a flat argmax. A flat argmax over the normalized correlation is
    still vulnerable to a strong harmonic outscoring the true fundamental
    within the search band — walking forward and taking the FIRST peak
    (rather than the tallest) favors the fundamental, since the
    fundamental period is always the shortest genuine periodicity.
    """
    max_lag = min(max_lag, len(corr) - 1)
    if min_lag >= max_lag:
        return None

    search = corr[min_lag:max_lag]

    i = 1
    while i < len(search) - 1 and search[i] < search[i - 1]:
        i += 1

    while i < len(search) - 1:
        if search[i] >= search[i - 1] and search[i] >= search[i + 1]:
            return i + min_lag
        i += 1

    return None


def pitch_from_autocorrelation(samples, sample_rate, min_freq=150.0, max_freq=1500.0):
    corr = autocorrelate_normalized(samples)

    min_lag = int(sample_rate / max_freq)
    max_lag = int(sample_rate / min_freq)

    peak_lag = find_fundamental_lag(corr, min_lag, max_lag)
    if peak_lag is None or peak_lag == 0:
        return None

    refined_lag = parabolic_interpolation(corr, peak_lag)
    if refined_lag <= 0:
        return None

    return sample_rate / refined_lag


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

stream = sd.InputStream(
    samplerate=sample_rate,
    channels=1,
    blocksize=buffer_size
)

stream.start()

hann_window = np.hanning(buffer_size)

# Setup Butterworth bandpass filter
lowcut = 150.0
highcut = 1500.0
nyq = 0.5 * sample_rate
low = lowcut / nyq
high = highcut / nyq
sos = signal.butter(4, [low, high], btype='band', output='sos')

# Steady-state initial conditions instead of zeros: avoids a settling
# transient in the filter's output on the very first buffer. Scaled
# by 0 here since we don't know the first sample's DC level yet;
# sosfilt_zi gives the correct *shape*, scaling happens per-signal.
zi = signal.sosfilt_zi(sos)
zi = np.tile(zi[:, :, None], (1, 1, 1)).reshape(sos.shape[0], 2) * 0.0

while plt.fignum_exists(fig.number):

    data, overflowed = stream.read(buffer_size)
    if overflowed:
        # Lost samples this block — treat any pitch reading from this
        # frame with suspicion; don't silently pretend it's clean data.
        print("[warning] input overflow — buffer underrun, samples dropped")

    samples = data[:, 0]

    # Remove DC offset
    samples = samples - np.mean(samples)

    # Apply bandpass filter (150-1500 Hz)
    samples, zi = signal.sosfilt(sos, samples, zi=zi)

    # Update time domain
    line1.set_ydata(samples)

    # Apply Hann window
    windowed_samples = samples * hann_window

    # Compute FFT with zero-padding (for visualization only — pitch
    # detection below uses autocorrelation, not this FFT)
    fft_result = np.fft.rfft(windowed_samples, n=fft_size)
    magnitude = np.abs(fft_result) / (buffer_size / 2)

    # Silence gate: based on time-domain frame energy, NOT FFT peak
    # magnitude. The old code gated on the tallest FFT bin, which has
    # no necessary relationship to the pitch autocorrelation reports —
    # a strong unrelated bin could pass the gate, or a clean quiet
    # note could fail it.
    energy = frame_energy(samples)

    if energy > ENERGY_THRESHOLD:
        peak_freq = pitch_from_autocorrelation(windowed_samples, sample_rate, lowcut, highcut)
        if peak_freq is not None:
            note = freq_to_note(peak_freq)
            if note:
                ax2.set_title(f"Frequency Domain | Flute/Note: {note} ({peak_freq:.1f} Hz)")
                print(f"Detected: {note} at {peak_freq:.1f} Hz")
        else:
            ax2.set_title("Frequency Domain | (no confident pitch)")
    else:
        ax2.set_title("Frequency Domain")

    # Update frequency domain
    line2.set_ydata(magnitude)
    ax2.set_ylim(0, max(0.01, np.max(magnitude) * 1.2))

    fig.canvas.draw()
    fig.canvas.flush_events()

stream.stop()
stream.close()