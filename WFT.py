import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import get_window
from scipy.fftpack import fft, fftshift
import pywt

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import get_window
from scipy.fftpack import fft, fftshift
import pywt  # For the CWT

# === Parameters ===
fs = 1000           # Sampling frequency (Hz)
duration = 1.0      # Signal duration (seconds)
window_type = 'hann'
window_size = 50   # WFT window size

# Hej

# === Generate sine wave ===
t = np.linspace(0, duration, int(fs * duration), endpoint=False)
signal = np.sin(50.0 * 2.0*np.pi*t) + 0.5*np.sin(80.0 * 2.0*np.pi*t)

# === WFT computation ===
step = window_size // 4
window = get_window(window_type, window_size)
n_windows = (len(signal) - window_size) // step + 1
spec = np.zeros((n_windows, window_size), dtype=complex)

for i in range(n_windows):
    start = i * step
    segment = signal[start:start + window_size] * window
    spec[i, :] = fftshift(fft(segment))

frequencies_wft = np.linspace(-fs / 2, fs / 2, window_size)
times_wft = np.arange(n_windows) * step / fs

# === CWT computation ===
wavelet = "cmor1.5-1.0"
widths = np.geomspace(1, 128, num=500)
sampling_period = 1.0 / fs
cwtmatr, freqs = pywt.cwt(signal, widths, wavelet, sampling_period=sampling_period)
cwtmatr = np.abs(cwtmatr)

# === Plotting ===
fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

# Plot original signal
axs[0].plot(t, signal)
axs[0].set_title("Signal")
axs[0].set_ylabel("Amplitude")
axs[0].grid()

# Plot WFT spectrogram
im1 = axs[1].imshow(np.abs(spec.T), aspect='auto',
                    extent=[times_wft[0], times_wft[-1], frequencies_wft[0], frequencies_wft[-1]],
                    origin='lower', cmap='viridis')
axs[1].set_title("Windowed Fourier Transform (Spectrogram)")
axs[1].set_ylabel("Frequency [Hz]")
fig.colorbar(im1, ax=axs[1], label='Magnitude')

# Plot CWT scalogram
T, F = np.meshgrid(t[:cwtmatr.shape[1]], freqs)
pcm = axs[2].pcolormesh(T, F, cwtmatr, shading='auto')
axs[2].set_yscale("log")
axs[2].set_xlabel("Time (s)")
axs[2].set_ylabel("Frequency (Hz)")
axs[2].set_title("Continuous Wavelet Transform (Scaleogram)")
fig.colorbar(pcm, ax=axs[0])

plt.tight_layout()
plt.show()
