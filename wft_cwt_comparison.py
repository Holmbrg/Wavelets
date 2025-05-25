import matplotlib.pyplot as plt
import numpy as np
import pywt
import scipy.io.wavfile as wavfile
from scipy import signal as sgn
import os

# Parameters
fs = 1000           # Sampling frequency (Hz)
duration = 1.0      # Signal duration (seconds)
window_type = 'hann' # WFT window shape
window_size = 200   # WFT window size
function_freq = 50 # Base frequency of input signal
function_type = 'chirp' # Choose 'sine_wave', 'chirp' or 'noisy_sine'
wavelet = "cmor1.5-1.0"
scales = np.geomspace(1, 512, num=200)

# Generate input signal
def get_signal():

    if function_type == 'sine_wave':
        return np.sin(function_freq * 2.0*np.pi*t)
    if function_type == 'chirp':
        return np.sin(function_freq * 2.0*np.pi*(t**2))
    if function_type == 'noisy_sine':
        return np.sin(function_freq * 2.0*np.pi*t) + np.random.normal(scale=0.5, size=t.shape)

t = np.linspace(0, duration, int(fs * duration), endpoint=False)
signal = get_signal()

# Compute WFT
wft_freqs, wft_times, wft_coeffs = sgn.stft(signal, fs, window_type, window_size)

# Compute CWT
cwt_coeffs, cwt_freqs = pywt.cwt(signal, scales, wavelet, 1.0 / fs)


# --- PLOTTING --- #

fig, axs = plt.subplots(2, 1, figsize=(9, 9), sharex=False)

# Plot WFT
im1 = axs[0].imshow(np.abs(wft_coeffs), aspect='auto',
                    extent=[wft_times[0], wft_times[-1], wft_freqs[0], wft_freqs[-1]],
                    origin='lower', cmap='viridis')
axs[0].set_title("Windowed Fourier Transform (Spectrogram)")
axs[0].set_ylim([0,400])
axs[0].set_ylabel("Frequency [Hz]")
#fig.colorbar(im1, ax=axs[0], label='Magnitude')

pcm = axs[1].pcolormesh(t, cwt_freqs, np.abs(cwt_coeffs))
axs[1].set_ylim([0,400])
axs[1].set_yscale("linear")
axs[1].set_xlabel("Time (s)")
axs[1].set_ylabel("Frequency (Hz)")
axs[1].set_title("Continuous Wavelet Transform (Scaleogram)")

plt.tight_layout()
fig.subplots_adjust(hspace=0.3)

plt.show()
