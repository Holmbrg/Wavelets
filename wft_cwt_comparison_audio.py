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
window_size = 20000   # WFT window size
wavelet = "cmor1.5-1.0"
scales = np.geomspace(1, 512, num=100)

# Load the .wav file
fs, data = wavfile.read(r'C:\Users\45298\Documents\GitHub\wavelet-cwt\samples\sangdrossel.wav')

# Converting to mono
if data.ndim > 1:
    data = data.mean(axis=1)

# Normalize
if np.issubdtype(data.dtype, np.integer):
    max_val = np.iinfo(data.dtype).max
    data = data / max_val

# Create time array
t = np.arange(len(data)) / fs

# Trim sample
duration = 5  # seconds
samples_to_use = int(duration * fs)
data = data[:samples_to_use]
t = t[:samples_to_use]

# Compute WFT
wft_freqs, wft_times, wft_coeffs = sgn.stft(data, fs, window_type, window_size)

# Compute CWT
cwt_coeffs, cwt_freqs = pywt.cwt(data, scales, wavelet, 1.0 / fs)


# --- PLOTTING --- #

fig, axs = plt.subplots(2, 1, figsize=(9, 9), sharex=False)

# Plot WFT
im1 = axs[0].imshow(np.abs(wft_coeffs), aspect='auto',
                    extent=[wft_times[0], wft_times[-1], wft_freqs[0], wft_freqs[-1]],
                    origin='lower', cmap='viridis')
axs[0].set_title("Windowed Fourier Transform (Spectrogram)")
axs[0].set_ylim([0,10000])
axs[0].set_ylabel("Frequency [Hz]")
#fig.colorbar(im1, ax=axs[0], label='Magnitude')

pcm = axs[1].pcolormesh(t, cwt_freqs, np.abs(cwt_coeffs))
axs[1].set_ylim([0,10000])
axs[1].set_yscale("linear")
axs[1].set_xlabel("Time (s)")
axs[1].set_ylabel("Frequency (Hz)")
axs[1].set_title("Continuous Wavelet Transform (Scaleogram)")

plt.tight_layout()
fig.subplots_adjust(hspace=0.3)

plt.show()
