import matplotlib.pyplot as plt
import numpy as np
import pywt
import scipy.io.wavfile as wavfile
from scipy import signal as sgn
import os

# Parameters
window_type = 'hann' # WFT window shape
window_size = 1200   # WFT window size
wavelet = "cmor1.5-1.0"
scales = np.geomspace(1, 512, num=100)

# Load the .wav file
fs, data = wavfile.read(r'C:\Users\45298\Documents\GitHub\wavelet-cwt\samples\nattergal.wav')

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

# First plot: the signal
plt.figure(figsize=(10, 4))
plt.plot(t, data)
plt.title('Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.tight_layout()
plt.show()

# Second plot: WFT and CWT
fig, axs = plt.subplots(2, 1, figsize=(9, 9), sharex=False)

# PLOT 1: WFT (STFT)
axs[0].imshow(np.abs(wft_coeffs), aspect='auto',
              extent=[wft_times[0], wft_times[-1], wft_freqs[0], wft_freqs[-1]],
              origin='lower', cmap='viridis')
axs[0].set_title("WFT Spectrogram (window size " + str(window_size) + " out of " + str(len(data)) + ")")
axs[0].set_ylim([0, fs // 2])
axs[0].set_ylabel("Frequency [Hz]")

# PLOT 2: CWT
axs[1].pcolormesh(t, cwt_freqs, np.abs(cwt_coeffs), shading="auto", cmap="viridis")
axs[1].set_title("CWT Scaleogram")
axs[1].set_ylim([0, fs // 2])
axs[1].set_xlabel("Time (s)")
axs[1].set_ylabel("Frequency [Hz]")


plt.tight_layout()
fig.subplots_adjust(hspace=0.3)

plt.show()
