import matplotlib.pyplot as plt
import numpy as np
import pywt
import scipy.io.wavfile as wavfile
import os

# Parameters
wavelet = "cmor1.5-1.0"

# Load the .wav file
fs, data = wavfile.read(r'C:\Users\45298\Desktop\wavelet_programmer\Katydid.wav')

# Normalize
if np.issubdtype(data.dtype, np.integer):
    max_val = np.iinfo(data.dtype).max
    data = data / max_val

# Create time array
time = np.arange(len(data)) / fs

# Trim sample
duration = 5  # seconds
samples_to_use = int(duration * fs)
data = data[:samples_to_use]
time = time[:samples_to_use]

# Perform CWT
widths = np.geomspace(1, 512, num=50)
sampling_period = 1.0 / fs
cwtmatr, freqs = pywt.cwt(data, widths, wavelet, sampling_period=sampling_period)
cwtmatr = np.abs(cwtmatr)

# Perform FFT
n = len(data)
fft_freq_bins = np.fft.fftfreq(n, 1/fs)
fft_data = np.fft.fft(data)
#positive_freqs = fft_freqs[:n//2]
#positive_fft = np.abs(fft_data[:n//2])

# Plot the FFT
fig, axs = plt.subplots(2, 1, figsize=(12, 8))

axs[0].plot(fft_freq_bins, fft_data)
axs[0].set_title("FFT of signal")
axs[0].set_xlabel("Frequency (Hz)")
axs[0].set_ylabel("Amplitude")

# Plot the CWT
T, F = np.meshgrid(time[:cwtmatr.shape[1]], freqs)
pcm = axs[1].pcolormesh(T, F, cwtmatr, shading='auto')
axs[1].set_yscale("linear")
axs[1].set_xlabel("Time (s)")
axs[1].set_ylabel("Frequency (Hz)")
axs[1].set_title("Continuous Wavelet Transform (Scaleogram)")

fig.colorbar(pcm, ax=axs[0])
plt.tight_layout()
plt.show()
