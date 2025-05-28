import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sgn
import pywt

# Parameters
fs = 1000           # Sampling frequency (Hz)
duration = 1.0      # Signal duration (seconds)
window_type = 'hann'
window_size = 100   # WFT window size
function_freq = 70
function_type = 'chirp' # Choose 'sine_wave', 'chirp' or 'noisy_sine'

def get_signal():

    if function_type == 'sine_wave':
        return np.sin(function_freq * 2.0*np.pi*t)
    if function_type == 'chirp':
        return np.sin(function_freq * 2.0*np.pi*(t**2))
    if function_type == 'noisy_sine':
        return np.sin(function_freq * 2.0*np.pi*t) + np.random.normal(scale=0.5, size=t.shape)

# Generating the input signal
t = np.linspace(0, duration, int(fs * duration), endpoint=False)
signal = get_signal()

# Compute wft
wft_freqs, wft_times, wft_coeffs = sgn.stft(signal, fs, window_type, window_size)

# Compute fft
N = len(signal)
freq_bins = np.fft.fftfreq(N, 1 / fs) # Returns the array of sample frequencies used
fft_coeffs = np.fft.fft(signal) # Returns the fft coefficients
fft_magnitudes = np.abs(fft_coeffs) / N

# === Plotting ===

# First plot: the signal
plt.figure(figsize=(10, 4))
plt.plot(t, signal)
plt.title('Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.tight_layout()
plt.show()

# Second plot: FFT magnitude and WFT spectrogram
fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=False)

axs[0].plot(freq_bins, fft_magnitudes)
axs[0].set_title("FFT")
axs[0].set_xlabel("Frequency [Hz]")
axs[0].set_ylabel("Magnitude")
axs[0].set_xlim(0, 500)

im1 = axs[1].imshow(np.abs(wft_coeffs), aspect='auto',
                    extent=[wft_times[0], wft_times[-1], wft_freqs[0], wft_freqs[-1]],
                    origin='lower', cmap='viridis')
axs[1].set_title("Windowed Fourier Transform (Spectrogram)")
axs[1].set_ylabel("Frequency [Hz]")
#fig.colorbar(im1, ax=axs[2], label='Magnitude')

plt.tight_layout()
fig.subplots_adjust(hspace=0.5)

plt.show()


