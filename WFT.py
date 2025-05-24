import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sgn

# Parameters
fs = 1000           # Sampling frequency (Hz)
duration = 1.0      # Signal duration (seconds)
window_type = 'hann'
window_size = 200   # WFT window size
function_freq = 50
function_type = 'noisy_sine' # Choose 'sine_wave', 'chirp' or 'noisy_sine'

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

# PLOTTING
fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# Plot original signal
axs[0].plot(t, signal)
axs[0].set_title("Signal")
axs[0].set_ylabel("Amplitude")
axs[0].grid()

# Plot WFT spectrogram
im1 = axs[1].imshow(np.abs(wft_coeffs), aspect='auto',
                    extent=[wft_times[0], wft_times[-1], wft_freqs[0], wft_freqs[-1]],
                    origin='lower', cmap='viridis')
axs[1].set_title("Windowed Fourier Transform (Spectrogram)")
axs[1].set_ylabel("Frequency [Hz]")
fig.colorbar(im1, ax=axs[1], label='Magnitude')

plt.tight_layout()
plt.show()
