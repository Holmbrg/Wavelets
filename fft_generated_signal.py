import numpy as np
import matplotlib.pyplot as plt

# Parameters
fs = 1000   # Hz
duration = 0.1       # seconds
function_freq1 = 50 
function_freq2 = 70 
function_type = 'sine_sum' # Choose 'sine_wave', 'chirp', 'sine_sum' or 'noisy_sine'

def get_signal():

    if function_type == 'sine_wave':
        return np.sin(function_freq1 * 2.0*np.pi*t)
    if function_type == 'chirp':
        return np.sin(function_freq1 * 2.0*np.pi*(t**2))
    if function_type == 'noisy_sine':
        return np.sin(function_freq1 * 2.0*np.pi*t) + np.random.normal(scale=0.5, size=t.shape)
    if function_type == 'sine_sum':
        return np.sin(function_freq1 * 2.0*np.pi*t) + np.sin(function_freq2 * 2.0*np.pi*t)

# Generating the input signal
t = np.linspace(0, duration, int(fs * duration), endpoint=False)
signal = get_signal()

# Compute FFT
N = len(signal)
freq_bins = np.fft.fftfreq(N, 1 / fs) # Returns the array of sample frequencies used
fft_coeffs = np.fft.fft(signal) # Returns the fft coefficients
fft_magnitudes = np.abs(fft_coeffs) / N

# Plot
plt.figure(figsize=(12, 5))

# Plotting the signal
plt.subplot(1, 2, 1)
plt.plot(t, signal)
plt.title("Signal (Time Domain)")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")

# Plotting the fft
plt.subplot(1, 2, 2)
plt.plot(freq_bins[:N//2], fft_magnitudes[:N//2])
plt.title("Signal (Frequency Domain)")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Magnitude")

plt.tight_layout()
plt.show()
