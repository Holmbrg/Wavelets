import numpy as np
import matplotlib.pyplot as plt

# Compares the fft of two piecewise functions.
# Signal A is a sine wave of frequency f1 on the first half of the sampling period,
# and a sine wave of frequency f2 on the second half.
# Signal B is a sine wave of frequency f2 on the first half of the sampling period,
# and a sine wave of frequency f1 on the second half.

# Parameters
f1 = 20     # First frequency
f2 = 100    # Second frequency
fs = 1000
duration = 1.0
t = np.linspace(0, duration, int(fs * duration), endpoint=False)
N = len(t)
half = N // 2

# Signal A: f1 in first half, f2 in second half
signal_a = np.zeros(N)
signal_a[:half] = np.sin(2 * np.pi * f1 * t[:half])
signal_a[half:] = np.sin(2 * np.pi * f2 * t[half:])

# Signal B: f2 in first half, f1 in second half
signal_b = np.zeros(N)
signal_b[:half] = np.sin(2 * np.pi * f2 * t[:half])
signal_b[half:] = np.sin(2 * np.pi * f1 * t[half:])

# FFTs
fft_a = np.fft.fft(signal_a)
fft_b = np.fft.fft(signal_b)
freq_bins = np.fft.fftfreq(N, 1 / fs)

# Magnitude spectra (should be identical or nearly identical)
mag_a = np.abs(fft_a) / N
mag_b = np.abs(fft_b) / N

# Plot time-domain signals
plt.figure(figsize=(12, 6))

plt.subplot(2, 2, 1)
plt.plot(t, signal_a)
plt.title("Signal A (20 Hz then 40 Hz)")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")

plt.subplot(2, 2, 2)
plt.plot(t, signal_b)
plt.title("Signal B (40 Hz then 20 Hz)")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")

# Plot FFT magnitudes
plt.subplot(2, 2, 3)
plt.plot(freq_bins[:N//2], mag_a[:N//2])
plt.title("FFT Magnitude of Signal A")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Magnitude")

plt.subplot(2, 2, 4)
plt.plot(freq_bins[:N//2], mag_b[:N//2])
plt.title("FFT Magnitude of Signal B")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Magnitude")

plt.tight_layout()
plt.show()


