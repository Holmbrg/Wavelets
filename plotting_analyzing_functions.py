import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.fft import fft
import pywt

# Parameters
fs = 1000           # Sampling frequency (Hz)
duration = 1.0      # Signal duration (seconds)
omega = 100
window_type = 'hann'
window_size = 50   # WFT window size
wavelet = pywt.ContinuousWavelet('cmor1.0-0.5') # y=0, t=1

# Time vector
t = np.linspace(0, duration, int(fs * duration), endpoint=False)

# --- Signals ---

# Complex exponential
ew = np.exp(2j * np.pi * omega * t)

# Window function (Hann)
gwt = signal.windows.get_window(window_type, window_size)

# Wavelet function
psi, x = wavelet.wavefun()

# --- FFTs ---

# FFT of ew
N1 = len(ew)
freq_bins1 = np.fft.fftfreq(N1, 1 / fs)
fft_coeffs1 = np.fft.fft(ew)
fft_magnitudes1 = np.abs(fft_coeffs1) / N1

# FFT of gwt
N2 = len(gwt)
freq_bins2 = np.fft.fftfreq(N2, 1 / fs)
fft_coeffs2 = np.fft.fft(gwt)
fft_magnitudes2 = np.abs(fft_coeffs2) / N2

# FFT of wavelet psi
N3 = len(psi)
dt = x[1] - x[0]  # Wavelet's sampling interval
freq_bins3 = np.fft.fftfreq(N3, dt)
fft_coeffs3 = np.fft.fft(psi)
fft_magnitudes3 = np.abs(fft_coeffs3) / N3

# --- PLOTTING ---

# First plot: ew and its FFT
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(t, np.abs(ew))
plt.title("Complex Exponential (Time Domain)")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")

plt.subplot(1, 2, 2)
plt.plot(np.fft.fftshift(freq_bins1), np.fft.fftshift(fft_magnitudes1))
plt.title("Complex Exponential (Frequency Domain)")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Magnitude")

# Second plot: gwt and its FFT
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(np.linspace(0, window_size / fs, window_size), gwt)
plt.title("Hann Window (Time Domain)")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")

plt.subplot(1, 2, 2)
plt.plot(np.fft.fftshift(freq_bins2), np.fft.fftshift(fft_magnitudes2))
plt.title("Hann Window (Frequency Domain)")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Magnitude")


# Third plot: psi and its FFT
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(x, np.abs(psi))
plt.title("Wavelet Function (Time Domain)")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")

plt.subplot(1, 2, 2)
plt.plot(np.fft.fftshift(freq_bins3), np.fft.fftshift(fft_magnitudes3))
plt.title("Wavelet Function (Frequency Domain)")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Magnitude")

plt.tight_layout()
plt.show()

