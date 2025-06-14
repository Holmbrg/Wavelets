import numpy as np
import pywt
import matplotlib.pyplot as plt
from scipy.io import wavfile
import random
import sounddevice as sd
import time
from scipy import signal
import math


def main_denoise_tit(z: list[float], wavelet: str, uni_or_adap: str, hard_or_soft: str, noise_level_perc: float=75) -> None:
    """Takes true signal and adds noise corresponding to noise_level_perc (perc. of abs. mean of signal).
    Choose 'uni', 'uni_perc' or 'adap' and 'hard' or 'soft' thresholding"""

    noise_level = (noise_level_perc/100)*np.mean(np.abs(z))
    noisy_signal = add_noise(z,noise_level)

    a = sum([ abs(x)**2 for x in z ])
    b = sum([ abs(z[k]-noisy_signal[k])**2 for k in range(len(z)) ])

    SNR_pre = a/b

    if uni_or_adap == 'uni':
        if hard_or_soft == 'hard':
            denoised_audio = idwt(denoise_wavl_uni_hard(dwt(noisy_signal, wavelet),len(z), noise_level), wavelet)
        elif hard_or_soft == 'soft':
            denoised_audio = idwt(denoise_wavl_uni_soft(dwt(noisy_signal, wavelet),len(z), noise_level), wavelet)
        else:
            print('Invalid input: Write "hard" or "soft"')
    elif uni_or_adap == 'uni_perc':
        thresh = float(input('Enter largest percentile of coeffs to keep: '))
        if hard_or_soft == 'hard':
            denoised_audio = idwt(denoise_wavl_uniperc_hard(dwt(noisy_signal, wavelet),thresh), wavelet)
        elif hard_or_soft == 'soft':
            denoised_audio = idwt(denoise_wavl_uniperc_soft(dwt(noisy_signal, wavelet),thresh), wavelet)
        else:
            print('Invalid input: Write "hard" or "soft"')
    elif uni_or_adap == 'adap':
        thresh = float(input('Enter largest percentile to keep for each scale: '))
        if hard_or_soft == 'hard':
            denoised_audio = idwt(denoise_wavl_adap_hard(dwt(noisy_signal, wavelet),thresh), wavelet)
        elif hard_or_soft == 'soft':
            denoised_audio = idwt(denoise_wavl_adap_soft(dwt(noisy_signal, wavelet),thresh), wavelet)
        else:
            print('Invalid input: Write "hard" or "soft"')
    else:
        print('Invalid input: Write "uni", "uni_perc" or "adap"')


    c = sum([ abs(z[k]-denoised_audio[k])**2 for k in range(len(z)) ])

    SNR_pos = a/c

    print('SNR_pre = ' + str(SNR_pre) + ' and SNR_pos = ' + str(SNR_pos) + ' SNR-dif = ' + str(SNR_pos-SNR_pre))

    MSE = c/len(z)

    print('MSE = ' + str(MSE))

    plt.plot(time ,denoised_audio)
    plt.title('Great tit denoised, ' + str(hard_or_soft)+' ' + str(uni_or_adap)+'. ' + str(thresh)+'%, ' + str(wavelet))
    plt.ylabel('Amplitude')
    plt.xlabel('Time [s]')
    plt.show()

    play_audio(denoised_audio)
    


def main_denoise_speech(z: list[float], wavelet: str, uni_or_adap: str, hard_or_soft: str) -> None:
    """Takes noisy signal (speech in loud office).
    Choose 'uni' or 'adap' and 'hard' or 'soft' thresholding"""

    noisy_signal = z

    a = sum([ abs(x)**2 for x in audio_speech ])
    b = sum([ abs(x)**2 for x in audio_office ])

    SNR_pre = a/b

    if uni_or_adap == 'uni':
        if hard_or_soft == 'hard':
            denoised_audio = idwt(denoise_wavl_uni_hard(dwt(noisy_signal, wavelet),len(z), 0), wavelet)
        elif hard_or_soft == 'soft':
            denoised_audio = idwt(denoise_wavl_uni_soft(dwt(noisy_signal, wavelet),len(z), 0), wavelet)
        else:
            print('Invalid input: Write "hard" or "soft"')
    elif uni_or_adap == 'uni_perc':
        thresh = float(input('Enter largest percentile of coeffs to keep: '))
        if hard_or_soft == 'hard':
            denoised_audio = idwt(denoise_wavl_uniperc_hard(dwt(noisy_signal, wavelet),thresh), wavelet)
        elif hard_or_soft == 'soft':
            denoised_audio = idwt(denoise_wavl_uniperc_soft(dwt(noisy_signal, wavelet),thresh), wavelet)
        else:
            print('Invalid input: Write "hard" or "soft"')
    elif uni_or_adap == 'adap':
        thresh = float(input('Enter largest percentile to keep for each scale: '))
        if hard_or_soft == 'hard':
            denoised_audio = idwt(denoise_wavl_adap_hard(dwt(noisy_signal, wavelet),thresh), wavelet)
        elif hard_or_soft == 'soft':
            denoised_audio = idwt(denoise_wavl_adap_soft(dwt(noisy_signal, wavelet),thresh), wavelet)
        else:
            print('Invalid input: Write "hard" or "soft"')
    else:
        print('Invalid input: Write "uni" or "adap"')


    c = sum([ abs(audio_speech[k]-denoised_audio[k])**2 for k in range(len(z)) ])

    SNR_pos = a/c

    print('SNR_pre = ' + str(SNR_pre) + ' and SNR_pos = ' + str(SNR_pos) + ' SNR-dif = ' + str(SNR_pos-SNR_pre))

    MSE = c/len(z)

    print('MSE = ' + str(MSE))

    plt.plot(time ,denoised_audio)
    plt.title('Speech denoised, ' + str(hard_or_soft)+' ' + str(uni_or_adap)+'. ' + str(thresh)+'%, ' + str(wavelet))
    plt.ylabel('Amplitude')
    plt.xlabel('Time [s]')
    plt.show()

    play_audio(denoised_audio)
  


def dwt(z: list[float], wavelet: str) -> list[list[float]]:
    """Applies DWT on signal z of lenght 2^p.
    Returns transformed z divided into scales"""

    return pywt.wavedec(z, wavelet)


def idwt(z: list[list[float]], wavelet: str) -> list[float]:
    """Applies IDWT on wavelet coeffs. z"""


    return pywt.waverec(z, wavelet)


def add_noise(z: list[float], noise_level: float) -> list[float]:
    """Adds noise to true signal z with std. dev. as noise_level"""

    abs_z = [ abs(x) for x in z ]

    noise = np.random.normal(0,noise_level,len(z))

    noisy_signal = [ z[k] + noise[k] for k in range(len(z)) ]


    plt.plot(time, noisy_signal)
    plt.title('Great tit, noisy signal, σ = 75% of abs. mean')
    plt.ylabel('Amplitude')
    plt.xlabel('Time [s]')
    plt.show()

    return noisy_signal




def denoise_wavl_uni_hard(wavelet_coef: list[list[float]], len_z: int, noise_level: float = 0) -> list[list[int]]:
    """Denoises input signal with universal hard thresholding. For wavelets."""

    denoised = [row[:] for row in wavelet_coef]

    #Calculate estimate for noise std. deviation
    median_finescale = np.median(wavelet_coef[-1])
    appr_std_dev = np.median( [ np.abs(x-median_finescale) for x in wavelet_coef[-1] ] ) / 0.6745

    thresh = appr_std_dev * np.sqrt(2*np.log(len_z))

    #thresh = noise_level * np.sqrt(2*(np.log(len_z)))  ##If std. dev. is known, use this thresh instead
    print(thresh)
    
    for j in range(len(wavelet_coef)):
        for k in range(len(wavelet_coef[j])):
            if abs(wavelet_coef[j][k]) <= float(thresh):
                denoised[j][k] = 0

    return denoised



def denoise_wavl_uni_soft(wavelet_coef: list[list[float]], len_z: int, noise_level: float = 0) -> list[list[int]]:
    """Denoises input signal with universal soft thresholding. For wavelets."""

    denoised = [row[:] for row in wavelet_coef]

    #Calculate estimate for noise std. deviation
    median_finescale = np.median(wavelet_coef[-1])
    appr_std_dev = np.median( [ np.abs(x-median_finescale) for x in wavelet_coef[-1] ] ) / 0.6745
    
    thresh = appr_std_dev * np.sqrt(2*np.log(len_z))

    #thresh = noise_level * np.sqrt(2*(np.log(len_z)))  ##If std. dev. is known, use this thresh instead
    
    for j in range(len(wavelet_coef)):
        for k in range(len(wavelet_coef[j])):
            if abs(wavelet_coef[j][k]) <= float(thresh):
                denoised[j][k] = 0
            else:
                denoised[j][k] = np.sign(denoised[j][k]) * (abs(denoised[j][k]) - thresh)

    return denoised



def denoise_wavl_adap_hard(wavelet_coef: list[list[float]], perc: float, scales: list[int] = [] ) -> list[list[int]]:
    """Denoises input signal with hard thresholding, retaining only perc % largest coeffs. at specified scales.
    If no scales are chosen, all are denoised. For wavelets."""

    if scales == []:
        scales = range(len(wavelet_coef))

    denoised = [row[:] for row in wavelet_coef]
    
    for j in scales:
        thresh = float(np.percentile(np.abs(wavelet_coef[j]), 100-perc))
        for k in range(len(wavelet_coef[j])):
            if abs(wavelet_coef[j][k]) <= thresh:
                denoised[j][k] = 0

    return denoised



def denoise_wavl_adap_soft(wavelet_coef: list[list[float]], perc: float, scales: list[int] = [] ) -> list[list[int]]:
    """Denoises input signal with soft thresholding, retaining only perc % largest coeffs. at specified scales.
    If no scales are chosen, all are denoised. For wavelets."""

    if scales == []:
        scales = range(len(wavelet_coef))

    denoised = [row[:] for row in wavelet_coef]

    for j in scales:
        thresh = float(np.percentile(np.abs(wavelet_coef[j]), 100-perc))
        for k in range(len(wavelet_coef[j])):
            if abs(wavelet_coef[j][k]) <= thresh:
                denoised[j][k] = 0
            else:
                denoised[j][k] = np.sign(denoised[j][k]) * (abs(denoised[j][k]) - thresh)

    return denoised


def denoise_wavl_uniperc_hard(wavelet_coef: list[list[float]], perc: float) -> list[list[int]]:
    """Denoises input signal with hard thresholding, retaining only perc % largest coeffs. calculated
    across all scales. For wavelets."""


    denoised = [row[:] for row in wavelet_coef]
    nr_scales = len(wavelet_coef)

    #Calculate percentile threshold
    all_coeff = []
    for scale in denoised:
        all_coeff = all_coeff + list(scale)
        
    thresh = np.percentile(np.abs(all_coeff), 100-perc)

    for j in range(nr_scales):
        for k in range(len(wavelet_coef[j])):
            if abs(wavelet_coef[j][k]) <= thresh:
                denoised[j][k] = 0

    return denoised


def denoise_wavl_uniperc_soft(wavelet_coef: list[list[float]], perc: float) -> list[list[int]]:
    """Denoises input signal with soft thresholding, retaining only perc % largest coeffs. calculated
    across all scales. For wavelets."""


    denoised = [row[:] for row in wavelet_coef]
    nr_scales = len(wavelet_coef)

    #Calculate percentile threshold
    all_coeff = []
    for scale in denoised:
        all_coeff = all_coeff + list(scale)
        
    thresh = np.percentile(np.abs(all_coeff), 100-perc)

    for j in range(nr_scales):
        for k in range(len(wavelet_coef[j])):
            if abs(wavelet_coef[j][k]) <= thresh:
                denoised[j][k] = 0
            else:
                denoised[j][k] = np.sign(denoised[j][k]) * (abs(denoised[j][k]) - thresh)

    return denoised


def wft_tit(z: list[float], perc: float, noise_level_perc: float = 75) -> None:
    """Adds noise to musvit.wav corresponding to noise_level_perc
    (perc. of abs. mean of z). Denoises by keeping thresh % largest coefficients
    of the WFT"""

    noise_level = (noise_level_perc/100)*np.mean(np.abs(z))
    noisy_signal = add_noise(z,noise_level)

    a = sum([ abs(x)**2 for x in z ])
    b = sum([ abs(z[k]-noisy_signal[k])**2 for k in range(len(z)) ])

    SNR_pre = a/b

    f, t, Zxx = signal.stft(noisy_signal, samplerate, 'hann', nperseg=256, noverlap=256//2)

    thresh = np.percentile(np.abs(Zxx), 100-perc)

    for i in range(len(f)):
        for j in range(len(t)):
            if abs(Zxx[i,j]) <= thresh:
                Zxx[i,j] = 0
##            else:                           #Comment out this else to use hard thresholding.
##                angle = np.angle(Zxx[i,j])
##                Zxx[i,j] = (abs(Zxx[i,j]) - thresh) * np.exp(1j*angle)

                
    t, denoised_audio = signal.istft(Zxx, samplerate, 'hann', nperseg=256, noverlap=256//2)


    c = sum([ abs(z[k]-denoised_audio[k])**2 for k in range(len(z)) ])

    SNR_pos = a/c


    print('SNR_pre = ' + str(SNR_pre) + ' and SNR_pos = ' + str(SNR_pos) + 'SNR_dif = ' + str(SNR_pos-SNR_pre))

    MSE = c/len(z)

    print('MSE = ' + str(MSE))
    
    plt.plot(time ,denoised_audio)
    plt.title('Great tit denoised, soft uni.' + str(perc)+'%, ' +'WFT')
    plt.ylabel('Amplitude')
    plt.xlabel('Time [s]')
    plt.show()


    play_audio(denoised_audio)



def wft_speech(z: list[float], perc: float) -> None:
    """Denoises speech in loud office by keeping thresh % largest coefficients
    of the WFT"""

    noisy_signal = z

    a = sum([ abs(x)**2 for x in audio_speech ])
    b = sum([ abs(x)**2 for x in audio_office ])

    SNR_pre = a/b

    f, t, Zxx = signal.stft(noisy_signal, samplerate, 'hann', nperseg=256, noverlap=256//2)

    thresh = np.percentile(np.abs(Zxx), 100-perc)

    for i in range(len(f)):
        for j in range(len(t)):
            if abs(Zxx[i,j]) <= thresh:
                Zxx[i,j] = 0
##            else:
##                angle = np.angle(Zxx[i,j])
##                Zxx[i,j] = (abs(Zxx[i,j]) - thresh) * np.exp(1j*angle)

                
    t, denoised_audio = signal.istft(Zxx, samplerate, 'hann', nperseg=256, noverlap=256//2)


    c = sum([ abs(audio_speech[k]-denoised_audio[k])**2 for k in range(len(z)) ])

    SNR_pos = a/c


    print('SNR_pre = ' + str(SNR_pre) + ' and SNR_pos = ' + str(SNR_pos) + 'SNR_dif = ' + str(SNR_pos-SNR_pre))

    MSE = c/len(z)

    print('MSE = ' + str(MSE))
    
    plt.plot(time ,denoised_audio)
    plt.title('Speech denoised, soft uni. '+ str(perc)+'%, '+ 'WFT')
    plt.ylabel('Amplitude')
    plt.xlabel('Time [s]')
    plt.show()


    play_audio(denoised_audio)

           

def play_audio(samples: list[float], samplerate: int = 44100):
    """Plays audio (samples)"""
    
    samples = samples / np.max(np.abs(samples))
    samples = np.array(samples, dtype=np.float32)
    sd.play(samples, samplerate)
    sd.wait()



def next_power_of_two(n):
    """Retunds next power of 2 for trimming wav files"""
    
    return 1 << (n.bit_length() - 1)


# Load and prepare audio
samplerate, data_office = wavfile.read("office_noise.wav")
samplerate, data_speech = wavfile.read("speech.wav")
samplerate, data_musvit = wavfile.read("musvit.wav")

# Convert to mono if stereo
if len(data_office.shape) > 1:
    data_office = data_office[:, 0]

if len(data_speech.shape) > 1:
    data_speech = data_speech[:, 0]

if len(data_musvit.shape) > 1:
    data_musvit = data_musvit[:, 0]

# Convert to float
data_office = data_office.astype(float)
data_speech = data_speech.astype(float)
data_musvit = data_musvit.astype(float)

# Trim to length 2^k
N = next_power_of_two(len(data_speech))
data_office = data_office[:N]
data_speech = data_speech[:N]
data_musvit = data_musvit[:N]

data_office = data_office.tolist()
data_speech = data_speech.tolist()
data_musvit = data_musvit.tolist()

# Normalize data
normalizer_office = max(np.abs(data_office))
normalizer_speech = max(np.abs(data_speech))
normalizer_musvit = max(np.abs(data_musvit))

#Normalize and scale to fitting noise level for office noise
audio_office = [(data_office[i]/normalizer_office)*0.15 for i in range(N)]

#Normalize speech
audio_speech = [data_speech[i]/normalizer_speech for i in range(N)]

#Normalize musvit and removes ambient noise at the ends of the audio file
audio_musvit = [data_musvit[i]/normalizer_musvit if (1.055e5 < i < 1.993e5) else 0 for i in range(N)]

audio_noisy_speech = [audio_office[i] + audio_speech[i] for i in range(N)]

 

# Plot chosen audio, just change the names below

time = [ (1/samplerate)*n for n in range(N) ]
##plt.plot(time ,audio_speech)
##plt.title('Speech, true signal')
##plt.ylabel('Amplitude')
##plt.xlabel('Time [s]')
##plt.show()
##
##plt.plot(time ,audio)
##plt.title('Speech, with noise')
##plt.ylabel('Amplitude')
##plt.xlabel('Time [s]')
##plt.show()



## From here and down just unused functions


def plot_all_denoised(z: list[float], wavelet: str, noise_level: float, thresh: float) -> None:

    noisy_signal = add_noise(z,(noise_level/100)*np.mean(np.abs(z)))
    
    uni_hard = idwt(denoise_wavl_uni_hard(dwt(noisy_signal, wavelet),len(z), noise_level), wavelet)
    uni_soft = idwt(denoise_wavl_uni_soft(dwt(noisy_signal, wavelet),len(z), noise_level), wavelet)
    adap_hard = idwt(denoise_wavl_adap_hard(dwt(noisy_signal, wavelet),thresh), wavelet)
    adap_soft = idwt(denoise_wavl_adap_soft(dwt(noisy_signal, wavelet),thresh), wavelet)

    plt.subplot(2,2,1)
    plt.plot(time, uni_hard)
    plt.title('Universal, hard thresholding')
    plt.ylabel('Amplitude')
    plt.xlabel('Time [s]')

    plt.subplot(2,2,2)
    plt.plot(time, uni_soft)
    plt.title('Universal, soft thresholding')
    plt.ylabel('Amplitude')
    plt.xlabel('Time [s]')

    plt.subplot(2,2,3)
    plt.plot(time, adap_hard)
    plt.title('Adaptive, hard thresholding')
    plt.ylabel('Amplitude')
    plt.xlabel('Time [s]')

    plt.subplot(2,2,4)
    plt.plot(time, adap_soft)
    plt.title('Adaptive, soft thresholding')
    plt.ylabel('Amplitude')
    plt.xlabel('Time [s]')

    plt.show()


def detect_event(wavelet_coef: list[list[float]], thresh: float) -> list[list[int]]:
    """Detects event and returns window of event."""
    

    nr_scales = len(wavelet_coef)
    #nr_coeffs = 2**(nr_scales-1)

    events = []
    
    for j in range(nr_scales):
        for k in range(len(wavelet_coef[j])):
            if abs(wavelet_coef[j][k] - wavelet_coef[j][k-1]) > thresh:
                events = events + [[j,k]]
            else:
                wavelet_coef[j][k] = 0
    
    plt.plot(idwt(wavelet_coef, 'db1'))
    plt.show()
    
    return events
