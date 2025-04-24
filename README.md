# Wavelet-CWT

A tiny, self-contained Python helper for computing and visualising the **Continuous Wavelet Transform (CWT)** on 1-D signals such as audio waveforms.  
Powered by **NumPy · SciPy · PyWavelets · Matplotlib**.

---

## Features

- Robust path-handling (file or directory) & WAV loading (int/float, stereo→mono)
- Fast CWT via PyWavelets, plus scale→frequency conversion in Hz
- Publication-quality **scalogram** plot with log-frequency axis and colour-bar
- One-file turnkey CLI _or_ import as a library
- Pure-Python, no compiled extensions

---

## Quick start

```bash
# 1 – clone the repository
git clone https://github.com/your-org/wavelet-cwt.git
cd wavelet-cwt

# 2 – create & activate a fresh venv  (optional but recommended)
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 3 – install dependencies
pip install -r requirements.txt

# 4 – run the demo
python wavelet_transform.py samples/Katydid.wav --wavelet cmor1.5-1.0 --seconds 5 --fft
```

## Comments

To run:
python wavelet_transform.py samples/Katydid.wav --wavelet cmor1.5-1.0 --seconds 5 --fft

You'll need to open a terminal run: cd "path/to/wavelet-cwt", then run the above command.
You can adjust the following:

- samples/Katydid.wav <=> Choose between samples/Katydid.wav, samples/RedeyedVireo.wav, samples/Gibbon.wav
  OR add more sample files to folder 'samples'

- for --wavelet "name of wavelet" you can choose between:
  ['cgau1', 'cgau2', 'cgau3', 'cgau4', 'cgau5',
  'cgau6', 'cgau7', 'cgau8', 'cmor', 'fbsp',
  'gaus1', 'gaus2', 'gaus3', 'gaus4', 'gaus5',
  'gaus6', 'gaus7', 'gaus8', 'mexh', 'morl', 'shan']

  Complex Gaussian:
  cgau1 … cgau8 ~ Analytic derivatives of a Gaussian.

  Complex Morlet:
  cmor, cmorB-C ~ Plain cmor uses the library’s default bandwidth/centre;
  you can dial your own, e.g. cmor1.5-1.0.

  Frequency B-spline:
  fbsp, fbspM-FB-FC ~ Default fbsp; or specify order M, bandwidth FB, centre FC: fbsp2-1-1.5.

  Gaussian (real):
  gaus1 … gaus8 ~ nth derivative of a Gaussian, real-valued.

  Mexican hat:
  mexh ~ Also called the Ricker wavelet.

  Morlet (real):
  morl ~ Classic real Morlet.

  Shannon:
  shan, shanB-C ~ Default shan; or give bandwidth B and centre C, e.g. shan1-1.5.

- for --seconds "number of seconds" ~ Should never exceed the sound files duration and doesn't have
  to equal it either. Ideally short durations for less computation (e.g. 5 seconds)

- finally, for --fft ~ this is an optional boolean flag, meaning if you want an fft on top of
  the scalogram you can add this part to the end of the command, else, exclude it, like such:
  python wavelet_transform.py samples/Katydid.wav --wavelet cmor1.5-1.0 --seconds 5
