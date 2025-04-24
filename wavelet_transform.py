"""
wavelet_transform.py
====================
A compact helper for **Continuous Wavelet Transform (CWT)** analysis of 1-D signals
(e.g. WAV audio).  Highlights:

* Robust path handling – pass either a **.wav file** or a **directory** that
  contains exactly one WAV.
* Fast CWT via *PyWavelets* with 50 log-spaced scales by default
  (≈ the settings in the “quick” handwritten demo).
* Optional FFT overlay → two-panel figure (scalogram + magnitude spectrum).
* CLI flags for wavelet family/parameters, scale count, cropping, FFT, headless
  mode, and NumPy *.cwt.npy* export.
* Pure-Python and dependency-light: NumPy ≥ 1.21, SciPy ≥ 1.7,
  PyWavelets ≥ 1.3, Matplotlib ≥ 3.5.

Example — as a **library**

```python
from wavelet_transform import WaveletTransform

wt = WaveletTransform("samples/Katydid.wav")
sr, sig          = wt.load_audio(crop_seconds=5)         # 5-s window
coefs, freqs_hz  = wt.cwt(sig, sr, wavelet="cmor1.5-1.0")
wt.plot_cwt_and_fft(coefs, freqs_hz, sr, sig)            # 2-panel plot
```

Command line use:
    python wavelet_transform.py samples/Katydid.wav --wavelet cmor1.5-1.0 --seconds 5 --fft
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pywt
from numpy.typing import NDArray
from scipy.io import wavfile


# --------------------------------------------------------------------- #
#  Helper class                                                         #
# --------------------------------------------------------------------- #
class WaveletTransform:
    """
    Lightweight wrapper that bundles **loading**, **CWT / FFT computation**
    and **plot helpers** for a single 1-D WAV signal.

    ----------
    Parameters
    ----------
    audio_location : str | pathlib.Path
        Either of the following is accepted –

        • Path to a single **.wav** file *or*
        • Directory that contains **exactly one** ``*.wav`` file
          (useful for quick demos: that file is auto-selected).

    ----------
    Notes
    -----
    * When given a directory the first and only WAV is resolved and stored in
      :pyattr:`self.audio_file`; otherwise the provided file path is used.
    * All heavy work happens lazily:
        - :py:meth:`load_audio` decodes, normalises (±1), stereo→mono and
          optionally time-crops the signal.
        - :py:meth:`cwt` computes the continuous wavelet transform and converts
          scale→frequency (Hz).
        - :py:meth:`fft` returns the positive-frequency magnitude spectrum.
    * Plot helpers (`plot_scalogram`, `plot_cwt_and_fft`) visualise the data
      with sensible defaults and a colour-bar.
    * The complementary :py:meth:`cli` classmethod turns the module into a
      one-shot command-line utility:

        ``python wavelet_transform.py <file_or_dir> [--wavelet morl] [--fft]``

    The class is intentionally minimal—no external state beyond the resolved
    WAV path—so it’s easy to instantiate repeatedly inside batch pipelines.
    """

    # ---------- construction ----------------------------------------- #
    def __init__(self, audio_location: str | Path):
        """
        Create a :class:`WaveletTransform` and verify *audio_location*.
        """

        self.audio_file = self._resolve_audio_file(audio_location)

    @staticmethod
    def _resolve_audio_file(location: str | Path) -> Path:
        """
        Return an **absolute** path to a WAV file or raise a clear error.

        * If *location* points at a directory:
        * exactly one ``*.wav`` must exist – that file is returned.
        * If *location* points at a file:
        * it must exist and have suffix ``.wav``.

        Raises
        ------
        FileNotFoundError
            No WAV in directory / file does not exist.
        FileExistsError
            Directory contained multiple WAVs (ambiguous).
        ValueError
            File exists but is not a ``.wav`` container.
        """

        loc = Path(location).expanduser().resolve()

        if loc.is_dir():
            wavs = sorted(loc.glob("*.wav"))
            if not wavs:
                raise FileNotFoundError(f"No WAV files in {loc}")
            if len(wavs) > 1:
                raise FileExistsError(
                    f"{loc} contains multiple WAVs: {', '.join(w.name for w in wavs)}"
                )
            return wavs[0]

        if not loc.exists():
            raise FileNotFoundError(f"{loc} does not exist")
        if loc.suffix.lower() != ".wav":
            raise ValueError(f"{loc} is not a WAV file")
        return loc

    # ---------- audio loading ---------------------------------------- #
    def load_audio(
        self, *, crop_seconds: int | float | None = None
    ) -> Tuple[int, NDArray[np.floating]]:
        """
        Read the WAV file, convert to **mono float32 in [-1, 1]**, optionally
        return only the first *crop_seconds* seconds.

        Parameters
        ----------
        crop_seconds : int | float | None, default ``None``
            If given, truncates the decoded signal to that many seconds.  Use
            this to speed up exploratory plots or conserve RAM.

        Returns
        -------
        sr : int
            Sample-rate in Hz.
        signal : numpy.ndarray[float32]
            1-D mono waveform, already normalised to ±1.

        Notes
        -----
        • Integer PCM (e.g. ``int16``) is divided by its full-scale max.
        • Multichannel audio is averaged to mono.
        """

        sr, data = wavfile.read(self.audio_file)

        if data.dtype.kind in "iu":
            data = data.astype(np.float32) / np.iinfo(data.dtype).max
        elif data.dtype != np.float32:
            data = data.astype(np.float32)

        if data.ndim == 2:
            data = data.mean(axis=1)

        if crop_seconds is not None:
            samples = int(sr * crop_seconds)
            data = data[:samples]

        return sr, data

    # ---------- CWT --------------------------------------------------- #
    @staticmethod
    def cwt(
        signal: NDArray[np.floating],
        sr: int,
        *,
        wavelet: str = "morl",
        scales: Sequence[float] | None = None,
    ) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
        """
        Compute a **continuous wavelet transform**.

        Parameters
        ----------
        signal : ndarray
            1-D mono waveform.
        sr : int
            Sample-rate (Hz).
        wavelet : str, default ``"morl"``
            Any name returned by ``pywt.wavelist(kind="continuous")`` – e.g.
            ``"cmor1.5-1.0"``, ``"gaus4"``, ``"mexh"``.
        scales : sequence of float | None
            List/array of scales.  If ``None``  →  50 log-spaced scales
            ``np.geomspace(1,512,50)``.

        Returns
        -------
        coeffs : ndarray[complex128]  (n_scales × n_samples)
            CWT coefficients.
        freqs_hz : ndarray[float]     (n_scales,)
            Centre frequency of each scale **in Hz**.
        """

        if scales is None:
            scales = np.geomspace(1, 512, num=50)
        else:
            scales = np.asarray(scales, dtype=float)

        coeffs, _ = pywt.cwt(signal, scales, wavelet, sampling_period=1 / sr)
        freqs = pywt.scale2frequency(pywt.ContinuousWavelet(wavelet), scales) * sr
        return coeffs, freqs

    # ---------- FFT --------------------------------------------------- #
    @staticmethod
    def fft(
        signal: NDArray[np.floating],
        sr: int,
    ) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
        """
        Return the **positive-frequency half** of the FFT magnitude spectrum.

        Parameters
        ----------
        signal : ndarray
            1-D time-domain signal.
        sr : int
            Sample-rate (Hz).

        Returns
        -------
        freqs : ndarray[float]
            Positive frequency bins (Hz).
        magnitude : ndarray[float]
            |FFT(signal)| matching *freqs*.
        """

        n = signal.size
        fft_data = np.fft.fft(signal)
        freqs = np.fft.fftfreq(n, d=1 / sr)
        pos = freqs > 0
        return freqs[pos], np.abs(fft_data[pos])

    # ---------- plots ------------------------------------------------- #
    @staticmethod
    def plot_scalogram(
        coeffs: NDArray[np.floating],
        freqs: NDArray[np.floating],
        sr: int,
        *,
        t_start: float = 0.0,
        title: str = "CWT Scalogram",
        cmap: str | None = None,
    ) -> None:
        """
        Display a **log-frequency scalogram** with a colour-bar.

        Parameters
        ----------
        coeffs, freqs : ndarray
            Output of :py:meth:`cwt`.
        sr : int
            Sample-rate (Hz) – used to convert sample index to seconds.
        t_start : float, default 0.0
            Offset for the x-axis (useful when plotting successive windows).
        title : str
            Figure title.
        cmap : str | None
            Colormap forwarded to ``matplotlib.pyplot.pcolormesh``.
        """

        t = np.arange(coeffs.shape[1]) / sr + t_start
        fig, ax = plt.subplots(figsize=(12, 5))
        pcm = ax.pcolormesh(t, freqs, np.abs(coeffs), shading="auto", cmap=cmap)
        ax.set_yscale("log")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
        ax.set_title(title)
        fig.colorbar(pcm, ax=ax, label="|Coefficient|")
        fig.tight_layout()
        plt.show()

    @staticmethod
    def plot_cwt_and_fft(
        coeffs: NDArray[np.floating],
        freqs: NDArray[np.floating],
        sr: int,
        signal: NDArray[np.floating],
        *,
        t_start: float = 0.0,
        title: str = "Continuous Wavelet Transform (Scalogram)",
        cmap: str | None = None,
    ) -> None:
        """
        Two-panel figure: **scalogram** (top) + **FFT magnitude** (bottom).

        Useful for quick “what frequencies are present?” diagnostics.

        Parameters are identical to :py:meth:`plot_scalogram` with the addition
        of *signal* – the raw time-domain samples for the FFT panel.
        """

        t = np.arange(coeffs.shape[1]) / sr + t_start
        f_fft, mag_fft = WaveletTransform.fft(signal, sr)

        fig, (ax_cwt, ax_fft) = plt.subplots(2, 1, figsize=(12, 8))

        pcm = ax_cwt.pcolormesh(t, freqs, np.abs(coeffs), shading="auto", cmap=cmap)
        ax_cwt.set_yscale("log")
        ax_cwt.set_xlabel("Time (s)")
        ax_cwt.set_ylabel("Frequency (Hz)")
        ax_cwt.set_title(title)
        fig.colorbar(pcm, ax=ax_cwt, label="|Coefficient|")

        ax_fft.plot(f_fft, mag_fft)
        ax_fft.set_xlabel("Frequency (Hz)")
        ax_fft.set_ylabel("Amplitude")
        ax_fft.set_title("FFT of signal")

        fig.tight_layout()
        plt.show()

    # ---------- CLI --------------------------------------------------- #
    @staticmethod
    def cli() -> None:
        """
        Parse ``sys.argv`` and run **one-shot analysis**.

        Flags
        -----
        * ``location``   – WAV file or directory containing one
        * ``--wavelet``  – continuous wavelet name (default *morl*)
        * ``--scales``   – number of log-spaced scales (default 50)
        * ``--seconds``  – crop first *N* seconds before analysis
        * ``--fft``      – add FFT subplot beneath scalogram
        * ``--no-plot``  – skip plotting, still save ``.cwt.npy``

        A NumPy array of CWT coefficients is always saved next to the input file
        as ``<audio>.cwt.npy`` for downstream processing.
        """

        parser = argparse.ArgumentParser(description="Quick CWT demo + optional FFT")
        parser.add_argument("location", help="WAV file or directory containing one")
        parser.add_argument("--wavelet", default="morl", help="PyWavelets CWT name")
        parser.add_argument(
            "--scales",
            type=int,
            default=50,
            help="number of log-spaced scales (default 50)",
        )
        parser.add_argument(
            "--seconds",
            type=float,
            help="crop first N seconds of audio (speeds things up)",
        )
        parser.add_argument("--fft", action="store_true", help="add FFT subplot")
        parser.add_argument("--no-plot", action="store_true", help="skip plotting")
        args = parser.parse_args()

        wt = WaveletTransform(args.location)
        sr, sig = wt.load_audio(crop_seconds=args.seconds)

        scales = np.geomspace(1, 512, num=args.scales)
        coeffs, freqs = wt.cwt(sig, sr, wavelet=args.wavelet, scales=scales)
        np.save(Path(args.location).with_suffix(".cwt.npy"), coeffs)

        if not args.no_plot:
            if args.fft:
                wt.plot_cwt_and_fft(coeffs, freqs, sr, sig, title=args.location)
            else:
                wt.plot_scalogram(coeffs, freqs, sr, title=args.location)

    # ---------- repr -------------------------------------------------- #
    def __repr__(self) -> str:  # noqa: D401
        """Return ``WaveletTransform(audio_file=<path>)`` for quick introspection."""
        return f"WaveletTransform(audio_file={self.audio_file!s})"


# --------------------------------------------------------------------- #
#  Run from command line                                                #
# --------------------------------------------------------------------- #
if __name__ == "__main__":
    try:
        WaveletTransform.cli()
    except Exception as exc:  # pylint: disable=broad-except
        print(f"\n\x1b[31mError:\x1b[0m {exc}", file=sys.stderr)
        sys.exit(1)
