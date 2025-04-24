"""
wavelet_transform.py
--------------------
Fast Continuous Wavelet Transform utilities + optional FFT overlay.

    python wavelet_transform.py AUDIO.wav --wavelet cmor1.5-1.0 --seconds 5 --fft
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
    """Compute CWT (and FFT) on 1-D signals with robust WAV handling."""

    # ---------- construction ----------------------------------------- #
    def __init__(self, audio_location: str | Path):
        self.audio_file = self._resolve_audio_file(audio_location)

    @staticmethod
    def _resolve_audio_file(location: str | Path) -> Path:
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
        sr, data = wavfile.read(self.audio_file)

        # normalise ints to ±1
        if data.dtype.kind in "iu":
            data = data.astype(np.float32) / np.iinfo(data.dtype).max
        elif data.dtype != np.float32:
            data = data.astype(np.float32)

        # stereo → mono
        if data.ndim == 2:
            data = data.mean(axis=1)

        # optional crop
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
        if scales is None:
            # 50 log-spaced scales ≈ geomspace in the old fast script
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
    def __repr__(self) -> str:
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
