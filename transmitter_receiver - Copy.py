"""Direct-detection (IM/DD) transmitter and receiver.

All waveforms are sampled at fs = bit_rate * samples_per_bit. Filters are applied in the frequency
domain with real (zero-phase) responses, so the sampling instant stays at the bit centre. The bit
sequence is treated as periodic (FFT wrap-around); this is negligible for hundreds of bits.
Optical powers are in W, photocurrents in A.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.special import erfc


def nebw_factor(order: int, kind: str = "Bessel") -> float:
    """Noise-equivalent bandwidth / 3 dB bandwidth of the receiver filter."""
    if kind == "Butterworth":
        x = math.pi / (2.0 * order)
        return x / math.sin(x)
    return 0.5 * math.sqrt(math.pi / math.log(2.0))      # Gaussian (Bessel-like) response: 1.0645


def generate_bits(n: int, seed: int = 42, use_seed: bool = True) -> np.ndarray:
    rng = np.random.default_rng(seed if use_seed else None)
    bits = rng.integers(0, 2, n).astype(int)
    if n >= 2:
        bits[0], bits[1] = 0, 1          # guarantee both symbols are present
    return bits


def tx_levels(p_avg_w: float, ext_db: float, modulation: str):
    """Optical power of a '1' and a '0' for a given AVERAGE power and extinction ratio.

    NRZ: duty d = 1. RZ: duty d = 0.5 (a '1' is a pulse in the first half of the bit).
    Average over equiprobable bits = p_avg.
    """
    r = 10.0 ** (ext_db / 10.0)
    d = 0.5 if modulation.startswith("RZ") else 1.0
    p0 = 2.0 * p_avg_w / (d * r + 2.0 - d)
    return r * p0, p0


def _apply(x: np.ndarray, H: np.ndarray) -> np.ndarray:
    return np.fft.irfft(np.fft.rfft(x) * H, n=len(x))


def gaussian_filter(x: np.ndarray, sigma_samples: float) -> np.ndarray:
    """Convolve with a unit-area Gaussian of RMS width sigma_samples (zero-phase)."""
    if sigma_samples <= 1e-9:
        return x.copy()
    f = np.fft.rfftfreq(len(x))
    return _apply(x, np.exp(-2.0 * (np.pi * f * sigma_samples) ** 2))


def butterworth_filter(x: np.ndarray, fc_norm: float, order: int) -> np.ndarray:
    """Butterworth magnitude response with 3 dB cut-off fc_norm (cycles/sample), zero-phase."""
    f = np.fft.rfftfreq(len(x))
    return _apply(x, 1.0 / np.sqrt(1.0 + (f / fc_norm) ** (2 * order)))


def receiver_filter(x: np.ndarray, fc_norm: float, order: int = 4, kind: str = "Bessel") -> np.ndarray:
    """Receiver electrical filter, zero-phase magnitude response with 3 dB cut-off fc_norm.

    'Bessel' uses |H|^2 = 2^-(f/fc)^2 (Gaussian, the low-ringing shape of a Bessel-Thomson filter);
    'Butterworth' uses the n-th order maximally-flat response (more ringing, i.e. more ISI).
    """
    if kind == "Butterworth":
        return butterworth_filter(x, fc_norm, order)
    f = np.fft.rfftfreq(len(x))
    return _apply(x, 2.0 ** (-0.5 * (f / fc_norm) ** 2))


def build_tx_waveform(bits: np.ndarray, sps: int, p1: float, p0: float, modulation: str,
                      rise_frac: float) -> np.ndarray:
    """Optical power waveform (W) with finite transmitter rise time (10-90 % = rise_frac x bit period)."""
    on = np.repeat(bits, sps).astype(float)
    if modulation.startswith("RZ"):
        phase = np.tile(np.arange(sps) / sps, len(bits))
        on = on * (phase < 0.5)
    w = p0 + (p1 - p0) * on
    sigma = rise_frac * sps / 2.563          # 10-90 rise of a Gaussian-filtered step = 2.563 sigma
    return gaussian_filter(w, sigma)


def decision_threshold(i1: float, i0: float, s1: float, s0: float, kind: str = "Optimum") -> float:
    if kind == "Midpoint" or (s1 + s0) <= 0:
        return 0.5 * (i1 + i0)
    return (s0 * i1 + s1 * i0) / (s0 + s1)


def ber_from_levels(i1: float, i0: float, s1: float, s0: float, i_th: float) -> float:
    """Gaussian-noise BER for equiprobable bits and decision threshold i_th."""
    if s1 <= 0 or s0 <= 0:
        return 0.0 if (i1 > i_th > i0) else 0.5
    return float(0.25 * (erfc((i1 - i_th) / (s1 * math.sqrt(2.0))) + erfc((i_th - i0) / (s0 * math.sqrt(2.0)))))


def q_factor(i1: float, i0: float, s1: float, s0: float) -> float:
    return (i1 - i0) / (s1 + s0) if (s1 + s0) > 0 else float("inf")


def ber_from_q(q: float) -> float:
    return float(0.5 * erfc(q / math.sqrt(2.0)))


def sample_indices(n_bits: int, sps: int) -> np.ndarray:
    return np.arange(n_bits) * sps + sps // 2


def recover_bits(wave: np.ndarray, n_bits: int, sps: int, threshold: float) -> np.ndarray:
    return (wave[sample_indices(n_bits, sps)] > threshold).astype(int)


def eye_traces(wave: np.ndarray, sps: int, n_ui: int = 2, max_traces: int = 300, skip: int = 2):
    """Overlay traces spanning n_ui bit periods. Returns (time axis in UI, 2-D array)."""
    seg = n_ui * sps
    starts = np.arange(skip * sps, len(wave) - seg - sps, sps)[:max_traces]
    traces = np.array([wave[s: s + seg] for s in starts])
    return np.arange(seg) / sps, traces
