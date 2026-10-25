"""Receiver noise models (direct detection, one-sided power spectral densities, SI units).

Noise current variances are computed for an electrical noise bandwidth Be (Hz):

  thermal   : 4 k T F / R_L * Be                                   (A^2)
  shot      : 2 q (I_total + I_dark) * Be
  laser RIN : RIN * I_sig^2 * Be            (RIN in 1/Hz, I_sig = R P)
  signal-ASE beat : 4 R^2 P S_sp Be
  ASE-ASE beat    : 4 R^2 S_sp^2 Bo Be (1 - Be / 2 Bo)
  crosstalk : R^2 sum_j (T_ij P_j kappa_j)^2     (incoherent, other channels carry random data)

S_sp is the ASE PSD per polarisation (W/Hz), Bo the noise-equivalent optical bandwidth.
References: Agrawal, Fiber-Optic Communication Systems; Olsson, J. Lightwave Technol. 7 (1989).
"""
from __future__ import annotations

from typing import Callable, Dict

import numpy as np

from config import K_B, Q_E

NOISE_NAMES = ["thermal", "shot", "laser_rin", "ase", "crosstalk", "background"]
NOISE_LABELS = {
    "thermal": "Thermal", "shot": "Shot", "laser_rin": "Laser RIN", "ase": "Optical-amplifier ASE",
    "crosstalk": "Inter-channel crosstalk", "background": "Background optical noise",
    "jitter": "Timing jitter",
}


def thermal_psd(temperature_k: float, load_ohm: float, nf_lin: float = 1.0) -> float:
    """One-sided thermal-noise current PSD, A^2/Hz."""
    return 4.0 * K_B * temperature_k / load_ohm * nf_lin


def thermal_variance(temperature_k: float, load_ohm: float, be_hz: float, nf_lin: float = 1.0) -> float:
    return thermal_psd(temperature_k, load_ohm, nf_lin) * be_hz


def shot_variance(i_mean_a: float, i_dark_a: float, be_hz: float) -> float:
    return 2.0 * Q_E * (i_mean_a + i_dark_a) * be_hz


def rin_variance(i_sig_a: float, rin_lin_per_hz: float, be_hz: float) -> float:
    return rin_lin_per_hz * i_sig_a ** 2 * be_hz


def sig_ase_variance(R: float, p_w: float, s_sp: float, be_hz: float) -> float:
    return 4.0 * R ** 2 * p_w * s_sp * be_hz


def ase_ase_variance(R: float, s_sp: float, bo_hz: float, be_hz: float) -> float:
    return 4.0 * R ** 2 * s_sp ** 2 * bo_hz * be_hz * max(1.0 - be_hz / (2.0 * bo_hz), 0.0)


def level_variances(p_level_w: float, p: dict, flags: dict) -> Dict[str, float]:
    """Noise-current variance (A^2) of each source at one logic level (optical power p_level_w).

    A source whose flag is False contributes exactly zero (it is excluded from the total).
    p keys: R, Be, Bo, T, RL, F, Id, rin_lin, S_a, S_b, p_xt_mean, var_xt
    """
    R, Be, Bo = p["R"], p["Be"], p["Bo"]
    s_a = p["S_a"] if flags.get("ase") else 0.0
    s_b = p["S_b"] if flags.get("background") else 0.0
    i_sig = R * p_level_w
    i_xt = R * p["p_xt_mean"] if flags.get("crosstalk") else 0.0
    i_ase = R * 2.0 * (s_a + s_b) * Bo
    i_tot = i_sig + i_xt + i_ase
    comp = {k: 0.0 for k in NOISE_NAMES}
    if flags.get("thermal"):
        comp["thermal"] = thermal_variance(p["T"], p["RL"], Be, p["F"])
    if flags.get("shot"):
        comp["shot"] = shot_variance(i_tot, p["Id"], Be)
    if flags.get("laser_rin"):
        comp["laser_rin"] = rin_variance(i_sig, p["rin_lin"], Be)
    if s_a > 0:
        comp["ase"] = sig_ase_variance(R, p_level_w, s_a, Be) + \
            ase_ase_variance(R, s_a, Bo, Be) + \
            2.0 * (4.0 * R ** 2 * s_a * s_b * Bo * Be * max(1.0 - Be / (2.0 * Bo), 0.0))
    if s_b > 0:
        comp["background"] = sig_ase_variance(R, p_level_w, s_b, Be) + ase_ase_variance(R, s_b, Bo, Be)
    if flags.get("crosstalk"):
        comp["crosstalk"] = p["var_xt"]
    return comp


def noise_waveforms(p_rx_t: np.ndarray, i_unf: np.ndarray, p: dict, flags: dict, fs: float,
                    rng: np.random.Generator, filt: Callable[[np.ndarray], np.ndarray],
                    xt_unf: np.ndarray | None = None):
    """Time-domain noise-current waveforms (A), already passed through the receiver filter `filt`.

    White Gaussian noise with variance (one-sided PSD) * fs / 2 per sample is generated and filtered,
    so the filtered variance equals PSD * (noise-equivalent bandwidth). Disabled sources are all-zero.
    """
    n = len(p_rx_t)
    R, Bo = p["R"], p["Bo"]
    zero = np.zeros(n)
    s_a = p["S_a"] if flags.get("ase") else 0.0
    s_b = p["S_b"] if flags.get("background") else 0.0
    dc_xt = R * p["p_xt_mean"] if flags.get("crosstalk") else 0.0
    dc_ase = R * 2.0 * (s_a + s_b) * Bo
    i_tot_t = i_unf + dc_xt + dc_ase

    def white(var_t):
        return np.sqrt(np.maximum(var_t, 0.0)) * rng.standard_normal(n)

    comps = {k: zero for k in NOISE_NAMES}
    if flags.get("thermal"):
        comps["thermal"] = filt(white(np.full(n, thermal_psd(p["T"], p["RL"], p["F"]) * fs / 2.0)))
    if flags.get("shot"):
        comps["shot"] = filt(white(Q_E * (i_tot_t + p["Id"]) * fs))
    if flags.get("laser_rin"):
        comps["laser_rin"] = filt(white(p["rin_lin"] * i_unf ** 2 * fs / 2.0))
    if s_a > 0:
        comps["ase"] = filt(white(2.0 * R ** 2 * p_rx_t * s_a * fs + 2.0 * R ** 2 * s_a ** 2 * Bo * fs))
    if s_b > 0:
        comps["background"] = filt(white(2.0 * R ** 2 * p_rx_t * s_b * fs + 2.0 * R ** 2 * s_b ** 2 * Bo * fs))
    if flags.get("crosstalk") and xt_unf is not None:
        comps["crosstalk"] = filt(xt_unf - xt_unf.mean())
    combined = sum(comps.values()) if comps else zero
    return comps, combined, dc_xt + dc_ase
