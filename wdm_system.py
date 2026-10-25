"""WDM channel grids (CWDM / DWDM), validation, and multiplexer / demultiplexer models."""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
from scipy.special import gamma as _gamma

from config import C, WDMConfig, FiberConfig
from fiber_models import WAVELENGTH_RANGE_NM

Issue = Tuple[str, str]


def wavelength_to_frequency_hz(wl_nm):
    return C / (np.asarray(wl_nm, dtype=float) * 1e-9)


def frequency_to_wavelength_nm(f_hz):
    return C / np.asarray(f_hz, dtype=float) * 1e9


def channel_wavelengths(w: WDMConfig) -> np.ndarray:
    """Channel wavelengths in nm, sorted ascending (channel 1 = shortest wavelength)."""
    n = int(w.n_channels)
    if w.custom_enabled:
        return np.sort(np.asarray(w.custom_wavelengths_nm[:n], dtype=float))
    if w.scheme == "CWDM":
        return w.cwdm_start_nm + w.cwdm_spacing_nm * np.arange(n, dtype=float)
    f0 = C / (w.center_nm * 1e-9)
    df = w.dwdm_spacing_ghz * 1e9
    f = f0 + (np.arange(n) - (n - 1) / 2.0) * df
    return np.sort(C / f * 1e9)


def channel_frequencies_hz(w: WDMConfig) -> np.ndarray:
    return wavelength_to_frequency_hz(channel_wavelengths(w))


def nominal_capacity_gbps(w: WDMConfig) -> float:
    """Total nominal data rate = sum of per-channel bit rates (NOT received throughput)."""
    return float(sum(w.bitrates_gbps[: int(w.n_channels)]))


def min_spacing_ghz(w: WDMConfig) -> float:
    f = np.sort(channel_frequencies_hz(w))
    return float(np.min(np.diff(f)) / 1e9) if len(f) > 1 else float("inf")


def validate_wdm(w: WDMConfig, fiber: FiberConfig | None = None, amp_enabled: bool = False) -> List[Issue]:
    out: List[Issue] = []
    n = int(w.n_channels)
    if n < 1 or n > 16:
        out.append(("error", "Number of channels must be between 1 and 16."))
        return out
    if w.custom_enabled and len(w.custom_wavelengths_nm) < n:
        out.append(("error", "Fewer custom wavelengths than channels."))
        return out
    lam = channel_wavelengths(w)
    if np.any(lam <= 0):
        out.append(("error", "Wavelengths must be positive."))
        return out
    if n > 1 and np.min(np.diff(lam)) < 1e-6:
        out.append(("error", "Duplicate channel wavelengths are not allowed."))
        return out
    rates = w.bitrates_gbps[:n]
    if any(r <= 0 for r in rates):
        out.append(("error", "Every channel needs a positive bit rate."))
    spacing = min_spacing_ghz(w)
    max_rate = max(rates) if rates else 10.0
    if spacing < max_rate:
        out.append(("error", f"Minimum channel spacing {spacing:.1f} GHz is below the bit rate "
                             f"({max_rate:.0f} Gb/s): channel spectra overlap."))
    elif spacing < 2 * max_rate:
        out.append(("warning", f"Spacing {spacing:.1f} GHz is below about 2x the bit rate "
                               f"({2 * max_rate:.0f} GHz): expect strong inter-channel crosstalk."))
    if not w.custom_enabled:
        if w.scheme == "CWDM":
            if n > 18:
                out.append(("error", "CWDM supports at most 18 channels (1271-1611 nm)."))
            if abs(w.cwdm_spacing_nm - 20.0) > 1.0:
                out.append(("warning", "ITU-T G.694.2 CWDM nominal spacing is 20 nm."))
            if lam[0] < 1260 or lam[-1] > 1625:
                out.append(("warning", "CWDM channels outside 1260-1625 nm (outside typical fibre bands)."))
            off = (lam[0] - 1271.0) / 20.0
            if abs(off - round(off)) > 0.05:
                out.append(("info", "Start wavelength is not on the ITU CWDM grid (1271 + 20 k nm)."))
        else:
            if w.dwdm_spacing_ghz not in (12.5, 25.0, 50.0, 100.0, 200.0):
                out.append(("info", "DWDM spacing is not a standard ITU-T G.694.1 value "
                                    "(12.5/25/50/100/200 GHz)."))
            if lam[0] < 1530.0 or lam[-1] > 1625.0:
                out.append(("warning", "DWDM channels leave the C+L band (1530-1625 nm)."))
            if amp_enabled and (lam[0] < 1528.0 or lam[-1] > 1565.0):
                out.append(("warning", "Channels leave the typical EDFA C-band gain window (1528-1565 nm); "
                                       "flat gain over all channels is not realistic."))
    if fiber is not None:
        lo, hi = WAVELENGTH_RANGE_NM[fiber.material]
        n_bad = int(np.sum((lam < lo) | (lam > hi)))
        if n_bad:
            out.append(("warning", f"{n_bad} channel(s) lie outside the {fiber.material} fibre model "
                                   f"range ({lo:.0f}-{hi:.0f} nm)."))
    bw = demux_bw3db_hz(wavelength_to_frequency_hz(lam), w.demux_bw_frac)
    if np.any(bw < 1.2 * np.array(rates + [rates[-1]] * (len(bw) - len(rates)))[: len(bw)] * 1e9):
        out.append(("warning", "Demux 3 dB bandwidth is narrower than ~1.2x the bit rate: filter-induced "
                               "ISI is not modelled, results are optimistic."))
    return out


# ------------------------------------------------------------------------- mux / demux
def demux_bw3db_hz(freqs_hz: np.ndarray, frac: float) -> np.ndarray:
    """Per-channel demux 3 dB bandwidth = frac x distance to the nearest neighbour channel."""
    f = np.asarray(freqs_hz, dtype=float)
    if len(f) == 1:
        return np.array([frac * 100e9])
    d = np.abs(f[:, None] - f[None, :])
    np.fill_diagonal(d, np.inf)
    return frac * d.min(axis=1)


def optical_noise_bandwidth_hz(bw3db_hz: np.ndarray, order: int) -> np.ndarray:
    """Noise-equivalent bandwidth of exp(-ln2 (2 f / BW)^(2 n)) = BW Gamma(1 + 1/2n) / ln2^(1/2n)."""
    k = 1.0 / (2.0 * order)
    return bw3db_hz * _gamma(1.0 + k) / np.log(2.0) ** k


def demux_matrix(freqs_hz: np.ndarray, bw3db_hz: np.ndarray, order: int, isolation_db: float) -> np.ndarray:
    """T[i, j] = relative power transmission of demux port i for the light of channel j (T[i, i] = 1)."""
    f = np.asarray(freqs_hz, dtype=float)
    df = f[None, :] - f[:, None]
    bw = np.asarray(bw3db_hz, dtype=float)[:, None]
    T = np.exp(-np.log(2.0) * (2.0 * np.abs(df) / bw) ** (2 * order))
    T = T + 10.0 ** (-isolation_db / 10.0) * (1.0 - np.eye(len(f)))
    np.fill_diagonal(T, 1.0)
    return T
