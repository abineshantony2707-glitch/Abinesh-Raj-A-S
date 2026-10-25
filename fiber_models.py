"""Optical fibre models: geometry, attenuation, chromatic / modal / polarisation-mode dispersion.

Simplified, physically interpretable models. See README for equations and limitations.
"""
from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np

from config import C, FiberConfig

SM_V_CUTOFF = 2.405   # first zero of J0: single-mode condition for a step-index fibre

Issue = Tuple[str, str]   # (level, message) with level in {"error", "warning", "info"}


def _p(core, clad, n1, n2, A, B, oh=0.0, pmd=0.0, emb=0.0, lam0=1313.0, s0=0.092,
       status="ok", note=""):
    return dict(core_um=core, clad_um=clad, n_core=n1, n_clad=n2, A=A, B=B, oh=oh, pmd=pmd,
                emb=emb, lam0=lam0, s0=s0, status=status, note=note)


_SM_SILICA = dict(core=8.2, clad=125.0, n1=1.4682, n2=1.4633)

FIBER_PRESETS: Dict[Tuple[str, str, str], dict] = {
    ("Silica", "Single-mode", "Step-index"): _p(
        8.2, 125.0, 1.4682, 1.4633, 0.75, 0.04, oh=0.12, pmd=0.1,
        note="G.652-like single-mode fibre (representative values)."),
    ("Silica", "Single-mode", "Graded-index"): _p(
        8.2, 125.0, 1.4682, 1.4633, 0.75, 0.04, oh=0.12, pmd=0.1, status="warn",
        note="Graded-index profiles are not used in conventional single-mode fibre; "
             "V-number and dispersion are computed as for a step-index core (approximation)."),
    ("Silica", "Multimode", "Step-index"): _p(
        50.0, 125.0, 1.4800, 1.4636, 1.3, 0.04, oh=0.10, lam0=1330.0,
        note="Step-index silica multimode fibre (representative; large modal dispersion)."),
    ("Silica", "Multimode", "Graded-index"): _p(
        50.0, 125.0, 1.4800, 1.4665, 1.3, 0.04, oh=0.10, emb=500.0, lam0=1330.0,
        note="OM2-like 50/125 graded-index fibre; effective modal bandwidth 500 MHz km assumed."),
    ("POF", "Single-mode", "Step-index"): _p(
        6.0, 125.0, 1.4920, 1.4898, 0.0, 0.0, status="invalid",
        note="Single-mode POF is not a commercial product (research only); parameters are illustrative."),
    ("POF", "Single-mode", "Graded-index"): _p(
        6.0, 125.0, 1.4920, 1.4898, 0.0, 0.0, status="invalid",
        note="Single-mode / graded-index POF combination is not a commercial product."),
    ("POF", "Multimode", "Step-index"): _p(
        980.0, 1000.0, 1.4920, 1.4020, 0.0, 0.0,
        note="PMMA step-index POF (core 980 um, NA about 0.51). Usable band roughly 450-800 nm."),
    ("POF", "Multimode", "Graded-index"): _p(
        500.0, 750.0, 1.4920, 1.4780, 0.0, 0.0, emb=150.0,
        note="PMMA graded-index POF, representative; modal bandwidth assumed."),
    ("PCS", "Single-mode", "Step-index"): _p(
        8.2, 125.0, 1.4682, 1.4633, 1.0, 3.5, status="invalid", pmd=0.1,
        note="PCS fibres have large cores (>= 100 um); single-mode PCS is physically unusual."),
    ("PCS", "Single-mode", "Graded-index"): _p(
        8.2, 125.0, 1.4682, 1.4633, 1.0, 3.5, status="invalid", pmd=0.1,
        note="PCS single-mode / graded-index combination is not a standard product."),
    ("PCS", "Multimode", "Step-index"): _p(
        200.0, 230.0, 1.4570, 1.4105, 1.0, 3.5, lam0=1310.0,
        note="Plastic-clad silica 200/230 um step-index fibre (polymer cladding), representative."),
    ("PCS", "Multimode", "Graded-index"): _p(
        200.0, 230.0, 1.4570, 1.4105, 1.0, 3.5, emb=100.0, lam0=1310.0, status="warn",
        note="Graded-index PCS is not a standard commercial product; modal bandwidth assumed."),
}

# valid operating windows (nm) for the attenuation models
WAVELENGTH_RANGE_NM = {"Silica": (600.0, 1700.0), "PCS": (600.0, 1650.0), "POF": (450.0, 800.0)}

# PMMA POF attenuation, dB/km (representative shape of published spectra - ASSUMED)
POF_WL = np.array([450.0, 500.0, 520.0, 570.0, 600.0, 650.0, 700.0, 750.0, 800.0])
POF_AT = np.array([200.0, 120.0, 100.0, 90.0, 110.0, 150.0, 400.0, 1200.0, 3000.0])
POF_OPAQUE_DB_KM = 20000.0   # beyond ~850 nm PMMA is effectively opaque for km links


def preset(material: str, mode: str, profile: str) -> dict:
    return FIBER_PRESETS[(material, mode, profile)]


def defaults_for(material: str, mode: str, profile: str) -> dict:
    """Default FiberConfig values (and operating wavelength) for a material/mode/profile choice."""
    p = preset(material, mode, profile)
    if material == "POF":
        wl = 650.0
    elif material == "PCS":
        wl = 850.0
    elif mode == "Multimode":
        wl = 850.0
    else:
        wl = 1550.0
    return dict(core_um=p["core_um"], clad_um=p["clad_um"], n_core=p["n_core"], n_clad=p["n_clad"],
                pmd_ps_sqrtkm=p["pmd"], modal_bw_mhz_km=p["emb"], wavelength_nm=wl)


# ------------------------------------------------------------------------- geometry
def numerical_aperture(n_core: float, n_clad: float) -> float:
    return math.sqrt(max(n_core ** 2 - n_clad ** 2, 0.0))


def v_number(core_um: float, na: float, wl_nm: float) -> float:
    """Normalised frequency V = 2 pi a NA / lambda."""
    return 2.0 * math.pi * (core_um / 2.0) * na / (wl_nm / 1000.0)


def cutoff_wavelength_nm(core_um: float, na: float) -> float:
    """Wavelength below which a step-index fibre supports more than one mode."""
    return 2.0 * math.pi * (core_um / 2.0) * na / SM_V_CUTOFF * 1000.0


def num_modes(v: float, profile: str) -> float:
    """Approximate number of guided modes (large-V estimate; 1 for V <= 2.405)."""
    if v <= SM_V_CUTOFF:
        return 1.0
    return v ** 2 / 4.0 if profile == "Graded-index" else v ** 2 / 2.0


def fiber_geometry(f: FiberConfig, wl_nm: float | None = None) -> dict:
    wl = f.wavelength_nm if wl_nm is None else wl_nm
    na = numerical_aperture(f.n_core, f.n_clad)
    v = v_number(f.core_um, na, wl)
    return dict(na=na, v=v, modes=num_modes(v, f.profile),
                cutoff_nm=cutoff_wavelength_nm(f.core_um, na),
                delta=(f.n_core - f.n_clad) / f.n_core if f.n_core else 0.0,
                single_mode=v <= SM_V_CUTOFF)


# ------------------------------------------------------------------------- attenuation
def _atten_model(f: FiberConfig, wl_nm) -> np.ndarray:
    wl = np.atleast_1d(np.asarray(wl_nm, dtype=float))
    lam = wl / 1000.0
    p = preset(f.material, f.mode, f.profile)
    if f.material == "POF":
        a = np.exp(np.interp(wl, POF_WL, np.log(POF_AT)))
        a = np.where(wl > 850.0, POF_OPAQUE_DB_KM, a)
        return a
    ir = 7.81e11 * np.exp(-48.48 / lam)                      # infrared absorption tail
    oh = p["oh"] * np.exp(-0.5 * ((lam - 1.383) / 0.025) ** 2)  # OH absorption peak
    return p["A"] / lam ** 4 + p["B"] + ir + oh


def attenuation_db_km(f: FiberConfig, wl_nm):
    """Attenuation coefficient in dB/km at wavelength(s) wl_nm (wavelength dependent)."""
    scalar = np.ndim(wl_nm) == 0
    a = _atten_model(f, wl_nm)
    if f.atten_override_db_km is not None:
        ref = _atten_model(f, f.wavelength_nm)[0]
        a = a * (f.atten_override_db_km / ref if ref > 0 else 1.0)
    return float(a[0]) if scalar else a


# ------------------------------------------------------------------------- dispersion
def _d_model(f: FiberConfig, wl_nm) -> np.ndarray:
    wl = np.atleast_1d(np.asarray(wl_nm, dtype=float))
    p = preset(f.material, f.mode, f.profile)
    if f.material == "POF":
        return np.full_like(wl, -300.0)
    return p["s0"] / 4.0 * (wl - p["lam0"] ** 4 / wl ** 3)


def chromatic_dispersion_ps_nm_km(f: FiberConfig, wl_nm):
    """Chromatic dispersion parameter D in ps/(nm km)."""
    scalar = np.ndim(wl_nm) == 0
    d = _d_model(f, wl_nm)
    if f.disp_override_ps_nm_km is not None:
        d = d + (f.disp_override_ps_nm_km - _d_model(f, f.wavelength_nm)[0])
    return float(d[0]) if scalar else d


def modal_dispersion_rms_ps(f: FiberConfig, length_km: float | None = None,
                            wl_nm: float | None = None) -> float:
    """RMS modal pulse broadening (ps). Zero when the fibre is physically single-mode."""
    L = f.length_km if length_km is None else length_km
    g = fiber_geometry(f, wl_nm)
    if g["single_mode"]:
        return 0.0
    n1, delta = f.n_core, g["delta"]
    Lm = L * 1000.0
    if f.profile == "Step-index":
        s = Lm * n1 * delta / (2.0 * math.sqrt(3.0) * C)
    else:
        s_ideal = Lm * n1 * delta ** 2 / (20.0 * math.sqrt(3.0) * C)
        s_emb = 0.1874 / (f.modal_bw_mhz_km * 1e6 / L) if f.modal_bw_mhz_km > 0 and L > 0 else 0.0
        s = max(s_ideal, s_emb)
    return s * 1e12


def pmd_rms_ps(f: FiberConfig, length_km: float | None = None, wl_nm: float | None = None) -> float:
    L = f.length_km if length_km is None else length_km
    if not fiber_geometry(f, wl_nm)["single_mode"]:
        return 0.0
    return f.pmd_ps_sqrtkm * math.sqrt(max(L, 0.0))


def dispersion_budget(f: FiberConfig, wl_nm: float, bitrate_bps: float, source_rms_nm: float,
                      linewidth_hz: float, enabled: bool = True, comp_enabled: bool = False,
                      comp_fraction: float = 0.9, length_km: float | None = None) -> dict:
    """Pulse-broadening budget (RMS widths, ps) added by the fibre.

    Chromatic: sigma = |D| L sigma_lambda, with sigma_lambda the quadrature sum of the source
    chirp width, the laser-linewidth width and the transform-limited modulation width.
    Modal and PMD terms are added in quadrature (Gaussian-pulse approximation).
    """
    L = f.length_km if length_km is None else length_km
    tb = 1.0 / bitrate_bps
    sig_in = tb / math.sqrt(12.0)                       # RMS width of an ideal NRZ bit
    d_lw = (wl_nm ** 2 * 1e-18 / C) * linewidth_hz * 1e9
    d_mod = (wl_nm ** 2 * 1e-18) / (4.0 * math.pi * C * sig_in) * 1e9
    sig_lam = math.sqrt(source_rms_nm ** 2 + d_lw ** 2 + d_mod ** 2)
    D = float(chromatic_dispersion_ps_nm_km(f, wl_nm))
    s_chrom = abs(D) * L * sig_lam
    s_modal = modal_dispersion_rms_ps(f, L, wl_nm)
    s_pmd = pmd_rms_ps(f, L, wl_nm)
    if not enabled:
        s_chrom = s_modal = s_pmd = 0.0
    s_chrom_c = s_chrom * (1.0 - comp_fraction) if (comp_enabled and enabled) else s_chrom
    raw = math.sqrt(s_chrom ** 2 + s_modal ** 2 + s_pmd ** 2)
    comp = math.sqrt(s_chrom_c ** 2 + s_modal ** 2 + s_pmd ** 2)
    pen = lambda s_ps: 5.0 * math.log10(1.0 + 2.0 * math.pi * (bitrate_bps * s_ps * 1e-12) ** 2)
    return dict(D=D, sigma_lambda_nm=sig_lam, chrom_ps=s_chrom, modal_ps=s_modal, pmd_ps=s_pmd,
                chrom_comp_ps=s_chrom_c, raw_ps=raw, extra_ps=comp,
                sigma_in_ps=sig_in * 1e12, sigma_out_ps=math.sqrt((sig_in * 1e12) ** 2 + comp ** 2),
                penalty_db=pen(comp), penalty_raw_db=pen(raw))


# ------------------------------------------------------------------------- validation
def validate_fiber(f: FiberConfig, channel_wl_nm: List[float] | None = None) -> List[Issue]:
    """Return configuration issues. 'error' items make the model meaningless; 'warning' items are caveats."""
    out: List[Issue] = []
    key = (f.material, f.mode, f.profile)
    p = FIBER_PRESETS[key]
    if p["status"] == "invalid":
        out.append(("warning", f"Invalid/unusual combination {key}: {p['note']}"))
    elif p["status"] == "warn":
        out.append(("warning", p["note"]))
    if f.clad_um <= f.core_um:
        out.append(("error", "Cladding diameter must be larger than the core diameter."))
    if f.n_core <= f.n_clad:
        out.append(("error", "Core index must exceed cladding index for guiding (n_core > n_clad)."))
        return out
    wls = [f.wavelength_nm] + list(channel_wl_nm or [])
    lo, hi = WAVELENGTH_RANGE_NM[f.material]
    bad = [w for w in wls if not lo <= w <= hi]
    if bad:
        out.append(("warning", f"{len(bad)} wavelength(s) outside the {f.material} model range "
                               f"{lo:.0f}-{hi:.0f} nm (e.g. {bad[0]:.1f} nm): attenuation/dispersion unreliable."))
    na = numerical_aperture(f.n_core, f.n_clad)
    if na > 0.6:
        out.append(("warning", f"Numerical aperture {na:.2f} is unusually large."))
    shortest = min(wls)
    g = fiber_geometry(f, shortest)
    if f.mode == "Single-mode" and not g["single_mode"]:
        out.append(("warning", f"Labelled single-mode, but V = {g['v']:.2f} > 2.405 at {shortest:.0f} nm "
                               f"(cutoff {g['cutoff_nm']:.0f} nm): the fibre is physically multimode there, "
                               f"so modal dispersion is applied."))
    if f.mode == "Multimode" and g["single_mode"]:
        out.append(("warning", f"Labelled multimode, but V = {g['v']:.2f} <= 2.405: the fibre is "
                               f"physically single-mode at this wavelength."))
    if f.profile == "Graded-index" and f.mode == "Multimode" and f.modal_bw_mhz_km <= 0:
        out.append(("info", "Graded-index fibre without an effective modal bandwidth: ideal parabolic "
                            "profile assumed (optimistic)."))
    if f.length_km <= 0:
        out.append(("error", "Fibre length must be positive."))
    return out
