"""Link budget and per-channel performance (SNR, Q-factor, BER, eye, dispersion penalty).

Pipeline per channel:  P_launch -> MUX -> [booster] -> fibre (+ in-line amps) -> [DCF] -> [pre-amp]
                       -> DEMUX -> photodiode -> receiver filter -> decision.

Definitions
  SNR (dB) = 10 log10( ((I1 - I0)/2)^2 / ((var1 + var0)/2) )   (electrical, at the decision instant)
  Q        = (I1 - I0)/(sigma1 + sigma0),   BER = 0.5 erfc(Q / sqrt 2)   (Gaussian approximation)
For equal sigma, SNR = Q^2. BER values below ~1e-15 are reported as bounds, never as exactly zero.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import amplifier as amp_m
import fiber_models as fm
import noise_models as nm
import transmitter_receiver as trx
import wdm_system as ws
from config import BER_FAIL, BER_OK, C, SystemConfig

BER_REPORT_FLOOR = 1e-15


def dbm_to_w(dbm):
    return 1e-3 * 10.0 ** (np.asarray(dbm, dtype=float) / 10.0)


def w_to_dbm(w):
    return 10.0 * np.log10(np.maximum(np.asarray(w, dtype=float), 1e-30) / 1e-3)


def variant(cfg: SystemConfig, **changes) -> SystemConfig:
    """Deep copy of cfg with section fields replaced, e.g. variant(cfg, fiber={'length_km': 50})."""
    c = copy.deepcopy(cfg)
    for section, kv in changes.items():
        for k, v in kv.items():
            setattr(getattr(c, section), k, v)
    return c


def noise_flags(cfg: SystemConfig) -> dict:
    n = cfg.noise
    return dict(thermal=n.thermal, shot=n.shot, laser_rin=n.laser_rin, ase=n.ase,
                crosstalk=n.crosstalk, background=n.background, jitter=n.jitter)


# ============================================================================= link budget
@dataclass
class LinkResult:
    n: int
    amp_on: bool
    lam_nm: np.ndarray
    nu_hz: np.ndarray
    failed: np.ndarray
    p_launch_w: np.ndarray
    p_rx_w: np.ndarray
    s_sp_rx: np.ndarray            # ASE PSD per polarisation at the receiver input (W/Hz)
    alpha_db_km: np.ndarray
    fiber_db: np.ndarray
    conn_db: float
    splice_db: float
    margin_db: float
    dcf_db: float
    mux_db: float
    demux_db: float
    passive_loss_db: np.ndarray    # all passive losses (dB, positive)
    amps: List[dict] = field(default_factory=list)
    first_in_w: Optional[np.ndarray] = None
    first_out_w: Optional[np.ndarray] = None
    profile_z: np.ndarray = None
    profile_p: np.ndarray = None   # (points, channels), W
    bw3_hz: np.ndarray = None
    bo_hz: np.ndarray = None
    T: np.ndarray = None
    stages: List[dict] = field(default_factory=list)


def compute_link(cfg: SystemConfig, amp_enabled: bool | None = None) -> LinkResult:
    f, w, a, d = cfg.fiber, cfg.wdm, cfg.amp, cfg.disp
    amp_on = a.enabled if amp_enabled is None else amp_enabled
    lam = ws.channel_wavelengths(w)
    n = len(lam)
    nu = C / (lam * 1e-9)
    failed_set = set(cfg.fault.failed_channels)
    failed = np.array([i in failed_set for i in range(n)], dtype=bool)
    p_launch = dbm_to_w(np.array(w.powers_dbm[:n], dtype=float))
    p_launch = np.where(failed, 0.0, p_launch)

    L = f.length_km
    alpha = np.atleast_1d(fm.attenuation_db_km(f, lam))
    fiber_db = alpha * L
    conn = f.n_connectors * f.connector_loss_db
    spl = math.floor(L / f.splice_spacing_km) * f.splice_loss_db if f.splice_spacing_km > 0 else 0.0
    marg = f.margin_db
    dcf = d.dcf_loss_db if (d.enabled and d.compensation and d.method == "DCF") else 0.0
    span_db = fiber_db + conn + spl + marg
    n_spans = a.n_inline + 1 if (amp_on and a.placement == "In-line") else 1

    stages: List[dict] = [dict(name="MUX", kind="loss", loss=np.full(n, w.mux_il_db))]
    if amp_on and a.placement == "Booster":
        stages.append(dict(name="Booster EDFA", kind="amp"))
    for s in range(n_spans):
        stages.append(dict(name="Fibre" if n_spans == 1 else f"Fibre span {s + 1}", kind="loss",
                           loss=span_db / n_spans, z1=L * (s + 1) / n_spans))
        if amp_on and a.placement == "In-line" and s < n_spans - 1:
            stages.append(dict(name=f"In-line EDFA {s + 1}", kind="amp"))
    if dcf > 0:
        stages.append(dict(name="DCF module", kind="loss", loss=np.full(n, dcf)))
    if amp_on and a.placement == "Pre-amplifier":
        stages.append(dict(name="Pre-amplifier EDFA", kind="amp"))
    stages.append(dict(name="DEMUX", kind="loss", loss=np.full(n, w.demux_il_db)))

    P = p_launch.copy()
    S = np.zeros(n)
    prof_z, prof_p = [0.0], [P.copy()]
    amps: List[dict] = []
    first_in = first_out = None
    for st in stages:
        if st["kind"] == "loss":
            t = 10.0 ** (-np.asarray(st["loss"], dtype=float) / 10.0)
            P = P * t
            S = S * t
            prof_z.append(st.get("z1", prof_z[-1]))
            prof_p.append(P.copy())
        else:
            g_db = amp_m.effective_gain_db(a.gain_db, float(P.sum()), a.psat_dbm, a.saturation)
            g = 10.0 ** (g_db / 10.0)
            S = S * g + amp_m.ase_psd_per_pol(g, a.nf_db, nu)
            if first_in is None:
                first_in = P.copy()
            P = P * g
            if first_out is None:
                first_out = P.copy()
            amps.append(dict(name=st["name"], gain_db=g_db, requested_gain_db=a.gain_db,
                             n_sp=amp_m.n_sp_from_nf(a.nf_db, g), saturated=g_db < a.gain_db - 1e-9,
                             z_km=prof_z[-1]))
            prof_z.append(prof_z[-1])
            prof_p.append(P.copy())

    bw3 = ws.demux_bw3db_hz(nu, w.demux_bw_frac)
    bo = ws.optical_noise_bandwidth_hz(bw3, w.demux_order)
    T = ws.demux_matrix(nu, bw3, w.demux_order, w.isolation_db)
    passive = span_db + dcf + w.mux_il_db + w.demux_il_db
    return LinkResult(n=n, amp_on=amp_on, lam_nm=lam, nu_hz=nu, failed=failed, p_launch_w=p_launch,
                      p_rx_w=P, s_sp_rx=S, alpha_db_km=alpha, fiber_db=fiber_db, conn_db=conn,
                      splice_db=spl, margin_db=marg, dcf_db=dcf, mux_db=w.mux_il_db,
                      demux_db=w.demux_il_db, passive_loss_db=passive, amps=amps,
                      first_in_w=first_in, first_out_w=first_out, profile_z=np.array(prof_z),
                      profile_p=np.array(prof_p), bw3_hz=bw3, bo_hz=bo, T=T, stages=stages)


# ============================================================================= channel evaluation
def _rx_noise_params(cfg: SystemConfig, link: LinkResult, i: int) -> dict:
    tx = cfg.txrx
    B = cfg.wdm.bitrates_gbps[i] * 1e9
    be = B * tx.rx_bw_frac * trx.nebw_factor(tx.rx_order, tx.rx_filter)
    bo = max(float(link.bo_hz[i]), 2.0 * be)
    lam_m = link.lam_nm[i] * 1e-9
    dnu_1nm = C * 1e-9 / lam_m ** 2
    s_b = 1e-3 * 10.0 ** (cfg.noise.bg_dbm_per_nm / 10.0) / dnu_1nm
    p1, p0 = trx.tx_levels(1.0, tx.extinction_db, tx.modulation)
    kappa = (p1 - p0) / 2.0
    others = ~link.failed
    others = others.copy()
    others[i] = False
    p_x = link.p_rx_w * others * link.T[i]
    R = tx.responsivity
    return dict(R=R, Be=be, Bo=bo, T=tx.temperature_k, RL=tx.load_ohm, F=10.0 ** (tx.rx_nf_db / 10.0),
                Id=tx.dark_current_na * 1e-9, rin_lin=10.0 ** (tx.rin_db_hz / 10.0),
                S_a=float(link.s_sp_rx[i]), S_b=s_b, p_xt_mean=float(p_x.sum()),
                var_xt=float(R ** 2 * np.sum((p_x * kappa) ** 2)), p_x=p_x, others=others)


def _levels(w_tx, sig_extra_ps, gain, tx, fs, bits, idx, jitter_rng=None, jitter_on=False):
    """Noiseless photocurrent at the decision instants. Returns dict with levels and waveforms."""
    sps = tx.sps
    w = trx.gaussian_filter(w_tx, sig_extra_ps * 1e-12 * fs)
    i_unf = tx.responsivity * gain * w
    i_cl = trx.receiver_filter(i_unf, tx.rx_bw_frac / sps, tx.rx_order, tx.rx_filter)
    if jitter_on and tx.jitter_ps > 0 and jitter_rng is not None:
        reps = 8
        delta = jitter_rng.normal(0.0, tx.jitter_ps * 1e-12 * fs, (reps, len(idx)))
        pos = np.clip(idx[None, :] + delta, 0, len(i_cl) - 1)
        samples = np.interp(pos.ravel(), np.arange(len(i_cl)), i_cl).reshape(reps, len(idx))
        bb = np.tile(bits, (reps, 1))
    else:
        samples, bb = i_cl[idx][None, :], bits[None, :]
    s1, s0 = samples[bb == 1], samples[bb == 0]
    i1, i0 = float(s1.mean()), float(s0.mean())
    eye = (float(s1.min()) - float(s0.max())) / (i1 - i0) if i1 > i0 else 0.0
    return dict(i1=i1, i0=i0, isi1=float(s1.std()), isi0=float(s0.std()), eye=max(eye, 0.0),
                i_cl=i_cl, i_unf=i_unf, w=w)


def _dead_metrics(cfg, link, i) -> dict:
    nan = float("nan")
    return dict(channel=i + 1, wavelength_nm=float(link.lam_nm[i]),
                bitrate_gbps=cfg.wdm.bitrates_gbps[i], failed=True, status="FAILED",
                p_launch_dbm=nan, p_rx_dbm=-120.0, passive_loss_db=float(link.passive_loss_db[i]),
                q=0.0, ber=0.5, ber_display="0.5 (channel down)", snr_db=-99.0, eye_open_pct=0.0,
                xt_db=nan, broadening_ps=nan, penalty_formula_db=nan, penalty_sim_db=nan,
                sim_ber=0.5, noise_std_ua={}, i1_ua=0.0, i0_ua=0.0, sigma1_ua=nan, sigma0_ua=nan,
                disp={})


def fmt_ber(ber: float) -> str:
    if ber <= 0:
        return f"< {BER_REPORT_FLOOR:.0e}"
    if ber < BER_REPORT_FLOOR:
        return f"< {BER_REPORT_FLOOR:.0e}"
    return f"{ber:.2e}"


def classify(m: dict, cfg: SystemConfig) -> str:
    if m.get("failed"):
        return "FAILED"
    if m["ber"] > BER_FAIL:
        return "FAILED"
    if m["ber"] > BER_OK or m["p_rx_dbm"] < cfg.txrx.sensitivity_dbm:
        return "DEGRADED"
    return "OK"


def evaluate_channel(cfg: SystemConfig, link: LinkResult, i: int, flags: dict | None = None,
                     with_waveforms: bool = False, link_noamp: LinkResult | None = None,
                     n_show_bits: int = 32) -> dict:
    """Full analytic (and optionally time-domain Monte-Carlo) evaluation of channel i."""
    flags = noise_flags(cfg) if flags is None else flags
    tx = cfg.txrx
    if link.failed[i]:
        return _dead_metrics(cfg, link, i)
    lam = float(link.lam_nm[i])
    B = cfg.wdm.bitrates_gbps[i] * 1e9
    sps = tx.sps
    fs = B * sps
    nb = max(tx.ber_sim_bits, 256) if with_waveforms else max(tx.n_bits, 128)
    bits = trx.generate_bits(nb, tx.seed + i, tx.use_seed)
    idx = trx.sample_indices(nb, sps)
    p1l, p0l = trx.tx_levels(float(link.p_launch_w[i]), tx.extinction_db, tx.modulation)
    w_tx = trx.build_tx_waveform(bits, sps, p1l, p0l, tx.modulation, tx.rise_frac)
    disp = fm.dispersion_budget(cfg.fiber, lam, B, tx.spectral_width_nm, tx.linewidth_mhz * 1e6,
                                cfg.disp.enabled, cfg.disp.compensation and cfg.disp.method in ("DSP", "DCF"),
                                cfg.disp.comp_fraction)
    gain = float(link.p_rx_w[i] / link.p_launch_w[i])
    jr = np.random.default_rng(tx.seed + 500 + i if tx.use_seed else None)
    lv = _levels(w_tx, disp["extra_ps"], gain, tx, fs, bits, idx, jr, flags.get("jitter", False))
    lv_ref = _levels(w_tx, 0.0, gain, tx, fs, bits, idx)

    p = _rx_noise_params(cfg, link, i)
    R = p["R"]
    p1_eff, p0_eff = lv["i1"] / R, max(lv["i0"], 0.0) / R
    c1 = nm.level_variances(p1_eff, p, flags)
    c0 = nm.level_variances(p0_eff, p, flags)
    var1 = sum(c1.values()) + lv["isi1"] ** 2
    var0 = sum(c0.values()) + lv["isi0"] ** 2
    s1, s0 = math.sqrt(var1), math.sqrt(var0)
    i1, i0 = lv["i1"], lv["i0"]
    th = trx.decision_threshold(i1, i0, s1, s0, tx.threshold)
    ber = min(trx.ber_from_levels(i1, i0, s1, s0, th), 0.5)
    q = max(trx.q_factor(i1, i0, s1, s0), 0.0)
    snr = 10.0 * math.log10(max(((i1 - i0) / 2.0) ** 2, 1e-60) / max((var1 + var0) / 2.0, 1e-60))
    d_ref, d_now = lv_ref["i1"] - lv_ref["i0"], i1 - i0
    pen_sim = 10.0 * math.log10(d_ref / d_now) if d_now > 0 and d_ref > 0 else float("inf")
    p_rx_i = float(link.p_rx_w[i])
    m = dict(channel=i + 1, wavelength_nm=lam, bitrate_gbps=B / 1e9, failed=False,
             p_launch_dbm=float(w_to_dbm(link.p_launch_w[i])), p_rx_dbm=float(w_to_dbm(p_rx_i)),
             passive_loss_db=float(link.passive_loss_db[i]), q=q, ber=ber, ber_display=fmt_ber(ber),
             snr_db=snr, eye_open_pct=100.0 * lv["eye"],
             xt_db=10.0 * math.log10(p["p_xt_mean"] / p_rx_i) if p["p_xt_mean"] > 0 and p_rx_i > 0 else -100.0,
             broadening_ps=disp["extra_ps"], penalty_formula_db=disp["penalty_db"],
             penalty_sim_db=max(pen_sim, 0.0), sim_ber=float("nan"),
             noise_std_ua={k: math.sqrt(v) * 1e6 for k, v in c1.items()},
             i1_ua=i1 * 1e6, i0_ua=i0 * 1e6, sigma1_ua=s1 * 1e6, sigma0_ua=s0 * 1e6,
             threshold_ua=th * 1e6, disp=disp)
    m["status"] = classify(m, cfg)
    if not with_waveforms:
        return m

    # ------------------------------------------------------------ time-domain Monte Carlo
    rng = np.random.default_rng(tx.seed + 1000 + i if tx.use_seed else None)
    fc_norm = tx.rx_bw_frac / sps

    def filt(x):
        return trx.receiver_filter(x, fc_norm, tx.rx_order, tx.rx_filter)

    xt_unf = None
    if flags.get("crosstalk") and p["others"].any():
        xt_unf = np.zeros(len(w_tx))
        for j in np.where(p["others"])[0]:
            if p["p_x"][j] <= 1e-6 * p_rx_i:
                continue
            bj = trx.generate_bits(nb, tx.seed + 17 + int(j), tx.use_seed)
            a1, a0 = trx.tx_levels(float(p["p_x"][j]), tx.extinction_db, tx.modulation)
            xt_unf += R * trx.build_tx_waveform(bj, sps, a1, a0, tx.modulation, tx.rise_frac)
    p_rx_t = lv["w"] * gain
    comps, combined, dc = nm.noise_waveforms(p_rx_t, lv["i_unf"], p, flags, fs, rng, filt, xt_unf)
    noisy = lv["i_cl"] + combined
    if flags.get("jitter") and tx.jitter_ps > 0:
        pos = np.clip(idx + rng.normal(0.0, tx.jitter_ps * 1e-12 * fs, len(idx)), 0, len(noisy) - 1)
        samp = np.interp(pos, np.arange(len(noisy)), noisy)
    else:
        samp = noisy[idx]
    rec = (samp > th).astype(int)
    errors = int(np.sum(rec != bits))
    m["sim_errors"] = errors
    m["sim_bits"] = nb
    m["sim_ber"] = errors / nb
    m["sim_ber_upper95"] = 3.0 / nb if errors == 0 else float("nan")
    ns = min(n_show_bits, nb)
    ns_pts = ns * sps
    t_ns = np.arange(ns_pts) / fs * 1e9
    g_noamp = float(link_noamp.p_rx_w[i] / link_noamp.p_launch_w[i]) if link_noamp is not None else gain
    wave = dict(
        t_ns=t_ns, bits=bits[:ns], rec=rec[:ns], sps=sps, n_show=ns, threshold_a=th,
        w_tx=w_tx[:ns_pts], p_rx_noamp=(lv["w"] * g_noamp)[:ns_pts], p_rx_amp=p_rx_t[:ns_pts],
        i_clean=lv["i_cl"][:ns_pts], noisy=noisy[:ns_pts], combined=combined[:ns_pts],
        comps={k: v[:ns_pts] for k, v in comps.items()}, dc_a=dc,
        eye_t=None, eye_traces=None, flags=flags,
        p_amp_in=(lv["w"] * float(link.first_in_w[i] / link.p_launch_w[i]))[:ns_pts] if link.first_in_w is not None else None,
        p_amp_out=(lv["w"] * float(link.first_out_w[i] / link.p_launch_w[i]))[:ns_pts] if link.first_out_w is not None else None,
    )
    wave["eye_t"], wave["eye_traces"] = trx.eye_traces(noisy[: 420 * sps], sps)
    wave["eye_t_clean"], wave["eye_traces_clean"] = trx.eye_traces(lv["i_cl"][: 420 * sps], sps)
    m["wave"] = wave
    return m


def evaluate_all(cfg: SystemConfig, link: LinkResult, flags: dict | None = None) -> pd.DataFrame:
    rows = []
    for i in range(link.n):
        m = evaluate_channel(cfg, link, i, flags)
        rows.append({k: v for k, v in m.items() if k not in ("noise_std_ua", "disp", "wave")})
    return pd.DataFrame(rows)


# ============================================================================= sweeps / comparisons
def ber_vs_snr_curve(snr_db_axis: np.ndarray) -> np.ndarray:
    q = np.sqrt(10.0 ** (np.asarray(snr_db_axis) / 10.0))
    return np.array([max(trx.ber_from_q(x), 1e-300) for x in q])


def sweep_length(cfg: SystemConfig, ch: int, lengths: np.ndarray) -> pd.DataFrame:
    rows = []
    for L in lengths:
        c = variant(cfg, fiber={"length_km": float(L)})
        for label, amp_on in (("No amplifier", False), ("With amplifier", True)):
            link = compute_link(c, amp_enabled=amp_on)
            m = evaluate_channel(c, link, ch)
            rows.append(dict(length_km=float(L), case=label, p_rx_dbm=m["p_rx_dbm"],
                             snr_db=m["snr_db"], q=m["q"], ber=max(m["ber"], 1e-300)))
    return pd.DataFrame(rows)


def sweep_spacing(cfg: SystemConfig, ch: int) -> pd.DataFrame:
    rows = []
    lam0 = ws.channel_wavelengths(cfg.wdm)          # centre the regular grid on the actual channels
    centre = {"center_nm": float(lam0.mean()), "cwdm_start_nm": float(lam0.min())}
    if cfg.wdm.scheme == "DWDM":
        values, key, unit = [12.5, 25.0, 37.5, 50.0, 75.0, 100.0, 150.0, 200.0], "dwdm_spacing_ghz", "GHz"
    else:
        values, key, unit = [2.5, 5.0, 10.0, 15.0, 20.0, 30.0], "cwdm_spacing_nm", "nm"
    for v in values:
        c = variant(cfg, wdm={key: v, "custom_enabled": False, **centre})
        link = compute_link(c)
        n = link.n
        qs = []
        for i in range(n):
            qs.append(evaluate_channel(c, link, i)["q"])
        mm = evaluate_channel(c, link, min(ch, n - 1))
        rows.append(dict(spacing=v, unit=unit, worst_q=min(qs), sel_q=mm["q"], sel_xt_db=mm["xt_db"],
                         sel_ber=max(mm["ber"], 1e-300)))
    return pd.DataFrame(rows)


def sweep_rate(cfg: SystemConfig, ch: int) -> pd.DataFrame:
    rows = []
    for r in (1.0, 2.5, 5.0, 10.0, 25.0, 40.0):
        c = variant(cfg, wdm={"bitrates_gbps": [r] * 16})
        link = compute_link(c)
        m = evaluate_channel(c, link, ch)
        rows.append(dict(rate_gbps=r, q=m["q"], snr_db=m["snr_db"], ber=max(m["ber"], 1e-300),
                         penalty_db=m["penalty_formula_db"], eye_pct=m["eye_open_pct"]))
    return pd.DataFrame(rows)


def compare_amplifier(cfg: SystemConfig, ch: int) -> pd.DataFrame:
    rows = []
    for label, on in (("Amplifier OFF", False), ("Amplifier ON", True)):
        link = compute_link(cfg, amp_enabled=on)
        m = evaluate_channel(cfg, link, ch)
        rows.append(dict(case=label, p_rx_dbm=m["p_rx_dbm"], snr_db=m["snr_db"], q=m["q"], ber=m["ber"],
                         ase_std_ua=m["noise_std_ua"].get("ase", 0.0) if m["noise_std_ua"] else 0.0))
    return pd.DataFrame(rows)


def compare_dispersion(cfg: SystemConfig, ch: int) -> pd.DataFrame:
    rows = []
    cases = [("Dispersion OFF", dict(enabled=False, compensation=False)),
             ("Dispersion ON, no compensation", dict(enabled=True, compensation=False)),
             (f"Dispersion ON + {cfg.disp.method} compensation", dict(enabled=True, compensation=True))]
    for label, kv in cases:
        c = variant(cfg, disp=kv)
        link = compute_link(c)
        m = evaluate_channel(c, link, ch)
        rows.append(dict(case=label, broadening_ps=m["broadening_ps"], eye_pct=m["eye_open_pct"],
                         penalty_db=m["penalty_formula_db"], p_rx_dbm=m["p_rx_dbm"], q=m["q"], ber=m["ber"]))
    return pd.DataFrame(rows)


def compare_noise(cfg: SystemConfig, ch: int, link: LinkResult) -> pd.DataFrame:
    base = noise_flags(cfg)
    off = {k: False for k in base}
    rows = [dict(case="No noise (ISI only)", **_q_ber(cfg, link, ch, off))]
    for k in list(nm.NOISE_NAMES) + ["jitter"]:
        fl = dict(off)
        fl[k] = True
        rows.append(dict(case=f"Only {nm.NOISE_LABELS[k]}", **_q_ber(cfg, link, ch, fl)))
    rows.append(dict(case="All configured sources", **_q_ber(cfg, link, ch, base)))
    return pd.DataFrame(rows)


def _q_ber(cfg, link, ch, flags) -> dict:
    m = evaluate_channel(cfg, link, ch, flags)
    return dict(q=m["q"], snr_db=m["snr_db"], ber=m["ber"])


# ============================================================================= pulse shapes
def pulse_shapes(cfg: SystemConfig, ch: int, sps: int = 32) -> dict:
    """Isolated '1' bit before / after fibre, without and with compensation (normalised)."""
    link = compute_link(cfg)
    lam = float(link.lam_nm[ch])
    B = cfg.wdm.bitrates_gbps[ch] * 1e9
    tx = cfg.txrx
    bits = np.zeros(16, dtype=int)
    bits[8] = 1
    p1, p0 = trx.tx_levels(1.0, 40.0, "NRZ-OOK")
    w = trx.build_tx_waveform(bits, sps, p1, p0, "NRZ-OOK", tx.rise_frac)
    fs = B * sps
    raw = fm.dispersion_budget(cfg.fiber, lam, B, tx.spectral_width_nm, tx.linewidth_mhz * 1e6, True, False)
    cmp_ = fm.dispersion_budget(cfg.fiber, lam, B, tx.spectral_width_nm, tx.linewidth_mhz * 1e6, True, True,
                                cfg.disp.comp_fraction)
    out = dict(t_ps=(np.arange(len(w)) / fs) * 1e12, input=w / w.max())
    peak = w.max()
    out["no_comp"] = trx.gaussian_filter(w, raw["raw_ps"] * 1e-12 * fs) / peak
    out["comp"] = trx.gaussian_filter(w, cmp_["extra_ps"] * 1e-12 * fs) / peak
    out["raw_ps"], out["comp_ps"] = raw["raw_ps"], cmp_["extra_ps"]
    return out