"""WDM Optical Communication Simulator - Streamlit dashboard.

Run:  streamlit run app.py
"""
from __future__ import annotations

import io
import json
from dataclasses import fields
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

import fiber_models as fm
import performance as perf
import visualizations as viz
import wdm_system as ws
import noise_models as nm
import transmitter_receiver as trx
from config import (BER_OK, MAX_CH, PARAMETER_PROVENANCE, SystemConfig, _SECTIONS)
from simulation import run_simulation

st.set_page_config(page_title="WDM Optical Communication Simulator", layout="wide", page_icon="🔆")

DEFAULT = SystemConfig()
LIST_FIELDS = {("wdm", "custom_wavelengths_nm"), ("wdm", "bitrates_gbps"), ("wdm", "powers_dbm"),
               ("traffic", "demand_gbps"), ("traffic", "priority"), ("traffic", "group"),
               ("fault", "failed_channels")}
OPTIONAL_FIELDS = {("fiber", "atten_override_db_km"), ("fiber", "disp_override_ps_nm_km")}
STATUS_ICON = {"OK": "🟢", "DEGRADED": "🟠", "FAILED": "🔴"}



_plot_counter = [0]


def plot(container, fig, **kw):
    """Render a Plotly figure with an automatically unique key (the same figure may appear in two tabs)."""
    _plot_counter[0] += 1
    kw.setdefault("width", "stretch")
    kw.setdefault("theme", None)
    return container.plotly_chart(fig, key=f"plot_{_plot_counter[0]}", **kw)


PRETTY = {
    "channel": "Channel", "wavelength_nm": "Wavelength (nm)", "bitrate_gbps": "Bit rate (Gb/s)",
    "p_launch_dbm": "Launch power (dBm)", "p_rx_dbm": "Received power (dBm)", "passive_loss_db": "Total passive loss (dB)",
    "q": "Q-factor", "ber": "BER (numeric)", "ber_display": "BER", "snr_db": "SNR (dB)",
    "eye_open_pct": "Eye opening (%)", "xt_db": "Crosstalk level (dB)", "broadening_ps": "Pulse broadening (ps)",
    "penalty_formula_db": "Dispersion penalty, formula (dB)", "penalty_sim_db": "Dispersion penalty, simulated (dB)",
    "sim_ber": "Monte-Carlo BER", "failed": "Failed", "status": "Status", "alert": "Alert",
    "i1_ua": "Current at logic 1 (uA)", "i0_ua": "Current at logic 0 (uA)", "sigma1_ua": "Noise RMS at logic 1 (uA)",
    "sigma0_ua": "Noise RMS at logic 0 (uA)", "threshold_ua": "Decision threshold (uA)",
    "priority": "Priority", "group": "Group", "demand_gbps": "Demand (Gb/s)", "capacity_gbps": "Capacity (Gb/s)",
    "placed_gbps": "Placed traffic (Gb/s)", "carried_gbps": "Carried traffic (Gb/s)",
    "goodput_gbps": "Goodput (Gb/s)", "utilisation_pct": "Utilisation (%)", "offered_util_pct": "Offered utilisation (%)",
    "dropped_gbps": "Dropped traffic (Gb/s)", "latency_us": "Latency (us)", "state": "State",
}


def pretty(df: pd.DataFrame) -> pd.DataFrame:
    """Replace code-style column names (with underscores) by readable names with units."""
    return df.rename(columns=lambda c: PRETTY.get(c, str(c).replace("_", " ").capitalize()))


# ============================================================================ state handling
def init_state(force: bool = False):
    s = st.session_state
    for sec, cls in _SECTIONS.items():
        d = getattr(DEFAULT, sec)
        for f in fields(cls):
            key = f"{sec}.{f.name}"
            if (sec, f.name) in LIST_FIELDS or (sec, f.name) in OPTIONAL_FIELDS:
                continue
            if force or key not in s:
                s[key] = getattr(d, f.name)
    if force or "wdm_df" not in s:
        s["wdm_df"] = pd.DataFrame({
            "Bit rate (Gb/s)": DEFAULT.wdm.bitrates_gbps, "Launch power (dBm)": DEFAULT.wdm.powers_dbm,
            "Custom wavelength (nm)": DEFAULT.wdm.custom_wavelengths_nm})
    if force or "traffic_df" not in s:
        s["traffic_df"] = pd.DataFrame({
            "Demand (Gb/s)": DEFAULT.traffic.demand_gbps, "Priority (1=high)": DEFAULT.traffic.priority,
            "Group": DEFAULT.traffic.group})
    if force or "fault.failed_channels" not in s:
        s["fault.failed_channels"] = []
    for k, v in {"ui.atten_manual": False, "ui.atten_value": 0.2, "ui.disp_manual": False, "ui.disp_value": 17.0,
                 "applied_cfg": None, "auto_update": True, "ui.op_auto": True, "fiber.length_slider": 20.0,
                 "fiber.length_number": 20.0, "selected_ch": 1}.items():
        if force or k not in s:
            s[k] = v


def reset_defaults():
    for k in list(st.session_state.keys()):
        if k.startswith(("fiber.", "wdm.", "txrx.", "noise.", "amp.", "disp.", "traffic.", "fault.", "ui.")) \
                or k in ("wdm_df", "traffic_df", "applied_cfg", "selected_ch"):
            del st.session_state[k]
    init_state(force=True)


def collect_config() -> SystemConfig:
    s = st.session_state
    parts = {}
    for sec, cls in _SECTIONS.items():
        kw = {}
        for f in fields(cls):
            key = f"{sec}.{f.name}"
            if key in s and (sec, f.name) not in LIST_FIELDS:
                kw[f.name] = s[key]
        parts[sec] = cls(**kw)
    fb = parts["fiber"]
    fb.atten_override_db_km = float(s["ui.atten_value"]) if s["ui.atten_manual"] else None
    fb.disp_override_ps_nm_km = float(s["ui.disp_value"]) if s["ui.disp_manual"] else None
    w, df = parts["wdm"], s["wdm_df"]
    w.bitrates_gbps = [float(x) for x in df["Bit rate (Gb/s)"]]
    w.powers_dbm = [float(x) for x in df["Launch power (dBm)"]]
    w.custom_wavelengths_nm = [float(x) for x in df["Custom wavelength (nm)"]]
    t, tdf = parts["traffic"], s["traffic_df"]
    t.demand_gbps = [float(x) for x in tdf["Demand (Gb/s)"]]
    t.priority = [int(x) for x in tdf["Priority (1=high)"]]
    t.group = [int(x) for x in tdf["Group"]]
    parts["fault"].failed_channels = sorted(int(x) for x in s["fault.failed_channels"])
    if s.get("ui.op_auto", True):                  # operating wavelength = average of the WDM channels
        try:
            fb.wavelength_nm = float(np.clip(np.mean(ws.channel_wavelengths(w)), 400.0, 1700.0))
        except Exception:
            pass
    n = int(parts["wdm"].n_channels)
    return SystemConfig(**parts, selected_channel=min(int(s["selected_ch"]) - 1, max(n - 1, 0)))


@st.cache_data(show_spinner="Running simulation ...", max_entries=8)
def cached_run(cfg_json: str):
    return run_simulation(SystemConfig.from_json(cfg_json))


# ============================================================================ widget helpers
def num(label, key, lo, hi, step, help=None, fmt=None, container=None):
    c = container or st
    return c.number_input(label, min_value=lo, max_value=hi, step=step, key=key, help=help, format=fmt)


def issues_box(issues):
    for level, msg in issues:
        {"error": st.error, "warning": st.warning, "info": st.info}[level](msg)


def edit_table(df_key: str, n: int, column_config: dict, key: str, on_change=None):
    full = st.session_state[df_key]
    edited = st.data_editor(full.iloc[:n].reset_index(drop=True), column_config=column_config, key=key,
                            num_rows="fixed", width="stretch", hide_index=False, on_change=on_change)
    for c in edited.columns:
        full.loc[: n - 1, c] = edited[c].values
    st.session_state[df_key] = full


def on_fiber_choice():
    s = st.session_state
    d = fm.defaults_for(s["fiber.material"], s["fiber.mode"], s["fiber.profile"])
    for k, v in d.items():
        s[f"fiber.{k}"] = v
    s["ui.atten_manual"] = False
    s["ui.disp_manual"] = False


def sync_len_slider():
    st.session_state["fiber.length_km"] = float(st.session_state["fiber.length_slider"])
    st.session_state["fiber.length_number"] = float(st.session_state["fiber.length_slider"])


def sync_len_number():
    v = float(min(max(st.session_state["fiber.length_number"], 0.1), 400.0))
    st.session_state["fiber.length_km"] = v
    st.session_state["fiber.length_slider"] = float(min(max(v, 1.0), 100.0))


def on_wdm_edit():
    """Editing a custom wavelength cell turns 'Custom wavelengths' on (otherwise the edit would be ignored)."""
    ed = st.session_state.get("wdm_editor") or {}
    rows = ed.get("edited_rows", {}) if isinstance(ed, dict) else {}
    sig = tuple(sorted((int(r), float(v["Custom wavelength (nm)"])) for r, v in rows.items()
                       if "Custom wavelength (nm)" in v and v["Custom wavelength (nm)"] is not None))
    if sig and sig != st.session_state.get("_custom_sig"):
        st.session_state["wdm.custom_enabled"] = True
    st.session_state["_custom_sig"] = sig


def fault_trigger(idx: int):
    cur = set(st.session_state["fault.failed_channels"])
    cur.add(idx)
    st.session_state["fault.failed_channels"] = sorted(cur)


def fault_restore():
    st.session_state["fault.failed_channels"] = []


# ============================================================================ sidebar
_plot_counter[0] = 0
init_state()
st.sidebar.title("🔆 WDM Simulator")
st.sidebar.caption("Physics-based, simplified model. Results are simulated, not hardware measurements.")
run_clicked = st.sidebar.button("▶ Run Simulation", type="primary", width="stretch")
st.sidebar.checkbox("Auto-update when settings change", key="auto_update",
                    help="When off, results only change after pressing Run Simulation.")
st.sidebar.button("↺ Reset to Defaults", width="stretch", on_click=reset_defaults)
cfg_now = collect_config()
n_ch = int(cfg_now.wdm.n_channels)
for _k in ("selected_ch", "ui.victim", "ui.backup_ch"):
    if _k in st.session_state and st.session_state[_k] > n_ch:
        st.session_state[_k] = n_ch
st.sidebar.selectbox("Channel inspected in detail", list(range(1, n_ch + 1)), key="selected_ch")
st.sidebar.checkbox("Reproducible random seed", key="txrx.use_seed")
st.sidebar.number_input("Seed", 0, 10 ** 6, key="txrx.seed", step=1)
cfg_now = collect_config()

if run_clicked or st.session_state["auto_update"] or st.session_state["applied_cfg"] is None:
    st.session_state["applied_cfg"] = cfg_now.to_json()
cfg: SystemConfig = SystemConfig.from_json(st.session_state["applied_cfg"])
if st.session_state["applied_cfg"] != cfg_now.to_json():
    st.sidebar.warning("Settings changed - press Run Simulation to update results.")

blocking = [m for lvl, m in (fm.validate_fiber(cfg.fiber) + ws.validate_wdm(cfg.wdm, cfg.fiber, cfg.amp.enabled))
            if lvl == "error"]
if blocking:
    st.error("The simulation cannot run with this configuration:")
    for m in blocking:
        st.error(m)
    st.stop()
res = cached_run(cfg.to_json())
n = res.link.n
ch = res.ch
sel = res.sel
if res.all_failed:
    st.error("All channels have failed - there is no optical signal to analyse.")
    st.button("♻ Restore failed channels", on_click=fault_restore)
    st.stop()
if res.ch_note:
    st.info(res.ch_note)

st.markdown(
    """
    <style>
    html { font-size: 112.5%; }                                   /* all text a little larger */
    button[data-baseweb="tab"] p { font-size: 1.12rem; font-weight: 600; }
    [data-testid="stMetricValue"] { font-size: 2.1rem; }
    [data-testid="stMetricLabel"] p { font-size: 1.0rem; }
    .wdm-title { font-weight: 800; font-size: 2.7rem; line-height: 1.15; margin: 0 0 0.6rem 0; }
    </style>
    <div class="wdm-title">Wavelength Division Multiplexing (WDM)</div>
    """, unsafe_allow_html=True)

TABS = ["Overview", "Fibre", "WDM", "Tx/Rx", "Noise", "Amplifier", "Dispersion",
        "Signals", "Performance", "Traffic", "Failure", "Summary"]
tabs = st.tabs(TABS)

# ============================================================================ 1 overview
with tabs[0]:
    st.subheader("System overview")
    plot(st, viz.fig_block_diagram(cfg, res.link, res.table, res.pipeline), width="stretch")
    ok = int((res.table.status == "OK").sum())
    deg = int((res.table.status == "DEGRADED").sum())
    fail = int((res.table.status == "FAILED").sum())
    c = st.columns(5)
    c[0].metric("Nominal capacity", f"{res.nominal_gbps:.0f} Gb/s", help="Sum of channel bit rates")
    c[1].metric("Received goodput", f"{res.received_gbps:.1f} Gb/s",
                help="Traffic actually carried after congestion, failures and packet errors")
    c[2].metric("Channels OK / degraded / failed", f"{ok} / {deg} / {fail}")
    worst = res.table[res.table.status != "FAILED"]
    c[3].metric("Worst-channel Q", f"{worst.q.min():.1f}" if len(worst) else "n/a")
    c[4].metric(f"Ch{ch + 1} Rx power", f"{sel['p_rx_dbm']:.1f} dBm")
    issues_box(res.issues)
    st.dataframe(pretty(res.status_table), width="stretch", hide_index=True)
    st.caption("Nominal capacity is the sum of channel rates; received goodput also reflects traffic demand, "
               "congestion, packet errors and failed channels.")

# ============================================================================ 2 fibre
with tabs[1]:
    st.subheader("Fibre configuration")
    c1, c2, c3 = st.columns(3)
    c1.selectbox("Material", ["Silica", "POF", "PCS"], key="fiber.material", on_change=on_fiber_choice,
                 help="Silica glass, plastic optical fibre (PMMA) or plastic-clad silica")
    c2.selectbox("Structure", ["Single-mode", "Multimode"], key="fiber.mode", on_change=on_fiber_choice)
    c3.selectbox("Refractive-index profile", ["Step-index", "Graded-index"], key="fiber.profile",
                 on_change=on_fiber_choice)
    st.caption(fm.preset(cfg_now.fiber.material, cfg_now.fiber.mode, cfg_now.fiber.profile)["note"])
    cl1, cl2 = st.columns([3, 1])
    cl1.slider("Fibre length (km)", 1.0, 100.0, key="fiber.length_slider", step=1.0, on_change=sync_len_slider)
    cl2.number_input("Length (km, numeric)", 0.1, 400.0, key="fiber.length_number", step=1.0, on_change=sync_len_number)
    g1, g2, g3, g4 = st.columns(4)
    num("Core diameter (um)", "fiber.core_um", 1.0, 2000.0, 0.5, container=g1)
    num("Cladding diameter (um)", "fiber.clad_um", 2.0, 3000.0, 1.0, container=g2)
    num("Core refractive index", "fiber.n_core", 1.2, 2.0, 0.0001, fmt="%.4f", container=g3)
    num("Cladding refractive index", "fiber.n_clad", 1.2, 2.0, 0.0001, fmt="%.4f", container=g4)
    o1, o2, o3 = st.columns(3)
    o1.checkbox("Operating wavelength follows the WDM channels", key="ui.op_auto",
                help="On: the operating wavelength is the average of the WDM channel wavelengths (tab WDM)")
    if st.session_state["ui.op_auto"]:
        st.session_state["fiber.wavelength_nm"] = float(round(collect_config().fiber.wavelength_nm, 3))
    o1.number_input("Operating wavelength (nm)", 400.0, 1700.0, step=10.0, key="fiber.wavelength_nm",
                    disabled=bool(st.session_state["ui.op_auto"]), format="%.2f",
                    help="Used for V-number, cut-off check and the attenuation/dispersion override reference")
    num("PMD coefficient (ps/sqrt km)", "fiber.pmd_ps_sqrtkm", 0.0, 2.0, 0.01, container=o2)
    num("Modal bandwidth (MHz km, graded-index MMF)", "fiber.modal_bw_mhz_km", 0.0, 20000.0, 50.0, container=o3)
    a1, a2 = st.columns(2)
    a1.checkbox("Manual attenuation at operating wavelength", key="ui.atten_manual")
    if st.session_state["ui.atten_manual"]:
        a1.number_input("Attenuation (dB/km)", 0.01, 50000.0, key="ui.atten_value", step=0.01)
    a2.checkbox("Manual chromatic dispersion at operating wavelength", key="ui.disp_manual")
    if st.session_state["ui.disp_manual"]:
        a2.number_input("D (ps/nm/km)", -1000.0, 1000.0, key="ui.disp_value", step=0.5)
    l1, l2, l3, l4 = st.columns(4)
    num("Connectors (count)", "fiber.n_connectors", 0, 20, 1, container=l1)
    num("Loss per connector (dB, assumed)", "fiber.connector_loss_db", 0.0, 3.0, 0.1, container=l2)
    num("Splice loss (dB, assumed)", "fiber.splice_loss_db", 0.0, 1.0, 0.01, container=l3)
    num("Extra margin (dB)", "fiber.margin_db", 0.0, 10.0, 0.5, container=l4)
    cf = collect_config().fiber
    geo = fm.fiber_geometry(cf)
    rows = pd.DataFrame([
        ("Numerical aperture (calculated)", f"{geo['na']:.4f}"),
        ("V-number at operating wavelength (calculated)", f"{geo['v']:.2f}"),
        ("Guided modes (calculated, approx.)", f"{geo['modes']:.0f}"),
        ("Single-mode cut-off wavelength (calculated)", f"{geo['cutoff_nm']:.0f} nm"),
        ("Attenuation at operating wavelength (model)", f"{fm.attenuation_db_km(cf, cf.wavelength_nm):.3f} dB/km"),
        ("Chromatic dispersion D (model)", f"{fm.chromatic_dispersion_ps_nm_km(cf, cf.wavelength_nm):.2f} ps/nm/km"),
        ("Modal dispersion over length (RMS)", f"{fm.modal_dispersion_rms_ps(cf):.1f} ps"),
        ("PMD over length (mean DGD)", f"{fm.pmd_rms_ps(cf):.2f} ps"),
        ("Fibre loss over length at operating wavelength", f"{fm.attenuation_db_km(cf, cf.wavelength_nm) * cf.length_km:.2f} dB"),
    ], columns=["Quantity", "Value"])
    st.dataframe(rows, hide_index=True, width="stretch")
    issues_box(fm.validate_fiber(cf, list(res.link.lam_nm)))
    lam_axis = np.linspace(*fm.WAVELENGTH_RANGE_NM[cf.material], 200)
    figa = viz.fig_lines(pd.DataFrame(dict(w=lam_axis, a=fm.attenuation_db_km(cf, lam_axis))), "w", {"Attenuation": "a"},
                         "Attenuation vs wavelength", "Wavelength (nm)", "dB/km", logy=True)
    figd = viz.fig_lines(pd.DataFrame(dict(w=lam_axis, d=fm.chromatic_dispersion_ps_nm_km(cf, lam_axis))), "w",
                         {"D": "d"}, "Chromatic dispersion vs wavelength", "Wavelength (nm)", "ps/(nm km)")
    p1, p2 = st.columns(2)
    plot(p1, figa, width="stretch")
    plot(p2, figd, width="stretch")

# ============================================================================ 3 WDM
with tabs[2]:
    st.subheader("WDM configuration")
    w1, w2, w3 = st.columns(3)
    w1.radio("Scheme", ["DWDM", "CWDM"], key="wdm.scheme", horizontal=True)
    w2.slider("Number of channels", 1, MAX_CH, key="wdm.n_channels")
    w3.checkbox("Custom wavelengths", key="wdm.custom_enabled",
                help="On: the wavelengths typed in the table below are used instead of the regular grid")
    custom_on = bool(st.session_state["wdm.custom_enabled"])
    if custom_on:
        st.info("Custom wavelengths are active: the channel wavelengths come from the column "
                "'Custom wavelength (nm)' below, and the grid settings in this section are ignored.")
    else:
        st.caption("Tip: typing a value in the 'Custom wavelength (nm)' column switches custom wavelengths on automatically.")
    if st.session_state["wdm.scheme"] == "DWDM":
        d1, d2 = st.columns(2)
        d1.number_input("Central wavelength (nm)", 1200.0, 1700.0, step=0.1, format="%.3f", key="wdm.center_nm",
                        disabled=custom_on)
        d2.selectbox("Frequency spacing (GHz)", [12.5, 25.0, 50.0, 100.0, 200.0], key="wdm.dwdm_spacing_ghz",
                     disabled=custom_on)
    else:
        d1, d2 = st.columns(2)
        d1.number_input("Start wavelength (nm)", 1260.0, 1625.0, step=1.0, key="wdm.cwdm_start_nm", disabled=custom_on)
        d2.number_input("Spacing (nm)", 2.0, 60.0, step=1.0, key="wdm.cwdm_spacing_nm", disabled=custom_on)
    m1, m2, m3, m4, m5 = st.columns(5)
    num("MUX insertion loss (dB)", "wdm.mux_il_db", 0.0, 15.0, 0.5, container=m1)
    num("DEMUX insertion loss (dB)", "wdm.demux_il_db", 0.0, 15.0, 0.5, container=m2)
    num("DEMUX 3 dB BW / spacing", "wdm.demux_bw_frac", 0.1, 1.0, 0.05, container=m3)
    m4.selectbox("DEMUX filter order", [1, 2, 3], key="wdm.demux_order")
    num("Isolation floor (dB)", "wdm.isolation_db", 10.0, 70.0, 1.0, container=m5)
    st.markdown("**Per-channel settings**")
    edit_table("wdm_df", n_ch, {
        "Bit rate (Gb/s)": st.column_config.NumberColumn(min_value=0.1, max_value=100.0, step=0.5),
        "Launch power (dBm)": st.column_config.NumberColumn(min_value=-20.0, max_value=10.0, step=0.5),
        "Custom wavelength (nm)": st.column_config.NumberColumn(min_value=400.0, max_value=1700.0, step=0.01,
                                                                format="%.3f", help="Used only if 'Custom wavelengths' is on")},
        key="wdm_editor", on_change=on_wdm_edit)
    cw = collect_config().wdm
    issues_box(ws.validate_wdm(cw, collect_config().fiber, collect_config().amp.enabled))
    lam = ws.channel_wavelengths(cw)
    grid = pd.DataFrame(dict(Channel=np.arange(1, len(lam) + 1), **{"Wavelength (nm)": lam.round(4),
                                                                       "Frequency (THz)": (ws.wavelength_to_frequency_hz(lam) / 1e12).round(4)}))
    st.dataframe(grid, hide_index=True, width="stretch")
    st.metric("Total nominal data rate", f"{ws.nominal_capacity_gbps(cw):.1f} Gb/s")
    st.caption("Nominal capacity = sum of channel rates. It is not the received throughput (see Traffic and Performance).")

# ============================================================================ 4 Tx/Rx
with tabs[3]:
    st.subheader("Transmitter and receiver")
    t1, t2, t3 = st.columns(3)
    t1.selectbox("Modulation format", ["NRZ-OOK", "RZ-OOK"], key="txrx.modulation")
    num("Extinction ratio (dB)", "txrx.extinction_db", 3.0, 25.0, 0.5, container=t2)
    num("Tx rise time (fraction of bit)", "txrx.rise_frac", 0.05, 0.8, 0.05, container=t3)
    r1, r2, r3, r4 = st.columns(4)
    num("Samples per bit", "txrx.sps", 8, 64, 8, container=r1)
    num("Bits displayed in waveforms", "txrx.n_bits", 128, 1024, 64, container=r2)
    num("Monte-Carlo BER bits", "txrx.ber_sim_bits", 1000, 200000, 1000, container=r3)
    r4.selectbox("Decision threshold", ["Optimum", "Midpoint"], key="txrx.threshold")
    q1, q2, q3, q4 = st.columns(4)
    num("Laser RIN (dB/Hz)", "txrx.rin_db_hz", -170.0, -110.0, 1.0, container=q1)
    num("Laser linewidth (MHz)", "txrx.linewidth_mhz", 0.01, 1000.0, 1.0, container=q2)
    num("Source RMS spectral width (nm)", "txrx.spectral_width_nm", 0.0, 5.0, 0.01, container=q3)
    q4.caption("Phase noise (linewidth) does not directly affect IM/DD OOK detection; here it only widens the "
               "spectrum and so increases chromatic-dispersion broadening.")
    p1, p2, p3, p4 = st.columns(4)
    num("Responsivity (A/W)", "txrx.responsivity", 0.1, 1.2, 0.05, container=p1)
    num("Receiver BW / bit rate", "txrx.rx_bw_frac", 0.3, 1.2, 0.05, container=p2)
    p3.selectbox("Receiver filter", ["Bessel", "Butterworth"], key="txrx.rx_filter",
                 help="Bessel (Gaussian) has little ringing; Butterworth rings and adds ISI")
    num("Sensitivity (dBm)", "txrx.sensitivity_dbm", -50.0, 0.0, 1.0, container=p4)
    if st.session_state["txrx.rx_filter"] == "Butterworth":
        st.selectbox("Butterworth order", [2, 3, 4, 5, 6], key="txrx.rx_order")
    h1, h2, h3, h4 = st.columns(4)
    num("Temperature (K)", "txrx.temperature_k", 200.0, 400.0, 5.0, container=h1)
    num("Load resistance (ohm)", "txrx.load_ohm", 10.0, 10000.0, 10.0, container=h2)
    num("Receiver noise figure (dB)", "txrx.rx_nf_db", 0.0, 15.0, 0.5, container=h3)
    num("Dark current (nA)", "txrx.dark_current_na", 0.0, 1000.0, 1.0, container=h4)
    st.info("Direct-detection model: photocurrent I = R x P. All noise sources are in A^2; optical power in W.")
    cex = collect_config()
    st.write(f"Receiver 3 dB bandwidth = {cex.txrx.rx_bw_frac * cex.wdm.bitrates_gbps[ch]:.2f} GHz at "
             f"{cex.wdm.bitrates_gbps[ch]:.1f} Gb/s; noise-equivalent bandwidth factor "
             f"{trx.nebw_factor(cex.txrx.rx_order, cex.txrx.rx_filter):.3f}.")

# ============================================================================ 5 noise
with tabs[4]:
    st.subheader("Noise configuration")
    nz1, nz2, nz3, nz4 = st.columns(4)
    nz1.toggle("Thermal noise", key="noise.thermal")
    nz1.toggle("Shot noise", key="noise.shot")
    nz2.toggle("Laser RIN noise", key="noise.laser_rin")
    nz2.toggle("Amplifier ASE noise", key="noise.ase", help="Only active while the amplifier is ON")
    nz3.toggle("Inter-channel crosstalk", key="noise.crosstalk")
    nz3.toggle("Background optical noise", key="noise.background")
    nz4.toggle("Timing jitter", key="noise.jitter")
    if st.session_state["noise.background"]:
        num("Background noise PSD (dBm/nm per polarisation, assumed)", "noise.bg_dbm_per_nm", -80.0, -10.0, 1.0)
    if st.session_state["noise.jitter"]:
        num("RMS timing jitter (ps)", "txrx.jitter_ps", 0.0, 100.0, 0.5)
    plot(st, viz.fig_noise_contrib(sel["noise_std_ua"]), width="stretch")
    st.markdown("**Separate and combined effect on the inspected channel**")
    cmp_n = res.comparisons["noise"].copy()
    cmp_n["ber"] = cmp_n["ber"].map(perf.fmt_ber)
    st.dataframe(cmp_n.rename(columns={"case": "Case", "q": "Q-factor", "snr_db": "SNR (dB)", "ber": "BER (analytic)"}),
                 hide_index=True, width="stretch")
    st.caption("Disabled sources contribute exactly zero to every variance. 'Only X' rows enable one source on top of "
               "the always-present ISI of the link.")

# ============================================================================ 6 amplifier
with tabs[5]:
    st.subheader("Optical amplifier (EDFA)")
    e1, e2, e3, e4 = st.columns(4)
    e1.toggle("Amplifier ON", key="amp.enabled")
    e2.selectbox("Placement", ["Booster", "In-line", "Pre-amplifier"], key="amp.placement")
    num("Gain (dB)", "amp.gain_db", 0.0, 40.0, 0.5, container=e3)
    num("Noise figure (dB)", "amp.nf_db", 3.0, 12.0, 0.5, container=e4)
    f1, f2, f3 = st.columns(3)
    num("In-line amplifiers (count)", "amp.n_inline", 1, 8, 1, container=f1)
    f2.toggle("Output power limit (saturation)", key="amp.saturation")
    num("Total output power limit (dBm)", "amp.psat_dbm", 0.0, 30.0, 0.5, container=f3)
    cmp_a = res.comparisons["amplifier"]
    st.dataframe(cmp_a.assign(ber=cmp_a.ber.map(perf.fmt_ber)).rename(columns={
        "case": "Case", "p_rx_dbm": "Rx power (dBm)", "snr_db": "SNR (dB)", "q": "Q", "ber": "BER",
        "ase_std_ua": "ASE noise (uA RMS)"}), hide_index=True, width="stretch")
    pc1, pc2 = st.columns(2)
    plot(pc1, viz.fig_power_profile(res.link, ch), width="stretch")
    plot(pc2, viz.fig_rx_before_after(sel["wave"], res.link.amp_on), width="stretch")
    plot(st, viz.fig_amp_in_out(sel["wave"]), width="stretch")
    for a in res.link.amps:
        st.write(f"**{a['name']}**: effective gain {a['gain_db']:.1f} dB (requested {a['requested_gain_db']:.1f} dB), "
                 f"n_sp = {a['n_sp']:.2f}" + (" - output-power limited" if a["saturated"] else ""))
    st.info("Amplification raises the signal power but also adds ASE noise (signal-ASE and ASE-ASE beating). "
            "It improves quality when the link is limited by receiver thermal noise, and may not help (or may "
            "hurt) when the signal is already well above the thermal noise floor or the ASE adds more than it gains.")

# ============================================================================ 7 dispersion
with tabs[6]:
    st.subheader("Dispersion and compensation")
    ds1, ds2, ds3, ds4 = st.columns(4)
    ds1.toggle("Fibre dispersion ON", key="disp.enabled")
    ds2.toggle("Dispersion compensation ON", key="disp.compensation")
    ds3.selectbox("Method", ["DSP", "DCF"], key="disp.method")
    num("Compensated fraction of chromatic broadening", "disp.comp_fraction", 0.0, 1.0, 0.05, container=ds4)
    num("DCF insertion loss (dB, assumed)", "disp.dcf_loss_db", 0.0, 15.0, 0.5)
    d = sel["disp"]
    cdm = st.columns(5)
    cdm[0].metric("D (ps/nm/km)", f"{d['D']:.2f}")
    cdm[1].metric("Chromatic RMS", f"{d['chrom_ps']:.1f} ps")
    cdm[2].metric("Modal / PMD RMS", f"{d['modal_ps']:.1f} / {d['pmd_ps']:.1f} ps")
    cdm[3].metric("Net broadening (after comp.)", f"{d['extra_ps']:.1f} ps")
    cdm[4].metric("Dispersion penalty", f"{d['penalty_db']:.2f} dB")
    plot(st, viz.fig_pulses(res.pulses), width="stretch")
    cd = res.comparisons["dispersion"]
    st.dataframe(cd.assign(ber=cd.ber.map(perf.fmt_ber)).rename(columns={
        "case": "Case", "broadening_ps": "Broadening (ps)", "eye_pct": "Eye opening (%)", "penalty_db": "Penalty (dB)",
        "p_rx_dbm": "Rx power (dBm)", "q": "Q", "ber": "BER"}), hide_index=True, width="stretch")
    st.info("WDM itself does NOT remove dispersion: every wavelength is dispersed independently. Pulse broadening "
            "uses a Gaussian-pulse approximation (RMS widths added in quadrature); compensation here is idealised "
            "(a fixed fraction of the chromatic term is removed; modal dispersion and PMD are not compensated).")

# ============================================================================ 8 signals
with tabs[7]:
    st.subheader(f"Signal visualisation - channel {ch + 1} ({sel['wavelength_nm']:.2f} nm)")
    wv = sel["wave"]
    flg = wv["flags"]
    st.caption(f"All time-domain panels use the same {wv['n_show']}-bit window at {sel['bitrate_gbps']:.1f} Gb/s. "
               f"Photocurrent panels are AC-coupled (DC offsets removed). Monte-Carlo run: {sel['sim_bits']} bits.")
    s1, s2 = st.columns(2)
    plot(s1, viz.fig_data(wv["bits"]), width="stretch")
    plot(s2, viz.fig_optical(wv["t_ns"], wv["w_tx"]), width="stretch")
    s3, s4 = st.columns(2)
    plot(s3, viz.fig_rx_before_after(wv, res.link.amp_on), width="stretch")
    plot(s4, viz.fig_amp_in_out(wv), width="stretch")
    n1, n2 = st.columns(2)
    plot(n1, viz.fig_noise(wv, "thermal", "Thermal", flg["thermal"]), width="stretch")
    plot(n2, viz.fig_noise(wv, "shot", "Shot", flg["shot"]), width="stretch")
    n3, n4 = st.columns(2)
    plot(n3, viz.fig_noise(wv, "laser_rin", "Laser (RIN)", flg["laser_rin"]), width="stretch")
    plot(n4, viz.fig_noise(wv, "ase", "ASE (signal-ASE + ASE-ASE beat)", flg["ase"] and res.link.amp_on),
                    width="stretch")
    n5, n6 = st.columns(2)
    plot(n5, viz.fig_noise(wv, "crosstalk", "Inter-channel crosstalk", flg["crosstalk"]), width="stretch")
    plot(n6, viz.fig_combined_noise(wv), width="stretch")
    plot(st, viz.fig_clean_vs_noisy(wv), width="stretch")
    r1, r2 = st.columns(2)
    plot(r1, viz.fig_recovered(wv), width="stretch")
    met = f"Q = {sel['q']:.2f}<br>Eye opening (ISI) = {sel['eye_open_pct']:.0f} %"
    plot(r2, viz.fig_eye(wv, True, met), width="stretch")
    sp1, sp2 = st.columns(2)
    plot(sp1, viz.fig_spectrum(res.link, ch), width="stretch")
    plot(sp2, viz.fig_rx_power_bars(res.table), width="stretch")
    ber_txt = (f"{sel['sim_errors']} errors in {sel['sim_bits']} bits (simulated BER {sel['sim_ber']:.2e})"
               if sel["sim_errors"] else
               f"0 errors in {sel['sim_bits']} bits: the true BER is NOT zero; 95 % upper bound about "
               f"{sel['sim_ber_upper95']:.1e}")
    st.write(f"**Monte-Carlo result:** {ber_txt}.  **Analytic (Gaussian) BER:** {sel['ber_display']}.")

# ============================================================================ 9 performance
with tabs[8]:
    st.subheader("Performance analysis")
    pm = st.columns(6)
    pm[0].metric("Tx power", f"{sel['p_launch_dbm']:.1f} dBm")
    pm[1].metric("Rx power", f"{sel['p_rx_dbm']:.1f} dBm")
    pm[2].metric("Total passive loss", f"{sel['passive_loss_db']:.1f} dB")
    pm[3].metric("SNR", f"{sel['snr_db']:.1f} dB")
    pm[4].metric("BER (analytic)", sel["ber_display"])
    pm[5].metric("Crosstalk level", f"{sel['xt_db']:.1f} dB")
    pm2 = st.columns(5)
    pm2[0].metric("Q-factor", f"{sel['q']:.2f}")
    pm2[1].metric("Eye opening (ISI)", f"{sel['eye_open_pct']:.0f} %")
    pm2[2].metric("Pulse broadening", f"{sel['broadening_ps']:.1f} ps")
    pm2[3].metric("Dispersion penalty", f"{sel['penalty_formula_db']:.2f} dB")
    pm2[4].metric("Aggregate goodput", f"{res.received_gbps:.1f} Gb/s")
    show = res.table[["channel", "wavelength_nm", "bitrate_gbps", "p_rx_dbm", "snr_db", "q", "ber_display",
                      "eye_open_pct", "xt_db", "broadening_ps", "status"]].copy()
    show.columns = ["Ch", "λ (nm)", "Rate (Gb/s)", "Rx (dBm)", "SNR (dB)", "Q", "BER", "Eye %", "XT (dB)",
                    "Broadening (ps)", "Status"]
    show["Status"] = show["Status"].map(lambda x: f"{STATUS_ICON[x]} {x}")
    st.dataframe(show.round(3), hide_index=True, width="stretch")
    st.caption("BER below 1e-15 is shown as a bound. Q and BER come from a Gaussian-noise model (see README).")
    sw = res.sweeps
    snr_ax = np.linspace(0, 24, 100)
    g1, g2 = st.columns(2)
    plot(g1, viz.fig_ber_vs_snr(snr_ax, perf.ber_vs_snr_curve(snr_ax), sel["snr_db"], sel["ber"]),
                    width="stretch")
    plot(g2, viz.fig_lines(sw["length"], "length_km", {"BER": "ber"}, "BER vs fibre length", "Length (km)",
                                  "BER", logy=True, split="case"), width="stretch")
    g3, g4 = st.columns(2)
    plot(g3, viz.fig_lines(sw["length"], "length_km", {"P": "p_rx_dbm"}, "Received power vs fibre length",
                                  "Length (km)", "dBm", split="case"), width="stretch")
    plot(g4, viz.fig_lines(sw["length"], "length_km", {"S": "snr_db"}, "SNR vs fibre length",
                                  "Length (km)", "dB", split="case"), width="stretch")
    g5, g6 = st.columns(2)
    unit = sw["spacing"]["unit"].iloc[0]
    plot(g5, viz.fig_lines(sw["spacing"], "spacing", {"Worst channel Q": "worst_q", "Inspected channel Q": "sel_q"},
                                  "Performance vs channel spacing", f"Spacing ({unit})", "Q-factor"),
                    width="stretch")
    plot(g6, viz.fig_lines(sw["rate"], "rate_gbps", {"Q": "q", "Eye (%)": "eye_pct"},
                                  "Performance vs data rate", "Rate per channel (Gb/s)", "Q / eye %"),
                    width="stretch")
    st.markdown("**Before / after comparisons (inspected channel)**")
    cc1, cc2 = st.columns(2)
    ca, cdsp = res.comparisons["amplifier"], res.comparisons["dispersion"]
    plot(cc1, viz.fig_bars(ca["case"], ca["q"], "Amplifier OFF vs ON: Q-factor", "Q"), width="stretch")
    plot(cc2, viz.fig_bars(cdsp["case"], cdsp["q"], "Dispersion / compensation: Q-factor", "Q"),
                     width="stretch")
    cn = res.comparisons["noise"]
    plot(st, viz.fig_bars(cn["case"], cn["q"].clip(upper=40), "Noise sources: Q-factor (capped at 40)", "Q"),
                    width="stretch")

# ============================================================================ 10 traffic
with tabs[9]:
    st.subheader("Network traffic and adaptive load balancing")
    st.caption("NETWORK-LEVEL simulation: there is no physical optical switching fabric. Traffic may move only "
               "between channels of the same compatibility group.")
    tc1, tc2, tc3, tc4 = st.columns(4)
    tc1.toggle("Adaptive load balancing", key="traffic.load_balancing")
    tc2.toggle("Priority scheduling", key="traffic.priority_scheduling")
    num("Congestion threshold", "traffic.threshold", 0.3, 1.0, 0.05, container=tc3)
    num("Re-route latency (us, assumed)", "traffic.reroute_overhead_us", 0.0, 1000.0, 10.0, container=tc4)
    edit_table("traffic_df", n_ch, {
        "Demand (Gb/s)": st.column_config.NumberColumn(min_value=0.0, max_value=200.0, step=0.5),
        "Priority (1=high)": st.column_config.NumberColumn(min_value=1, max_value=5, step=1),
        "Group": st.column_config.NumberColumn(min_value=1, max_value=8, step=1, help="Compatibility group")},
        key="traffic_editor")
    toff, ton = res.traffic_off, res.traffic_on
    tm_ = st.columns(5)
    tm_[0].metric("Offered", f"{ton.offered_gbps:.1f} Gb/s")
    tm_[1].metric("Carried", f"{ton.carried_gbps:.1f} Gb/s", f"{ton.carried_gbps - toff.carried_gbps:+.1f} vs no balancing")
    tm_[2].metric("Goodput (after packet errors)", f"{ton.goodput_gbps:.2f} Gb/s")
    tm_[3].metric("Dropped", f"{ton.dropped_gbps:.2f} Gb/s", f"{ton.loss_pct:.1f} % loss")
    tm_[4].metric("Congested channels", f"{ton.n_congested}", f"{ton.n_congested - toff.n_congested:+d} vs no balancing",
                  delta_color="inverse")
    tv1, tv2 = st.columns(2)
    plot(tv1, viz.fig_traffic(toff, ton), width="stretch")
    plot(tv2, viz.fig_throughput(toff, ton, ("Without balancing", "With balancing")), width="stretch")
    tab = ton.table.copy()
    tab["state"] = tab["state"].map(lambda s: {"OK": "🟢 OK", "CONGESTED": "🟠 CONGESTED", "OVERLOAD": "🔴 OVERLOAD",
                                               "FAILED": "🔴 FAILED"}[s])
    st.dataframe(pretty(tab).round(3), hide_index=True, width="stretch")
    if ton.moved.sum() > 0:
        mv = pd.DataFrame([(i + 1, j + 1, ton.moved[i, j]) for i in range(n) for j in range(n) if ton.moved[i, j] > 1e-9],
                          columns=["From channel", "To channel", "Moved (Gb/s)"])
        st.markdown("**Re-routed traffic**")
        st.dataframe(mv.round(3), hide_index=True)
    st.caption("Latency uses an M/M/1 queue per channel plus the re-route overhead; 'nan' means the channel is "
               "saturated or down. Packet loss includes dropped traffic; goodput also subtracts packet errors "
               "implied by the channel BER.")

# ============================================================================ 11 failure
with tabs[10]:
    st.subheader("Channel failure and recovery")
    fc1, fc2, fc3 = st.columns([1, 1, 2])
    victim = fc1.selectbox("Channel to fail", list(range(1, n_ch + 1)), key="ui.victim")
    fc2.button("⚡ Trigger failure", on_click=fault_trigger, args=(victim - 1,), width="stretch")
    fc2.button("♻ Restore failed channels", on_click=fault_restore, width="stretch")
    fc3.toggle("Protection switching", key="fault.protection")
    fc3.selectbox("Backup route", ["Reroute to remaining channels", "Dedicated backup channel"], key="fault.backup_mode")
    if st.session_state["fault.backup_mode"].startswith("Dedicated"):
        fc3.selectbox("Backup channel", list(range(1, n_ch + 1)), index=min(n_ch - 1, 6), key="ui.backup_ch")
        st.session_state["fault.backup_channel"] = int(st.session_state["ui.backup_ch"]) - 1
    tm1, tm2, tm3, tm4, tm5 = st.columns(5)
    num("Failure time (ms)", "fault.t_fail_ms", 0.0, 10000.0, 10.0, container=tm1)
    num("Monitor poll period (ms)", "fault.poll_ms", 1.0, 1000.0, 1.0, container=tm2)
    num("Debounce (polls)", "fault.debounce", 1, 20, 1, container=tm3)
    num("Switch time (ms, assumed)", "fault.switch_ms", 0.0, 5000.0, 5.0, container=tm4)
    num("Timeline end (ms)", "fault.t_end_ms", 100.0, 20000.0, 50.0, container=tm5)
    fa = res.fault
    for m_ in fa.messages:
        (st.warning if fa.detected else st.info)(m_)
    fm_ = st.columns(5)
    fm_[0].metric("Failure detected", "Yes" if fa.detected else "No")
    fm_[1].metric("Traffic recovered", "Yes" if fa.recovered else ("No" if fa.detected else "-"))
    fm_[2].metric("Simulated recovery time",
                  f"{fa.recovery_time_ms:.0f} ms" if not np.isnan(fa.recovery_time_ms) else "-")
    fm_[3].metric("Traffic lost in outage", f"{fa.outage_lost_mbit:.0f} Mbit")
    fm_[4].metric("Still lost after recovery", f"{fa.steady_lost_gbps:.2f} Gb/s")
    st.dataframe(pretty(res.status_table), hide_index=True, width="stretch")
    plot(st, viz.fig_fault_timeline(fa), width="stretch")
    q1, q2 = st.columns(2)
    plot(q1, viz.fig_traffic(fa.during, fa.after, ("During outage", "After recovery")), width="stretch")
    plot(q2, viz.fig_throughput(fa.before, fa.after, ("Before fault", "After recovery")), width="stretch")
    st.caption("Recovery time is a SIMULATED value from the configured polling, debounce and switch time. "
               "Real hardware switching times depend on the equipment and are not predicted here. A failed channel "
               "also stops interfering with its neighbours and changes amplifier loading when saturation is on.")

# ============================================================================ 12 summary
with tabs[11]:
    st.subheader("Results summary and export")
    st.markdown(f"""
**Configuration:** {cfg.wdm.scheme}, {n} channels, {cfg.fiber.material} {cfg.fiber.mode} ({cfg.fiber.profile}),
{cfg.fiber.length_km:g} km, amplifier {'ON (' + cfg.amp.placement + ')' if cfg.amp.enabled else 'OFF'},
dispersion {'ON' if cfg.disp.enabled else 'OFF'}{' + compensation' if cfg.disp.enabled and cfg.disp.compensation else ''}.

**Result:** nominal capacity {res.nominal_gbps:.0f} Gb/s, received goodput {res.received_gbps:.1f} Gb/s;
{int((res.table.status == 'OK').sum())} of {n} channels healthy. Inspected channel {ch + 1}: Rx {sel['p_rx_dbm']:.1f} dBm,
SNR {sel['snr_db']:.1f} dB, Q {sel['q']:.2f}, BER {sel['ber_display']} (analytic).
""")
    e1, e2, e3 = st.columns(3)
    csv = pretty(res.table.drop(columns=["noise_std_ua", "disp"], errors="ignore")).to_csv(index=False).encode()
    e1.download_button("Download channel results (CSV)", csv, "wdm_channel_results.csv", "text/csv")
    e2.download_button("Download configuration (JSON)", cfg.to_json().encode(), "wdm_config.json", "application/json")
    buf = io.StringIO()
    buf.write("# WDM simulation report\n\n")
    buf.write(f"Configuration JSON:\n\n```json\n{json.dumps(cfg.to_dict(), indent=1)}\n```\n\n")
    buf.write(pretty(res.table.drop(columns=["noise_std_ua", "disp"], errors="ignore")).round(4).to_markdown(index=False)
              if hasattr(res.table, "to_markdown") else res.table.to_string())
    buf.write("\n\nAll values are simulated by a simplified model; none are hardware measurements.\n")
    e3.download_button("Download report (Markdown)", buf.getvalue().encode(), "wdm_report.md", "text/markdown")
    st.markdown("**Parameter provenance (sourced / calculated / assumed)**")
    st.dataframe(pd.DataFrame(PARAMETER_PROVENANCE, columns=["Parameter / model", "Basis", "Type"]),
                 hide_index=True, width="stretch")
    st.markdown("""
**Limitations.** Intensity-modulation / direct-detection only; fibre nonlinearities (SPM, XPM, FWM, SRS) are not
modelled; amplifier gain is flat and instantaneous; dispersion uses Gaussian-pulse RMS widths; BER uses a Gaussian
approximation (a zero-error Monte-Carlo run gives only an upper bound); traffic balancing and protection switching are
network-level simulations without a physical switching fabric. No value here is a measurement of real hardware.
""")
    st.markdown("**Documents**")
    doc_cols = st.columns(2)
    for doc_col, doc_name, doc_key in (
            (doc_cols[0], "WDM_System_Working_and_Theory_Tutorial.pdf", "dl_doc_tutorial"),
            (doc_cols[1], "WDM_System_Configuration_Reference.pdf", "dl_doc_config_reference")):
        try:
            doc_bytes = (Path(__file__).parent / doc_name).read_bytes()
        except OSError:
            doc_col.info(f"{doc_name} not found in the project folder")
            continue
        doc_col.download_button(f"Download {doc_name[:-4]} (PDF)", doc_bytes, doc_name, "application/pdf", key=doc_key)
