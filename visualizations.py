"""Plotly figure builders for the dashboard (signals, spectra, eye, performance, traffic, faults)."""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import plotly.graph_objects as go

import noise_models as nm

TEMPLATE = "plotly_white"
CH_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf", "#8c564b", "#e377c2",
             "#7f7f7f", "#bcbd22", "#393b79", "#ad494a", "#637939", "#8c6d31", "#843c39", "#7b4173"]
STATE_COLORS = {"ok": "#2ca02c", "warn": "#ff9f1c", "error": "#d62728", "off": "#9aa0a6"}


def _base(title: str, xlabel: str, ylabel: str, height: int = 280, **kw) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(template=TEMPLATE, title=dict(text=title, font=dict(size=17)), height=height,
                      font=dict(size=14), xaxis_title=xlabel, yaxis_title=ylabel,
                      margin=dict(l=60, r=15, t=50, b=50), legend=dict(orientation="h", y=-0.3), **kw)
    return fig


def _disabled(fig: go.Figure, text: str) -> go.Figure:
    fig.add_annotation(text=text, xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
                       font=dict(size=18, color="#d62728"), bgcolor="rgba(255,255,255,0.85)")
    return fig


# ------------------------------------------------------------------------- pipeline
def fig_pipeline(stages: List[dict]) -> go.Figure:
    n = len(stages)
    fig = go.Figure()
    per_row = 6
    for k, s in enumerate(stages):
        row, col = divmod(k, per_row)
        x0, y0 = col * 1.0, -row * 1.1
        fig.add_shape(type="rect", x0=x0 + 0.04, x1=x0 + 0.92, y0=y0, y1=y0 + 0.8,
                      line=dict(color=STATE_COLORS[s["state"]], width=2),
                      fillcolor=STATE_COLORS[s["state"]], opacity=0.18)
        fig.add_annotation(x=x0 + 0.48, y=y0 + 0.5, text=f"<b>{s['stage']}</b>", showarrow=False, font=dict(size=11))
        fig.add_annotation(x=x0 + 0.48, y=y0 + 0.2, text=s["detail"], showarrow=False, font=dict(size=10, color="#444"))
        if k < n - 1 and col < per_row - 1:
            fig.add_annotation(x=x0 + 1.04, y=y0 + 0.4, ax=x0 + 0.93, ay=y0 + 0.4, xref="x", yref="y",
                               axref="x", ayref="y", showarrow=True, arrowhead=2, arrowcolor="#666")
    rows = (n + per_row - 1) // per_row
    fig.update_xaxes(visible=False, range=[-0.05, per_row + 0.05])
    fig.update_yaxes(visible=False, range=[-(rows - 1) * 1.1 - 0.15, 1.0])
    fig.update_layout(template=TEMPLATE, height=95 + 110 * rows, margin=dict(l=5, r=5, t=5, b=5))
    return fig


# ------------------------------------------------------------------------- block diagram
def fig_block_diagram(cfg, link, table, pipeline: List[dict]) -> go.Figure:
    """WDM system block diagram drawn with component symbols (laser, MUX, fibre, amplifier, DEMUX, photodiode)."""
    ok, bad, warn, off = "#2ca02c", "#d62728", "#ff9f1c", "#9aa0a6"
    ink = "#1f2933"
    fig = go.Figure()
    shapes, notes = [], []

    def rect(x0, y0, x1, y1, fill, line=ink, dash=None, opacity=1.0, rounded=True):
        shapes.append(dict(type="rect", x0=x0, y0=y0, x1=x1, y1=y1, fillcolor=fill, line=dict(color=line, width=2, dash=dash or "solid"),
                           opacity=opacity, layer="below"))

    def poly(pts, fill, line=ink, dash=None, opacity=1.0):
        path = "M " + " L ".join(f"{x},{y}" for x, y in pts) + " Z"
        shapes.append(dict(type="path", path=path, fillcolor=fill, line=dict(color=line, width=2, dash=dash or "solid"),
                           opacity=opacity))

    def line(x0, y0, x1, y1, color=ink, width=2, dash=None):
        shapes.append(dict(type="line", x0=x0, y0=y0, x1=x1, y1=y1, line=dict(color=color, width=width, dash=dash or "solid")))

    def arrow(x0, y0, x1, y1, color=ink):
        notes.append(dict(x=x1, y=y1, ax=x0, ay=y0, xref="x", yref="y", axref="x", ayref="y", showarrow=True,
                          arrowhead=2, arrowsize=1.2, arrowwidth=2, arrowcolor=color, text=""))

    def label(x, y, text, size=14, color=ink, bold=False):
        notes.append(dict(x=x, y=y, text=f"<b>{text}</b>" if bold else text, showarrow=False,
                          font=dict(size=size, color=color), align="center"))

    n = link.n
    show_n = n
    statuses = list(table["status"])
    col_of = {"OK": ok, "DEGRADED": warn, "FAILED": bad}
    ys = [52.0, 44.5, 37.0, 29.5][:show_n] if show_n > 1 else [40.0]
    if show_n == 4:
        ys = [53.0, 45.0, 37.0, 29.0]
    yc = 40.0
    # layout constants: unchanged for n <= 4; computed for n > 4 (every channel is drawn)
    dy, up, hh, f, fsz = 0, 0, 3, 1, 13
    if n > 4:
        pitch = max(3.6, 32 / (n - 1))
        ys = [yc + (n - 1) / 2 * pitch - i * pitch for i in range(n)]
        hh = min(3, 0.4 * pitch)
        f = hh / 3                                   # scale of the laser / photodiode symbols
        fsz = 13 if n <= 8 else 11
        dy = 25.0 - (min(ys) - 4)                    # extra height needed below the channel rows
        up = max(0.0, max(ys) + 4 - 57.0)            # extra height needed above (header labels)
    mid = float(np.mean(ys))

    # --- transmitters: laser symbols
    for i, y in enumerate(ys):
        failed = bool(link.failed[i])
        rect(1, y - hh, 9, y + hh, "#e8f1fb" if not failed else "#f3d6d6", line=bad if failed else "#1f77b4")
        label(3.4, y, f"<b>λ{i + 1}</b>", fsz)
        # laser-diode symbol (triangle + bar) and emitted wave
        poly([(5.3, y - 1.6 * f), (5.3, y + 1.6 * f), (7.3, y)], bad if failed else "#1f77b4", line=ink)
        line(7.3, y - 1.8 * f, 7.3, y + 1.8 * f)
        line(9, y, 14.5, y, color=off if failed else "#1f77b4", width=3)
        arrow(12.0, y, 14.5, y, off if failed else "#1f77b4")
    label(5, 68.0 + up, "Transmitters", 15, bold=True)
    label(5, 64.6 + up, "Lasers (data ➜ light)", 12, "#52606d")

    # --- multiplexer: trapezoid
    top, bot = max(ys) + 4, min(ys) - 4
    poly([(14.5, top), (24, yc + 4), (24, yc - 4), (14.5, bot)], "#ece7fb", line="#7f77dd")
    label(18.5, yc, "MUX", 15, "#4b3fb0", bold=True)
    label(19, bot - 3.5, f"insertion loss {cfg.wdm.mux_il_db:g} dB", 12, "#52606d")
    line(24, yc, 34, yc, "#1f77b4", 3)

    # --- booster amplifier
    def edfa(x0, x1, name, active, detail=True):
        c = "#ff9f1c" if active else off
        poly([(x0, yc + 5), (x0, yc - 5), (x1, yc)], c, line=ink, dash=None if active else "dot", opacity=1.0 if active else 0.35)
        label((x0 + x1) / 2 - 0.5, yc - 8.2, f"{name}", 12, "#8a5a00" if active else "#6b7280", bold=True)
        if active and detail:
            label((x0 + x1) / 2 - 0.5, yc - 11.4, f"+{cfg.amp.gain_db:g} dB, NF {cfg.amp.nf_db:g} dB", 11, "#52606d")

    amp_on, place = cfg.amp.enabled, cfg.amp.placement
    fx0, fx1 = 34.0, 66.0
    if amp_on and place == "Booster":
        edfa(26.0, 32.0, "Booster EDFA", True)
        line(24, yc, 26, yc, "#1f77b4", 3)
        line(32, yc, fx0, yc, "#1f77b4", 3)
    # --- optical fibre with attenuation / dispersion / noise
    rect(fx0, yc - 1.6, fx1, yc + 1.6, "#dbe9f6", line="#1f77b4")
    xs = np.linspace(fx0 + 0.5, fx1 - 0.5, 160)
    fig.add_trace(go.Scatter(x=xs, y=yc + 0.9 * np.sin((xs - fx0) * 1.6), mode="lines", hoverinfo="skip",
                             line=dict(color="#1f77b4", width=1.6), showlegend=False))
    alpha = float(np.mean(link.alpha_db_km))
    label((fx0 + fx1) / 2, yc + 11.5, f"Optical fibre ({cfg.fiber.material}, {cfg.fiber.mode.lower()})", 14, ink, bold=True)
    label((fx0 + fx1) / 2, yc + 8.6, f"{cfg.fiber.length_km:g} km  •  {alpha:.2f} dB/km", 13, "#52606d")
    chips = [("Attenuation", True), ("Dispersion", cfg.disp.enabled), ("Noise", True)]
    cx = fx0 + 3.0
    for k, (txt, on) in enumerate(chips):
        x0 = cx + k * 10.4
        rect(x0, 20.0 - dy, x0 + 9.6, 24.4 - dy, "#eaf6ea" if on else "#eceff1", line=ok if on else off, dash=None if on else "dot")
        label(x0 + 4.8, 22.2 - dy, txt, 12, ink if on else "#6b7280")
    label((fx0 + fx1) / 2, 18.3 - dy, "impairments added along the link", 11, "#7b8794")
    if amp_on and place == "In-line":
        for k in range(cfg.amp.n_inline):
            xa = fx0 + (fx1 - fx0) * (k + 1) / (cfg.amp.n_inline + 1)
            edfa(xa - 2.6, xa + 2.6, "EDFA", True, detail=False)
        label((fx0 + fx1) / 2, yc - 11.4, f"{cfg.amp.n_inline} in-line EDFA(s): +{cfg.amp.gain_db:g} dB, NF {cfg.amp.nf_db:g} dB each",
              11, "#52606d")
    # --- dispersion compensation block
    xdem0 = 82.0
    xamp0, xamp1 = 72.0, 78.0
    comp_on = cfg.disp.enabled and cfg.disp.compensation
    if comp_on:
        rect(fx1, yc - 3.2, fx1 + 4.8, yc + 3.2, "#e6f4ea", line=ok)
        label(fx1 + 2.4, yc, cfg.disp.method, 12, "#1e6b2f", bold=True)
        label(fx1 + 2.4, yc - 5.4, "compensator", 11, "#52606d")
        line(fx1 + 4.8, yc, xamp0 - 0.5, yc, "#1f77b4", 3)
    else:
        line(fx1, yc, xamp0 - 0.5, yc, "#1f77b4", 3)
    # --- pre-amplifier (or bypassed placeholder)
    if amp_on and place == "Pre-amplifier":
        edfa(xamp0, xamp1, "Pre-amp EDFA", True)
    elif not amp_on:
        edfa(xamp0, xamp1, "EDFA (OFF)", False)
    else:
        line(xamp0 - 0.5, yc, xamp1, yc, "#1f77b4", 3)
    line(xamp1, yc, xdem0, yc, "#1f77b4", 3)
    # --- demultiplexer
    poly([(xdem0, yc + 4), (xdem0 + 9.5, top), (xdem0 + 9.5, bot), (xdem0, yc - 4)], "#ece7fb", line="#7f77dd")
    label(xdem0 + 4.2, yc, "DEMUX", 14, "#4b3fb0", bold=True)
    label(xdem0 + 4.7, bot - 3.5, f"insertion loss {cfg.wdm.demux_il_db:g} dB", 12, "#52606d")
    # --- photodiodes and receivers
    for i, y in enumerate(ys):
        c = col_of.get(statuses[i], off)
        line(xdem0 + 9.5, y, xdem0 + 14.0, y, c, 3)
        poly([(xdem0 + 14.0, y + 1.8 * f), (xdem0 + 14.0, y - 1.8 * f), (xdem0 + 16.4, y)], "#f1f5f9", line=ink)
        line(xdem0 + 16.4, y - 2.0 * f, xdem0 + 16.4, y + 2.0 * f)
        line(xdem0 + 16.4, y, xdem0 + 18.2, y, c, 2)
        rect(xdem0 + 18.2, y - hh, xdem0 + 26.6, y + hh, c, line=ink, opacity=0.85)
        label(xdem0 + 22.4, y, f"Rx{i + 1}", fsz, "white", bold=True)
    label(xdem0 + 14.5, 68.0 + up, "Receivers", 15, bold=True)
    label(xdem0 + 14.5, 64.6 + up, "Photodiode (light ➜ current)", 12, "#52606d")
    label(xdem0 + 14.5, 61.4 + up, "box colour = channel status", 11, "#7b8794")

    # --- processing chain (second row)
    st_map = {s["stage"]: s["state"] for s in pipeline}
    blocks = [("Configuration<br>dashboard", st_map.get("Configuration", "ok")),
              ("Data<br>generation", st_map.get("Data generation", "ok")),
              ("Signal analysis<br>SNR / BER", st_map.get("Receiver", "ok")),
              ("Adaptive traffic<br>management", st_map.get("Traffic mgmt", "ok")),
              ("Fault detection<br>and recovery", st_map.get("Fault recovery", "ok")),
              ("Results<br>dashboard", "ok")]
    fill = {"ok": "#e6f4ea", "warn": "#fff1d6", "error": "#f9dede", "off": "#eceff1"}
    edge = {"ok": ok, "warn": warn, "error": bad, "off": off}
    w, gap, x = 15.5, 3.4, 1.0
    y0, y1 = 2.0 - dy, 11.0 - dy
    centers = []
    for k, (txt, state) in enumerate(blocks):
        rect(x, y0, x + w, y1, fill[state], line=edge[state], dash=None if state != "off" else "dot")
        label(x + w / 2, (y0 + y1) / 2, txt, 13, ink, bold=True)
        centers.append(x + w / 2)
        if k < len(blocks) - 1:
            arrow(x + w + 0.2, (y0 + y1) / 2, x + w + gap - 0.2, (y0 + y1) / 2)
        x += w + gap
    # receivers feed the signal-analysis block
    xr = xdem0 + 22.4
    line(xr, min(ys) - 9.0, xr, 15.0 - dy, "#52606d", 2, "dash")
    line(xr, 15.0 - dy, centers[2], 15.0 - dy, "#52606d", 2, "dash")
    arrow(centers[2], 15.0 - dy, centers[2], y1 + 0.2, "#52606d")
    label(centers[2] + 14.5, 17.6 - dy, "detected data", 11, "#7b8794")
    label(55, -1.2 - dy, "Processing chain (software)", 11, "#7b8794")

    fig.update_layout(shapes=shapes, annotations=notes, template=TEMPLATE, height=560 + 30 * max(0, n - 4),
                      font=dict(size=14), margin=dict(l=5, r=5, t=5, b=5), showlegend=False)
    fig.update_xaxes(visible=False, range=[-1, 111], fixedrange=True)
    fig.update_yaxes(visible=False, range=[-3 - dy, 71 + up], fixedrange=True)
    return fig


# ------------------------------------------------------------------------- waveforms
def fig_data(bits, sps_unused=None) -> go.Figure:
    x = np.arange(len(bits) + 1)
    y = np.append(bits, bits[-1])
    fig = _base("Original transmitted data", "Bit index", "Logic level", 240)
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", line=dict(shape="hv", color="#1f77b4"), name="Data bits"))
    fig.update_yaxes(tickvals=[0, 1], range=[-0.2, 1.3])
    return fig


def fig_optical(t_ns, p_w, title="Transmitted optical signal", name="Optical power") -> go.Figure:
    fig = _base(title, "Time (ns)", "Optical power (mW)")
    fig.add_trace(go.Scatter(x=t_ns, y=np.asarray(p_w) * 1e3, mode="lines", name=name, line=dict(color="#d62728")))
    return fig


def fig_rx_before_after(wave: dict, amp_on: bool) -> go.Figure:
    fig = _base("Received optical signal: before vs after amplification", "Time (ns)", "Optical power (uW)")
    fig.add_trace(go.Scatter(x=wave["t_ns"], y=wave["p_rx_noamp"] * 1e6, name="Without amplifier (bypassed)",
                             line=dict(color="#7f7f7f")))
    if amp_on:
        fig.add_trace(go.Scatter(x=wave["t_ns"], y=wave["p_rx_amp"] * 1e6, name="With amplifier",
                                 line=dict(color="#2ca02c")))
    else:
        _disabled(fig, "Amplifier OFF: gain and ASE bypassed")
    return fig


def fig_amp_in_out(wave: dict) -> go.Figure:
    fig = _base("Signal at the first amplifier: input vs output", "Time (ns)", "Optical power (mW)")
    if wave.get("p_amp_in") is None:
        return _disabled(fig, "Amplifier OFF")
    fig.add_trace(go.Scatter(x=wave["t_ns"], y=wave["p_amp_in"] * 1e3, name="Amplifier input", line=dict(color="#7f7f7f")))
    fig.add_trace(go.Scatter(x=wave["t_ns"], y=wave["p_amp_out"] * 1e3, name="Amplifier output", line=dict(color="#2ca02c")))
    return fig


def fig_noise(wave: dict, key: str, label: str, enabled: bool, extra_note: str = "") -> go.Figure:
    fig = _base(f"{label} noise waveform", "Time (ns)", "Noise current (uA)", 240)
    y = wave["comps"][key] * 1e6
    fig.add_trace(go.Scatter(x=wave["t_ns"], y=y, mode="lines", name=label, line=dict(width=1)))
    rms = float(np.std(y)) if enabled else 0.0
    fig.add_annotation(text=f"RMS = {rms:.3f} uA", xref="paper", yref="paper", x=0.99, y=0.97,
                       showarrow=False, font=dict(size=11))
    if not enabled:
        _disabled(fig, f"{label} noise DISABLED (excluded from total)" + extra_note)
    return fig


def fig_combined_noise(wave: dict) -> go.Figure:
    fig = _base("Combined noise (sum of enabled sources)", "Time (ns)", "Noise current (uA)", 260)
    fig.add_trace(go.Scatter(x=wave["t_ns"], y=wave["combined"] * 1e6, line=dict(color="#444", width=1), name="Combined"))
    fig.add_annotation(text=f"RMS = {float(np.std(wave['combined'])) * 1e6:.3f} uA", xref="paper", yref="paper",
                       x=0.99, y=0.97, showarrow=False)
    return fig


def fig_clean_vs_noisy(wave: dict) -> go.Figure:
    fig = _base("Clean vs noisy received signal (photocurrent after receiver filter)", "Time (ns)", "Current (uA)", 300)
    fig.add_trace(go.Scatter(x=wave["t_ns"], y=wave["i_clean"] * 1e6, name="Clean", line=dict(color="#2ca02c")))
    fig.add_trace(go.Scatter(x=wave["t_ns"], y=wave["noisy"] * 1e6, name="Noisy", line=dict(color="#d62728", width=1)))
    fig.add_hline(y=wave["threshold_a"] * 1e6, line=dict(dash="dash", color="#444"),
                  annotation_text="Decision threshold", annotation_position="top left")
    return fig


def fig_recovered(wave: dict) -> go.Figure:
    n = wave["n_show"]
    x = np.arange(n + 1)
    fig = _base("Recovered data vs original", "Bit index", "Logic level", 240)
    fig.add_trace(go.Scatter(x=x, y=np.append(wave["bits"], wave["bits"][-1]), name="Original",
                             line=dict(shape="hv", color="#1f77b4")))
    fig.add_trace(go.Scatter(x=x, y=np.append(wave["rec"], wave["rec"][-1]) + 0.03, name="Recovered",
                             line=dict(shape="hv", color="#ff7f0e", dash="dot")))
    err = np.where(wave["bits"] != wave["rec"])[0]
    if len(err):
        fig.add_trace(go.Scatter(x=err + 0.5, y=[1.15] * len(err), mode="markers", name="Bit error",
                                 marker=dict(color="red", size=9, symbol="x")))
    fig.update_yaxes(tickvals=[0, 1], range=[-0.2, 1.35])
    return fig


def fig_eye(wave: dict, noisy: bool = True, metrics: str = "") -> go.Figure:
    t, tr = (wave["eye_t"], wave["eye_traces"]) if noisy else (wave["eye_t_clean"], wave["eye_traces_clean"])
    fig = _base("Eye diagram" + (" (with noise)" if noisy else " (noise-free)"), "Time (UI)", "Current (uA)", 340)
    for k in range(min(len(tr), 150)):
        fig.add_trace(go.Scattergl(x=t, y=tr[k] * 1e6, mode="lines", line=dict(color="rgba(31,119,180,0.18)", width=1),
                                   showlegend=False, hoverinfo="skip"))
    fig.add_hline(y=wave["threshold_a"] * 1e6, line=dict(dash="dash", color="#d62728"))
    if metrics:
        fig.add_annotation(text=metrics, xref="paper", yref="paper", x=0.99, y=0.97, showarrow=False, align="right",
                           bgcolor="rgba(255,255,255,0.8)")
    return fig


def fig_spectrum(link, ch: int = 0) -> go.Figure:
    """Optical spectrum at the MUX output (before link) and at the receiver input (after link)."""
    from performance import w_to_dbm
    fig = _base("Optical spectrum before and after the fibre link", "Wavelength (nm)", "Power (dBm)", 340)
    lam = link.lam_nm
    for name, p, col in (("Before link (launch)", link.p_launch_w, "#1f77b4"),
                         ("After link (receiver input)", link.p_rx_w, "#d62728")):
        x, y = [], []
        for l, v in zip(lam, p):
            x += [l, l, l]
            y += [-80, float(w_to_dbm(v)) if v > 0 else -80, None]
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=name, line=dict(color=col, width=3)))
    if link.amp_on:
        dnu = 299792458.0 / (lam * 1e-9) ** 2 * 0.1e-9
        ase = w_to_dbm(2.0 * link.s_sp_rx * dnu)
        fig.add_trace(go.Scatter(x=lam, y=ase, mode="lines+markers", name="ASE in 0.1 nm", line=dict(color="#ff7f0e", dash="dot")))
    fig.update_yaxes(range=[-80, max(5, float(w_to_dbm(link.p_launch_w.max())) + 5)])
    return fig


def fig_rx_power_bars(table) -> go.Figure:
    fig = _base("Per-channel received power", "Channel", "Received power (dBm)", 300)
    cols = [{"OK": "#2ca02c", "DEGRADED": "#ff9f1c", "FAILED": "#d62728"}[s] for s in table["status"]]
    fig.add_trace(go.Bar(x=[f"Ch{c} ({w:.1f} nm)" for c, w in zip(table["channel"], table["wavelength_nm"])],
                         y=table["p_rx_dbm"].clip(lower=-60), marker_color=cols, name="Rx power"))
    return fig


def fig_power_profile(link, ch: int) -> go.Figure:
    from performance import w_to_dbm
    fig = _base("Power along the link (selected channel)", "Distance along fibre (km)", "Power (dBm)", 300)
    fig.add_trace(go.Scatter(x=link.profile_z, y=w_to_dbm(link.profile_p[:, ch]), mode="lines+markers",
                             line=dict(color="#1f77b4"), name="Signal power"))
    for a in link.amps:
        fig.add_vline(x=a["z_km"], line=dict(dash="dot", color="#ff7f0e"), annotation_text=a["name"].replace(" EDFA", ""))
    return fig


# ------------------------------------------------------------------------- performance charts
def fig_ber_vs_snr(snr_axis, ber_axis, op_snr, op_ber) -> go.Figure:
    fig = _base("BER versus SNR (Gaussian-noise model)", "SNR (dB)", "BER", 320)
    fig.add_trace(go.Scatter(x=snr_axis, y=ber_axis, name="BER = 0.5 erfc(sqrt(SNR/2))", line=dict(color="#1f77b4")))
    if op_snr > -50:
        fig.add_trace(go.Scatter(x=[op_snr], y=[max(op_ber, 1e-300)], mode="markers", name="This channel",
                                 marker=dict(size=12, color="#d62728", symbol="star")))
    fig.update_yaxes(type="log", range=[-15, 0])
    return fig


def fig_lines(df, x: str, ys: Dict[str, str], title: str, xlabel: str, ylabel: str, logy: bool = False,
              split: str | None = None, height: int = 300) -> go.Figure:
    fig = _base(title, xlabel, ylabel, height)
    for label, col in ys.items():
        if split:
            for case, g in df.groupby(split, sort=False):
                fig.add_trace(go.Scatter(x=g[x], y=g[col], mode="lines+markers", name=f"{case}"))
        else:
            fig.add_trace(go.Scatter(x=df[x], y=df[col], mode="lines+markers", name=label))
    if logy:
        fig.update_yaxes(type="log")
    return fig


def fig_bars(labels: Sequence[str], values: Sequence[float], title: str, ylabel: str, logy=False,
             height: int = 300, color="#1f77b4") -> go.Figure:
    fig = _base(title, "", ylabel, height)
    fig.add_trace(go.Bar(x=list(labels), y=list(values), marker_color=color))
    if logy:
        fig.update_yaxes(type="log")
    return fig


def fig_noise_contrib(std_ua: Dict[str, float]) -> go.Figure:
    labs = [nm.NOISE_LABELS[k] for k in nm.NOISE_NAMES]
    vals = [std_ua.get(k, 0.0) for k in nm.NOISE_NAMES]
    return fig_bars(labs, vals, "Noise contribution at the '1' level (RMS current)", "RMS noise (uA)", height=300)


def fig_pulses(p: dict) -> go.Figure:
    fig = _base("Pulse shape: input vs after fibre, with and without compensation", "Time (ps)",
                "Normalised power", 320)
    fig.add_trace(go.Scatter(x=p["t_ps"], y=p["input"], name="Input pulse", line=dict(color="#7f7f7f")))
    fig.add_trace(go.Scatter(x=p["t_ps"], y=p["no_comp"], name=f"After fibre (extra RMS {p['raw_ps']:.0f} ps)",
                             line=dict(color="#d62728")))
    fig.add_trace(go.Scatter(x=p["t_ps"], y=p["comp"], name=f"After compensation (extra RMS {p['comp_ps']:.0f} ps)",
                             line=dict(color="#2ca02c")))
    return fig


# ------------------------------------------------------------------------- traffic / fault
def fig_traffic(before, after, labels=("Without balancing", "With balancing")) -> go.Figure:
    fig = _base("Channel utilisation before / after", "Channel", "Utilisation (% of capacity)", 340)
    ch = [f"Ch{c}" for c in before.table["channel"]]
    fig.add_trace(go.Bar(x=ch, y=before.table["offered_util_pct"].replace(np.inf, 300).clip(upper=300), name=labels[0],
                         marker_color="#9aa0a6"))
    fig.add_trace(go.Bar(x=ch, y=after.table["offered_util_pct"].replace(np.inf, 300).clip(upper=300), name=labels[1],
                         marker_color="#1f77b4"))
    fig.add_hline(y=100, line=dict(color="#d62728", dash="dash"), annotation_text="Capacity")
    fig.update_layout(barmode="group")
    return fig


def fig_throughput(before, after, labels=("Before", "After")) -> go.Figure:
    fig = _base("Per-channel carried throughput", "Channel", "Throughput (Gb/s)", 320)
    ch = [f"Ch{c}" for c in before.table["channel"]]
    fig.add_trace(go.Bar(x=ch, y=before.table["carried_gbps"], name=labels[0], marker_color="#9aa0a6"))
    fig.add_trace(go.Bar(x=ch, y=after.table["carried_gbps"], name=labels[1], marker_color="#2ca02c"))
    fig.update_layout(barmode="group")
    return fig


def fig_fault_timeline(f) -> go.Figure:
    fig = _base("Throughput during failure and recovery (simulated timeline)", "Time (ms)", "Throughput (Gb/s)", 340)
    tl = f.timeline
    fig.add_trace(go.Scatter(x=tl.t_ms, y=tl.aggregate_gbps, name="Aggregate", line=dict(color="#1f77b4", shape="hv")))
    fig.add_trace(go.Scatter(x=tl.t_ms, y=tl.failed_traffic_gbps, name="Traffic of failed channel(s)",
                             line=dict(color="#d62728", shape="hv")))
    if f.detected:
        fig.add_vline(x=f.t_fail_ms, line=dict(color="#d62728", dash="dash"), annotation_text="Failure")
        fig.add_vline(x=f.t_detect_ms, line=dict(color="#ff9f1c", dash="dash"), annotation_text="Detected")
        if not np.isnan(f.t_recover_ms):
            fig.add_vline(x=f.t_recover_ms, line=dict(color="#2ca02c", dash="dash"), annotation_text="Recovered")
    return fig
