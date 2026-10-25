"""Orchestrates the complete WDM pipeline and collects everything the dashboard shows.

Configuration -> data generation -> transmitter -> MUX -> fibre (attenuation, dispersion, noise)
-> amplifier -> DEMUX -> receiver -> SNR/BER -> traffic management -> fault detection/recovery.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd

import amplifier as amp_m
import fault_recovery as fr
import fiber_models as fm
import performance as perf
import traffic_management as tm
import wdm_system as ws
from config import SystemConfig


@dataclass
class SimResults:
    cfg: SystemConfig
    link: perf.LinkResult
    link_noamp: perf.LinkResult
    table: pd.DataFrame
    issues: List[tuple]
    sel: dict
    traffic_off: tm.TrafficResult
    traffic_on: tm.TrafficResult
    fault: fr.FaultResult
    status_table: pd.DataFrame
    pipeline: List[dict]
    sweeps: Dict[str, pd.DataFrame] = field(default_factory=dict)
    comparisons: Dict[str, pd.DataFrame] = field(default_factory=dict)
    pulses: dict = field(default_factory=dict)
    nominal_gbps: float = 0.0
    received_gbps: float = 0.0
    ch: int = 0                       # channel shown in detail (0-based)
    ch_note: str = ""
    all_failed: bool = False


def collect_issues(cfg: SystemConfig, link: perf.LinkResult) -> List[tuple]:
    lam = list(link.lam_nm)
    issues = fm.validate_fiber(cfg.fiber, lam) + ws.validate_wdm(cfg.wdm, cfg.fiber, cfg.amp.enabled)
    a = cfg.amp
    if a.enabled:
        g = 10 ** (a.gain_db / 10)
        nfmin = amp_m.quantum_limit_nf_db(g)
        if a.nf_db < nfmin - 1e-6:
            issues.append(("warning", f"Amplifier noise figure {a.nf_db:.1f} dB is below the quantum limit "
                                      f"{nfmin:.1f} dB for {a.gain_db:.0f} dB gain; n_sp clamped to 1."))
        for rec in link.amps:
            if rec["saturated"]:
                issues.append(("info", f"{rec['name']} is output-power limited: gain reduced from "
                                       f"{rec['requested_gain_db']:.1f} to {rec['gain_db']:.1f} dB (total power limit)."))
    if cfg.txrx.rx_bw_frac * 2 >= cfg.txrx.sps:
        issues.append(("error", "Receiver bandwidth too high for the samples-per-bit setting; increase samples per bit."))
    if cfg.noise.ase and not a.enabled:
        issues.append(("info", "ASE noise is enabled but the amplifier is OFF: ASE contributes nothing (bypassed)."))
    if cfg.fiber.material in ("POF", "PCS") and cfg.txrx.responsivity > 0.6 and cfg.fiber.wavelength_nm < 900:
        issues.append(("info", "Responsivity 0.9 A/W is typical for InGaAs at 1550 nm; a silicon PIN at "
                               "650-850 nm is more like 0.4-0.5 A/W."))
    return issues


def pipeline_status(cfg: SystemConfig, link: perf.LinkResult, table: pd.DataFrame) -> List[dict]:
    n_fail = int(link.failed.sum())
    worst = table["status"].tolist()
    q_state = "error" if "FAILED" in worst else "warn" if "DEGRADED" in worst else "ok"
    return [
        dict(stage="Configuration", state="ok", detail=f"{cfg.wdm.scheme}, {link.n} ch"),
        dict(stage="Data generation", state="ok", detail=f"{cfg.txrx.modulation}"),
        dict(stage="WDM transmitter", state="error" if n_fail == link.n else "warn" if n_fail else "ok",
             detail=f"{n_fail} down" if n_fail else "all up"),
        dict(stage="Multiplexer", state="ok", detail=f"IL {cfg.wdm.mux_il_db:.1f} dB"),
        dict(stage="Optical fibre", state="ok", detail=f"{cfg.fiber.material} {cfg.fiber.length_km:g} km"),
        dict(stage="Dispersion / noise", state="ok" if cfg.disp.enabled else "off",
             detail="dispersion " + ("ON" if cfg.disp.enabled else "OFF")),
        dict(stage="Optical amplifier", state="ok" if cfg.amp.enabled else "off",
             detail=cfg.amp.placement if cfg.amp.enabled else "bypassed"),
        dict(stage="Demultiplexer", state="ok", detail=f"IL {cfg.wdm.demux_il_db:.1f} dB"),
        dict(stage="Receiver", state=q_state, detail="BER/SNR"),
        dict(stage="Traffic mgmt", state="ok" if cfg.traffic.load_balancing else "off",
             detail="balancing " + ("ON" if cfg.traffic.load_balancing else "OFF")),
        dict(stage="Fault recovery", state="warn" if n_fail else ("ok" if cfg.fault.protection else "off"),
             detail="fault active" if n_fail else "monitoring"),
    ]


def run_simulation(cfg: SystemConfig, with_sweeps: bool = True) -> SimResults:
    link = perf.compute_link(cfg)
    link_na = perf.compute_link(cfg, amp_enabled=False)
    flags = perf.noise_flags(cfg)
    table = perf.evaluate_all(cfg, link, flags)
    n = link.n
    ch = min(max(cfg.selected_channel, 0), n - 1)
    note, all_failed = "", bool(link.failed.all())
    if link.failed[ch] and not all_failed:
        alt = int(np.where(~link.failed)[0][0])
        note = (f"Channel {ch + 1} is failed (no light), so the detailed views show the first healthy "
                f"channel, {alt + 1}.")
        ch = alt
    if all_failed:
        sel = perf.evaluate_channel(cfg, link, ch, flags)
    else:
        sel = perf.evaluate_channel(cfg, link, ch, flags, with_waveforms=True, link_noamp=link_na)
    issues = collect_issues(cfg, link)

    rates = list(cfg.wdm.bitrates_gbps[:n])
    bers = table["ber"].tolist()
    t = cfg.traffic
    common = dict(threshold=t.threshold, priority_scheduling=t.priority_scheduling, packet_bytes=t.packet_bytes,
                  reroute_overhead_us=t.reroute_overhead_us, failed=cfg.fault.failed_channels, ber=bers,
                  protection=cfg.fault.protection,
                  backup=cfg.fault.backup_channel if cfg.fault.backup_mode.startswith("Dedicated") else None)
    d, p, g = t.demand_gbps[:n], t.priority[:n], t.group[:n]
    tr_off = tm.allocate_traffic(d, rates, p, g, load_balancing=False, **common)
    tr_on = tm.allocate_traffic(d, rates, p, g, load_balancing=t.load_balancing, **common)
    fault = fr.simulate_fault(cfg, rates, bers)
    status = fr.channel_status_table(table, cfg.fault.failed_channels)

    res = SimResults(cfg=cfg, link=link, link_noamp=link_na, table=table, issues=issues, sel=sel,
                     traffic_off=tr_off, traffic_on=tr_on, fault=fault, status_table=status,
                     pipeline=pipeline_status(cfg, link, table),
                     nominal_gbps=ws.nominal_capacity_gbps(cfg.wdm), received_gbps=tr_on.goodput_gbps,
                     ch=ch, ch_note=note, all_failed=all_failed)
    if with_sweeps and not all_failed:
        lmax = max(100.0, cfg.fiber.length_km * 1.1)
        res.sweeps["length"] = perf.sweep_length(cfg, ch, np.linspace(1.0, lmax, 24))
        res.sweeps["spacing"] = perf.sweep_spacing(cfg, ch)
        res.sweeps["rate"] = perf.sweep_rate(cfg, ch)
        res.comparisons["amplifier"] = perf.compare_amplifier(cfg, ch)
        res.comparisons["dispersion"] = perf.compare_dispersion(cfg, ch)
        res.comparisons["noise"] = perf.compare_noise(cfg, ch, link)
        res.pulses = perf.pulse_shapes(cfg, ch)
    return res
