"""Channel failure detection, protection switching and recovery (simulated timeline).

IMPORTANT: all times here are SIMULATED values derived from the configured monitoring period,
debounce count and switch time. They are NOT measurements of real hardware switching time.

Detection model: a power monitor polls every `poll_ms`. A failure at t_fail is first seen at the next
poll and is declared after `debounce` consecutive polls below threshold:
    t_detect = (floor(t_fail / poll) + debounce) * poll
Recovery completes after the protection-switch time:  t_recover = t_detect + switch_ms.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from config import SystemConfig
from traffic_management import TrafficResult, allocate_traffic


@dataclass
class FaultResult:
    failed: List[int]
    detected: bool
    recovered: bool
    t_fail_ms: float
    t_detect_ms: float
    t_recover_ms: float
    recovery_time_ms: float          # t_recover - t_fail (simulated)
    outage_lost_mbit: float          # traffic lost during the detection + switching window
    steady_lost_gbps: float          # traffic still lost after recovery (no capacity left)
    rerouted_gbps: float
    before: TrafficResult
    during: TrafficResult
    after: TrafficResult
    timeline: pd.DataFrame
    messages: List[str]


def detection_time_ms(t_fail: float, poll: float, debounce: int) -> float:
    return (math.floor(t_fail / poll) + max(int(debounce), 1)) * poll


def channel_status_table(df: pd.DataFrame, failed: List[int]) -> pd.DataFrame:
    """Status + alert text per channel from the performance table (columns: channel, status, ber, p_rx_dbm)."""
    out = df[["channel", "wavelength_nm", "p_rx_dbm", "snr_db", "ber_display", "status"]].copy()
    alerts = []
    for _, r in df.iterrows():
        if r["status"] == "FAILED" and (int(r["channel"]) - 1) in failed:
            alerts.append("LOSS OF SIGNAL: channel disabled")
        elif r["status"] == "FAILED":
            alerts.append("BER above 1e-3: channel unusable")
        elif r["status"] == "DEGRADED":
            alerts.append("Degraded: BER > 1e-9 or Rx power below receiver sensitivity")
        else:
            alerts.append("")
    out["alert"] = alerts
    return out


def simulate_fault(cfg: SystemConfig, rates_gbps: List[float], ber: List[float]) -> FaultResult:
    t, fa = cfg.traffic, cfg.fault
    n = len(rates_gbps)
    failed = [i for i in fa.failed_channels if i < n]
    demand = np.array(t.demand_gbps[:n], dtype=float)
    prio = t.priority[:n]
    grp = t.group[:n]
    kw = dict(threshold=t.threshold, load_balancing=t.load_balancing, priority_scheduling=t.priority_scheduling,
              packet_bytes=t.packet_bytes, reroute_overhead_us=t.reroute_overhead_us)
    ber0 = np.zeros(n)
    before = allocate_traffic(demand, rates_gbps, prio, grp, ber=ber0, **kw)
    during = allocate_traffic(demand, rates_gbps, prio, grp, failed=failed, protection=False, ber=ber0, **kw)
    backup = None
    if fa.backup_mode.startswith("Dedicated") and fa.backup_channel < n and fa.backup_channel not in failed:
        backup = fa.backup_channel
    after = allocate_traffic(demand, rates_gbps, prio, grp, failed=failed, protection=fa.protection,
                             backup=backup, ber=ber0, **kw)

    msgs: List[str] = []
    detected = bool(failed)
    t_det = detection_time_ms(fa.t_fail_ms, fa.poll_ms, fa.debounce) if detected else float("nan")
    t_rec = t_det + fa.switch_ms if (detected and fa.protection) else float("nan")
    lost_during = float(sum(demand[i] for i in failed))
    rerouted = float(after.moved[failed, :].sum()) if failed else 0.0
    steady_lost = max(lost_during - rerouted, 0.0) if failed else 0.0
    recovered = bool(failed) and fa.protection and steady_lost <= 1e-9
    if failed:
        msgs.append(f"Failure of channel(s) {[i + 1 for i in failed]} detected after "
                    f"{t_det - fa.t_fail_ms:.0f} ms (simulated monitoring model).")
        if not fa.protection:
            msgs.append("Protection switching is OFF: the failed traffic is lost until the channel is restored.")
        elif recovered:
            msgs.append(f"Traffic fully recovered (simulated recovery time {t_rec - fa.t_fail_ms:.0f} ms).")
        else:
            msgs.append(f"Traffic only partly recovered: {steady_lost:.2f} Gb/s cannot be accommodated.")
        msgs.append("Times are simulated from the configured parameters, not measured hardware switching times.")
    else:
        msgs.append("No channel has been failed. Use 'Trigger failure' to simulate a fault.")

    end = max(fa.t_end_ms, (t_rec if not math.isnan(t_rec) else t_det if not math.isnan(t_det) else 0) + 100)
    tt = np.arange(0.0, end + 1, 1.0)
    agg = np.where(tt < fa.t_fail_ms, before.carried_gbps, during.carried_gbps)
    if not math.isnan(t_rec):
        agg = np.where(tt >= t_rec, after.carried_gbps, agg)
    fch = np.zeros_like(tt)
    for i in failed:
        fch += np.where(tt < fa.t_fail_ms, before.table.carried_gbps[i], 0.0)
    if not math.isnan(t_rec):
        for i in failed:
            fch += np.where(tt >= t_rec, rerouted / max(len(failed), 1), 0.0)
    timeline = pd.DataFrame(dict(t_ms=tt, aggregate_gbps=agg, failed_traffic_gbps=fch))
    outage_mbit = 0.0
    if failed:
        end_out = t_rec if not math.isnan(t_rec) else end
        outage_mbit = lost_during * (end_out - fa.t_fail_ms) / 1000.0 * 1000.0   # Gb/s * s = Gb -> Mb
    return FaultResult(failed=failed, detected=detected, recovered=recovered, t_fail_ms=fa.t_fail_ms,
                       t_detect_ms=t_det, t_recover_ms=t_rec,
                       recovery_time_ms=(t_rec - fa.t_fail_ms) if not math.isnan(t_rec) else float("nan"),
                       outage_lost_mbit=outage_mbit, steady_lost_gbps=steady_lost, rerouted_gbps=rerouted,
                       before=before, during=during, after=after, timeline=timeline, messages=msgs)
