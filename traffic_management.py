"""Network-level traffic allocation, priority scheduling and adaptive load balancing.

This is a NETWORK-LEVEL simulation. There is no physical optical switching fabric: traffic can be
moved from one wavelength to another only between channels in the same compatibility `group`
(think of a wavelength-converting / electronic grooming node), and every move adds a configurable
re-routing latency. Capacity of a channel = its bit rate, derated by link quality (failed channel = 0).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


@dataclass
class TrafficResult:
    table: pd.DataFrame
    moved: np.ndarray                 # moved[i, j] = Gb/s moved from channel i to channel j
    offered_gbps: float
    carried_gbps: float
    goodput_gbps: float
    dropped_gbps: float
    loss_pct: float
    unaccommodated_gbps: float        # traffic that could not be placed anywhere (after rebalancing)
    n_congested: int


def packet_error_rate(ber: float, packet_bytes: int) -> float:
    """PER = 1 - (1 - BER)^bits  ~  bits x BER for small BER."""
    bits = packet_bytes * 8
    if ber <= 0:
        return 0.0
    if ber >= 0.5:
        return 1.0
    return float(-math.expm1(bits * math.log1p(-ber)))


def allocate_traffic(demand: Sequence[float], capacity: Sequence[float], priority: Sequence[int],
                     group: Sequence[int], threshold: float = 0.8, load_balancing: bool = True,
                     priority_scheduling: bool = True, failed: Sequence[int] = (), protection: bool = False,
                     backup: Optional[int] = None, ber: Sequence[float] | None = None,
                     packet_bytes: int = 1500, reroute_overhead_us: float = 50.0) -> TrafficResult:
    """Allocate offered traffic to channels.

    * Without balancing the original allocation is kept (traffic of channel i stays on channel i).
    * With balancing, traffic above `threshold x capacity` is moved to channels of the same group that
      still have headroom below their own threshold. Sources are handled in priority order, and the
      lowest-priority flows leave an overloaded channel first.
    * With protection, traffic of FAILED channels is re-routed to `backup` (up to its full capacity)
      or, if no backup is given, to the remaining channels of the same group (up to threshold).
    * Whatever still exceeds a channel's capacity is dropped: lowest priority first when priority
      scheduling is on, otherwise proportionally.
    """
    n = len(demand)
    demand = np.asarray(demand, dtype=float)
    cap = np.asarray(capacity, dtype=float).copy()
    prio = np.asarray(priority, dtype=int)
    grp = np.asarray(group, dtype=int)
    failed_set = set(int(i) for i in failed)
    for i in failed_set:
        cap[i] = 0.0
    thr_cap = threshold * cap
    # flows[j] = list of [origin, amount, priority] currently placed on channel j
    flows: List[List[list]] = [[[i, float(demand[i]), int(prio[i])]] if demand[i] > 0 else [] for i in range(n)]
    load = demand.copy()
    moved = np.zeros((n, n))

    def room(j: int, full: bool = False) -> float:
        lim = cap[j] if full else thr_cap[j]
        return max(lim - load[j], 0.0)

    def move(i: int, amount: float, targets: List[int], full: bool = False) -> float:
        """Move up to `amount` from channel i; lowest-priority flows first. Returns amount moved."""
        left = amount
        for j in sorted(targets, key=lambda t: -room(t, full)):
            if left <= 1e-12:
                break
            if j == i or j in failed_set:
                continue
            h = room(j, full)
            if h <= 1e-12:
                continue
            x = min(h, left)
            taken = x
            for fl in sorted(flows[i], key=lambda f: -f[2]):         # lowest priority first
                if taken <= 1e-12:
                    break
                t = min(fl[1], taken)
                fl[1] -= t
                flows[j].append([fl[0], t, fl[2]])
                taken -= t
            flows[i] = [f for f in flows[i] if f[1] > 1e-12]
            load[i] -= x
            load[j] += x
            moved[i, j] += x
            left -= x
        return amount - left

    # 1) protection switching of failed channels
    if protection:
        for i in sorted(failed_set, key=lambda k: prio[k]):
            if load[i] <= 0:
                continue
            if backup is not None and backup not in failed_set and backup != i:
                move(i, load[i], [backup], full=True)
            if load[i] > 1e-12:
                peers = [j for j in range(n) if grp[j] == grp[i] and j != i]
                move(i, load[i], peers, full=False)

    # 2) adaptive load balancing of congested channels
    if load_balancing:
        order = sorted([i for i in range(n) if cap[i] > 0 and load[i] > thr_cap[i] + 1e-12],
                       key=lambda k: (prio[k], -(load[k] - thr_cap[k])))
        for i in order:
            excess = load[i] - thr_cap[i]
            peers = [j for j in range(n) if grp[j] == grp[i] and j != i and cap[j] > 0]
            move(i, excess, peers, full=False)

    # 3) capacity enforcement (priority scheduling or proportional drop)
    carried = np.zeros(n)
    dropped_by_origin = np.zeros(n)
    for j in range(n):
        if load[j] <= 0:
            continue
        if load[j] <= cap[j] + 1e-12:
            carried[j] = load[j]
            continue
        carried[j] = cap[j]
        short = load[j] - cap[j]
        if priority_scheduling:
            for fl in sorted(flows[j], key=lambda f: -f[2]):       # drop lowest priority first
                if short <= 1e-12:
                    break
                d = min(fl[1], short)
                dropped_by_origin[fl[0]] += d
                short -= d
        else:
            tot = sum(f[1] for f in flows[j])
            for fl in flows[j]:
                dropped_by_origin[fl[0]] += (load[j] - cap[j]) * fl[1] / tot
    ber_arr = np.zeros(n) if ber is None else np.asarray(ber, dtype=float)
    per = np.array([packet_error_rate(b, packet_bytes) for b in ber_arr])
    goodput = carried * (1.0 - per)
    util = np.divide(carried, cap, out=np.zeros(n), where=cap > 0)
    util_off = np.divide(load, cap, out=np.where(load > 0, np.inf, 0.0), where=cap > 0)
    mu = cap * 1e9 / (packet_bytes * 8.0)                       # packets / s
    lam_ = carried * 1e9 / (packet_bytes * 8.0)
    lat = np.full(n, np.nan)
    moved_in = moved.sum(axis=0)
    for j in range(n):
        if cap[j] > 0 and mu[j] - lam_[j] > 1e-9 * max(mu[j], 1.0):
            lat[j] = 1e6 / (mu[j] - lam_[j])
            if carried[j] > 0:
                lat[j] += reroute_overhead_us * min(moved_in[j] / carried[j], 1.0)
    status = []
    for j in range(n):
        if j in failed_set:
            status.append("FAILED")
        elif load[j] > cap[j] + 1e-9:
            status.append("OVERLOAD")
        elif load[j] > thr_cap[j] + 1e-9:
            status.append("CONGESTED")
        else:
            status.append("OK")
    table = pd.DataFrame(dict(
        channel=np.arange(1, n + 1), priority=prio, group=grp, demand_gbps=demand, capacity_gbps=cap,
        placed_gbps=load, carried_gbps=carried, goodput_gbps=goodput,
        utilisation_pct=100.0 * util, offered_util_pct=100.0 * util_off,
        dropped_gbps=dropped_by_origin, latency_us=lat, state=status))
    # traffic left on a failed channel (cap 0) is dropped in step 3, so it is already counted here
    total_dropped = float(dropped_by_origin.sum())
    carried_total = float(carried.sum())
    offered = float(demand.sum())
    return TrafficResult(
        table=table, moved=moved, offered_gbps=offered, carried_gbps=carried_total,
        goodput_gbps=float(goodput.sum()), dropped_gbps=total_dropped,
        loss_pct=100.0 * total_dropped / offered if offered > 0 else 0.0,
        unaccommodated_gbps=total_dropped, n_congested=int(sum(s in ("CONGESTED", "OVERLOAD") for s in status)))
