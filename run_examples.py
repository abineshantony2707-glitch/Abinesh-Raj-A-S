"""Generate the example configurations (examples/*.json) and sample results (examples/sample_results.md).

Run:  python run_examples.py
Load a JSON file in the dashboard by copying its values, or in code with SystemConfig.from_json(...).
All numbers are simulated by this simplified model; none are hardware measurements.
"""
from __future__ import annotations

import os

import pandas as pd

import fiber_models as fm
import performance as perf
import simulation as sim
from config import SystemConfig

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "examples")


def fiber_cfg(material, mode, profile, length_km, **extra):
    d = fm.defaults_for(material, mode, profile)
    cfg = SystemConfig()
    for k, v in d.items():
        setattr(cfg.fiber, k, v)
    cfg.fiber.material, cfg.fiber.mode, cfg.fiber.profile, cfg.fiber.length_km = material, mode, profile, length_km
    for k, v in extra.items():
        setattr(cfg.fiber, k, v)
    return cfg


def custom_channels(cfg, wavelengths, rate_gbps, power_dbm=0.0):
    w = cfg.wdm
    w.custom_enabled, w.n_channels = True, len(wavelengths)
    w.custom_wavelengths_nm = list(wavelengths) + [wavelengths[-1] + 10.0 * (i + 1) for i in range(16 - len(wavelengths))]
    w.bitrates_gbps = [rate_gbps] * 16
    w.powers_dbm = [power_dbm] * 16
    return cfg


def examples() -> dict:
    ex = {}
    # 1. DWDM over standard single-mode fibre
    ex["dwdm_smf_20km"] = SystemConfig()
    # 2. CWDM over single-mode fibre
    c = SystemConfig()
    c.wdm.scheme, c.wdm.n_channels, c.wdm.cwdm_start_nm, c.wdm.demux_bw_frac = "CWDM", 8, 1471.0, 0.6
    c.fiber.length_km = 40.0
    ex["cwdm_smf_40km"] = c
    # 3. Long-haul DWDM with in-line amplifiers and DSP dispersion compensation
    c = SystemConfig()
    c.fiber.length_km = 150.0
    c.amp.enabled, c.amp.placement, c.amp.n_inline, c.amp.gain_db, c.amp.nf_db = True, "In-line", 3, 20.0, 5.0
    c.amp.psat_dbm = 20.0
    c.disp.compensation, c.disp.method, c.disp.comp_fraction = True, "DSP", 0.95
    ex["dwdm_smf_150km_amplified_dsp"] = c
    # 4. Multimode (graded-index OM2-like) short-wave WDM, 4 channels, 50 m
    c = fiber_cfg("Silica", "Multimode", "Graded-index", 0.05)
    custom_channels(c, [850.0, 880.0, 910.0, 940.0], 10.0)
    c.txrx.responsivity, c.txrx.sensitivity_dbm, c.wdm.demux_bw_frac = 0.5, -20.0, 0.6
    c.wdm.mux_il_db = c.wdm.demux_il_db = 1.5
    ex["mmf_graded_swdm4_50m"] = c
    # 5. Plastic optical fibre, two visible channels (50 Mb/s each), 50 m
    c = fiber_cfg("POF", "Multimode", "Step-index", 0.05)
    custom_channels(c, [520.0, 650.0], 0.05, -3.0)
    c.txrx.responsivity, c.txrx.sensitivity_dbm, c.wdm.mux_il_db, c.wdm.demux_il_db = 0.4, -30.0, 2.0, 2.0
    c.txrx.spectral_width_nm, c.wdm.demux_bw_frac = 5.0, 0.5
    ex["pof_si_2ch_50m"] = c
    # 6. Plastic-clad silica, two channels, 100 m
    c = fiber_cfg("PCS", "Multimode", "Step-index", 0.1)
    custom_channels(c, [850.0, 1300.0], 0.05, -3.0)
    c.txrx.responsivity, c.txrx.sensitivity_dbm, c.wdm.mux_il_db, c.wdm.demux_il_db = 0.5, -30.0, 2.0, 2.0
    ex["pcs_si_2ch_100m"] = c
    return ex


def row(name, cfg):
    r = sim.run_simulation(cfg, with_sweeps=False)
    t = r.table
    live = t[t.status != "FAILED"]
    return dict(example=name, channels=r.link.n, nominal_gbps=r.nominal_gbps,
                rx_dbm_ch1=round(float(t.p_rx_dbm.iloc[0]), 1), snr_db_ch1=round(float(t.snr_db.iloc[0]), 1),
                q_worst=round(float(live.q.min()), 2), ber_worst=perf.fmt_ber(float(live.ber.max())),
                broadening_ps=round(float(t.broadening_ps.iloc[0]), 1),
                ok=f"{int((t.status == 'OK').sum())}/{r.link.n}")


def material_length_table() -> pd.DataFrame:
    rows = []
    cases = [("Silica SM (1550 nm)", "Silica", "Single-mode", "Step-index", 1550.0, 0.9, 1.0),
             ("Silica MM-GI (850 nm)", "Silica", "Multimode", "Graded-index", 850.0, 0.5, 1.0),
             ("Silica MM-SI (850 nm)", "Silica", "Multimode", "Step-index", 850.0, 0.5, 1.0),
             ("PCS 200/230 (850 nm)", "PCS", "Multimode", "Step-index", 850.0, 0.5, 1.0),
             ("POF PMMA (650 nm)", "POF", "Multimode", "Step-index", 650.0, 0.4, 1.0)]
    for label, mat, mode, prof, wl, resp, rate in cases:
        for L in (0.05, 0.1, 0.5, 2.0, 10.0):
            c = fiber_cfg(mat, mode, prof, L)
            custom_channels(c, [wl], rate, 0.0)
            c.wdm.mux_il_db = c.wdm.demux_il_db = 0.0
            c.txrx.responsivity, c.fiber.wavelength_nm = resp, wl
            c.txrx.spectral_width_nm = 5.0 if mat == "POF" else 0.03
            link = perf.compute_link(c)
            m = perf.evaluate_channel(c, link, 0)
            rows.append({"fibre": label, "length (km)": L, "loss (dB)": round(m["passive_loss_db"], 2),
                         "Rx (dBm)": round(m["p_rx_dbm"], 1), "broadening (ps)": round(m["broadening_ps"], 1),
                         "eye (%)": round(m["eye_open_pct"]), "Q": round(m["q"], 2), "BER (analytic)": m["ber_display"]})
    return pd.DataFrame(rows)


def length_amp_table() -> pd.DataFrame:
    base = SystemConfig()
    rows = []
    for L in (20, 60, 80, 100, 150):
        for label, kv in (("no amplifier", dict()),
                          ("pre-amp 15 dB", dict(enabled=True, placement="Pre-amplifier", gain_db=15.0)),
                          ("pre-amp + DSP comp.", dict(enabled=True, placement="Pre-amplifier", gain_db=15.0))):
            c = perf.variant(base, fiber={"length_km": float(L)}, amp=kv,
                             disp={"compensation": label.endswith("comp."), "method": "DSP"})
            m = perf.evaluate_channel(c, perf.compute_link(c), 0)
            rows.append({"length (km)": L, "case": label, "Rx (dBm)": round(m["p_rx_dbm"], 1),
                         "SNR (dB)": round(m["snr_db"], 1), "Q": round(m["q"], 2), "BER (analytic)": m["ber_display"]})
    return pd.DataFrame(rows)


def main():
    os.makedirs(OUT, exist_ok=True)
    ex = examples()
    for name, cfg in ex.items():
        with open(os.path.join(OUT, f"{name}.json"), "w") as f:
            f.write(cfg.to_json())
    summary = pd.DataFrame([row(n, c) for n, c in ex.items()])
    with open(os.path.join(OUT, "sample_results.md"), "w") as f:
        f.write("# Sample simulation results\n\nGenerated by `python run_examples.py`. "
                "All values are simulated by a simplified model, not hardware measurements.\n\n")
        f.write("## Example configurations (channel 1 unless noted)\n\n")
        f.write(summary.to_markdown(index=False) + "\n\n")
        f.write("## Fibre materials and lengths (single 1 Gb/s channel, 0 dBm launch, no MUX/DEMUX loss)\n\n")
        f.write(material_length_table().to_markdown(index=False) + "\n\n")
        f.write("## DWDM over SMF: length, amplifier and dispersion compensation (channel 1, 8 x 10 Gb/s)\n\n")
        f.write(length_amp_table().to_markdown(index=False) + "\n")
    print(summary.to_string(index=False))
    print(f"\nWrote {len(ex)} configs and sample_results.md to {OUT}")


if __name__ == "__main__":
    main()
