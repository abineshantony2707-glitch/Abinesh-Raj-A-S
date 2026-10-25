"""Shared configuration dataclasses, physical constants and parameter provenance.

Every number the simulator uses is either
  * SOURCED   - a physical constant or a textbook / standards value,
  * CALCULATED - derived from other parameters by an equation, or
  * ASSUMED   - a representative default chosen by the author (NOT a measurement).
Nothing in this project is a measured value from real hardware.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict, fields
from typing import List

# ----------------------------------------------------------------- constants (SOURCED, CODATA)
C = 299_792_458.0            # speed of light, m/s
H_PLANCK = 6.62607015e-34    # Planck constant, J s
Q_E = 1.602176634e-19        # elementary charge, C
K_B = 1.380649e-23           # Boltzmann constant, J/K
MAX_CH = 16                  # maximum number of WDM channels supported by the UI

BER_OK = 1e-9                # channel considered healthy at or below this BER
BER_FAIL = 1e-3              # channel considered failed/unusable above this BER


def _rep(value, n: int = MAX_CH) -> list:
    return [value] * n


# ----------------------------------------------------------------- configuration sections
@dataclass
class FiberConfig:
    material: str = "Silica"            # Silica | POF | PCS
    mode: str = "Single-mode"           # Single-mode | Multimode
    profile: str = "Step-index"         # Step-index | Graded-index
    length_km: float = 20.0
    core_um: float = 8.2
    clad_um: float = 125.0
    n_core: float = 1.4682
    n_clad: float = 1.4633
    wavelength_nm: float = 1550.0       # operating (reference) wavelength
    atten_override_db_km: float | None = None
    disp_override_ps_nm_km: float | None = None
    pmd_ps_sqrtkm: float = 0.1
    modal_bw_mhz_km: float = 0.0        # effective modal bandwidth (graded-index multimode)
    n_connectors: int = 2
    connector_loss_db: float = 0.5
    splice_spacing_km: float = 2.0
    splice_loss_db: float = 0.05
    margin_db: float = 0.0


@dataclass
class WDMConfig:
    scheme: str = "DWDM"                # DWDM | CWDM
    n_channels: int = 8
    center_nm: float = 1550.12          # DWDM centre wavelength (193.4 THz)
    cwdm_start_nm: float = 1471.0
    cwdm_spacing_nm: float = 20.0
    dwdm_spacing_ghz: float = 100.0
    custom_enabled: bool = False
    custom_wavelengths_nm: List[float] = field(
        default_factory=lambda: [round(1546.92 + 0.8 * i, 3) for i in range(MAX_CH)])
    bitrates_gbps: List[float] = field(default_factory=lambda: _rep(10.0))
    powers_dbm: List[float] = field(default_factory=lambda: _rep(0.0))
    mux_il_db: float = 3.0
    demux_il_db: float = 3.0
    demux_bw_frac: float = 0.5          # demux 3 dB bandwidth as a fraction of channel spacing
    demux_order: int = 1                # filter order (1 = Gaussian)
    isolation_db: float = 40.0          # non-adjacent isolation floor (assumed)


@dataclass
class TxRxConfig:
    n_bits: int = 256                   # bits shown in waveforms / eye
    ber_sim_bits: int = 20000           # bits in the Monte-Carlo BER run
    sps: int = 16                       # samples per bit
    seed: int = 42
    use_seed: bool = True
    modulation: str = "NRZ-OOK"         # NRZ-OOK | RZ-OOK
    extinction_db: float = 10.0
    rise_frac: float = 0.25             # transmitter 10-90 % rise time / bit period
    rin_db_hz: float = -150.0
    linewidth_mhz: float = 10.0
    spectral_width_nm: float = 0.03     # RMS source spectral width (chirp), assumed
    responsivity: float = 0.9           # A/W
    rx_bw_frac: float = 0.7             # receiver 3 dB bandwidth / bit rate
    rx_filter: str = "Bessel"           # Bessel (Gaussian, low ringing) | Butterworth
    rx_order: int = 4                   # Butterworth order (ignored for Bessel)
    sensitivity_dbm: float = -24.0
    dark_current_na: float = 1.0
    temperature_k: float = 300.0
    load_ohm: float = 500.0
    rx_nf_db: float = 3.0
    jitter_ps: float = 2.0
    threshold: str = "Optimum"          # Optimum | Midpoint


@dataclass
class NoiseConfig:
    thermal: bool = True
    shot: bool = True
    laser_rin: bool = True
    ase: bool = True                    # only has an effect while an amplifier is ON
    crosstalk: bool = True
    background: bool = False
    jitter: bool = False
    bg_dbm_per_nm: float = -45.0        # background optical noise, per polarisation, assumed


@dataclass
class AmpConfig:
    enabled: bool = False
    placement: str = "Pre-amplifier"    # Booster | In-line | Pre-amplifier
    gain_db: float = 15.0
    nf_db: float = 5.0
    n_inline: int = 1
    saturation: bool = True
    psat_dbm: float = 17.0              # total output power limit


@dataclass
class DispConfig:
    enabled: bool = True
    compensation: bool = False
    method: str = "DSP"                 # DSP | DCF
    comp_fraction: float = 0.9          # fraction of chromatic broadening removed
    dcf_loss_db: float = 5.0            # DCF module insertion loss (assumed)


@dataclass
class TrafficConfig:
    demand_gbps: List[float] = field(
        default_factory=lambda: [9.5, 6.0, 3.0, 8.8, 2.0, 5.0, 4.0, 7.0] * 2)
    priority: List[int] = field(default_factory=lambda: [1, 2, 3, 1, 3, 2, 3, 2] * 2)
    group: List[int] = field(default_factory=lambda: _rep(1))
    threshold: float = 0.8
    load_balancing: bool = True
    priority_scheduling: bool = True
    packet_bytes: int = 1500
    reroute_overhead_us: float = 50.0


@dataclass
class FaultConfig:
    failed_channels: List[int] = field(default_factory=list)
    protection: bool = True
    backup_mode: str = "Reroute to remaining channels"   # or "Dedicated backup channel"
    backup_channel: int = 7
    t_fail_ms: float = 100.0
    poll_ms: float = 10.0
    debounce: int = 3
    switch_ms: float = 50.0
    t_end_ms: float = 400.0


_SECTIONS = {
    "fiber": FiberConfig, "wdm": WDMConfig, "txrx": TxRxConfig, "noise": NoiseConfig,
    "amp": AmpConfig, "disp": DispConfig, "traffic": TrafficConfig, "fault": FaultConfig,
}


@dataclass
class SystemConfig:
    fiber: FiberConfig = field(default_factory=FiberConfig)
    wdm: WDMConfig = field(default_factory=WDMConfig)
    txrx: TxRxConfig = field(default_factory=TxRxConfig)
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    amp: AmpConfig = field(default_factory=AmpConfig)
    disp: DispConfig = field(default_factory=DispConfig)
    traffic: TrafficConfig = field(default_factory=TrafficConfig)
    fault: FaultConfig = field(default_factory=FaultConfig)
    selected_channel: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    @staticmethod
    def from_dict(d: dict) -> "SystemConfig":
        kw = {}
        for name, cls in _SECTIONS.items():
            names = {f.name for f in fields(cls)}
            kw[name] = cls(**{k: v for k, v in d.get(name, {}).items() if k in names})
        kw["selected_channel"] = int(d.get("selected_channel", 0))
        return SystemConfig(**kw)

    @staticmethod
    def from_json(s: str) -> "SystemConfig":
        return SystemConfig.from_dict(json.loads(s))


# ----------------------------------------------------------------- parameter provenance
PARAMETER_PROVENANCE = [
    ("Physical constants c, h, q, k", "CODATA 2018 exact values", "Sourced"),
    ("Silica attenuation model A/lambda^4 + B + IR tail + OH peak",
     "Rayleigh + IR absorption form and IR constants from Agrawal, Fiber-Optic Communication Systems; "
     "A, B and OH amplitude are representative values (result ~0.19 dB/km at 1550 nm)", "Sourced form / Assumed values"),
    ("POF (PMMA) attenuation table", "Approximate shape of published PMMA-POF spectra; values rounded", "Assumed"),
    ("PCS attenuation", "Representative 4-6 dB/km class value", "Assumed"),
    ("Silica chromatic dispersion D = S0/4 (lambda - lambda0^4/lambda^3)",
     "ITU-T G.652 form, S0 = 0.092 ps/nm^2/km, lambda0 = 1313 nm", "Sourced"),
    ("POF chromatic dispersion (-300 ps/nm/km)", "Representative PMMA value", "Assumed"),
    ("Step-index modal dispersion", "Ray-optics result, RMS = L n1 Delta / (2 sqrt3 c)", "Calculated"),
    ("Graded-index modal bandwidth (EMB)", "OM2-like 500 MHz km for silica; POF/PCS values assumed", "Assumed"),
    ("Fibre geometry / refractive-index presets", "Representative commercial-class values", "Assumed"),
    ("Receiver: responsivity, load resistance, noise figure, dark current", "Typical PIN-receiver values", "Assumed"),
    ("RIN -150 dB/Hz, source RMS width 0.03 nm, jitter 2 ps", "Typical DFB / driver values", "Assumed"),
    ("EDFA gain, noise figure, saturation power", "Typical EDFA datasheet class values", "Assumed"),
    ("EDFA ASE: n_sp = (NF G - 1)/(2 (G - 1)), S_sp = n_sp (G - 1) h nu", "Standard EDFA noise theory", "Sourced"),
    ("Thermal, shot, RIN, signal-ASE and ASE-ASE beat noise", "Standard receiver noise theory (Agrawal; Olsson 1989)", "Sourced"),
    ("BER = 0.5 erfc(Q / sqrt 2), Q = (I1 - I0)/(s1 + s0)", "Gaussian-noise approximation", "Sourced"),
    ("Dispersion penalty 5 log10(1 + 2 pi (B sigma)^2)", "Gaussian-pulse ISI approximation (Agrawal)", "Sourced"),
    ("Connector / splice / margin losses", "Common engineering defaults", "Assumed"),
    ("MUX/DEMUX insertion loss, bandwidth, isolation", "Typical AWG-class values", "Assumed"),
    ("Protection switching 50 ms, monitoring period 10 ms", "Typical target figures; NOT hardware measurements", "Assumed"),
    ("All simulation outputs", "Produced by this model only", "Calculated (no measured data)"),
]
