"""Gas absorption: line-by-line cross sections and the high-resolution optical-depth database.

Two sources of spectroscopy are supported:

* **HITRAN line lists** (``.par`` 160-character format). Download them with
  :func:`fetch_hitran` (requires the optional ``hitran-api`` package and network
  access to hitran.org) or from https://hitran.org/lbl/ and load with
  :func:`read_hitran_par`. This is what you should use for science.
* **Synthetic demo line lists** from :func:`demo_linelist`. These are *not*
  real spectroscopy: they mimic the position, rotational structure and depth of
  the main VNIR-SWIR bands of H2O, O2, CO2 and CH4 so that the package, its
  tests and its examples run fully offline. Never report numbers derived from
  them.

Absorbers with broad electronic bands (O3, NO2, SO2) are described by
cross-section tables rather than line lists; see :class:`CrossSectionTable`.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.special import wofz

from .atmosphere import Atmosphere
from .constants import C2, HITRAN_ID, K_BOLTZMANN, MOLAR_MASS, N_AVOGADRO

C_LIGHT = 2.99792458e8
#: Absorbers described by cross-section tables instead of line lists.
_XSEC_GASES = ("O3", "NO2", "SO2")
T_HITRAN = 296.0
#: Exponent of the power-law approximation Q(T) ~ T**a for the partition function.
_Q_EXPONENT = {"CO2": 1.0, "O2": 1.0, "CH4": 1.5, "H2O": 1.5, "NO2": 1.5, "SO2": 1.5}


@dataclass
class LineList:
    """Parameters of absorption lines for one gas (HITRAN conventions)."""

    gas: str
    nu: np.ndarray  # line centre [cm^-1]
    s296: np.ndarray  # intensity at 296 K [cm^-1 / (molecule cm^-2)]
    gamma_air: np.ndarray  # air-broadened HWHM at 1 atm, 296 K [cm^-1/atm]
    gamma_self: np.ndarray
    elower: np.ndarray  # lower state energy [cm^-1]
    n_air: np.ndarray  # temperature exponent of gamma_air
    delta_air: np.ndarray  # pressure shift [cm^-1/atm]
    synthetic: bool = False

    def __len__(self) -> int:
        return len(self.nu)

    def select(self, nu_min: float, nu_max: float, s_min: float = 0.0) -> LineList:
        m = (self.nu >= nu_min) & (self.nu <= nu_max) & (self.s296 >= s_min)
        return LineList(
            self.gas,
            *(getattr(self, k)[m] for k in ("nu", "s296", "gamma_air", "gamma_self", "elower", "n_air", "delta_air")),
            synthetic=self.synthetic,
        )


def read_hitran_par(path: str | Path, gas: str) -> LineList:
    """Parse a HITRAN2004+ ``.par`` file (160-character records)."""
    nu, s, ga, gs, el, na, da = ([] for _ in range(7))
    mol_id = HITRAN_ID.get(gas)
    with open(path) as f:
        for line in f:
            if len(line) < 67:
                continue
            if mol_id is not None and int(line[0:2]) != mol_id:
                continue
            nu.append(float(line[3:15]))
            s.append(float(line[15:25]))
            ga.append(float(line[35:40]))
            gs.append(float(line[40:45]))
            el.append(float(line[45:55]))
            na.append(float(line[55:59]))
            da.append(float(line[59:67]))
    arr = [np.asarray(v, dtype=float) for v in (nu, s, ga, gs, el, na, da)]
    return LineList(gas, *arr)


def fetch_hitran(gas: str, nu_min: float, nu_max: float, out_dir: str | Path = ".") -> Path:
    """Download a HITRAN line list with ``hapi`` and return the ``.par`` path.

    Requires ``pip install hitran-api`` and network access to hitran.org.
    Only the principal isotopologue is fetched.
    """
    try:
        import hapi  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("fetch_hitran needs the optional 'hitran-api' package") from exc
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    hapi.db_begin(str(out_dir))
    table = f"{gas}_{int(nu_min)}_{int(nu_max)}"
    hapi.fetch(table, HITRAN_ID[gas], 1, nu_min, nu_max)
    return out_dir / f"{table}.data"


def demo_linelist(gas: str, seed: int = 0) -> LineList:
    """Synthetic, *non-physical* line list for offline demos and tests.

    Band centres follow the real VNIR-SWIR bands (H2O 0.72, 0.82, 0.94, 1.14,
    1.38, 1.88 and 2.7 um; O2 0.69, 0.76 and 1.27 um; CO2 1.6 and 2.0 um; CH4
    1.66 and 2.3 um) and intensities are tuned so that band depths at ~10 nm
    resolution are of realistic magnitude. The fine structure is invented.
    """
    rng = np.random.default_rng(seed + sum(map(ord, gas)))
    nus, ss = [], []

    def rot_band(nu0, b, s_band, jmax, t=296.0):
        j = np.arange(0, jmax)
        pop = (2 * j + 1) * np.exp(-C2 * b * j * (j + 1) / t)
        pop /= pop.sum()
        for sign in (-1, 1):  # P and R branches
            nus.append(nu0 + sign * 2 * b * (j + (sign > 0)))
            ss.append(0.45 * s_band * pop)
        nus.append(nu0 + rng.normal(0, 0.6 * b, size=jmax))  # Q-branch cluster
        ss.append(0.10 * s_band * pop)

    def random_band(nu0, width, s_band, n):
        nus.append(nu0 + rng.normal(0, width, size=n))
        w = rng.lognormal(0.0, 1.5, size=n)
        ss.append(s_band * w / w.sum())

    if gas == "CH4":
        rot_band(4320.0, 5.24, 3.0e-19, 22)  # 2.3 um combination bands
        random_band(4300.0, 110.0, 1.2e-18, 4000)  # dense overlapping manifolds
        rot_band(6005.0, 5.24, 3.0e-21, 22)  # 1.66 um 2nu3
        random_band(6000.0, 90.0, 1.0e-20, 2000)
    elif gas == "CO2":
        rot_band(4978.0, 0.39, 4.5e-21, 60)  # 2.01 um
        rot_band(5099.0, 0.39, 1.6e-21, 60)  # 1.96 um
        rot_band(6228.0, 0.39, 1.3e-23, 60)  # 1.6 um
        rot_band(6348.0, 0.39, 1.7e-23, 60)
    elif gas == "H2O":
        random_band(13830.0, 90.0, 3.0e-21, 600)  # 0.72 um
        random_band(12200.0, 100.0, 2.0e-21, 600)  # 0.82 um
        random_band(10650.0, 120.0, 1.0e-20, 1200)  # 0.94 um
        random_band(8800.0, 130.0, 7.0e-21, 1200)  # 1.14 um
        random_band(7250.0, 140.0, 6.0e-20, 1500)  # 1.38 um
        random_band(5330.0, 110.0, 8.0e-20, 1500)  # 1.88 um
        random_band(3756.0, 160.0, 8.0e-18, 1500)  # 2.7 um
        random_band(4400.0, 250.0, 1.0e-22, 600)  # weak lines across 2.1-2.5 um
    elif gas == "O2":
        rot_band(13120.0, 1.44, 2.2e-22, 35)  # A band, 0.76 um
        rot_band(14530.0, 1.44, 1.5e-23, 35)  # B band, 0.69 um
        rot_band(7880.0, 1.44, 1.0e-23, 35)  # 1.27 um
    else:
        raise ValueError(f"No demo line list for {gas}")
    nu = np.concatenate(nus)
    s = np.concatenate(ss)
    n = len(nu)
    j_like = rng.uniform(0, 800, size=n)
    return LineList(
        gas=gas,
        nu=nu,
        s296=s,
        gamma_air=rng.uniform(0.05, 0.08, size=n),
        gamma_self=rng.uniform(0.08, 0.12, size=n),
        elower=j_like,
        n_air=np.full(n, 0.75),
        delta_air=np.full(n, -0.005),
        synthetic=True,
    )


def partition_ratio(gas: str, t: float) -> float:
    """Q(296 K) / Q(T). Uses HAPI's TIPS when available, else a power law."""
    try:  # pragma: no cover - optional dependency
        import hapi  # type: ignore

        return hapi.partitionSum(HITRAN_ID[gas], 1, T_HITRAN) / hapi.partitionSum(HITRAN_ID[gas], 1, t)
    except Exception:
        return (T_HITRAN / t) ** _Q_EXPONENT.get(gas, 1.5)


def cross_section(
    lines: LineList,
    nu_grid: np.ndarray,
    pressure_pa: float,
    temperature: float,
    vmr: float = 0.0,
    wing_cutoff: float = 25.0,
) -> np.ndarray:
    """Voigt absorption cross section [cm^2/molecule] on ``nu_grid``.

    Line intensities are scaled to ``temperature``, widths to pressure and
    temperature, and each line is evaluated within ``wing_cutoff`` cm^-1.
    """
    p_atm = pressure_pa / 101325.0
    t = temperature
    nu_c = lines.nu + lines.delta_air * p_atm
    s = (
        lines.s296
        * partition_ratio(lines.gas, t)
        * np.exp(-C2 * lines.elower / t)
        / np.exp(-C2 * lines.elower / T_HITRAN)
        * (1 - np.exp(-C2 * lines.nu / t))
        / (1 - np.exp(-C2 * lines.nu / T_HITRAN))
    )
    gamma_l = (T_HITRAN / t) ** lines.n_air * (lines.gamma_air * p_atm * (1 - vmr) + lines.gamma_self * p_atm * vmr)
    m = MOLAR_MASS[lines.gas] / N_AVOGADRO
    alpha_d = lines.nu / C_LIGHT * np.sqrt(2 * K_BOLTZMANN * t / m)  # 1/e half width
    sigma = np.zeros_like(nu_grid, dtype=float)
    dnu = nu_grid[1] - nu_grid[0]
    half = int(np.ceil(wing_cutoff / dnu))
    idx = np.searchsorted(nu_grid, nu_c)
    for k in np.nonzero((idx > -half) & (idx < len(nu_grid) + half))[0]:
        lo, hi = max(idx[k] - half, 0), min(idx[k] + half, len(nu_grid))
        if lo >= hi:
            continue
        x = (nu_grid[lo:hi] - nu_c[k]) / alpha_d[k]
        y = gamma_l[k] / alpha_d[k]
        prof = wofz(x + 1j * y).real / (alpha_d[k] * np.sqrt(np.pi))
        sigma[lo:hi] += s[k] * prof
    return sigma


@dataclass
class CrossSectionTable:
    """A tabulated cross section sigma(lambda) [cm^2/molecule] (O3, NO2, SO2...)."""

    gas: str
    wavelength_nm: np.ndarray
    sigma: np.ndarray
    synthetic: bool = False

    @classmethod
    def from_ascii(
        cls, path: str | Path, gas: str, wl_col: int = 0, sigma_col: int = 1, vacuum_wavenumber: bool = False
    ) -> CrossSectionTable:
        """Load a two-column ASCII table such as Vandaele (NO2) or Bogumil (SO2)."""
        data = np.loadtxt(path, comments=("#", "*", ";"))
        x, s = data[:, wl_col], data[:, sigma_col]
        if vacuum_wavenumber:
            x = 1e7 / x
        order = np.argsort(x)
        return cls(gas, x[order], s[order])

    def __call__(self, wavelength_nm: np.ndarray) -> np.ndarray:
        return np.interp(wavelength_nm, self.wavelength_nm, self.sigma, left=0.0, right=0.0)


def demo_uvvis_cross_section(gas: str) -> CrossSectionTable:
    """Synthetic O3/NO2/SO2-like cross sections for offline demos. *Not* real data."""
    wl = np.arange(250.0, 2510.0, 0.02) if gas == "O3" else np.arange(280.0, 560.0, 0.02)
    if gas == "NO2":
        envelope = 4.5e-19 * np.exp(-0.5 * ((wl - 420.0) / 70.0) ** 2)
        bands = 1.0 + 0.45 * np.sin(2 * np.pi * (wl - 400.0) / 7.3) * np.exp(-(((wl - 440) / 60) ** 2))
        sigma = envelope * bands
    elif gas == "SO2":
        envelope = 1.0e-18 * np.exp(-0.5 * ((wl - 295.0) / 12.0) ** 2)
        bands = 1.0 + 0.6 * np.sin(2 * np.pi * (wl - 300.0) / 2.5)
        sigma = np.where(wl < 330, envelope * bands, 0.0)
    elif gas == "O3":  # Hartley band + Chappuis band
        sigma = 1.1e-17 * np.exp(-0.5 * ((wl - 255.0) / 18.0) ** 2) + 5e-21 * np.exp(-0.5 * ((wl - 600.0) / 60.0) ** 2)
    else:
        raise ValueError(f"No demo UV-VIS cross section for {gas}")
    return CrossSectionTable(gas, wl, np.clip(sigma, 0, None), synthetic=True)


class SpectralDatabase:
    """Monochromatic gas optical depths on a uniform wavenumber grid.

    For every gas the database stores the vertical optical depth of the whole
    atmosphere, the cross section at the surface layer (for column
    perturbations), and the cumulative column fraction with altitude, which
    gives the optical depth below an airborne or drone sensor.
    """

    def __init__(self, nu, gases, tau_vertical, sigma_surface, z_km, column_fraction, column_total, synthetic):
        self.nu = nu
        self.gases = list(gases)
        self.tau_vertical = tau_vertical  # gas -> (n_nu,)
        self.sigma_surface = sigma_surface  # gas -> (n_nu,) cm^2/molecule
        self.z_km = z_km  # altitude grid of the cumulative fractions
        self.column_fraction = column_fraction  # gas -> (n_z,) fraction of column below z
        self.column_total = column_total  # gas -> molecules/cm^2
        self.synthetic = synthetic

    @property
    def wavelength_nm(self) -> np.ndarray:
        return 1e7 / self.nu

    @classmethod
    def build(
        cls,
        atmosphere: Atmosphere,
        wl_range_nm: tuple[float, float] = (380.0, 2500.0),
        gases: tuple[str, ...] = ("H2O", "O2", "CO2", "CH4", "O3"),
        linelists: dict[str, LineList] | None = None,
        cross_sections: dict[str, CrossSectionTable] | None = None,
        dnu: float = 0.05,
        s_min: float = 1e-26,
        wing_cutoff: float = 15.0,
    ) -> SpectralDatabase:
        """Compute the database.

        Line absorbers use ``linelists`` (HITRAN), cross-section absorbers use
        ``cross_sections``. Anything missing falls back to the synthetic demo
        spectroscopy, and the database is flagged ``synthetic``.
        """
        nu_min, nu_max = 1e7 / wl_range_nm[1], 1e7 / wl_range_nm[0]
        nu = np.arange(nu_min - 1.0, nu_max + 1.0, dnu)
        wl = 1e7 / nu
        linelists = dict(linelists or {})
        cross_sections = dict(cross_sections or {})
        synthetic = False
        tau_v, sig_s, frac, total = {}, {}, {}, {}
        zb = atmosphere.z_bounds_km
        for g in gases:
            col = atmosphere.gas_column(g)
            total[g] = float(col.sum())
            frac[g] = np.concatenate([[0.0], np.cumsum(col)]) / col.sum()
            if g in _XSEC_GASES:
                table = cross_sections.get(g) or demo_uvvis_cross_section(g)
                synthetic |= table.synthetic
                sig = table(wl)
                tau_v[g] = (sig * col.sum()).astype(np.float32)
                sig_s[g] = sig.astype(np.float32)
                continue
            ll = linelists.get(g) or demo_linelist(g)
            synthetic |= ll.synthetic
            ll = ll.select(nu_min - wing_cutoff, nu_max + wing_cutoff, s_min)
            tau = np.zeros_like(nu)
            for i in range(atmosphere.n_layers):
                sig = cross_section(
                    ll,
                    nu,
                    atmosphere.pressure[i],
                    atmosphere.temperature[i],
                    vmr=atmosphere.vmr[g][i],
                    wing_cutoff=wing_cutoff,
                )
                tau += sig * col[i]
                if i == 0:
                    sig_s[g] = sig.astype(np.float32)
            tau_v[g] = tau.astype(np.float32)
        if synthetic:
            warnings.warn(
                "SpectralDatabase uses synthetic demo spectroscopy; results are illustrative only.", stacklevel=2
            )
        return cls(nu, gases, tau_v, sig_s, zb.copy(), frac, total, synthetic)

    def column_fraction_below(self, gas: str, altitude_m: float | None) -> float:
        """Fraction of the gas column below ``altitude_m`` (1 for ``None``: space-borne)."""
        if altitude_m is None:
            return 1.0
        return float(np.interp(altitude_m / 1e3, self.z_km, self.column_fraction[gas]))

    def vertical_tau(self, gas: str, below_altitude_m: float | None = None) -> np.ndarray:
        """Vertical optical depth, optionally only of the air below an altitude.

        The partial column assumes the spectral shape of the full column, which
        neglects the narrower upper-layer lines; adequate at instrument resolution.
        """
        return self.tau_vertical[gas].astype(float) * self.column_fraction_below(gas, below_altitude_m)

    def save(self, path: str | Path) -> None:
        arrays = {
            "nu": self.nu,
            "z_km": self.z_km,
            "gases": np.array(self.gases),
            "synthetic": np.array(self.synthetic),
        }
        for g in self.gases:
            arrays[f"tau_{g}"] = self.tau_vertical[g]
            arrays[f"sigma_{g}"] = self.sigma_surface[g]
            arrays[f"frac_{g}"] = self.column_fraction[g]
            arrays[f"total_{g}"] = np.array(self.column_total[g])
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path: str | Path) -> SpectralDatabase:
        d = np.load(path)
        gases = [str(g) for g in d["gases"]]
        return cls(
            d["nu"],
            gases,
            {g: d[f"tau_{g}"] for g in gases},
            {g: d[f"sigma_{g}"] for g in gases},
            d["z_km"],
            {g: d[f"frac_{g}"] for g in gases},
            {g: float(d[f"total_{g}"]) for g in gases},
            bool(d["synthetic"]),
        )
