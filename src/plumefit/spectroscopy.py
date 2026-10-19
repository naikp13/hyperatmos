"""Line-by-line absorption cross sections and the high-resolution spectral database.

Two sources of spectroscopy are supported:

* **HITRAN line lists** (``.par`` 160-character format). Download them with
  :func:`fetch_hitran` (requires the optional ``hitran-api`` package and network
  access to hitran.org) or from https://hitran.org/lbl/ and load with
  :func:`read_hitran_par`. This is what you should use for science.
* **Synthetic demo line lists** from :func:`demo_linelist`. These are *not*
  real spectroscopy: they mimic the position, rotational structure and depth of
  the main SWIR bands of CH4, CO2 and H2O so that the package, its tests and its
  examples run fully offline. Never report numbers derived from them.

UV-visible absorbers (NO2, SO2) are described by measured cross-section tables
rather than line lists; see :class:`CrossSectionTable`.
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
T_HITRAN = 296.0
#: Exponent of the power-law approximation Q(T) ~ T**a for the partition function.
_Q_EXPONENT = {"CO2": 1.0, "CH4": 1.5, "H2O": 1.5, "NO2": 1.5, "SO2": 1.5}


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

    Band centres and widths follow the real SWIR bands (CH4 2.3 and 1.65 um,
    CO2 2.0 and 1.6 um, H2O 1.4, 1.9 and 2.7 um) and intensities are tuned so
    that band depths at ~7 nm resolution are of realistic magnitude. The fine
    structure is invented.
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
        random_band(5330.0, 110.0, 8.0e-20, 1500)  # 1.88 um
        random_band(7250.0, 140.0, 6.0e-20, 1500)  # 1.38 um
        random_band(3756.0, 160.0, 8.0e-18, 1500)  # 2.7 um
        random_band(4400.0, 250.0, 1.0e-22, 600)  # weak lines across 2.1-2.5 um
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
    """A tabulated cross section sigma(lambda) [cm^2/molecule] (UV-VIS absorbers)."""

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
    """Synthetic NO2/SO2-like cross sections for offline demos. *Not* real data."""
    wl = np.arange(280.0, 560.0, 0.02)
    if gas == "NO2":
        envelope = 4.5e-19 * np.exp(-0.5 * ((wl - 420.0) / 70.0) ** 2)
        bands = 1.0 + 0.45 * np.sin(2 * np.pi * (wl - 400.0) / 7.3) * np.exp(-(((wl - 440) / 60) ** 2))
        sigma = envelope * bands
    elif gas == "SO2":
        envelope = 1.0e-18 * np.exp(-0.5 * ((wl - 295.0) / 12.0) ** 2)
        bands = 1.0 + 0.6 * np.sin(2 * np.pi * (wl - 300.0) / 2.5)
        sigma = np.where(wl < 330, envelope * bands, 0.0)
    elif gas == "O3":
        sigma = 1.1e-17 * np.exp(-0.5 * ((wl - 255.0) / 18.0) ** 2) + 5e-21 * np.exp(-0.5 * ((wl - 600.0) / 60.0) ** 2)
    else:
        raise ValueError(f"No demo UV-VIS cross section for {gas}")
    return CrossSectionTable(gas, wl, np.clip(sigma, 0, None), synthetic=True)


class SpectralDatabase:
    """Monochromatic optical properties for a set of gases on a wavenumber grid.

    Stores, for every gas, the layer optical depths of the background
    atmosphere (vertical) and the cross section at the plume layer
    (surface pressure/temperature by default).
    """

    def __init__(
        self,
        nu: np.ndarray,
        gases: list[str],
        layer_tau: dict,
        plume_sigma: dict,
        z_bounds_km: np.ndarray,
        synthetic: bool,
    ):
        self.nu = nu
        self.gases = list(gases)
        self.layer_tau = layer_tau  # gas -> (n_layers, n_nu)
        self.plume_sigma = plume_sigma  # gas -> (n_nu,)
        self.z_bounds_km = z_bounds_km
        self.synthetic = synthetic

    @property
    def wavelength_nm(self) -> np.ndarray:
        return 1e7 / self.nu

    @classmethod
    def build(
        cls,
        atmosphere: Atmosphere,
        wl_range_nm: tuple[float, float] = (2100.0, 2450.0),
        gases: tuple[str, ...] = ("CH4", "CO2", "H2O"),
        linelists: dict[str, LineList] | None = None,
        dnu: float = 0.02,
        plume_layer: int = 0,
        s_min: float = 1e-26,
    ) -> SpectralDatabase:
        """Compute the database. Without ``linelists``, demo spectroscopy is used."""
        nu_min, nu_max = 1e7 / wl_range_nm[1], 1e7 / wl_range_nm[0]
        nu = np.arange(nu_min - 1.0, nu_max + 1.0, dnu)
        linelists = dict(linelists or {})
        synthetic = False
        for g in gases:
            if g not in linelists:
                linelists[g] = demo_linelist(g)
            synthetic |= linelists[g].synthetic
        if synthetic:
            warnings.warn(
                "SpectralDatabase uses synthetic demo spectroscopy; results are illustrative only.", stacklevel=2
            )
        layer_tau, plume_sigma = {}, {}
        for g in gases:
            ll = linelists[g].select(nu_min - 30, nu_max + 30, s_min)
            col = atmosphere.gas_column(g)
            taus = np.empty((atmosphere.n_layers, nu.size))
            for i in range(atmosphere.n_layers):
                sig = cross_section(ll, nu, atmosphere.pressure[i], atmosphere.temperature[i], vmr=atmosphere.vmr[g][i])
                taus[i] = sig * col[i]
                if i == plume_layer:
                    plume_sigma[g] = sig
            layer_tau[g] = taus
        return cls(nu, gases, layer_tau, plume_sigma, atmosphere.z_bounds_km.copy(), synthetic)

    def vertical_tau(self, gas: str, below_altitude_m: float | None = None) -> np.ndarray:
        """Background vertical optical depth, optionally only below an altitude."""
        taus = self.layer_tau[gas]
        if below_altitude_m is None:
            return taus.sum(axis=0)
        zb = self.z_bounds_km * 1e3
        frac = np.clip((below_altitude_m - zb[:-1]) / (zb[1:] - zb[:-1]), 0.0, 1.0)
        return frac @ taus

    def save(self, path: str | Path) -> None:
        arrays = {
            "nu": self.nu,
            "z_bounds_km": self.z_bounds_km,
            "gases": np.array(self.gases),
            "synthetic": np.array(self.synthetic),
        }
        for g in self.gases:
            arrays[f"tau_{g}"] = self.layer_tau[g].astype(np.float32)
            arrays[f"sigma_{g}"] = self.plume_sigma[g]
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path: str | Path) -> SpectralDatabase:
        d = np.load(path)
        gases = [str(g) for g in d["gases"]]
        return cls(
            d["nu"],
            gases,
            {g: d[f"tau_{g}"].astype(float) for g in gases},
            {g: d[f"sigma_{g}"] for g in gases},
            d["z_bounds_km"],
            bool(d["synthetic"]),
        )
