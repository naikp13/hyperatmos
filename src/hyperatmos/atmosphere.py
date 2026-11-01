"""Layered background atmospheres.

A deliberately small model: a US-1976-like temperature profile, hydrostatic
pressure, and simple gas profiles (well mixed O2/CO2/CH4, exponential H2O,
a stratospheric O3 layer). It is enough to compute band-averaged gas
transmittances. For operational work, replace it with a reanalysis profile
(e.g. ERA5), or scale it to the water vapour that ISOFIT retrieved (see
:mod:`hyperatmos.isofit_interop`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .constants import DOBSON_UNIT, K_BOLTZMANN, MOLAR_MASS, N_AVOGADRO

G0 = 9.80665
R_AIR = 287.05


def us_standard_temperature(z_km: np.ndarray, t_surface: float = 288.15) -> np.ndarray:
    """Piecewise-linear US-1976 temperature profile up to 47 km [K]."""
    z = np.asarray(z_km, dtype=float)
    t11 = t_surface - 6.5 * 11.0
    t = np.where(z <= 11.0, t_surface - 6.5 * z, t11)
    t = np.where(z > 20.0, t11 + 1.0 * (z - 20.0), t)
    t = np.where(z > 32.0, t11 + 12.0 + 2.8 * (z - 32.0), t)
    return t


@dataclass
class Atmosphere:
    """A layered atmosphere.

    Attributes
    ----------
    z_bounds_km : layer boundaries [km], increasing, first value is the surface.
    vmr : volume mixing ratio per gas at layer centres (dimensionless).
    """

    z_bounds_km: np.ndarray
    surface_pressure: float = 101325.0
    surface_temperature: float = 288.15
    vmr: dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self):
        self.z_bounds_km = np.asarray(self.z_bounds_km, dtype=float)
        zc = self.z_centres_km
        self.temperature = us_standard_temperature(zc, self.surface_temperature)
        # Integrate hydrostatic equation on a fine grid for boundary pressures.
        zf = np.linspace(self.z_bounds_km[0], self.z_bounds_km[-1], 4001)
        tf = us_standard_temperature(zf, self.surface_temperature)
        integrand = G0 / (R_AIR * tf) * 1e3
        lnp = np.log(self.surface_pressure) - np.concatenate(
            [[0.0], np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(zf))]
        )
        self.p_bounds = np.exp(np.interp(self.z_bounds_km, zf, lnp))
        self.pressure = np.sqrt(self.p_bounds[1:] * self.p_bounds[:-1])

    @property
    def z_centres_km(self) -> np.ndarray:
        return 0.5 * (self.z_bounds_km[1:] + self.z_bounds_km[:-1])

    @property
    def n_layers(self) -> int:
        return len(self.z_bounds_km) - 1

    def air_column(self) -> np.ndarray:
        """Air partial column per layer [molecules/cm^2] (hydrostatic)."""
        dp = self.p_bounds[:-1] - self.p_bounds[1:]
        m_air = 28.9647e-3 / 6.02214076e23  # kg/molecule
        return dp / (G0 * m_air) * 1e-4

    def gas_column(self, gas: str) -> np.ndarray:
        """Gas partial column per layer [molecules/cm^2]."""
        return self.vmr[gas] * self.air_column()

    def column_fraction_below(self, gas: str, altitude_m: float) -> float:
        """Fraction of the total gas column below ``altitude_m``.

        Used for airborne/drone geometry, where the upward (view) path only
        traverses the atmosphere between the surface and the sensor.
        """
        col = self.gas_column(gas)
        zb = self.z_bounds_km * 1e3
        frac = np.clip((altitude_m - zb[:-1]) / (zb[1:] - zb[:-1]), 0.0, 1.0)
        return float(np.sum(col * frac) / np.sum(col))

    def set_column(self, gas: str, molec_cm2: float) -> None:
        """Rescale a gas profile to a total column [molecules/cm^2], keeping its shape."""
        self.vmr[gas] = self.vmr[gas] * molec_cm2 / self.gas_column(gas).sum()

    def number_density(self) -> np.ndarray:
        """Air number density at layer centres [molecules/cm^3]."""
        return self.pressure / (K_BOLTZMANN * self.temperature) * 1e-6


def standard_atmosphere(
    h2o_g_cm2: float = 1.5,
    o3_du: float = 300.0,
    co2_ppm: float = 420.0,
    ch4_ppm: float = 1.92,
    h2o_scale_height_km: float = 2.0,
    surface_pressure: float = 101325.0,
    surface_temperature: float = 288.15,
    fine: bool = False,
) -> Atmosphere:
    """A mid-latitude-like atmosphere from the surface to 40 km.

    ``fine=False`` uses 15 layers (enough for band-averaged transmittance);
    ``fine=True`` uses 30.
    """
    if fine:
        zb = np.concatenate([np.arange(0.0, 20.0, 1.0), np.arange(20.0, 40.1, 2.0)])
    else:
        zb = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0, 25.0, 30.0, 40.0])
    atm = Atmosphere(zb, surface_pressure, surface_temperature)
    zc = atm.z_centres_km
    h2o_shape = np.maximum(np.exp(-zc / h2o_scale_height_km), 1e-3)
    o3_shape = np.exp(-0.5 * ((zc - 22.0) / 6.0) ** 2) + 1e-3
    atm.vmr = {
        "CO2": np.full(atm.n_layers, co2_ppm * 1e-6),
        "CH4": np.full(atm.n_layers, ch4_ppm * 1e-6),
        "O2": np.full(atm.n_layers, 0.2095),
        "H2O": h2o_shape,
        "O3": o3_shape,
        "NO2": np.full(atm.n_layers, 1e-10),
        "SO2": np.full(atm.n_layers, 1e-11),
    }
    atm.set_column("H2O", h2o_g_cm2 / (MOLAR_MASS["H2O"] * 1e3) * N_AVOGADRO)
    atm.set_column("O3", o3_du * DOBSON_UNIT)
    return atm
