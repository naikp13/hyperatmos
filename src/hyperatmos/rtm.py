"""Fast radiative transfer for imaging spectrometers on drones, aircraft and satellites.

The model follows the standard decoupled formulation used by atmospheric
correction codes (e.g. 6S, ATREM, and the "4-component" view of ISOFIT's
lookup tables). For a Lambertian surface of reflectance ``rho``, the
reflectance seen by the sensor is::

    rho_s = T_g * [ rho_path + T_dn * T_up * rho / (1 - S * rho) ]

and the radiance is ``L = mu0 * E0 / pi * rho_s``, where

* ``T_g`` is the band-averaged gas transmittance along the sun-surface-sensor
  path, from line-by-line spectroscopy convolved with the instrument response;
* ``rho_path`` is the path reflectance from Rayleigh and aerosol single
  scattering in the air between the surface and the sensor;
* ``T_dn``, ``T_up`` are total (direct + forward-diffuse) transmittances;
* ``S`` is the spherical albedo of the atmosphere (two-stream estimate).

Gas absorption is computed at high spectral resolution, while scattering is
smooth and evaluated at band centres. For a sensor at altitude ``h`` (drone,
aircraft) the upward path, the path reflectance and the gas transmittance
only include the air below ``h``.

The scattering part is a single-scattering / two-stream approximation:
accurate to a few percent of path reflectance for AOD below ~0.3 in the red
and beyond, less accurate in the blue. Use it for fast inversions, sensitivity
studies and instrument design; for the highest accuracy, swap in lookup
tables from a full multiple-scattering code (see the roadmap).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .constants import DOBSON_UNIT, molec_cm2_to_h2o_g_cm2
from .instrument import Instrument
from .spectroscopy import SpectralDatabase

RAYLEIGH_SCALE_HEIGHT_KM = 8.0


@dataclass
class Geometry:
    """Observation geometry.

    ``relative_azimuth_deg`` is 0 when the sensor looks along the solar beam
    direction from the sun side (backscatter / hot-spot side) and 180 on the
    forward-scattering side. ``sensor_altitude_m=None`` means space-borne.
    """

    solar_zenith_deg: float = 30.0
    view_zenith_deg: float = 0.0
    relative_azimuth_deg: float = 0.0
    sensor_altitude_m: float | None = None

    @property
    def mu0(self) -> float:
        return float(np.cos(np.deg2rad(self.solar_zenith_deg)))

    @property
    def muv(self) -> float:
        return float(np.cos(np.deg2rad(self.view_zenith_deg)))

    @property
    def cos_scattering_angle(self) -> float:
        s0 = np.sin(np.deg2rad(self.solar_zenith_deg))
        sv = np.sin(np.deg2rad(self.view_zenith_deg))
        return float(-self.mu0 * self.muv - s0 * sv * np.cos(np.deg2rad(self.relative_azimuth_deg)))

    @property
    def airmass(self) -> float:
        """Geometric two-way air-mass factor ``1/mu0 + 1/mu_v``."""
        return 1.0 / self.mu0 + 1.0 / self.muv


@dataclass
class Aerosol:
    """Spectrally parameterised aerosol: Angstrom law, single scattering albedo,
    a two-term Henyey-Greenstein phase function and an exponential profile.

    The phase function is ``f HG(g1) + (1 - f) HG(-g2)``; ``f`` is set so the
    asymmetry parameter equals ``asymmetry``. The backward lobe gives the
    realistic side-/back-scattering that a single HG function lacks.
    """

    aod550: float = 0.1
    angstrom: float = 1.3
    ssa: float = 0.93
    asymmetry: float = 0.68
    forward_g: float = 0.78
    backward_g: float = 0.35
    scale_height_km: float = 2.0

    def phase(self, cos_theta: float) -> float:
        f = (self.asymmetry + self.backward_g) / (self.forward_g + self.backward_g)
        return f * henyey_greenstein(cos_theta, self.forward_g) + (1 - f) * henyey_greenstein(
            cos_theta, -self.backward_g
        )

    def tau(self, wavelength_nm: np.ndarray, aod550=None) -> np.ndarray:
        aod = self.aod550 if aod550 is None else aod550
        return np.multiply.outer(np.asarray(aod, dtype=float), (wavelength_nm / 550.0) ** (-self.angstrom))


def rayleigh_optical_depth(wavelength_nm: np.ndarray, surface_pressure_pa: float = 101325.0) -> np.ndarray:
    """Rayleigh optical depth (Hansen and Travis, 1974), scaled with surface pressure."""
    lam = np.asarray(wavelength_nm) / 1e3
    return 0.008569 * lam**-4 * (1 + 0.0113 * lam**-2 + 0.00013 * lam**-4) * surface_pressure_pa / 101325.0


def rayleigh_phase(cos_theta: float) -> float:
    return 0.75 * (1 + cos_theta**2)


def henyey_greenstein(cos_theta: float, g: float) -> float:
    return (1 - g**2) / (1 + g**2 - 2 * g * cos_theta) ** 1.5


@dataclass
class RTCoefficients:
    """Band coefficients of the decoupled radiative transfer equation."""

    t_gas: np.ndarray
    rho_path: np.ndarray
    t_down: np.ndarray
    t_up: np.ndarray
    s_alb: np.ndarray

    def toa_reflectance(self, rho: np.ndarray) -> np.ndarray:
        tt = self.t_down * self.t_up
        return self.t_gas * (self.rho_path + tt * rho / (1 - self.s_alb * rho))

    def surface_reflectance(self, rho_s: np.ndarray) -> np.ndarray:
        """Algebraic inversion of :meth:`toa_reflectance`."""
        y = rho_s / self.t_gas - self.rho_path
        return y / (self.t_down * self.t_up + self.s_alb * y)

    def d_toa_d_rho(self, rho: np.ndarray) -> np.ndarray:
        return self.t_gas * self.t_down * self.t_up / (1 - self.s_alb * rho) ** 2


@dataclass
class RadiativeTransferModel:
    """Radiative transfer for one instrument, geometry and background atmosphere.

    Parameters
    ----------
    db : high-resolution gas optical depths.
    instrument : bands outside the database range are dropped (see ``band_mask``).
    geometry : solar and viewing geometry, sensor altitude.
    aerosol : aerosol optical model; ``aod550`` is the default AOD.
    surface_pressure : Pa, scales the Rayleigh optical depth.
    solar_irradiance : high-resolution TOA irradiance on ``db.wavelength_nm``
        [uW cm^-2 nm^-1]; a 5778 K blackbody by default.
    """

    db: SpectralDatabase
    instrument: Instrument
    geometry: Geometry
    aerosol: Aerosol = field(default_factory=Aerosol)
    surface_pressure: float = 101325.0
    solar_irradiance: np.ndarray | None = None
    h2o_grid: np.ndarray = field(default_factory=lambda: np.geomspace(0.05, 8.0, 48))

    def __post_init__(self):
        db, g = self.db, self.geometry
        wl_hi = db.wavelength_nm
        lo, hi = wl_hi.min(), wl_hi.max()
        margin = 2 * self.instrument.fwhm_nm
        self.band_mask = (self.instrument.wavelength_nm - margin >= lo) & (self.instrument.wavelength_nm + margin <= hi)
        full = self.instrument
        self.instrument = Instrument(
            full.wavelength_nm[self.band_mask],
            full.fwhm_nm[self.band_mask],
            full.snr_ref,
            full.radiance_ref,
            full.read_noise_fraction,
            full.name,
        )
        self.wavelength_nm = self.instrument.wavelength_nm
        self._srf = self.instrument.srf_matrix(db.nu)
        e0 = self.solar_irradiance if self.solar_irradiance is not None else solar_irradiance_blackbody(wl_hi)
        self._e0_hi = e0
        self.e0 = self._srf @ e0
        # Slant optical depth per gas along sun -> surface -> sensor.
        h = g.sensor_altitude_m
        self._slant = {gas: db.vertical_tau(gas) / g.mu0 + db.vertical_tau(gas, h) / g.muv for gas in db.gases}
        self.h2o_reference = molec_cm2_to_h2o_g_cm2(db.column_total["H2O"]) if "H2O" in db.gases else None
        self._fixed = sum(v for k, v in self._slant.items() if k != "H2O")
        self._h2o_lut = None
        self._tau_r = rayleigh_optical_depth(self.wavelength_nm, self.surface_pressure)

    # ---------------------------------------------------------------- gases
    def gas_transmittance(self, h2o_g_cm2: float | None = None, scales: dict[str, float] | None = None) -> np.ndarray:
        """Band-averaged two-way gas transmittance at full spectral resolution.

        ``h2o_g_cm2`` sets the water vapour column; ``scales`` multiplies the
        column of any other gas (e.g. ``{"CH4": 1.02, "O3": 0.9}``).
        """
        tau = np.zeros_like(self.db.nu)
        scales = scales or {}
        for gas, t in self._slant.items():
            f = scales.get(gas, 1.0)
            if gas == "H2O" and h2o_g_cm2 is not None:
                f = h2o_g_cm2 / self.h2o_reference
            tau += f * t
        return (self._srf @ (self._e0_hi * np.exp(-tau))) / self.e0

    def h2o_transmittance(self, h2o_g_cm2) -> np.ndarray:
        """Fast gas transmittance for arrays of water vapour columns (lookup table).

        Returns ``(..., n_bands)``. All gases other than H2O stay at the
        background columns of the database.
        """
        if self._h2o_lut is None:
            lut = []
            for w in self.h2o_grid:
                tau = self._fixed + (w / self.h2o_reference) * self._slant["H2O"]
                lut.append(np.log(np.clip((self._srf @ (self._e0_hi * np.exp(-tau))) / self.e0, 1e-30, None)))
            self._h2o_lut = np.array(lut)
        lw = np.log(np.clip(np.asarray(h2o_g_cm2, dtype=float), self.h2o_grid[0], self.h2o_grid[-1]))
        grid = np.log(self.h2o_grid)
        i = np.clip(np.searchsorted(grid, lw) - 1, 0, len(grid) - 2)
        f = ((lw - grid[i]) / (grid[i + 1] - grid[i]))[..., None]
        return np.exp(self._h2o_lut[i] * (1 - f) + self._h2o_lut[i + 1] * f)

    def absorption_signature(self, gas: str, rel_change: float = 0.05) -> np.ndarray:
        """d ln(T_g) / d(column scale) per band: the spectral signature of a
        fractional change in the column of ``gas`` (for the matched filter)."""
        t0 = self.gas_transmittance()
        t1 = self.gas_transmittance(scales={gas: 1.0 + rel_change})
        return np.log(t1 / t0) / rel_change

    # ----------------------------------------------------------- scattering
    def scattering(self, aod550=None) -> dict[str, np.ndarray]:
        """Path reflectance, total transmittances and spherical albedo per band.

        ``aod550`` may be a scalar or an array (pixels); outputs broadcast to
        ``(..., n_bands)``.
        """
        g, a = self.geometry, self.aerosol
        wl = self.wavelength_nm
        tau_r = self._tau_r
        tau_a = a.tau(wl, aod550)
        h = g.sensor_altitude_m
        fr = 1.0 if h is None else 1 - np.exp(-h / 1e3 / RAYLEIGH_SCALE_HEIGHT_KM)
        fa = 1.0 if h is None else 1 - np.exp(-h / 1e3 / a.scale_height_km)
        fwd = (1 + a.asymmetry) / 2
        tau_eff_r, tau_eff_a = 0.5 * tau_r, tau_a * (1 - a.ssa * fwd)
        t_down = np.exp(-(tau_eff_r + tau_eff_a) / g.mu0)
        t_up = np.exp(-(fr * tau_eff_r + fa * tau_eff_a) / g.muv)
        # Single scattering by the air below the sensor, lit through the air
        # above it. Rayleigh and aerosol components are computed separately,
        # each with its own extinction: forward-scattering aerosol does not
        # remove light from the Rayleigh path, which keeps the path
        # reflectance increasing with AOD as multiple scattering would.
        cos_t = g.cos_scattering_angle
        m = g.airmass
        above = np.exp(-((1 - fr) * tau_eff_r + (1 - fa) * tau_eff_a) / g.mu0)

        def single(tau, omega_p):
            return omega_p / (4 * (g.mu0 + g.muv)) * (1 - np.exp(-tau * m))

        rho_path = (single(fr * tau_r, rayleigh_phase(cos_t)) + single(fa * tau_a, a.ssa * a.phase(cos_t))) * above
        tau_d = 0.75 * (tau_r + (1 - a.asymmetry) * a.ssa * tau_a)
        s_alb = tau_d / (1 + tau_d)
        return {"rho_path": rho_path, "t_down": t_down, "t_up": t_up, "s_alb": s_alb}

    # -------------------------------------------------------------- combined
    def coefficients(self, h2o_g_cm2=None, aod550=None, scales: dict[str, float] | None = None) -> RTCoefficients:
        """All band coefficients. Uses the H2O lookup table unless ``scales`` is given."""
        if h2o_g_cm2 is None:
            h2o_g_cm2 = self.h2o_reference
        if scales:
            t_gas = self.gas_transmittance(h2o_g_cm2, scales)
        else:
            t_gas = self.h2o_transmittance(h2o_g_cm2)
        sc = self.scattering(aod550)
        if np.ndim(h2o_g_cm2) and not np.ndim(aod550):
            sc = {k: np.broadcast_to(v, t_gas.shape) for k, v in sc.items()}
        return RTCoefficients(t_gas, sc["rho_path"], sc["t_down"], sc["t_up"], sc["s_alb"])

    def radiance(self, rho, h2o_g_cm2=None, aod550=None, scales=None) -> np.ndarray:
        """At-sensor radiance [uW cm^-2 sr^-1 nm^-1] for surface reflectance ``rho``."""
        c = self.coefficients(h2o_g_cm2, aod550, scales)
        return self.geometry.mu0 * self.e0 / np.pi * c.toa_reflectance(rho)

    def to_reflectance_factor(self, radiance: np.ndarray) -> np.ndarray:
        """Radiance to sensor-level reflectance factor ``pi L / (mu0 E0)``."""
        return np.pi * radiance / (self.geometry.mu0 * self.e0)

    def surface_reflectance(self, radiance, h2o_g_cm2=None, aod550=None, scales=None) -> np.ndarray:
        """Invert radiance to surface reflectance for a given atmospheric state."""
        c = self.coefficients(h2o_g_cm2, aod550, scales)
        return c.surface_reflectance(self.to_reflectance_factor(radiance))


def ozone_du(db: SpectralDatabase) -> float:
    """Total ozone column of the database atmosphere [DU]."""
    return db.column_total["O3"] / DOBSON_UNIT


def solar_irradiance_blackbody(wavelength_nm: np.ndarray, t_sun: float = 5778.0) -> np.ndarray:
    """Top-of-atmosphere solar irradiance [uW cm^-2 nm^-1] from a 5778 K blackbody.

    Smooth, without Fraunhofer lines. Supply a measured spectrum (for example
    the one ISOFIT ships) for real data.
    """
    h, c, k = 6.62607015e-34, 2.99792458e8, 1.380649e-23
    wl = wavelength_nm * 1e-9
    b = 2 * h * c**2 / wl**5 / np.expm1(h * c / (wl * k * t_sun))  # W m^-2 sr^-1 m^-1
    omega = 6.794e-5  # solid angle of the Sun from 1 AU [sr]
    return b * omega * 1e-9 * 1e2  # -> uW cm^-2 nm^-1
