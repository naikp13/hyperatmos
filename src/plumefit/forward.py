"""Forward model for at-sensor radiance with gas plume enhancements.

For a Lambertian surface with negligible scattering in strong absorption
windows (the standard assumption of SWIR plume retrievals), the radiance in
band ``b`` is::

    L_b = mu0 / pi * rho_b * [ SRF * ( E0 * exp(-tau_bg) * exp(-AMF * sum_g sigma_g dX_g) ) ]_b

with ``tau_bg = tau_vert / mu0 + tau_below_sensor / mu_v``. For a satellite the
second term is the full vertical column; for a drone flying at altitude ``h``
only the layer between the ground and the sensor is traversed on the way up.
Plume enhancements are assumed to sit below the sensor, so their air-mass
factor is ``1/mu0 + 1/mu_v`` in both cases.

The surface reflectance ``rho`` is either a low-order polynomial in wavelength,
or a prior spectrum (e.g. an ISOFIT ``_rfl`` product) times such a polynomial.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import ppmm_to_molec_cm2
from .instrument import Instrument
from .spectroscopy import SpectralDatabase


@dataclass
class Geometry:
    """Observation geometry. ``sensor_altitude_m=None`` means space-borne."""

    solar_zenith_deg: float = 30.0
    view_zenith_deg: float = 0.0
    sensor_altitude_m: float | None = None

    @property
    def mu0(self) -> float:
        return float(np.cos(np.deg2rad(self.solar_zenith_deg)))

    @property
    def muv(self) -> float:
        return float(np.cos(np.deg2rad(self.view_zenith_deg)))

    @property
    def plume_amf(self) -> float:
        """Two-way air-mass factor for an absorber below the sensor."""
        return 1.0 / self.mu0 + 1.0 / self.muv


def solar_irradiance_blackbody(wavelength_nm: np.ndarray, t_sun: float = 5778.0) -> np.ndarray:
    """Top-of-atmosphere solar irradiance [uW cm^-2 nm^-1] from a 5778 K blackbody.

    Smooth, without Fraunhofer lines. Supply a measured spectrum (for example
    the one ISOFIT uses) via ``ForwardModel(solar_irradiance=...)`` for real data.
    """
    h, c, k = 6.62607015e-34, 2.99792458e8, 1.380649e-23
    wl = wavelength_nm * 1e-9
    b = 2 * h * c**2 / wl**5 / np.expm1(h * c / (wl * k * t_sun))  # W m^-2 sr^-1 m^-1
    omega = 6.794e-5  # solid angle of the Sun from 1 AU [sr]
    return b * omega * 1e-9 * 1e2  # -> uW cm^-2 nm^-1


class ForwardModel:
    """Radiance forward model and analytic Jacobian.

    State vector layout: ``[dX_gas_1, ..., dX_gas_n, s_0, ..., s_K]`` where
    ``dX`` are column enhancements in ppm·m. With ``surface="polynomial"``
    the ``s_k`` are coefficients of a reflectance polynomial in normalised
    wavelength (optionally multiplying a fixed reflectance prior); with
    ``surface="bands"`` they are the reflectance in every band, to be
    constrained by a statistical surface prior (see
    :func:`plumefit.isofit_bridge.surface_prior_from_reflectance`).
    """

    def __init__(
        self,
        db: SpectralDatabase,
        instrument: Instrument,
        geometry: Geometry,
        retrieved_gases: tuple[str, ...] = ("CH4", "H2O"),
        poly_order: int = 2,
        reflectance_prior: np.ndarray | None = None,
        solar_irradiance: np.ndarray | None = None,
        surface: str = "polynomial",
    ):
        if surface not in ("polynomial", "bands"):
            raise ValueError("surface must be 'polynomial' or 'bands'")
        self.surface = surface
        self.db = db
        self.geometry = geometry
        wl_hi = db.wavelength_nm
        lo, hi = wl_hi.min(), wl_hi.max()
        self.instrument = instrument.subset(lo + 2 * instrument.fwhm_nm.max(), hi - 2 * instrument.fwhm_nm.max())
        self.band_mask = np.isin(instrument.wavelength_nm, self.instrument.wavelength_nm)
        self.gases = tuple(retrieved_gases)
        self.poly_order = poly_order
        self.srf = self.instrument.srf_matrix(db.nu)
        e0 = solar_irradiance if solar_irradiance is not None else solar_irradiance_blackbody(wl_hi)
        tau_bg = np.zeros_like(db.nu)
        for g in db.gases:
            tau_bg += db.vertical_tau(g) / geometry.mu0
            tau_bg += db.vertical_tau(g, geometry.sensor_altitude_m) / geometry.muv
        self._e0 = e0
        self._tau_bg = tau_bg
        conv = ppmm_to_molec_cm2(1.0)
        # Optical depth of 1 ppm·m along the two-way plume path, per gas.
        self._dtau = np.array([geometry.plume_amf * conv * db.plume_sigma[g] for g in self.gases])
        wl = self.instrument.wavelength_nm
        self._x_poly = (wl - wl.mean()) / (0.5 * (wl.max() - wl.min()))
        if surface == "bands":
            self._vander = np.eye(wl.size)
        else:
            self._vander = np.vander(self._x_poly, poly_order + 1, increasing=True)
        self.reflectance_prior = None
        self.set_reflectance_prior(reflectance_prior)

    def set_reflectance_prior(self, reflectance: np.ndarray | None) -> None:
        """Use a per-pixel reflectance spectrum (all instrument bands) as surface prior."""
        if reflectance is not None:
            reflectance = np.asarray(reflectance, dtype=float)
            if reflectance.size == self.band_mask.size:
                reflectance = reflectance[self.band_mask]
        self.reflectance_prior = reflectance

    @property
    def wavelength_nm(self) -> np.ndarray:
        return self.instrument.wavelength_nm

    @property
    def n_gas(self) -> int:
        return len(self.gases)

    @property
    def n_surface(self) -> int:
        return self._vander.shape[1]

    @property
    def n_state(self) -> int:
        return self.n_gas + self.n_surface

    def reflectance(self, coeffs: np.ndarray) -> np.ndarray:
        rho = self._vander @ coeffs
        if self.reflectance_prior is not None:
            rho = rho * self.reflectance_prior
        return rho

    def transmitted_irradiance(self, dx_ppmm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Band-convolved E0*T and its derivative w.r.t. each enhancement."""
        # Combine optical depths before exponentiating: a negative enhancement
        # (e.g. less H2O than the climatology) must not overflow where the
        # background is opaque.
        mono = self._e0 * np.exp(-(self._tau_bg + np.asarray(dx_ppmm) @ self._dtau))
        conv = self.srf @ mono
        dconv = -(self.srf @ (mono[None, :] * self._dtau).T).T  # (n_gas, n_bands)
        return conv, dconv

    def radiance(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        conv, _ = self.transmitted_irradiance(x[: self.n_gas])
        return self.geometry.mu0 / np.pi * self.reflectance(x[self.n_gas :]) * conv

    def jacobian(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return radiance and Jacobian (n_bands, n_state)."""
        x = np.asarray(x, dtype=float)
        conv, dconv = self.transmitted_irradiance(x[: self.n_gas])
        rho = self.reflectance(x[self.n_gas :])
        k = self.geometry.mu0 / np.pi
        y = k * rho * conv
        jac = np.empty((y.size, self.n_state))
        jac[:, : self.n_gas] = (k * rho * dconv).T
        dpoly = self._vander if self.reflectance_prior is None else self._vander * self.reflectance_prior[:, None]
        jac[:, self.n_gas :] = (k * conv)[:, None] * dpoly
        return y, jac

    def first_guess(self, y: np.ndarray) -> np.ndarray:
        """Zero enhancement and a least-squares reflectance polynomial."""
        conv, _ = self.transmitted_irradiance(np.zeros(self.n_gas))
        a = (self.geometry.mu0 / np.pi * conv)[:, None] * (
            self._vander if self.reflectance_prior is None else self._vander * self.reflectance_prior[:, None]
        )
        # Weight towards transparent bands so absorption does not bias the continuum.
        w = conv / conv.max()
        coeffs, *_ = np.linalg.lstsq(a * w[:, None], y * w, rcond=None)
        return np.concatenate([np.zeros(self.n_gas), coeffs])

    def unit_absorption_spectrum(self, gas: str = "CH4", delta_ppmm: float = 1000.0) -> np.ndarray:
        """d ln(L) / d(dX) [1/ppm·m] per band, linearised around ``delta_ppmm/2``.

        This is the target signature used by the matched filter; it only
        depends on spectroscopy, geometry and the instrument response.
        """
        i = self.gases.index(gas)
        dx0 = np.zeros(self.n_gas)
        dx1 = dx0.copy()
        dx1[i] = delta_ppmm
        c0, _ = self.transmitted_irradiance(dx0)
        c1, _ = self.transmitted_irradiance(dx1)
        return np.log(c1 / c0) / delta_ppmm
