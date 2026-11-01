"""Forward model for optimal estimation of atmosphere and surface together.

State vector, in order:

* ``H2O``: water vapour column [g/cm^2];
* ``AOD550``: aerosol optical depth at 550 nm;
* optional column scale factors of other gases (``scale_CH4``, ``scale_CO2``,
  ``scale_O3``...), 1 meaning the background atmosphere;
* surface reflectance in every band.

This is the same joint atmosphere-surface formulation as ISOFIT, written
around :class:`hyperatmos.rtm.RadiativeTransferModel` so that it runs without
licensed radiative transfer codes and supports drone geometry. Jacobians are
analytic for the surface and by finite differences for the atmosphere.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from .rtm import RadiativeTransferModel


class AtmosphereSurfaceModel:
    """Radiance as a function of (H2O, AOD, gas scales, band reflectances).

    Parameters
    ----------
    rtm : radiative transfer model.
    gases : extra gases whose column is retrieved as a scale factor.
    min_gas_transmittance : bands with lower transmittance (saturated H2O
        bands) are excluded from the measurement vector, as in ISOFIT's
        windows.
    """

    def __init__(self, rtm: RadiativeTransferModel, gases: tuple[str, ...] = (), min_gas_transmittance: float = 0.1):
        self.rtm = rtm
        self.gases = tuple(gases)
        self.state_names = ["H2O", "AOD550", *[f"scale_{g}" for g in self.gases]]
        self.n_atm = len(self.state_names)
        nb = rtm.wavelength_nm.size
        self.state_names += [f"rho_{w:.1f}" for w in rtm.wavelength_nm]
        self.n_state = self.n_atm + nb
        self.measurement_mask = rtm.h2o_transmittance(rtm.h2o_reference) >= min_gas_transmittance
        self._steps = np.array([0.02, 0.005] + [0.002] * len(self.gases))

    @property
    def wavelength_nm(self) -> np.ndarray:
        return self.rtm.wavelength_nm

    def _coefficients(self, x):
        scales = {g: x[2 + i] for i, g in enumerate(self.gases)} or None
        return self.rtm.coefficients(x[0], x[1], scales)

    def radiance(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        c = self._coefficients(x)
        return self.rtm.geometry.mu0 * self.rtm.e0 / np.pi * c.toa_reflectance(x[self.n_atm :])

    def jacobian(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Radiance and Jacobian (n_bands, n_state)."""
        x = np.asarray(x, dtype=float)
        k0 = self.rtm.geometry.mu0 * self.rtm.e0 / np.pi
        rho = x[self.n_atm :]
        c = self._coefficients(x)
        y = k0 * c.toa_reflectance(rho)
        jac = np.zeros((y.size, self.n_state))
        for i, h in enumerate(self._steps):
            xp = x.copy()
            xp[i] += h
            jac[:, i] = (self.radiance(xp) - y) / h
        jac[:, self.n_atm :] = np.diag(k0 * c.d_toa_d_rho(rho))
        return y, jac

    def surface_index(self) -> slice:
        return slice(self.n_atm, self.n_state)


def smooth_surface_prior(
    wavelength_nm: np.ndarray,
    mean: np.ndarray,
    rel_sigma: float = 0.1,
    abs_sigma: float = 0.01,
    correlation_nm: float = 40.0,
    nugget: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    """Gaussian surface prior with spectrally smooth errors.

    ``mean`` is a reflectance guess per band (e.g. a smoothed algebraic
    inversion, a library spectrum or ISOFIT output). Errors have standard
    deviation ``rel_sigma * mean + abs_sigma`` and a squared-exponential
    correlation of length ``correlation_nm``, plus a small uncorrelated part
    (``nugget``, as a fraction of the variance) for fine surface structure.
    The smoothness is what separates narrow gas absorption from the surface.
    """
    sig = rel_sigma * np.abs(mean) + abs_sigma
    d = wavelength_nm[:, None] - wavelength_nm[None, :]
    corr = (1 - nugget) * np.exp(-0.5 * (d / correlation_nm) ** 2) + nugget * np.eye(mean.size)
    return mean, sig[:, None] * sig[None, :] * corr


def surface_guess(
    fm: AtmosphereSurfaceModel, radiance: np.ndarray, h2o_g_cm2: float, aod550: float, smooth_bands: float = 2.0
) -> np.ndarray:
    """Reflectance first guess: algebraic inversion, interpolated across
    excluded (saturated) bands and lightly smoothed."""
    rho = fm.rtm.surface_reflectance(radiance, h2o_g_cm2, aod550)
    good = fm.measurement_mask & np.isfinite(rho)
    wl = fm.wavelength_nm
    rho = np.interp(wl, wl[good], rho[good])
    return ndimage.gaussian_filter1d(rho, smooth_bands) if smooth_bands else rho


def state_prior(
    fm: AtmosphereSurfaceModel,
    surface_mean: np.ndarray,
    h2o: tuple[float, float] = (1.5, 0.5),
    aod: tuple[float, float] = (0.1, 0.2),
    gas_scale_sigma: float = 0.1,
    surface_cov: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Prior mean and covariance for the full state vector.

    ``h2o`` and ``aod`` are (mean, sigma); gas scales have mean 1. Without
    ``surface_cov`` a :func:`smooth_surface_prior` around ``surface_mean`` is used.
    """
    if surface_cov is None:
        surface_mean, surface_cov = smooth_surface_prior(fm.wavelength_nm, surface_mean)
    xa = np.concatenate([[h2o[0], aod[0]], np.ones(len(fm.gases)), surface_mean])
    sa = np.zeros((fm.n_state, fm.n_state))
    sa[0, 0], sa[1, 1] = h2o[1] ** 2, aod[1] ** 2
    for i in range(len(fm.gases)):
        sa[2 + i, 2 + i] = gas_scale_sigma**2
    sa[fm.n_atm :, fm.n_atm :] = surface_cov
    return xa, sa
