"""Instrument trade studies for gas-sensing hyperspectral cameras.

For a candidate camera (spectral range, sampling, FWHM, SNR) and observation
geometry, the linear optimal-estimation posterior gives the single-pixel
precision of a gas enhancement retrieval, accounting for interference from
water vapour and from surface reflectance. Sweeping these parameters shows
which design choices buy the most sensitivity, before any hardware is built.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .forward import ForwardModel, Geometry
from .instrument import Instrument, uniform_instrument
from .spectroscopy import SpectralDatabase


@dataclass
class Sensitivity:
    sigma_ppmm: float  # 1-sigma single-pixel precision
    dofs: float  # degrees of freedom for the target gas
    n_bands: int


def single_pixel_precision(
    db: SpectralDatabase,
    instrument: Instrument,
    geometry: Geometry,
    gas: str = "CH4",
    reflectance: float = 0.25,
    retrieved_gases: tuple[str, ...] = ("CH4", "H2O"),
    poly_order: int = 2,
    prior_sigma_ppmm: float = 2e4,
) -> Sensitivity:
    """Posterior 1-sigma of ``gas`` for a flat surface of given ``reflectance``."""
    fm = ForwardModel(db, instrument, geometry, retrieved_gases, poly_order)
    x = np.zeros(fm.n_state)
    x[fm.n_gas] = reflectance
    y, k = fm.jacobian(x)
    se_inv = 1.0 / fm.instrument.noise_sigma(y) ** 2
    sa = np.full(fm.n_state, 10.0)
    for i, g in enumerate(fm.gases):
        sa[i] = prior_sigma_ppmm if g == gas else 5e6
    s_hat = np.linalg.inv((k.T * se_inv) @ k + np.diag(1 / sa**2))
    i = fm.gases.index(gas)
    a = s_hat @ (k.T * se_inv) @ k
    return Sensitivity(float(np.sqrt(s_hat[i, i])), float(a[i, i]), fm.instrument.n_bands)


def trade_study(
    db: SpectralDatabase,
    geometry: Geometry,
    fwhm_nm: np.ndarray,
    snr: np.ndarray,
    wl_range: tuple[float, float] = (2120.0, 2430.0),
    sampling_per_fwhm: float = 0.8,
    **kwargs,
) -> np.ndarray:
    """Precision [ppm·m] on a (len(fwhm), len(snr)) grid of camera designs."""
    out = np.empty((len(fwhm_nm), len(snr)))
    for i, fw in enumerate(fwhm_nm):
        for j, s in enumerate(snr):
            inst = uniform_instrument(*wl_range, sampling=fw * sampling_per_fwhm, fwhm=fw, snr_ref=s)
            out[i, j] = single_pixel_precision(db, inst, geometry, **kwargs).sigma_ppmm
    return out
