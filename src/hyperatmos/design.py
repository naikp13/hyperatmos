"""Instrument and mission design: how well can a camera retrieve the atmosphere?

For a candidate instrument (spectral range, sampling, FWHM, SNR), geometry
and platform altitude, the linearised optimal-estimation posterior gives the
precision of water vapour, aerosol, trace-gas columns and surface
reflectance, including the interference between all of them. Sweeping the
design parameters shows which choices buy the most information before any
hardware is built.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .forward import AtmosphereSurfaceModel, state_prior
from .instrument import Instrument, uniform_instrument
from .rtm import Geometry, RadiativeTransferModel
from .spectroscopy import SpectralDatabase


@dataclass
class Sensitivity:
    sigma: dict[str, float]  # posterior 1-sigma per atmospheric state element
    dofs: dict[str, float]  # degrees of freedom for signal per element
    reflectance_sigma: np.ndarray  # posterior 1-sigma of surface reflectance per band
    wavelength_nm: np.ndarray


def retrieval_precision(
    db: SpectralDatabase,
    instrument: Instrument,
    geometry: Geometry,
    reflectance: float | np.ndarray = 0.25,
    h2o_g_cm2: float = 1.5,
    aod550: float = 0.1,
    gases: tuple[str, ...] = (),
    **prior_kwargs,
) -> Sensitivity:
    """Posterior precision for a surface of given ``reflectance`` (scalar or per band)."""
    rtm = RadiativeTransferModel(db, instrument, geometry)
    fm = AtmosphereSurfaceModel(rtm, gases)
    rho = np.broadcast_to(np.asarray(reflectance, dtype=float), rtm.wavelength_nm.shape).copy()
    xa, sa = state_prior(
        fm,
        rho,
        h2o=(h2o_g_cm2, prior_kwargs.pop("h2o_sigma", 1.0)),
        aod=(aod550, prior_kwargs.pop("aod_sigma", 0.2)),
        **prior_kwargs,
    )
    y, k = fm.jacobian(xa)
    se_inv = np.where(fm.measurement_mask, 1.0 / rtm.instrument.noise_sigma(y) ** 2, 0.0)
    s_hat = np.linalg.inv((k.T * se_inv) @ k + np.linalg.inv(sa))
    a = s_hat @ (k.T * se_inv) @ k
    names = fm.state_names[: fm.n_atm]
    sig = {n: float(np.sqrt(s_hat[i, i])) for i, n in enumerate(names)}
    dofs = {n: float(a[i, i]) for i, n in enumerate(names)}
    return Sensitivity(sig, dofs, np.sqrt(np.diag(s_hat))[fm.n_atm :], rtm.wavelength_nm)


def trade_study(
    db: SpectralDatabase,
    geometry: Geometry,
    fwhm_nm: np.ndarray,
    snr: np.ndarray,
    quantity: str = "H2O",
    wl_range: tuple[float, float] = (850.0, 1250.0),
    sampling_per_fwhm: float = 0.8,
    **kwargs,
) -> np.ndarray:
    """Posterior 1-sigma of ``quantity`` on a (len(fwhm), len(snr)) grid of designs.

    SNR is defined at the median radiance of the reference scene in
    ``wl_range``, so it is the SNR the camera actually achieves on that scene.
    """
    probe = RadiativeTransferModel(db, uniform_instrument(*wl_range, sampling=5.0, fwhm=5.0), geometry)
    rho = kwargs.get("reflectance", 0.25)
    l_ref = float(np.median(probe.radiance(np.broadcast_to(rho, probe.wavelength_nm.shape))))
    out = np.empty((len(fwhm_nm), len(snr)))
    for i, fw in enumerate(fwhm_nm):
        for j, s in enumerate(snr):
            inst = uniform_instrument(
                *wl_range, sampling=fw * sampling_per_fwhm, fwhm=fw, snr_ref=s, radiance_ref=l_ref
            )
            out[i, j] = retrieval_precision(db, inst, geometry, **kwargs).sigma[quantity]
    return out
