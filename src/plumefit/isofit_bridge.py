"""Glue between ISOFIT products and plumefit retrievals.

ISOFIT retrieves surface reflectance, aerosol optical depth and water vapour
by optimal estimation, but treats CH4 and CO2 as fixed background absorbers.
plumefit picks up from there:

1. plume-free ISOFIT reflectance provides a statistical surface prior (band
   mean and covariance) for a joint retrieval of gas and reflectance, so
   surface spectral structure (clays, carbonates, paints) is not mistaken for
   gas absorption (:func:`surface_prior_from_reflectance`);
2. the ISOFIT water-vapour column (``H2OSTR``, g/cm^2) sets the background
   H2O of the spectroscopy instead of a climatology;
3. the same ISOFIT wavelength file describes the instrument in both codes.

Because ISOFIT does not model the plume, strong plumes leave a dent in its
reflectance at the gas bands, which is why plume pixels are excluded when the
prior is estimated. :func:`plume_safe_prior` offers a simpler alternative,
copying the reflectance of the nearest plume-free pixel; it only works where
the surface is uniform on the scale of the plume
(see ``examples/04_isofit_handoff.py``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import ndimage

from .atmosphere import Atmosphere
from .constants import MOLAR_MASS, N_AVOGADRO
from .io import header_wavelengths_nm, read_envi


def read_isofit_reflectance(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Read an ISOFIT ``*_rfl`` ENVI file. Returns (cube, wavelength_nm)."""
    cube, hdr = read_envi(path)
    return cube, header_wavelengths_nm(hdr)


def read_isofit_state(path: str | Path) -> dict[str, np.ndarray]:
    """Read the non-surface elements of an ISOFIT ``*_state`` file by band name.

    Returns e.g. ``{"AOT550": ..., "H2OSTR": ...}`` as (lines, samples) arrays.
    """
    cube, hdr = read_envi(path)
    names = hdr.get("band names")
    if not names:
        raise ValueError("ISOFIT state file has no band names")
    out = {}
    for i, n in enumerate(names):
        n = str(n).strip()
        if not n.replace(".", "", 1).isdigit():  # skip reflectance bands named by wavelength
            out[n] = cube[..., i]
    return out


def scale_atmosphere_h2o(atm: Atmosphere, h2o_g_cm2: float) -> Atmosphere:
    """Scale the H2O profile so the total column matches ``h2o_g_cm2``."""
    target = h2o_g_cm2 / (MOLAR_MASS["H2O"] * 1e3) * N_AVOGADRO  # molecules/cm^2
    current = atm.gas_column("H2O").sum()
    atm.vmr["H2O"] = atm.vmr["H2O"] * target / current
    return atm


def plume_safe_prior(reflectance: np.ndarray, plume_mask: np.ndarray, dilate_px: int = 2) -> np.ndarray:
    """Replace reflectance under (dilated) plume pixels by the nearest plume-free pixel.

    Only appropriate over spatially uniform surfaces; prefer
    :func:`surface_prior_from_reflectance` with ``ForwardModel(surface="bands")``.
    """
    mask = ndimage.binary_dilation(plume_mask, iterations=dilate_px) if dilate_px else plume_mask
    if not mask.any():
        return reflectance.copy()
    _, (ri, ci) = ndimage.distance_transform_edt(mask, return_indices=True)
    return reflectance[ri, ci]


def resample_to_bands(spectrum_wl: np.ndarray, spectrum: np.ndarray, target_wl: np.ndarray) -> np.ndarray:
    """Linear interpolation of spectra (..., bands) onto other band centres."""
    flat = spectrum.reshape(-1, spectrum.shape[-1])
    out = np.stack([np.interp(target_wl, spectrum_wl, s) for s in flat])
    return out.reshape(*spectrum.shape[:-1], target_wl.size)


def surface_prior_from_reflectance(
    reflectance: np.ndarray,
    exclude: np.ndarray | None = None,
    dilate_px: int = 3,
    shrinkage: float = 0.01,
    noise_floor: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray]:
    """Mean and covariance of band reflectance over plume-free pixels.

    ``reflectance`` is an ISOFIT ``_rfl`` cube (lines, samples, bands);
    ``exclude`` marks plume pixels (e.g. matched-filter detections), which are
    dilated and left out so the prior does not learn the plume itself. The
    covariance captures how surface features co-vary across bands (clay and
    carbonate absorptions near 2.2-2.35 um), which is what lets the joint
    retrieval separate them from gas absorption.
    """
    valid = np.all(np.isfinite(reflectance), axis=-1)
    if exclude is not None:
        ex = ndimage.binary_dilation(exclude, iterations=dilate_px) if dilate_px else exclude
        valid &= ~ex
    x = reflectance[valid]
    mean = x.mean(axis=0)
    cov = np.cov(x, rowvar=False)
    cov = (1 - shrinkage) * cov + shrinkage * np.diag(np.diag(cov))
    cov += np.eye(cov.shape[0]) * noise_floor**2
    return mean, cov
