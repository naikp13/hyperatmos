"""Interoperability with ISOFIT.

hyperatmos reads and writes ISOFIT's file conventions, so the two codes can
be swapped, chained or compared on the same flight line:

* :func:`read_isofit_reflectance`, :func:`read_isofit_state` read ISOFIT's
  ``*_rfl`` and ``*_state`` ENVI products;
* :func:`write_isofit_outputs` writes hyperatmos results in the same layout
  (state band names ``AOT550`` and ``H2OSTR``), for ISOFIT-based tooling;
* :func:`atmosphere_from_isofit` scales the background atmosphere to ISOFIT's
  retrieved water vapour, e.g. to compute gas transmittances consistent with
  an ISOFIT run;
* :func:`surface_prior_from_reflectance` turns a reflectance cube (ISOFIT or
  hyperatmos output) into a Gaussian surface prior for optimal estimation;
* :func:`compare_reflectance` gives per-band bias and RMSE between two
  reflectance products, e.g. hyperatmos vs ISOFIT.

Instruments are shared through ISOFIT wavelength files
(:meth:`hyperatmos.instrument.Instrument.from_isofit_wavelength_file`).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage

from .atmosphere import Atmosphere
from .constants import h2o_g_cm2_to_molec_cm2
from .io import header_wavelengths_nm, read_envi, write_envi


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


def write_isofit_outputs(
    prefix: str | Path,
    reflectance: np.ndarray,
    wavelength_nm: np.ndarray,
    h2o_g_cm2: np.ndarray,
    aod550: np.ndarray | float,
    fwhm_nm: np.ndarray | None = None,
) -> tuple[Path, Path]:
    """Write ``<prefix>_rfl`` and ``<prefix>_state`` in ISOFIT's layout."""
    prefix = Path(prefix)
    rfl = np.nan_to_num(reflectance, nan=-9999.0)
    write_envi(f"{prefix}_rfl", rfl, wavelength_nm, fwhm_nm=fwhm_nm, description="hyperatmos reflectance")
    aod = np.broadcast_to(np.asarray(aod550, dtype=float), h2o_g_cm2.shape)
    state = np.concatenate([rfl, aod[..., None], h2o_g_cm2[..., None]], axis=-1)
    names = [f"{w:.2f}" for w in wavelength_nm] + ["AOT550", "H2OSTR"]
    write_envi(f"{prefix}_state", state, band_names=names, description="hyperatmos state")
    return Path(f"{prefix}_rfl"), Path(f"{prefix}_state")


def atmosphere_from_isofit(atm: Atmosphere, h2ostr_g_cm2: float) -> Atmosphere:
    """Scale the H2O profile so the total column matches ISOFIT's ``H2OSTR``."""
    atm.set_column("H2O", h2o_g_cm2_to_molec_cm2(h2ostr_g_cm2))
    return atm


def surface_prior_from_reflectance(
    reflectance: np.ndarray,
    exclude: np.ndarray | None = None,
    dilate_px: int = 0,
    shrinkage: float = 0.01,
    noise_floor: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray]:
    """Mean and covariance of band reflectance over valid pixels.

    The covariance captures how surface features co-vary across bands, which
    is what lets a joint atmosphere-surface retrieval separate them from
    atmospheric absorption. ``exclude`` masks pixels to leave out (clouds,
    shadows, anomalies), optionally dilated.
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


@dataclass
class ReflectanceComparison:
    wavelength_nm: np.ndarray
    bias: np.ndarray
    rmse: np.ndarray
    n_pixels: int


def compare_reflectance(a: np.ndarray, b: np.ndarray, wavelength_nm: np.ndarray) -> ReflectanceComparison:
    """Per-band bias (a - b) and RMSE over pixels where both are valid."""
    diff = (a - b).reshape(-1, a.shape[-1])
    ok = np.all(np.isfinite(diff), axis=1)
    d = diff[ok]
    return ReflectanceComparison(wavelength_nm, d.mean(axis=0), np.sqrt((d**2).mean(axis=0)), int(ok.sum()))


def resample_to_bands(spectrum_wl: np.ndarray, spectrum: np.ndarray, target_wl: np.ndarray) -> np.ndarray:
    """Linear interpolation of spectra (..., bands) onto other band centres."""
    flat = spectrum.reshape(-1, spectrum.shape[-1])
    out = np.stack([np.interp(target_wl, spectrum_wl, s) for s in flat])
    return out.reshape(*spectrum.shape[:-1], target_wl.size)
