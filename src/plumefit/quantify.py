"""Emission-rate quantification from column enhancement maps.

Two standard methods are provided:

* **Integrated mass enhancement (IME)**, ``Q = U_eff * IME / L`` with
  ``L = sqrt(plume area)`` (Frankenberg et al., 2016; Varon et al., 2018).
* **Cross-sectional flux (CSF)**, ``Q = U * integral(dX dy)`` along transects
  perpendicular to the wind, averaged over several downwind distances.

``U_eff`` depends on the instrument, pixel size and plume masking, so it must
be calibrated for each sensor. :func:`calibrate_ueff` fits the linear relation
``U_eff = a * U10 + b`` to simulated plumes with known emission rates.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from .constants import ppmm_to_kg_m2


@dataclass
class PlumeMask:
    mask: np.ndarray
    n_pixels: int


def plume_mask(
    enhancement: np.ndarray,
    uncertainty: np.ndarray | float,
    source_rc: tuple[int, int],
    threshold_sigma: float = 2.0,
    smooth_px: float = 1.0,
    max_distance_px: int | None = None,
) -> PlumeMask:
    """Connected region above ``threshold_sigma`` that contains or touches the source.

    The map is lightly smoothed before thresholding to suppress isolated
    noise pixels; the connected component nearest the source is kept.
    """
    e = np.nan_to_num(enhancement, nan=0.0)
    if smooth_px > 0:
        e = ndimage.gaussian_filter(e, smooth_px)
    sig = np.broadcast_to(uncertainty, e.shape)
    sig_eff = sig / (2 * np.sqrt(np.pi) * smooth_px) if smooth_px > 0 else sig
    above = e > threshold_sigma * sig_eff
    labels, n = ndimage.label(above)
    if n == 0:
        return PlumeMask(np.zeros_like(above), 0)
    r0, c0 = source_rc
    win = labels[max(r0 - 2, 0) : r0 + 3, max(c0 - 2, 0) : c0 + 3]
    ids = np.unique(win[win > 0])
    if ids.size == 0:
        return PlumeMask(np.zeros_like(above), 0)
    mask = np.isin(labels, ids)
    if max_distance_px is not None:
        rr, cc = np.indices(mask.shape)
        mask &= np.hypot(rr - r0, cc - c0) <= max_distance_px
    return PlumeMask(mask, int(mask.sum()))


def integrated_mass_enhancement(
    enhancement_ppmm: np.ndarray, mask: np.ndarray, pixel_size_m: float, gas: str = "CH4"
) -> float:
    """IME [kg] over ``mask``."""
    kg_m2 = ppmm_to_kg_m2(np.nan_to_num(enhancement_ppmm[mask]), gas)
    return float(kg_m2.sum() * pixel_size_m**2)


def ime_emission_rate(
    enhancement_ppmm: np.ndarray, mask: np.ndarray, pixel_size_m: float, u_eff: float, gas: str = "CH4"
) -> float:
    """Emission rate [kg/h] by the IME method."""
    ime = integrated_mass_enhancement(enhancement_ppmm, mask, pixel_size_m, gas)
    length = np.sqrt(mask.sum()) * pixel_size_m
    return ime * u_eff / length * 3600.0


def ueff_linear(u10: float, a: float, b: float) -> float:
    """Effective wind speed ``a * U10 + b`` [m/s]."""
    return a * u10 + b


def calibrate_ueff(
    u10: np.ndarray, q_true_kg_h: np.ndarray, ime_kg: np.ndarray, length_m: np.ndarray
) -> tuple[float, float]:
    """Least-squares fit of ``U_eff = a U10 + b`` from plumes with known rates."""
    u_eff = q_true_kg_h / 3600.0 * length_m / ime_kg
    a, b = np.polyfit(u10, u_eff, 1)
    return float(a), float(b)


def cross_sectional_flux(
    enhancement_ppmm: np.ndarray,
    source_rc: tuple[int, int],
    wind_dir_deg: float,
    wind_speed: float,
    pixel_size_m: float,
    distances_m: np.ndarray,
    half_width_m: float = 500.0,
    gas: str = "CH4",
) -> tuple[float, np.ndarray]:
    """Emission rate [kg/h] from transects perpendicular to the wind.

    ``wind_dir_deg`` is the direction the plume travels towards, measured
    clockwise from image "up" (decreasing row index). Returns the median over
    transects and the individual transect estimates.
    """
    e = np.nan_to_num(ppmm_to_kg_m2(enhancement_ppmm, gas))
    th = np.deg2rad(wind_dir_deg)
    along = np.array([-np.cos(th), np.sin(th)])  # (drow, dcol)
    across = np.array([along[1], -along[0]])
    s = np.arange(-half_width_m, half_width_m + pixel_size_m, pixel_size_m / 2)
    fluxes = []
    for d in distances_m:
        pts = np.array(source_rc)[:, None] + (along[:, None] * d + across[:, None] * s[None, :]) / pixel_size_m
        vals = ndimage.map_coordinates(e, pts, order=1, mode="constant", cval=np.nan)
        if np.any(np.isnan(vals)):
            continue
        fluxes.append(wind_speed * np.trapezoid(vals, s) * 3600.0)
    fluxes = np.asarray(fluxes)
    return (float(np.median(fluxes)) if fluxes.size else float("nan")), fluxes


def monte_carlo_ime(
    enhancement_ppmm: np.ndarray,
    uncertainty_ppmm: np.ndarray,
    mask: np.ndarray,
    pixel_size_m: float,
    u10: float,
    u10_sigma: float,
    a: float,
    b: float,
    n: int = 500,
    gas: str = "CH4",
    seed: int = 0,
) -> tuple[float, float]:
    """IME emission rate with uncertainty from retrieval noise and wind [kg/h].

    Retrieval noise is treated as independent between pixels, which is
    optimistic where errors are spatially correlated (surface artefacts).
    """
    rng = np.random.default_rng(seed)
    q = np.empty(n)
    for i in range(n):
        e = enhancement_ppmm + rng.normal(0, 1, enhancement_ppmm.shape) * uncertainty_ppmm
        u = max(rng.normal(u10, u10_sigma), 0.1)
        q[i] = ime_emission_rate(e, mask, pixel_size_m, ueff_linear(u, a, b), gas)
    return float(np.mean(q)), float(np.std(q))
