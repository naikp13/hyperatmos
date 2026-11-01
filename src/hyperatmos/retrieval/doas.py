"""Differential optical absorption spectroscopy (DOAS) for UV-visible imagers.

Retrieves slant column densities (SCDs) of NO2, SO2 and similar absorbers
from a measured spectrum ``I`` and a reference ``I0`` (for an imaging
spectrometer, typically the mean of clean-background pixels in the same
detector column, or a zenith/solar reference)::

    ln(I0 / I) = sum_g sigma_g * SCD_g + P(lambda) [+ shift term]

``P`` is a low-order polynomial absorbing broadband extinction and surface
effects. A first-order wavelength shift of the measurement is optionally
fitted through the derivative of ``ln I``. Vertical columns follow as
``VCD = SCD / AMF``, where the air-mass factor comes from radiative transfer;
for absorbers in the boundary layer observed from low altitude the geometric
AMF ``1/mu0 + 1/mu_v`` is the usual first estimate. The retrieved columns are
then differential with respect to the reference.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..instrument import Instrument
from ..spectroscopy import CrossSectionTable

_SCALE = 1e19


def convolve_cross_section(
    table: CrossSectionTable, instrument: Instrument, i0_hires: np.ndarray | None = None
) -> np.ndarray:
    """Cross section at instrument resolution [cm^2/molecule].

    With ``i0_hires`` (a high-resolution solar spectrum on ``table.wavelength_nm``)
    the I0-effect correction of Aliwell et al. (2002) is applied.
    """
    wl = table.wavelength_nm
    sig_inst = instrument.fwhm_nm / (2 * np.sqrt(2 * np.log(2)))
    w = np.exp(-0.5 * ((wl[None, :] - instrument.wavelength_nm[:, None]) / sig_inst[:, None]) ** 2)
    w /= w.sum(axis=1, keepdims=True)
    if i0_hires is None:
        return w @ table.sigma
    scd_ref = 1e16
    i_conv = w @ (i0_hires * np.exp(-table.sigma * scd_ref))
    i0_conv = w @ i0_hires
    return np.log(i0_conv / i_conv) / scd_ref


@dataclass
class DOASResult:
    scd: dict[str, float]  # molecules/cm^2
    scd_error: dict[str, float]
    polynomial: np.ndarray
    shift_nm: float
    rms: float


def doas_fit(
    wavelength_nm: np.ndarray,
    intensity: np.ndarray,
    reference: np.ndarray,
    cross_sections: dict[str, np.ndarray],
    window: tuple[float, float] = (430.0, 465.0),
    poly_order: int = 3,
    fit_shift: bool = True,
) -> DOASResult:
    """Linear DOAS fit in ``window``.

    ``cross_sections`` maps gas name to the cross section already convolved
    to the instrument bands (see :func:`convolve_cross_section`).
    """
    m = (wavelength_nm >= window[0]) & (wavelength_nm <= window[1]) & (intensity > 0) & (reference > 0)
    wl = wavelength_nm[m]
    tau = np.log(reference[m] / intensity[m])
    xn = (wl - wl.mean()) / (0.5 * (wl.max() - wl.min()))
    cols, names = [], []
    for g, s in cross_sections.items():
        cols.append(s[m] * _SCALE)  # cross sections ~1e-19 cm^2; rescale for conditioning
        names.append(g)
    cols.extend(xn**k for k in range(poly_order + 1))
    if fit_shift:
        cols.append(np.gradient(np.log(intensity[m]), wl))
    a = np.column_stack(cols)
    coef, *_ = np.linalg.lstsq(a, tau, rcond=None)
    resid = tau - a @ coef
    dof = max(len(tau) - a.shape[1], 1)
    s2 = resid @ resid / dof
    cov = s2 * np.linalg.inv(a.T @ a)
    ng = len(names)
    scd = {g: float(coef[i] * _SCALE) for i, g in enumerate(names)}
    err = {g: float(np.sqrt(cov[i, i]) * _SCALE) for i, g in enumerate(names)}
    shift = float(coef[-1]) if fit_shift else 0.0
    return DOASResult(scd, err, coef[ng : ng + poly_order + 1], shift, float(np.sqrt(s2)))


def doas_image(
    wavelength_nm: np.ndarray,
    cube: np.ndarray,
    cross_sections: dict[str, np.ndarray],
    reference_rows: slice | np.ndarray,
    gas: str = "NO2",
    **kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """DOAS over a (rows, cols, bands) cube with a per-column reference spectrum.

    ``reference_rows`` selects along-track lines taken as the clean background
    (e.g. a rural part of the flight line); their mean in each column is ``I0``.
    """
    rows, cols, _ = cube.shape
    scd = np.full((rows, cols), np.nan)
    err = np.full((rows, cols), np.nan)
    for c in range(cols):
        i0 = cube[reference_rows, c, :].mean(axis=0)
        for r in range(rows):
            res = doas_fit(wavelength_nm, cube[r, c], i0, cross_sections, **kwargs)
            scd[r, c] = res.scd[gas]
            err[r, c] = res.scd_error[gas]
    return scd, err
