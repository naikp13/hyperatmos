"""Column-wise matched filter for gas enhancement maps.

Implements the clutter matched filter used for AVIRIS-NG and EMIT methane
products (Thompson et al., 2015, AMT; Foote et al., 2020, IEEE TGRS), with
three practical options:

* statistics per detector column (or groups of columns), which absorbs
  pushbroom striping and cross-track variations of the instrument response;
* shrinkage of the background covariance for numerical stability;
* iterative removal of plume pixels from the background statistics, which
  reduces the well-known under-estimation of strong plumes.

The output is the enhancement in ppm·m and its 1-sigma uncertainty from the
background clutter.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MatchedFilterResult:
    enhancement: np.ndarray  # (rows, cols) ppm·m
    uncertainty: np.ndarray  # (rows, cols) ppm·m, 1-sigma from clutter
    albedo: np.ndarray  # (rows, cols) relative brightness factor


def _shrunk_covariance(x: np.ndarray, shrinkage: float) -> np.ndarray:
    cov = np.cov(x, rowvar=False)
    target = np.diag(np.diag(cov))
    return (1 - shrinkage) * cov + shrinkage * target


def matched_filter(
    radiance: np.ndarray,
    unit_absorption: np.ndarray,
    column_group: int = 1,
    shrinkage: float = 1e-3,
    n_iter: int = 5,
    exclude_sigma: float = 2.0,
    albedo_correction: bool = True,
    valid: np.ndarray | None = None,
) -> MatchedFilterResult:
    """Run the matched filter over a radiance cube.

    Parameters
    ----------
    radiance : (rows, cols, bands) array, rows are along-track lines.
    unit_absorption : (bands,) d ln L / d(dX) in 1/ppm·m, from
        :meth:`plumefit.forward.ForwardModel.unit_absorption_spectrum`.
    column_group : number of adjacent columns pooled into one set of statistics.
    n_iter : passes; after each, pixels above ``exclude_sigma`` are left out of
        the background mean/covariance.
    albedo_correction : scale the target by each pixel's brightness relative to
        the column mean (Foote et al. 2020), so dark and bright pixels are not
        biased in opposite directions.
    valid : optional (rows, cols) boolean mask of usable pixels.
    """
    rows, cols, nb = radiance.shape
    if unit_absorption.shape != (nb,):
        raise ValueError("unit_absorption must have one value per band")
    if valid is None:
        valid = np.all(np.isfinite(radiance), axis=2)
    enh = np.full((rows, cols), np.nan)
    unc = np.full((rows, cols), np.nan)
    alb = np.full((rows, cols), np.nan)
    for c0 in range(0, cols, column_group):
        sl = slice(c0, min(c0 + column_group, cols))
        x = radiance[:, sl, :].reshape(-1, nb)
        v = valid[:, sl].reshape(-1)
        use = v.copy()
        if use.sum() <= nb:
            raise ValueError(
                f"Columns {c0}-{sl.stop - 1}: {use.sum()} valid pixels for {nb} bands; "
                "increase column_group or reduce the band count."
            )
        for _ in range(max(n_iter, 1)):
            mu = x[use].mean(axis=0)
            cov = _shrunk_covariance(x[use], shrinkage)
            target = mu * unit_absorption
            cinv_t = np.linalg.solve(cov, target)
            norm = target @ cinv_t
            a = np.ones(len(x))
            if albedo_correction:
                a = (x @ mu) / (mu @ mu)
            alpha = ((x - mu) @ cinv_t) / (a * norm)
            sigma = 1.0 / np.sqrt(norm) / a
            new_use = v & (alpha < exclude_sigma * np.nanstd(alpha[v]))
            if np.array_equal(new_use, use):
                break
            use = new_use
        alpha[~v] = np.nan
        shape = (rows, sl.stop - sl.start)
        enh[:, sl] = alpha.reshape(shape)
        unc[:, sl] = sigma.reshape(shape)
        alb[:, sl] = a.reshape(shape)
    return MatchedFilterResult(enh, unc, alb)
