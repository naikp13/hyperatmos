"""Image-based atmospheric correction: radiance cube to surface reflectance.

Three steps, each usable on its own:

1. **Aerosol** (:meth:`AtmosphericCorrection.estimate_aod`): dark dense
   vegetation constraint relating red and 2.1 um reflectance.
2. **Water vapour** (:meth:`AtmosphericCorrection.estimate_h2o`): per pixel,
   the column that makes the retrieved reflectance smoothest across an H2O
   band (0.94 or 1.14 um). This is the band-depth idea of ATREM-style codes
   phrased as a 1-D search over a transmittance lookup table, which works for
   any surface whose reflectance is smooth over ~100 nm.
3. **Inversion** (:meth:`AtmosphericCorrection.correct`): algebraic inversion
   of the radiative transfer equation per pixel.

The output can be written in ISOFIT's file layout
(:func:`hyperatmos.isofit_interop.write_isofit_outputs`) and used as a fast
first guess for, or as a comparison against, ISOFIT's optimal estimation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .rtm import RadiativeTransferModel

#: Candidate water vapour windows: (band centre region, left shoulder, right shoulder) [nm].
H2O_WINDOWS = {
    "940": ((915.0, 975.0), (860.0, 885.0), (1005.0, 1040.0)),
    "1140": ((1110.0, 1160.0), (1040.0, 1075.0), (1230.0, 1260.0)),
}


@dataclass
class CorrectionResult:
    reflectance: np.ndarray  # (rows, cols, bands)
    h2o_g_cm2: np.ndarray  # (rows, cols)
    aod550: float
    wavelength_nm: np.ndarray
    bad_bands: np.ndarray  # bands in saturated absorption, not trustworthy


class AtmosphericCorrection:
    """Atmospheric correction for one radiative transfer setup.

    Parameters
    ----------
    rtm : radiative transfer model (instrument, geometry, aerosol model).
    h2o_window : ``"940"``, ``"1140"`` or ``None`` (choose the first one the
        instrument covers).
    min_gas_transmittance : bands with lower gas transmittance (e.g. 1.38 and
        1.88 um) are flagged in ``bad_bands`` and set to NaN.
    """

    def __init__(self, rtm: RadiativeTransferModel, h2o_window: str | None = None, min_gas_transmittance: float = 0.1):
        self.rtm = rtm
        wl = rtm.wavelength_nm
        if h2o_window is None:
            for name, (_core, left, right) in H2O_WINDOWS.items():
                if wl.min() <= left[0] and wl.max() >= right[1]:
                    h2o_window = name
                    break
        self.h2o_window = h2o_window
        if h2o_window is not None:
            core, left, right = H2O_WINDOWS[h2o_window]
            self._core = (wl >= core[0]) & (wl <= core[1])
            self._shoulders = ((wl >= left[0]) & (wl <= left[1])) | ((wl >= right[0]) & (wl <= right[1]))
            self._win = (wl >= left[0]) & (wl <= right[1])
        t_ref = rtm.h2o_transmittance(rtm.h2o_reference)
        self.bad_bands = t_ref < min_gas_transmittance

    # ------------------------------------------------------------------ AOD
    def estimate_aod(
        self,
        radiance: np.ndarray,
        red_swir_ratio: float = 0.5,
        blue_swir_ratio: float = 0.25,
        ndvi_min: float = 0.7,
        aod_max: float = 1.5,
    ) -> float:
        """Dark dense vegetation (DDV) estimate of AOD at 550 nm.

        Over dark vegetation the red and blue reflectances are close to fixed
        fractions of the 2.1 um reflectance, which aerosol barely affects
        (Kaufman et al., 1997). The AOD is found by bisection so that the
        median red and blue reflectances of vegetated pixels match that
        prediction. Needs bands near 470, 660, 860 and 2130 nm; otherwise pass
        the AOD in (e.g. from a sun photometer or ISOFIT's ``AOT550``).
        """
        rtm = self.rtm
        wl = rtm.wavelength_nm
        need = (660.0, 860.0, 2130.0, 470.0)
        if wl.min() > 470 or wl.max() < 2130:
            raise ValueError("DDV needs bands at 470, 660, 860 and 2130 nm; pass aod550 explicitly")
        ir, inir, iswir, ib = (int(np.argmin(np.abs(wl - w))) for w in need)
        rho_s = rtm.to_reflectance_factor(radiance.reshape(-1, radiance.shape[-1]))
        ndvi = (rho_s[:, inir] - rho_s[:, ir]) / (rho_s[:, inir] + rho_s[:, ir])
        dark = (ndvi > ndvi_min) & (rho_s[:, iswir] > 0.01) & (rho_s[:, iswir] < 0.25)
        if dark.sum() < 10:
            raise ValueError("fewer than 10 dark vegetated pixels; pass aod550 explicitly")
        sub = rho_s[dark]

        def mismatch(aod):
            rho = rtm.coefficients(rtm.h2o_reference, aod).surface_reflectance(sub)
            return np.median(rho[:, ir] - red_swir_ratio * rho[:, iswir]) + np.median(
                rho[:, ib] - blue_swir_ratio * rho[:, iswir]
            )

        lo, hi = 0.0, aod_max
        if mismatch(lo) < 0:
            return 0.0
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if mismatch(mid) > 0 else (lo, mid)
        return 0.5 * (lo + hi)

    # ------------------------------------------------------------ water vapour
    def estimate_h2o(
        self, radiance: np.ndarray, aod550: float | None = None, grid: np.ndarray | None = None
    ) -> np.ndarray:
        """Per-pixel water vapour [g/cm^2] by smoothness of the retrieved reflectance.

        For each candidate column the reflectance in the window is inverted,
        a straight line is fitted through the shoulders, and the squared
        residual in the band core is minimised. A parabola through the best
        three candidates refines the minimum.
        """
        if self.h2o_window is None:
            raise ValueError("the instrument covers no water vapour window (0.94 or 1.14 um)")
        rtm = self.rtm
        shape = radiance.shape[:-1]
        rad = radiance.reshape(-1, radiance.shape[-1])
        grid = np.geomspace(0.1, 6.0, 40) if grid is None else grid
        win = self._win
        wl = rtm.wavelength_nm[win]
        rho_s = rtm.to_reflectance_factor(rad)[:, win]
        sc = rtm.scattering(aod550)
        sh = self._shoulders[win]
        core = self._core[win]
        x = np.column_stack([np.ones(sh.sum()), wl[sh]])
        proj = np.column_stack([np.ones(wl.size), wl]) @ np.linalg.pinv(x)  # line through shoulders
        cost = np.empty((rad.shape[0], grid.size))
        for k, w in enumerate(grid):
            tg = rtm.h2o_transmittance(w)[win]
            y = rho_s / tg - sc["rho_path"][win]
            rho = y / (sc["t_down"][win] * sc["t_up"][win] + sc["s_alb"][win] * y)
            cont = rho[:, sh] @ proj.T
            r = (rho - cont)[:, core] / np.maximum(np.abs(cont[:, core]), 1e-3)
            cost[:, k] = np.sum(r**2, axis=1)
        k = np.clip(np.argmin(cost, axis=1), 1, grid.size - 2)
        idx = np.arange(rad.shape[0])
        lg = np.log(grid)
        c0, c1, c2 = cost[idx, k - 1], cost[idx, k], cost[idx, k + 1]
        denom = c0 - 2 * c1 + c2
        step = np.where(denom > 0, 0.5 * (c0 - c2) / np.where(denom > 0, denom, 1), 0.0)
        step = np.clip(step, -1, 1)
        h2o = np.exp(lg[k] + step * (lg[k + 1] - lg[k - 1]) / 2)
        return h2o.reshape(shape)

    # ---------------------------------------------------------------- invert
    def correct(
        self, radiance: np.ndarray, aod550: float | None = None, h2o_g_cm2: np.ndarray | float | None = None
    ) -> CorrectionResult:
        """Full correction. Missing AOD and water vapour are estimated from the image."""
        rtm = self.rtm
        if radiance.shape[-1] != rtm.wavelength_nm.size:
            radiance = radiance[..., rtm.band_mask]
        if aod550 is None:
            aod550 = self.estimate_aod(radiance)  # raises if the scene has no usable dark vegetation
        if h2o_g_cm2 is None:
            h2o_g_cm2 = self.estimate_h2o(radiance, aod550)
        h2o_map = np.broadcast_to(np.asarray(h2o_g_cm2, dtype=float), radiance.shape[:-1])
        coeff = rtm.coefficients(h2o_map, aod550)
        rho = coeff.surface_reflectance(rtm.to_reflectance_factor(radiance))
        rho[..., self.bad_bands] = np.nan
        return CorrectionResult(rho, np.array(h2o_map), float(aod550), rtm.wavelength_nm, self.bad_bands)
