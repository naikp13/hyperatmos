"""Per-pixel optimal estimation (Rodgers, 2000) of gas enhancements.

Same statistical framework as ISOFIT, applied to a small state vector:
column enhancements of the target gases plus a reflectance polynomial. Each
retrieval reports the posterior covariance, averaging kernel and degrees of
freedom for signal, which make the result directly usable for validation and
error budgets.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..forward import ForwardModel


@dataclass
class OEResult:
    x: np.ndarray  # retrieved state
    s_hat: np.ndarray  # posterior covariance
    averaging_kernel: np.ndarray
    dofs: float
    chi2_reduced: float
    iterations: int
    converged: bool
    gases: tuple[str, ...]

    def enhancement(self, gas: str) -> float:
        return float(self.x[self.gases.index(gas)])

    def uncertainty(self, gas: str) -> float:
        i = self.gases.index(gas)
        return float(np.sqrt(self.s_hat[i, i]))


class OptimalEstimation:
    """Levenberg-Marquardt optimal estimation retrieval.

    Parameters
    ----------
    fm : forward model.
    prior_sigma_ppmm : prior 1-sigma per retrieved gas [ppm·m]. Prior means
        are zero (no enhancement).
    poly_prior_sigma : prior 1-sigma for the reflectance polynomial
        coefficients (loose by default). The prior mean is taken from the
        first-guess fit of each spectrum.
    surface_prior : ``(mean, covariance)`` of the band reflectances, required
        when the forward model uses ``surface="bands"``. Estimating it from
        plume-free ISOFIT reflectance turns the surface into a constrained
        part of the joint state, as in ISOFIT itself.
    """

    def __init__(
        self,
        fm: ForwardModel,
        prior_sigma_ppmm: dict[str, float] | None = None,
        poly_prior_sigma: float = 10.0,
        max_iter: int = 15,
        tol: float = 1e-3,
        surface_prior: tuple[np.ndarray, np.ndarray] | None = None,
    ):
        self.fm = fm
        default = {"CH4": 2e4, "CO2": 1e6, "H2O": 2e6}
        prior_sigma_ppmm = {**default, **(prior_sigma_ppmm or {})}
        gas_sig = np.array([prior_sigma_ppmm[g] for g in fm.gases], dtype=float)
        self.surface_mean = None
        if fm.surface == "bands":
            if surface_prior is None:
                raise ValueError("surface='bands' needs surface_prior=(mean, covariance)")
            mean, cov = (np.asarray(a, dtype=float) for a in surface_prior)
            if mean.size == fm.band_mask.size:
                mean, cov = mean[fm.band_mask], cov[np.ix_(fm.band_mask, fm.band_mask)]
            self.surface_mean = mean
            surf_chol = np.linalg.cholesky(cov)
        else:
            surf_chol = np.diag(np.full(fm.n_surface, poly_prior_sigma))
        # Square root of the prior covariance, block diagonal (gases, surface).
        self.sa_chol = np.zeros((fm.n_state, fm.n_state))
        self.sa_chol[: fm.n_gas, : fm.n_gas] = np.diag(gas_sig)
        self.sa_chol[fm.n_gas :, fm.n_gas :] = surf_chol
        self.max_iter = max_iter
        self.tol = tol

    def retrieve(self, y: np.ndarray, noise_sigma: np.ndarray | None = None) -> OEResult:
        fm = self.fm
        y = np.asarray(y, dtype=float)
        if y.size != fm.instrument.n_bands:
            y = y[fm.band_mask]
        if noise_sigma is None:
            noise_sigma = fm.instrument.noise_sigma(y)
        elif noise_sigma.size != y.size:
            noise_sigma = noise_sigma[fm.band_mask]
        se_inv = 1.0 / np.square(noise_sigma)
        if self.surface_mean is not None:
            xa = np.concatenate([np.zeros(fm.n_gas), self.surface_mean])
        else:
            xa = fm.first_guess(y)
        # Iterate on the prior-whitened state z = L^-1 (x - xa) with Sa = L L^T,
        # which keeps the normal equations well conditioned despite state
        # elements that differ by many orders of magnitude.
        sl = self.sa_chol
        z = np.zeros_like(xa)
        lam = 1e-2

        def cost(zz):
            r = y - fm.radiance(xa + sl @ zz)
            return r @ (se_inv * r) + zz @ zz

        c = cost(z)
        converged = False
        it = 0
        eye = np.eye(z.size)
        for it in range(1, self.max_iter + 1):  # noqa: B007 (reported in the result)
            fx, k = fm.jacobian(xa + sl @ z)
            kz = k @ sl
            kt_se = kz.T * se_inv
            h = kt_se @ kz + eye
            g = kt_se @ (y - fx) - z
            # Rodgers (2000) eq. 5.29: stop when the Gauss-Newton step is small
            # compared with the posterior uncertainty.
            if g @ np.linalg.solve(h, g) < self.tol * z.size:
                converged = True
                break
            while True:
                step = np.linalg.solve(h + lam * eye, g)  # Rodgers eq. 5.36
                c_new = cost(z + step)
                if c_new <= c or lam > 1e8:
                    break
                lam *= 10
            if not c_new <= c:
                break
            z = z + step
            if c - c_new < 1e-8 * c:
                c = c_new
                converged = True
                break
            c = c_new
            lam = max(lam / 10, 1e-6)
        x = xa + sl @ z
        fx, k = fm.jacobian(x)
        kt_se = k.T * se_inv
        kz = k @ sl
        s_hat = sl @ np.linalg.inv((kz.T * se_inv) @ kz + eye) @ sl.T
        a = s_hat @ kt_se @ k
        r = y - fx
        chi2 = float(r @ (se_inv * r)) / max(y.size - x.size, 1)
        return OEResult(x, s_hat, a, float(np.trace(a[: fm.n_gas, : fm.n_gas])), chi2, it, converged, fm.gases)

    def retrieve_cube(
        self,
        radiance: np.ndarray,
        gas: str = "CH4",
        mask: np.ndarray | None = None,
        reflectance_prior: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Retrieve every pixel of a (rows, cols, bands) cube.

        Returns enhancement and 1-sigma uncertainty maps for ``gas`` [ppm·m].
        ``mask`` restricts the (slow) retrieval to selected pixels, e.g. matched
        filter detections. ``reflectance_prior`` is an optional (rows, cols,
        bands) surface reflectance cube, typically ISOFIT output passed through
        :func:`plumefit.isofit_bridge.plume_safe_prior`.
        """
        rows, cols, _ = radiance.shape
        enh = np.full((rows, cols), np.nan)
        unc = np.full((rows, cols), np.nan)
        for r in range(rows):
            for c in range(cols):
                if mask is not None and not mask[r, c]:
                    continue
                if reflectance_prior is not None:
                    self.fm.set_reflectance_prior(reflectance_prior[r, c])
                res = self.retrieve(radiance[r, c])
                enh[r, c] = res.enhancement(gas)
                unc[r, c] = res.uncertainty(gas)
        return enh, unc
