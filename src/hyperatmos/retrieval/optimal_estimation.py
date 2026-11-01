"""Optimal estimation (Rodgers, 2000) for any forward model.

The forward model needs ``n_state``, ``state_names``, ``radiance(x)``,
``jacobian(x) -> (y, K)`` and optionally ``measurement_mask`` (bands to use).
:class:`hyperatmos.forward.AtmosphereSurfaceModel` is the main one.

Each retrieval returns the posterior covariance, averaging kernel, degrees of
freedom for signal and reduced chi-square, so results come with an error
budget, as in ISOFIT.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class OEResult:
    x: np.ndarray
    s_hat: np.ndarray
    averaging_kernel: np.ndarray
    chi2_reduced: float
    iterations: int
    converged: bool
    state_names: list[str]

    def value(self, name: str) -> float:
        return float(self.x[self.state_names.index(name)])

    def sigma(self, name: str) -> float:
        i = self.state_names.index(name)
        return float(np.sqrt(self.s_hat[i, i]))

    def dofs(self, names: list[str] | None = None) -> float:
        """Degrees of freedom for signal, optionally for a subset of the state."""
        if names is None:
            return float(np.trace(self.averaging_kernel))
        idx = [self.state_names.index(n) for n in names]
        return float(np.trace(self.averaging_kernel[np.ix_(idx, idx)]))


class OptimalEstimation:
    """Levenberg-Marquardt optimal estimation.

    Parameters
    ----------
    fm : forward model.
    max_iter, tol : iteration limit; convergence when the Gauss-Newton step is
        small compared with the posterior uncertainty (Rodgers eq. 5.29).
    model_error : fractional forward-model uncertainty added to the noise.
    """

    def __init__(self, fm, max_iter: int = 20, tol: float = 1e-3, model_error: float = 0.0):
        self.fm = fm
        self.max_iter = max_iter
        self.tol = tol
        self.model_error = model_error

    def retrieve(
        self,
        y: np.ndarray,
        xa: np.ndarray,
        sa: np.ndarray,
        noise_sigma: np.ndarray | None = None,
        x0: np.ndarray | None = None,
    ) -> OEResult:
        """Retrieve the state for measurement ``y`` with prior mean ``xa`` and covariance ``sa``."""
        fm = self.fm
        y = np.asarray(y, dtype=float)
        use = getattr(fm, "measurement_mask", np.ones(y.size, bool))
        if noise_sigma is None:
            noise_sigma = fm.rtm.instrument.noise_sigma(y)
        noise_sigma = np.sqrt(noise_sigma**2 + (self.model_error * y) ** 2)
        se_inv = np.where(use, 1.0 / np.square(noise_sigma), 0.0)
        # Iterate on the prior-whitened state z = L^-1 (x - xa), Sa = L L^T, which
        # keeps the normal equations well conditioned despite state elements
        # that differ by orders of magnitude.
        sl = np.linalg.cholesky(sa)
        z = np.zeros_like(xa) if x0 is None else np.linalg.solve(sl, x0 - xa)
        eye = np.eye(z.size)
        lam = 1e-2

        def cost(zz):
            r = y - fm.radiance(xa + sl @ zz)
            return r @ (se_inv * r) + zz @ zz

        c = cost(z)
        converged = False
        it = 0
        for it in range(1, self.max_iter + 1):  # noqa: B007 (reported in the result)
            fx, k = fm.jacobian(xa + sl @ z)
            kz = k @ sl
            kt_se = kz.T * se_inv
            h = kt_se @ kz + eye
            g = kt_se @ (y - fx) - z
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
                converged = True
                break
            c = c_new
            lam = max(lam / 10, 1e-6)
        x = xa + sl @ z
        fx, k = fm.jacobian(x)
        kz = k @ sl
        s_hat = sl @ np.linalg.inv((kz.T * se_inv) @ kz + eye) @ sl.T
        a = s_hat @ (k.T * se_inv) @ k
        r = y - fx
        chi2 = float(r @ (se_inv * r)) / max(int(use.sum()) - x.size, 1)
        return OEResult(x, s_hat, a, chi2, it, converged, list(fm.state_names))
