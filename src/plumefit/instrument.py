"""Instrument models: spectral response, noise, and sensor presets.

Band centres/FWHM can be read from an ISOFIT wavelength file, so that the
same instrument description drives ISOFIT and plumefit.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Instrument:
    """An imaging spectrometer.

    Parameters
    ----------
    wavelength_nm, fwhm_nm : band centres and Gaussian SRF widths.
    snr_ref : signal-to-noise ratio at ``radiance_ref``.
    radiance_ref : reference radiance [uW cm^-2 sr^-1 nm^-1] at which SNR is quoted.
    read_noise_fraction : share of the noise variance at ``radiance_ref`` that is
        signal independent (read/dark noise); the rest scales as shot noise.
    """

    wavelength_nm: np.ndarray
    fwhm_nm: np.ndarray
    snr_ref: float = 400.0
    radiance_ref: float = 0.5
    read_noise_fraction: float = 0.3
    name: str = "custom"

    def __post_init__(self):
        self.wavelength_nm = np.asarray(self.wavelength_nm, dtype=float)
        self.fwhm_nm = np.broadcast_to(np.asarray(self.fwhm_nm, dtype=float), self.wavelength_nm.shape).copy()

    @property
    def n_bands(self) -> int:
        return self.wavelength_nm.size

    def subset(self, wl_min: float, wl_max: float) -> Instrument:
        m = (self.wavelength_nm >= wl_min) & (self.wavelength_nm <= wl_max)
        return Instrument(
            self.wavelength_nm[m], self.fwhm_nm[m], self.snr_ref, self.radiance_ref, self.read_noise_fraction, self.name
        )

    def srf_matrix(self, nu_hires: np.ndarray) -> np.ndarray:
        """Row-normalised Gaussian SRF matrix (n_bands, n_hires) on a wavenumber grid.

        Integration weights account for the non-uniform wavelength spacing of
        a uniform wavenumber grid, so the matrix averages spectral radiance
        per unit wavelength.
        """
        wl = 1e7 / nu_hires
        dwl = np.abs(np.gradient(wl))
        sigma = self.fwhm_nm[:, None] / (2 * np.sqrt(2 * np.log(2)))
        w = np.exp(-0.5 * ((wl[None, :] - self.wavelength_nm[:, None]) / sigma) ** 2) * dwl[None, :]
        w[w < 1e-8 * w.max(axis=1, keepdims=True)] = 0.0
        return w / w.sum(axis=1, keepdims=True)

    def noise_sigma(self, radiance: np.ndarray) -> np.ndarray:
        """1-sigma noise for a radiance spectrum (shot + read noise)."""
        var_ref = (self.radiance_ref / self.snr_ref) ** 2
        a = (1 - self.read_noise_fraction) * var_ref / self.radiance_ref
        b = self.read_noise_fraction * var_ref
        return np.sqrt(a * np.clip(radiance, 0, None) + b)

    @classmethod
    def from_isofit_wavelength_file(cls, path: str | Path, **kwargs) -> Instrument:
        """Read an ISOFIT wavelength file (index, centre, FWHM; microns or nm)."""
        d = np.loadtxt(path)
        wl, fw = d[:, 1], d[:, 2]
        if wl.max() < 100:  # ISOFIT convention: microns
            wl, fw = wl * 1e3, fw * 1e3
        return cls(wl, fw, **kwargs)

    def to_isofit_wavelength_file(self, path: str | Path) -> None:
        """Write an ISOFIT-compatible wavelength file (microns)."""
        idx = np.arange(self.n_bands)
        np.savetxt(
            path, np.column_stack([idx, self.wavelength_nm / 1e3, self.fwhm_nm / 1e3]), fmt=["%d", "%.6f", "%.6f"]
        )


def uniform_instrument(
    wl_min: float, wl_max: float, sampling: float, fwhm: float | None = None, **kwargs
) -> Instrument:
    """Evenly sampled instrument; FWHM defaults to the sampling interval."""
    wl = np.arange(wl_min, wl_max + 1e-9, sampling)
    return Instrument(wl, fwhm or sampling, **kwargs)


#: Approximate nominal characteristics (SWIR) for orientation only. Check the
#: mission documentation, or load the real wavelength file, before using one
#: of these for anything quantitative.
PRESETS = {
    "EMIT": dict(wl_min=380, wl_max=2500, sampling=7.4, fwhm=8.5, snr_ref=300),
    "AVIRIS-NG": dict(wl_min=380, wl_max=2510, sampling=5.0, fwhm=5.5, snr_ref=400),
    "PRISMA": dict(wl_min=920, wl_max=2500, sampling=10.0, fwhm=10.0, snr_ref=150),
    "EnMAP": dict(wl_min=900, wl_max=2450, sampling=10.0, fwhm=10.0, snr_ref=180),
    "drone-swir": dict(wl_min=1600, wl_max=2450, sampling=3.0, fwhm=4.0, snr_ref=250),
}


def preset(name: str) -> Instrument:
    """Instrument from :data:`PRESETS` (approximate values, see note there)."""
    p = dict(PRESETS[name])
    inst = uniform_instrument(
        p.pop("wl_min"), p.pop("wl_max"), p.pop("sampling"), p.pop("fwhm"), snr_ref=p.pop("snr_ref"), name=name
    )
    return inst
