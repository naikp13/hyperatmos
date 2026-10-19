"""Synthetic scenes with known plumes, for closed-loop testing and instrument trade-offs.

A steady-state Gaussian plume provides the true column field, a mixture of
surface endmembers provides spatially varying reflectance (including a
2.2 um clay feature, a known source of matched-filter false positives), and
the forward model turns both into noisy radiance. Because the truth is known,
retrieval bias, detection limits and emission-rate errors can be measured.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from .constants import kg_m2_to_ppmm
from .forward import ForwardModel

#: Pasquill-Gifford horizontal dispersion, sigma_y = a * x_km**0.894 [m].
PG_SIGMA_Y = {"A": 213.0, "B": 156.0, "C": 104.0, "D": 68.0, "E": 50.5, "F": 34.0}


def gaussian_plume_column(
    shape: tuple[int, int],
    pixel_size_m: float,
    source_rc: tuple[float, float],
    q_kg_h: float,
    wind_speed: float,
    wind_dir_deg: float,
    stability: str = "D",
    gas: str = "CH4",
) -> np.ndarray:
    """Vertically integrated enhancement [ppm·m] of a steady ground-level source.

    ``wind_dir_deg``: direction the plume travels towards, clockwise from
    image "up" (decreasing row index).
    """
    rr, cc = np.indices(shape, dtype=float)
    dr, dc = (rr - source_rc[0]) * pixel_size_m, (cc - source_rc[1]) * pixel_size_m
    th = np.deg2rad(wind_dir_deg)
    x = -dr * np.cos(th) + dc * np.sin(th)  # downwind distance
    y = dr * np.sin(th) + dc * np.cos(th)  # crosswind distance
    x_eff = np.maximum(x, 0.5 * pixel_size_m)
    sy = PG_SIGMA_Y[stability] * (x_eff / 1000.0) ** 0.894
    q = q_kg_h / 3600.0
    col = q / (np.sqrt(2 * np.pi) * sy * wind_speed) * np.exp(-0.5 * (y / sy) ** 2)
    col = np.where(x > -0.5 * pixel_size_m, col, 0.0)
    return kg_m2_to_ppmm(col, gas)


def endmember_spectra(wavelength_nm: np.ndarray) -> dict[str, np.ndarray]:
    """Smooth, illustrative SWIR reflectance spectra (not from a library)."""
    x = (wavelength_nm - 2100.0) / 350.0
    clay = 0.08 * np.exp(-0.5 * ((wavelength_nm - 2205.0) / 12.0) ** 2)
    carbonate = 0.06 * np.exp(-0.5 * ((wavelength_nm - 2335.0) / 18.0) ** 2)
    return {
        "soil": 0.30 + 0.04 * x - clay,
        "vegetation": 0.20 - 0.08 * x + 0.03 * np.sin(6 * x),
        "asphalt": 0.08 + 0.01 * x,
        "limestone": 0.42 - 0.02 * x - carbonate,
    }


def random_surface(
    shape: tuple[int, int], wavelength_nm: np.ndarray, correlation_px: float = 6.0, seed: int = 0
) -> np.ndarray:
    """Reflectance cube (rows, cols, bands) from spatially correlated endmember mixtures."""
    rng = np.random.default_rng(seed)
    em = endmember_spectra(wavelength_nm)
    fields = np.stack([ndimage.gaussian_filter(rng.normal(size=shape), correlation_px) for _ in em])
    fields /= fields.std(axis=(1, 2), keepdims=True)
    w = np.exp(2.0 * fields)
    w /= w.sum(axis=0, keepdims=True)
    spectra = np.stack(list(em.values()))  # (n_em, bands)
    brightness = 1.0 + 0.1 * ndimage.gaussian_filter(rng.normal(size=shape), 1.0)
    return np.einsum("ers,eb->rsb", w, spectra) * brightness[..., None]


@dataclass
class SyntheticScene:
    radiance: np.ndarray  # (rows, cols, bands)
    reflectance: np.ndarray
    true_enhancement: np.ndarray  # ppm·m
    noise_sigma: np.ndarray
    wavelength_nm: np.ndarray
    pixel_size_m: float
    source_rc: tuple[float, float]
    q_kg_h: float
    wind_speed: float
    wind_dir_deg: float


def simulate_scene(
    fm: ForwardModel,
    shape: tuple[int, int] = (80, 80),
    pixel_size_m: float = 30.0,
    q_kg_h: float = 1000.0,
    wind_speed: float = 4.0,
    wind_dir_deg: float = 90.0,
    source_rc: tuple[float, float] | None = None,
    gas: str = "CH4",
    stability: str = "D",
    noise: bool = True,
    seed: int = 0,
    reflectance: np.ndarray | None = None,
) -> SyntheticScene:
    """Simulate radiance for a scene with one point source of ``gas``."""
    rng = np.random.default_rng(seed + 1)
    wl = fm.wavelength_nm
    if source_rc is None:
        source_rc = (shape[0] // 2, shape[1] // 4)
    truth = (
        gaussian_plume_column(shape, pixel_size_m, source_rc, q_kg_h, wind_speed, wind_dir_deg, stability, gas)
        if q_kg_h > 0
        else np.zeros(shape)
    )
    if reflectance is None:
        reflectance = random_surface(shape, wl, seed=seed)
    # Interpolate the convolved irradiance in log space over a grid of enhancements.
    gi = fm.gases.index(gas)
    grid = np.concatenate([[0.0], np.geomspace(1.0, max(truth.max(), 10.0) * 1.01, 96)])
    lut = []
    for v in grid:
        dx = np.zeros(fm.n_gas)
        dx[gi] = v
        lut.append(np.log(fm.transmitted_irradiance(dx)[0]))
    lut = np.array(lut)
    flat = truth.ravel()
    idx = np.clip(np.searchsorted(grid, flat) - 1, 0, len(grid) - 2)
    f = ((flat - grid[idx]) / (grid[idx + 1] - grid[idx]))[:, None]
    conv = np.exp(lut[idx] * (1 - f) + lut[idx + 1] * f).reshape(*shape, -1)
    rad = fm.geometry.mu0 / np.pi * reflectance * conv
    sigma = fm.instrument.noise_sigma(rad)
    if noise:
        rad = rad + rng.normal(size=rad.shape) * sigma
    return SyntheticScene(rad, reflectance, truth, sigma, wl, pixel_size_m, source_rc, q_kg_h, wind_speed, wind_dir_deg)
