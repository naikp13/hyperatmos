"""Synthetic scenes with known surface and atmosphere, for closed-loop testing.

Surface reflectance is a spatially correlated mixture of illustrative
endmembers (vegetation, soil, water, concrete, carbonate rock); water vapour
and aerosol vary smoothly across the scene. Because the truth is known,
atmospheric correction and retrieval errors can be measured exactly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from .rtm import RadiativeTransferModel


def _g(wl, c, w):
    return np.exp(-0.5 * ((wl - c) / w) ** 2)


def _sig(wl, c, w):
    return 1 / (1 + np.exp(-(wl - c) / w))


def endmember_spectra(wavelength_nm: np.ndarray) -> dict[str, np.ndarray]:
    """Smooth, illustrative VNIR-SWIR reflectance spectra (not from a spectral library).

    They carry the features that matter for atmospheric correction: the
    vegetation red edge and leaf-water bands at 0.98 and 1.2 um (which overlap
    the H2O retrieval windows), soil bound-water and clay features, dark water,
    and a carbonate feature at 2.34 um.
    """
    wl = np.asarray(wavelength_nm, dtype=float)
    veg = (
        0.02
        + 0.045 * _g(wl, 550, 35)
        + 0.025 * _sig(wl, 620, 20) * (1 - _sig(wl, 700, 10))
        + 0.44 * _sig(wl, 715, 12)
        - 0.03 * _g(wl, 980, 25)
        - 0.06 * _g(wl, 1200, 40)
        - 0.28 * _sig(wl, 1400, 30)
        + 0.08 * _g(wl, 1680, 90)
        - 0.12 * _sig(wl, 1880, 30)
        + 0.06 * _g(wl, 2220, 90)
    )
    soil = (
        0.08 + 0.30 * _sig(wl, 700, 180) - 0.05 * _g(wl, 1420, 40) - 0.07 * _g(wl, 1930, 50) - 0.05 * _g(wl, 2205, 15)
    )
    water = 0.07 * np.exp(-(((wl - 420) / 220) ** 2)) + 0.01 * np.exp(-(wl - 400) / 300) + 0.003
    concrete = 0.20 + 0.10 * _sig(wl, 600, 120) - 0.03 * _g(wl, 1930, 60)
    carbonate = 0.30 + 0.15 * _sig(wl, 550, 120) - 0.06 * _g(wl, 2340, 20) - 0.03 * _g(wl, 1930, 60)
    return {"vegetation": veg, "soil": soil, "water": water, "concrete": concrete, "carbonate": carbonate}


def smooth_field(
    shape: tuple[int, int], mean: float, rel_std: float, correlation_px: float, rng: np.random.Generator
) -> np.ndarray:
    """Positive, spatially correlated random field."""
    f = ndimage.gaussian_filter(rng.normal(size=shape), correlation_px)
    f = (f - f.mean()) / (f.std() + 1e-12)
    return mean * np.exp(rel_std * f - 0.5 * rel_std**2)


def random_surface(
    shape: tuple[int, int], wavelength_nm: np.ndarray, correlation_px: float = 6.0, seed: int = 0
) -> np.ndarray:
    """Reflectance cube (rows, cols, bands) from spatially correlated endmember mixtures."""
    rng = np.random.default_rng(seed)
    em = endmember_spectra(wavelength_nm)
    fields = np.stack([ndimage.gaussian_filter(rng.normal(size=shape), correlation_px) for _ in em])
    fields /= fields.std(axis=(1, 2), keepdims=True)
    w = np.exp(2.5 * fields)
    w /= w.sum(axis=0, keepdims=True)
    spectra = np.stack(list(em.values()))
    brightness = 1.0 + 0.08 * ndimage.gaussian_filter(rng.normal(size=shape), 1.0)
    return np.einsum("ers,eb->rsb", w, spectra) * brightness[..., None]


@dataclass
class SyntheticScene:
    radiance: np.ndarray  # (rows, cols, bands)
    reflectance: np.ndarray
    h2o_g_cm2: np.ndarray  # (rows, cols)
    aod550: np.ndarray  # (rows, cols)
    noise_sigma: np.ndarray
    wavelength_nm: np.ndarray


def simulate_scene(
    rtm: RadiativeTransferModel,
    shape: tuple[int, int] = (60, 60),
    h2o_mean: float = 1.5,
    h2o_rel_std: float = 0.15,
    aod_mean: float = 0.12,
    aod_rel_std: float = 0.0,
    noise: bool = True,
    seed: int = 0,
    reflectance: np.ndarray | None = None,
) -> SyntheticScene:
    """Simulate at-sensor radiance for a random scene."""
    rng = np.random.default_rng(seed + 1)
    wl = rtm.wavelength_nm
    if reflectance is None:
        reflectance = random_surface(shape, wl, seed=seed)
    h2o = smooth_field(shape, h2o_mean, h2o_rel_std, 12.0, rng)
    aod = smooth_field(shape, aod_mean, aod_rel_std, 20.0, rng) if aod_rel_std > 0 else np.full(shape, aod_mean)
    rad = rtm.radiance(reflectance, h2o, aod if aod_rel_std > 0 else aod_mean)
    sigma = rtm.instrument.noise_sigma(rad)
    if noise:
        rad = rad + rng.normal(size=rad.shape) * sigma
    return SyntheticScene(rad, reflectance, h2o, aod, sigma, wl)
