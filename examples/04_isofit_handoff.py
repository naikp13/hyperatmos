"""From ISOFIT products to an unbiased CH4 retrieval.

ISOFIT's atmospheric correction does not model the plume, so inside a strong
plume its reflectance has a dent at the CH4 bands. Using that reflectance
directly as the surface prior hides part of the plume. This example emulates
ISOFIT outputs (reflectance cube and H2OSTR state, in ISOFIT's ENVI layout),
reads them through ``plumefit.isofit_bridge``, and compares surface
treatments in the optimal-estimation retrieval:

1. no prior (quadratic reflectance only): surface features at 2.2-2.35 um
   (clays, carbonates) are mistaken for CH4;
2. the raw ISOFIT reflectance as a fixed surface: the plume is already in it,
   so nothing is retrieved;
3. the reflectance of the nearest plume-free pixel: fine over uniform
   surfaces, wrong where the surface changes on the scale of the plume;
4. a joint retrieval of CH4 and every band's reflectance, constrained by the
   mean and covariance of plume-free ISOFIT reflectance: unbiased. Its
   reported uncertainty is conservative, because one Gaussian fitted to the
   whole scene is broader than the local surface variability.

With real data, replace the emulation block with the ``*_rfl`` and
``*_state`` files written by ISOFIT's ``apply_oe`` for the same flight line.
"""

import tempfile
import warnings
from pathlib import Path

import numpy as np
from _common import demo_database

from plumefit import ForwardModel, Geometry, preset
from plumefit import quantify as qf
from plumefit.io import write_envi
from plumefit.isofit_bridge import (
    plume_safe_prior,
    read_isofit_reflectance,
    read_isofit_state,
    surface_prior_from_reflectance,
)
from plumefit.retrieval import OptimalEstimation, matched_filter
from plumefit.simulate import simulate_scene

warnings.filterwarnings("ignore", message="SpectralDatabase uses synthetic")

db = demo_database()
fm = ForwardModel(db, preset("EMIT"), Geometry(35.0, 5.0))
scene = simulate_scene(
    fm, (240, 60), 60.0, q_kg_h=3000.0, wind_speed=3.0, wind_dir_deg=90.0, source_rc=(120, 12), stability="C", seed=11
)

# --- Emulate ISOFIT outputs: a plume-unaware inversion of the radiance -------------------
conv0, _ = fm.transmitted_irradiance(np.zeros(fm.n_gas))
rfl_isofit = scene.radiance / (fm.geometry.mu0 / np.pi * conv0)
work = Path(tempfile.mkdtemp())
write_envi(work / "flight_rfl", rfl_isofit, fm.wavelength_nm)
state = np.concatenate(
    [rfl_isofit, np.full((*rfl_isofit.shape[:2], 1), 0.05), np.full((*rfl_isofit.shape[:2], 1), 1.4)], axis=-1
)
write_envi(work / "flight_state", state, band_names=[f"{w:.2f}" for w in fm.wavelength_nm] + ["AOT550", "H2OSTR"])
# -----------------------------------------------------------------------------------------

rfl, wl = read_isofit_reflectance(work / "flight_rfl")
h2o = read_isofit_state(work / "flight_state")["H2OSTR"]
print(f"ISOFIT H2OSTR: median {np.median(h2o):.2f} g/cm^2 (use it with scale_atmosphere_h2o)")

mf = matched_filter(scene.radiance, fm.unit_absorption_spectrum("CH4"))
truth = scene.true_enhancement
sigma = float(np.nanstd(mf.enhancement[truth < 20]))
mask = qf.plume_mask(mf.enhancement, sigma, (120, 12)).mask
safe = plume_safe_prior(rfl, mask, dilate_px=2)

fm_bands = ForwardModel(db, preset("EMIT"), Geometry(35.0, 5.0), surface="bands")
variants = {
    "no prior": (OptimalEstimation(fm), None),
    "raw ISOFIT rfl": (OptimalEstimation(fm), rfl),
    "nearest plume-free rfl": (OptimalEstimation(fm), safe),
    "joint, ISOFIT surface stats": (
        OptimalEstimation(fm_bands, surface_prior=surface_prior_from_reflectance(rfl, mask)),
        None,
    ),
}
print(f"Plume mask: {mask.sum()} pixels. Retrieved / true CH4 mass inside the mask:")
print(f"  {'matched filter':<34}{mf.enhancement[mask].sum() / truth[mask].sum():6.3f}")
for name, (oe, prior) in variants.items():
    oe.fm.set_reflectance_prior(None)
    enh, unc = oe.retrieve_cube(scene.radiance, mask=mask, reflectance_prior=prior)
    z = (enh[mask] - truth[mask]) / unc[mask]
    print(
        f"  OE, {name:<30}{np.nansum(enh[mask]) / truth[mask].sum():6.3f}"
        f"   (retrieved-true)/sigma: mean {np.nanmean(z):+6.2f}, std {np.nanstd(z):5.2f}"
    )
