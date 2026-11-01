import numpy as np
import pytest

from hyperatmos import AtmosphereSurfaceModel, AtmosphericCorrection, Geometry, uniform_instrument
from hyperatmos.design import retrieval_precision
from hyperatmos.forward import state_prior, surface_guess
from hyperatmos.retrieval import OptimalEstimation, matched_filter
from hyperatmos.simulate import simulate_scene


@pytest.fixture(scope="module")
def scene(rtm):
    return simulate_scene(rtm, (40, 40), aod_mean=0.15, seed=4)


def test_water_vapour_closed_loop(rtm, scene):
    h2o = AtmosphericCorrection(rtm).estimate_h2o(scene.radiance, 0.15)
    err = h2o - scene.h2o_g_cm2
    assert abs(err.mean()) < 0.05
    assert err.std() < 0.08


def test_reflectance_closed_loop_known_aerosol(rtm, scene):
    res = AtmosphericCorrection(rtm).correct(scene.radiance, aod550=0.15)
    e = (res.reflectance - scene.reflectance)[..., ~res.bad_bands]
    rmse = np.sqrt(np.mean(e**2, axis=(0, 1)))
    assert rmse.max() < 0.01
    assert np.all(np.isnan(res.reflectance[..., res.bad_bands]))


def test_ddv_aerosol_estimate(rtm, scene):
    aod = AtmosphericCorrection(rtm).estimate_aod(scene.radiance)
    assert aod == pytest.approx(0.15, abs=0.2)  # mixed pixels bias DDV high


def test_oe_recovers_atmosphere_over_known_surface(rtm, scene):
    fm = AtmosphereSurfaceModel(rtm)
    rho = scene.reflectance[5, 5]
    y = rtm.radiance(rho, 1.8, 0.2)
    tight = np.eye(rho.size) * 1e-8
    xa, sa = state_prior(fm, rho, h2o=(1.5, 1.0), aod=(0.1, 0.3), surface_cov=tight)
    res = OptimalEstimation(fm).retrieve(y, xa, sa, noise_sigma=rtm.instrument.noise_sigma(y) * 1e-3)
    assert res.converged
    assert res.value("H2O") == pytest.approx(1.8, abs=0.02)
    assert res.value("AOD550") == pytest.approx(0.2, abs=0.02)


def test_oe_uncertainty_covers_error(rtm, scene):
    fm = AtmosphereSurfaceModel(rtm)
    oe = OptimalEstimation(fm)
    z = []
    for r, c in [(3, 3), (10, 30), (25, 12), (35, 35)]:
        y = scene.radiance[r, c]
        xa, sa = state_prior(fm, surface_guess(fm, y, 1.5, 0.1), h2o=(1.5, 0.5))
        res = oe.retrieve(y, xa, sa)
        z.append((res.value("H2O") - scene.h2o_g_cm2[r, c]) / res.sigma("H2O"))
    assert np.all(np.abs(z) < 3)


def test_design_precision(db):
    inst = uniform_instrument(850, 1250, 4, 5, snr_ref=300)
    s = retrieval_precision(db, inst, Geometry(35, 0))
    assert 0 < s.sigma["H2O"] < 0.5
    assert s.dofs["H2O"] > 0.9
    better = retrieval_precision(db, uniform_instrument(850, 1250, 4, 5, snr_ref=900), Geometry(35, 0))
    assert better.sigma["H2O"] < s.sigma["H2O"]


def test_matched_filter_finds_water_vapour_anomaly(rtm, rng):
    shape = (60, 30)
    rho = simulate_scene(rtm, shape, h2o_rel_std=0.0, noise=False, seed=2).reflectance
    h2o = np.full(shape, 1.5)
    h2o[25:35, 10:20] = 1.8  # +20% local anomaly
    rad = rtm.radiance(rho, h2o, 0.1)
    rad += rng.normal(size=rad.shape) * rtm.instrument.noise_sigma(rad)
    win = (rtm.wavelength_nm > 850) & (rtm.wavelength_nm < 1250)
    k = rtm.absorption_signature("H2O")[win]
    res = matched_filter(rad[..., win], k, column_group=5)
    assert np.nanmean(res.enhancement[25:35, 10:20]) > 0.1
    assert abs(np.nanmean(res.enhancement[:20])) < 0.05
