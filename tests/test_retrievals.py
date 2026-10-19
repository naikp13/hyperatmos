import numpy as np
import pytest

from plumefit.instrument import uniform_instrument
from plumefit.retrieval import OptimalEstimation, convolve_cross_section, doas_fit, matched_filter
from plumefit.simulate import simulate_scene
from plumefit.spectroscopy import demo_uvvis_cross_section


@pytest.fixture(scope="module")
def scene(fm):
    return simulate_scene(fm, (60, 60), 60.0, q_kg_h=2000.0, wind_speed=3.0, wind_dir_deg=90.0, seed=3)


def test_matched_filter_closed_loop(fm, scene):
    k = fm.unit_absorption_spectrum("CH4")
    res = matched_filter(scene.radiance, k)
    truth = scene.true_enhancement
    bg = truth < 20
    assert abs(np.nanmean(res.enhancement[bg])) < 0.5 * np.nanstd(res.enhancement[bg])
    assert np.corrcoef(res.enhancement.ravel(), truth.ravel())[0, 1] > 0.85
    strong = truth > 500
    slope = np.polyfit(truth[strong], res.enhancement[strong], 1)[0]
    assert 0.5 < slope < 1.1  # linearised MF under-estimates strong plumes


def test_matched_filter_rejects_too_few_pixels(fm, scene):
    k = fm.unit_absorption_spectrum("CH4")
    with pytest.raises(ValueError):
        matched_filter(scene.radiance[:10], k)


def test_oe_noise_free_exact(fm):
    x_true = np.array([3000.0, 2e4, 0.3, 0.02, 0.0])
    res = OptimalEstimation(fm).retrieve(fm.radiance(x_true))
    assert res.converged
    assert res.enhancement("CH4") == pytest.approx(3000.0, abs=5.0)
    assert 1.5 < res.dofs <= 2.0


def test_oe_with_reflectance_prior_closed_loop(fm, scene):
    oe = OptimalEstimation(fm)
    rows, cols = np.nonzero(scene.true_enhancement > 1000)
    errs = []
    for r, c in list(zip(rows, cols, strict=True))[:12]:
        fm.set_reflectance_prior(scene.reflectance[r, c])
        res = oe.retrieve(scene.radiance[r, c])
        errs.append((res.enhancement("CH4") - scene.true_enhancement[r, c]) / res.uncertainty("CH4"))
    fm.set_reflectance_prior(None)
    errs = np.array(errs)
    assert np.abs(np.mean(errs)) < 1.5
    assert np.std(errs) < 2.0


def test_doas_recovers_slant_column():
    inst = uniform_instrument(400.0, 500.0, 0.5, 0.6)
    xs = demo_uvvis_cross_section("NO2")
    sig = convolve_cross_section(xs, inst)
    wl = inst.wavelength_nm
    i0 = 1000.0 * (1 + 0.2 * (wl - 450) / 50)
    scd = 4e16
    meas = i0 * np.exp(-sig * scd) * (0.95 + 0.0005 * (wl - 450))
    res = doas_fit(wl, meas, i0, {"NO2": sig}, window=(425.0, 490.0), fit_shift=False)
    assert res.scd["NO2"] == pytest.approx(scd, rel=0.01)


def test_joint_surface_retrieval_with_statistical_prior(db, fm, scene):
    from plumefit import ForwardModel, Geometry, preset
    from plumefit.isofit_bridge import surface_prior_from_reflectance

    plume = scene.true_enhancement > 300
    prior = surface_prior_from_reflectance(scene.reflectance, plume)
    fmb = ForwardModel(db, preset("EMIT"), Geometry(30.0, 0.0), surface="bands")
    oe = OptimalEstimation(fmb, surface_prior=prior)
    rows, cols = np.nonzero(scene.true_enhancement > 2000)
    z = []
    for r, c in list(zip(rows, cols, strict=True))[:10]:
        res = oe.retrieve(scene.radiance[r, c])
        z.append((res.enhancement("CH4") - scene.true_enhancement[r, c]) / res.uncertainty("CH4"))
    assert abs(np.mean(z)) < 1.0
    assert np.std(z) < 1.5


def test_bands_surface_requires_prior(db):
    from plumefit import ForwardModel, Geometry, preset

    fmb = ForwardModel(db, preset("EMIT"), Geometry(30.0, 0.0), surface="bands")
    with pytest.raises(ValueError):
        OptimalEstimation(fmb)
