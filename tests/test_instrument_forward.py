import numpy as np
import pytest

from plumefit import ForwardModel, Geometry, Instrument, preset


def test_srf_rows_normalised(db):
    srf = preset("EMIT").subset(2150, 2400).srf_matrix(db.nu)
    np.testing.assert_allclose(srf.sum(axis=1), 1.0)
    np.testing.assert_allclose(srf @ np.full(db.nu.size, 3.0), 3.0)


def test_isofit_wavelength_file_roundtrip(tmp_path):
    inst = preset("PRISMA")
    p = tmp_path / "wl.txt"
    inst.to_isofit_wavelength_file(p)
    back = Instrument.from_isofit_wavelength_file(p)
    np.testing.assert_allclose(back.wavelength_nm, inst.wavelength_nm, atol=1e-3)
    assert np.loadtxt(p)[:, 1].max() < 3  # written in microns like ISOFIT


def test_noise_model_matches_snr():
    inst = preset("EMIT")
    assert inst.radiance_ref / inst.noise_sigma(np.array([inst.radiance_ref]))[0] == pytest.approx(inst.snr_ref)


def test_jacobian_matches_finite_differences(fm):
    x = np.array([5000.0, 1e5, 0.3, 0.02, -0.01])
    y, k = fm.jacobian(x)
    for i, h in enumerate([1.0, 100.0, 1e-5, 1e-5, 1e-5]):
        xp = x.copy()
        xp[i] += h
        fd = (fm.radiance(xp) - y) / h
        np.testing.assert_allclose(fd, k[:, i], rtol=2e-3, atol=1e-6 * np.abs(k[:, i]).max())


def test_enhancement_reduces_radiance_in_band(fm):
    x0 = np.array([0.0, 0.0, 0.3, 0.0, 0.0])
    x1 = x0.copy()
    x1[0] = 2000.0
    ratio = fm.radiance(x1) / fm.radiance(x0)
    band = (fm.wavelength_nm > 2280) & (fm.wavelength_nm < 2360)
    assert np.all(ratio <= 1.0 + 1e-12)
    assert ratio[band].min() < 0.97


def test_drone_sees_less_background_absorption(db):
    inst = preset("drone-swir")
    sat = ForwardModel(db, inst, Geometry(30, 0, None))
    drone = ForwardModel(db, inst, Geometry(30, 0, 120.0))
    x = np.array([0.0, 0.0, 0.3, 0.0, 0.0])
    assert np.all(drone.radiance(x) >= sat.radiance(x))
    # Plume air-mass factor is the same: the plume is below both sensors.
    assert sat.geometry.plume_amf == drone.geometry.plume_amf


def test_unit_absorption_is_negative_in_band(fm):
    k = fm.unit_absorption_spectrum("CH4")
    band = (fm.wavelength_nm > 2280) & (fm.wavelength_nm < 2360)
    assert np.all(k[band] < 0)
    assert np.all(np.abs(k[fm.wavelength_nm < 2150]) < np.abs(k[band]).max() * 0.1)
