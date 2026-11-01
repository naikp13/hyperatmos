import numpy as np
import pytest

from hyperatmos import Geometry, Instrument, RadiativeTransferModel, preset
from hyperatmos.rtm import rayleigh_optical_depth


def test_rayleigh_optical_depth_reference():
    # Hansen & Travis (1974): ~0.097 at 550 nm, ~0.0155 at 860 nm for 1013 hPa
    assert rayleigh_optical_depth(np.array(550.0)) == pytest.approx(0.097, abs=0.002)
    assert rayleigh_optical_depth(np.array(860.0)) == pytest.approx(0.0155, abs=0.001)


def test_srf_rows_normalised(db):
    srf = preset("EMIT").subset(500, 2400).srf_matrix(db.nu)
    np.testing.assert_allclose(np.asarray(srf.sum(axis=1)).ravel(), 1.0)


def test_isofit_wavelength_file_roundtrip(tmp_path):
    inst = preset("PRISMA")
    p = tmp_path / "wl.txt"
    inst.to_isofit_wavelength_file(p)
    back = Instrument.from_isofit_wavelength_file(p)
    np.testing.assert_allclose(back.wavelength_nm, inst.wavelength_nm, atol=1e-3)
    assert np.loadtxt(p)[:, 1].max() < 3  # microns, like ISOFIT


def test_noise_model_matches_snr():
    inst = preset("EMIT")
    assert inst.radiance_ref / inst.noise_sigma(np.array([inst.radiance_ref]))[0] == pytest.approx(inst.snr_ref)


def test_forward_inverse_roundtrip(rtm, rng):
    rho = rng.uniform(0.02, 0.6, rtm.wavelength_nm.size)
    rad = rtm.radiance(rho, 2.1, 0.3)
    np.testing.assert_allclose(rtm.surface_reflectance(rad, 2.1, 0.3), rho, rtol=1e-10)


def test_h2o_lookup_table_matches_line_by_line(rtm):
    for w in (0.4, 1.3, 3.7):
        np.testing.assert_allclose(rtm.h2o_transmittance(w), rtm.gas_transmittance(w), rtol=3e-3, atol=1e-4)


def test_water_vapour_bands_deepen_with_column(rtm):
    i = int(np.argmin(np.abs(rtm.wavelength_nm - 940.0)))
    t = [rtm.h2o_transmittance(w)[i] for w in (0.5, 1.5, 3.0)]
    assert t[0] > t[1] > t[2]
    j = int(np.argmin(np.abs(rtm.wavelength_nm - 1650.0)))
    assert rtm.h2o_transmittance(1.5)[j] > 0.97  # atmospheric window


def test_path_reflectance_grows_with_aod_and_falls_with_wavelength(rtm):
    p0 = rtm.scattering(0.0)["rho_path"]
    p1 = rtm.scattering(0.4)["rho_path"]
    assert np.all(p1 > p0)
    assert p0[0] > 5 * p0[np.argmin(np.abs(rtm.wavelength_nm - 900))]


def test_drone_sees_less_atmosphere(db):
    inst = preset("drone-vnir")
    sat = RadiativeTransferModel(db, inst, Geometry(35, 0, 0, None))
    drone = RadiativeTransferModel(db, inst, Geometry(35, 0, 0, 120.0))
    s, d = sat.coefficients(1.5, 0.2), drone.coefficients(1.5, 0.2)
    assert np.all(d.rho_path < 0.1 * s.rho_path)
    assert np.all(d.t_up >= s.t_up)
    assert np.all(d.t_gas >= s.t_gas - 1e-12)


def test_absorption_signature_is_negative_in_band(db):
    rtm = RadiativeTransferModel(db, preset("EMIT"), Geometry(30, 0))
    k = rtm.absorption_signature("CH4")
    wl = rtm.wavelength_nm
    assert np.all(k[(wl > 2290) & (wl < 2360)] < 0)
    assert np.all(np.abs(k[(wl > 1000) & (wl < 1100)]) < 1e-3)
