import numpy as np
import pytest

from plumefit import constants as c
from plumefit.spectroscopy import LineList, SpectralDatabase, cross_section, read_hitran_par


def test_ppmm_roundtrip():
    assert c.molec_cm2_to_ppmm(c.ppmm_to_molec_cm2(1234.5)) == pytest.approx(1234.5)
    assert c.kg_m2_to_ppmm(c.ppmm_to_kg_m2(1000.0, "CH4"), "CH4") == pytest.approx(1000.0)


def test_ppmm_to_mass_reference_value():
    # 1000 ppm m CH4 at 1 atm, 288.15 K: 2.547e18 molec/cm^2 -> 6.78e-4 kg/m^2
    assert c.ppmm_to_molec_cm2(1000.0) == pytest.approx(2.547e18, rel=1e-3)
    assert c.ppmm_to_kg_m2(1000.0, "CH4") == pytest.approx(6.785e-4, rel=2e-3)


def _one_line(nu0=4300.0, s=1e-21):
    one = np.array
    return LineList("CH4", one([nu0]), one([s]), one([0.06]), one([0.08]), one([100.0]), one([0.75]), one([0.0]))


def test_voigt_line_integrates_to_intensity():
    nu = np.arange(4250.0, 4350.0, 0.002)
    sig = cross_section(_one_line(), nu, 101325.0, 296.0, wing_cutoff=49.0)
    # Lorentz wings beyond the cutoff hold ~0.08% of the area for this width.
    assert np.trapezoid(sig, nu) == pytest.approx(1e-21, rel=2e-3)
    assert nu[np.argmax(sig)] == pytest.approx(4300.0, abs=0.002)


def test_pressure_broadening_lowers_peak():
    nu = np.arange(4290.0, 4310.0, 0.001)
    high = cross_section(_one_line(), nu, 101325.0, 296.0)
    low = cross_section(_one_line(), nu, 10000.0, 296.0)
    assert low.max() > 3 * high.max()


def test_read_hitran_par(tmp_path):
    # HITRAN 160-char record for a CH4 line (values illustrative)
    fields = [
        " 6",
        "1",
        f"{4300.123456:12.6f}",
        f"{1.234e-21:10.3E}",
        f"{10.0:10.3E}",
        ".0600",
        "0.080",
        f"{100.0:10.4f}",
        "0.75",
        "-.005000",
    ]
    rec = "".join(fields)
    assert len(rec) == 67
    rec = rec + " " * 93 + "\n"
    f = tmp_path / "ch4.par"
    f.write_text(rec)
    ll = read_hitran_par(f, "CH4")
    assert len(ll) == 1
    assert ll.nu[0] == pytest.approx(4300.123456)
    assert ll.s296[0] == pytest.approx(1.234e-21)
    assert ll.gamma_air[0] == pytest.approx(0.06)
    assert ll.elower[0] == pytest.approx(100.0)
    assert ll.delta_air[0] == pytest.approx(-0.005)


def test_database_roundtrip(db, tmp_path):
    p = tmp_path / "db.npz"
    db.save(p)
    db2 = SpectralDatabase.load(p)
    assert db2.gases == db.gases and db2.synthetic
    np.testing.assert_allclose(db2.vertical_tau("CH4"), db.vertical_tau("CH4"), rtol=1e-5)


def test_partial_column_below_sensor(db):
    full = db.vertical_tau("CH4")
    below = db.vertical_tau("CH4", below_altitude_m=500.0)
    assert np.all(below <= full + 1e-12)
    # ~6% of a well-mixed column lies below 500 m
    assert below.sum() / full.sum() == pytest.approx(0.06, abs=0.015)
