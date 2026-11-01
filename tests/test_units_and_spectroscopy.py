import numpy as np
import pytest

from hyperatmos import constants as c
from hyperatmos import standard_atmosphere
from hyperatmos.spectroscopy import LineList, SpectralDatabase, cross_section, read_hitran_par


def test_h2o_unit_roundtrip():
    assert c.molec_cm2_to_h2o_g_cm2(c.h2o_g_cm2_to_molec_cm2(1.7)) == pytest.approx(1.7)
    # 1 g/cm^2 of water vapour is 3.343e22 molecules/cm^2
    assert c.h2o_g_cm2_to_molec_cm2(1.0) == pytest.approx(3.343e22, rel=1e-3)


def test_standard_atmosphere_columns():
    atm = standard_atmosphere(h2o_g_cm2=2.0, o3_du=320.0)
    assert c.molec_cm2_to_h2o_g_cm2(atm.gas_column("H2O").sum()) == pytest.approx(2.0)
    assert atm.gas_column("O3").sum() / c.DOBSON_UNIT == pytest.approx(320.0)
    # O2 column of a 1 atm atmosphere is about 4.5e24 molecules/cm^2
    assert atm.gas_column("O2").sum() == pytest.approx(4.5e24, rel=0.03)


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
    np.testing.assert_allclose(db2.vertical_tau("H2O"), db.vertical_tau("H2O"), rtol=1e-5)
    assert db2.column_total["O2"] == pytest.approx(db.column_total["O2"])


def test_partial_column_below_sensor(db):
    # ~6% of a well-mixed column lies below 500 m; water vapour is more concentrated
    assert db.column_fraction_below("CO2", 500.0) == pytest.approx(0.06, abs=0.01)
    assert db.column_fraction_below("H2O", 500.0) > 0.15
    assert db.column_fraction_below("CO2", None) == 1.0
