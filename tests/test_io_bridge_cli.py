import numpy as np
import pytest

from plumefit import standard_atmosphere
from plumefit.cli import main
from plumefit.io import read_envi, write_envi
from plumefit.isofit_bridge import plume_safe_prior, read_isofit_state, scale_atmosphere_h2o


@pytest.mark.parametrize("interleave", ["bil", "bip", "bsq"])
def test_envi_roundtrip(tmp_path, interleave, rng):
    a = rng.normal(size=(7, 5, 3)).astype(np.float32)
    write_envi(tmp_path / "x", a, np.array([2100.0, 2200.0, 2300.0]), interleave=interleave)
    b, hdr = read_envi(tmp_path / "x")
    np.testing.assert_allclose(a, b)
    assert hdr["wavelength"] == [2100.0, 2200.0, 2300.0]


def test_read_isofit_state_band_names(tmp_path, rng):
    cube = rng.uniform(size=(4, 4, 4))
    write_envi(tmp_path / "state", cube, band_names=["2100.0", "2200.0", "AOT550", "H2OSTR"])
    st = read_isofit_state(tmp_path / "state")
    assert set(st) == {"AOT550", "H2OSTR"}
    np.testing.assert_allclose(st["H2OSTR"], cube[..., 3], rtol=1e-6)


def test_scale_atmosphere_h2o():
    atm = scale_atmosphere_h2o(standard_atmosphere(), 2.5)
    grams = atm.gas_column("H2O").sum() / 6.02214076e23 * 18.015
    assert grams == pytest.approx(2.5, rel=1e-6)


def test_plume_safe_prior_replaces_masked_pixels():
    rfl = np.ones((10, 10, 3))
    rfl[4:6, 4:6] = 0.5  # plume imprint in ISOFIT reflectance
    mask = np.zeros((10, 10), bool)
    mask[4:6, 4:6] = True
    out = plume_safe_prior(rfl, mask, dilate_px=1)
    np.testing.assert_allclose(out, 1.0)


def test_cli_build_db_and_design(tmp_path, capsys):
    db = tmp_path / "db.npz"
    main(["build-db", "--out", str(db), "--wl", "2200", "2400", "--dnu", "0.1"])
    main(["design", "--db", str(db), "--fwhm-grid", "4", "8", "--snr-grid", "200", "--wl", "2220", "2380"])
    out = capsys.readouterr().out
    assert "precision" in out
