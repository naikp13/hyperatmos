import numpy as np
import pytest

from hyperatmos import standard_atmosphere
from hyperatmos.cli import main
from hyperatmos.constants import molec_cm2_to_h2o_g_cm2
from hyperatmos.io import read_envi, write_envi
from hyperatmos.isofit_interop import (
    atmosphere_from_isofit,
    compare_reflectance,
    read_isofit_reflectance,
    read_isofit_state,
    write_isofit_outputs,
)


@pytest.mark.parametrize("interleave", ["bil", "bip", "bsq"])
def test_envi_roundtrip(tmp_path, interleave, rng):
    a = rng.normal(size=(7, 5, 3)).astype(np.float32)
    write_envi(tmp_path / "x", a, np.array([2100.0, 2200.0, 2300.0]), interleave=interleave)
    b, hdr = read_envi(tmp_path / "x")
    np.testing.assert_allclose(a, b)
    assert hdr["wavelength"] == [2100.0, 2200.0, 2300.0]


def test_isofit_outputs_roundtrip(tmp_path, rng):
    rfl = rng.uniform(size=(4, 5, 3))
    h2o = rng.uniform(1, 2, size=(4, 5))
    wl = np.array([500.0, 900.0, 1600.0])
    write_isofit_outputs(tmp_path / "scene", rfl, wl, h2o, 0.12)
    back, wl2 = read_isofit_reflectance(tmp_path / "scene_rfl")
    np.testing.assert_allclose(back, rfl, rtol=1e-6)
    np.testing.assert_allclose(wl2, wl)
    st = read_isofit_state(tmp_path / "scene_state")
    assert set(st) == {"AOT550", "H2OSTR"}
    np.testing.assert_allclose(st["H2OSTR"], h2o, rtol=1e-6)
    np.testing.assert_allclose(st["AOT550"], 0.12, rtol=1e-6)


def test_atmosphere_from_isofit():
    atm = atmosphere_from_isofit(standard_atmosphere(), 2.5)
    assert molec_cm2_to_h2o_g_cm2(atm.gas_column("H2O").sum()) == pytest.approx(2.5)


def test_compare_reflectance():
    a = np.ones((3, 3, 2))
    b = a.copy()
    b[..., 1] = 0.9
    b[0, 0, 0] = np.nan
    cmp = compare_reflectance(a, b, np.array([1.0, 2.0]))
    assert cmp.n_pixels == 8
    np.testing.assert_allclose(cmp.bias, [0.0, 0.1])


def test_cli_end_to_end(tmp_path, capsys):
    db = tmp_path / "db.npz"
    main(["build-db", "--out", str(db), "--wl", "800", "1300", "--dnu", "0.2", "--gases", "H2O", "O2"])
    common = ["--db", str(db), "--sza", "35", "--sensor", "drone-vnir"]
    main(["simulate", *common, "--out", str(tmp_path / "s"), "--rows", "20", "--cols", "20"])
    main(["correct", str(tmp_path / "s_rdn"), *common, "--aod", "0.12", "--out", str(tmp_path / "s")])
    st = read_isofit_state(tmp_path / "s_state")
    truth, _ = read_envi(tmp_path / "s_true_h2o")
    assert np.abs(st["H2OSTR"] - truth[..., 0]).mean() < 0.1
    main(["design", *common, "--fwhm-grid", "4", "--snr-grid", "300"])
    assert "posterior 1-sigma of H2O" in capsys.readouterr().out
