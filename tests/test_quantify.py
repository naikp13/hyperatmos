import numpy as np
import pytest

from plumefit import quantify as qf
from plumefit.simulate import gaussian_plume_column


@pytest.fixture
def plume():
    q, u, px = 1500.0, 5.0, 20.0
    col = gaussian_plume_column((120, 200), px, (60, 10), q, u, 90.0, stability="D")
    return col, q, u, px


def test_cross_sectional_flux_recovers_rate(plume):
    col, q, u, px = plume
    est, transects = qf.cross_sectional_flux(col, (60, 10), 90.0, u, px, np.arange(400, 3000, 400), half_width_m=1100.0)
    assert len(transects) > 3
    assert est == pytest.approx(q, rel=0.05)


def test_ime_rectangular_mask(plume):
    col, q, u, px = plume
    mask = np.zeros(col.shape, bool)
    mask[:, 20:120] = True  # 100 px = 2 km segment starting 200 m downwind
    ime = qf.integrated_mass_enhancement(col, mask, px)
    # Steady plume: mass in a downwind segment of length L is Q * L / U.
    assert ime == pytest.approx(q / 3600.0 * 2000.0 / u, rel=0.05)


def test_plume_mask_finds_connected_plume(plume, rng):
    col, *_ = plume
    noisy = col + rng.normal(0, 50, col.shape)
    m = qf.plume_mask(noisy, 50.0, (60, 11))
    assert m.n_pixels > 50
    assert m.mask[60, 30]
    assert not m.mask[5, 150]


def test_calibrate_ueff_recovers_linear_law():
    u10 = np.array([2.0, 4.0, 6.0, 8.0])
    u_eff = 0.6 * u10 + 0.4
    length = np.full(4, 500.0)
    ime = np.full(4, 100.0)
    q = u_eff * ime / length * 3600
    a, b = qf.calibrate_ueff(u10, q, ime, length)
    assert (a, b) == pytest.approx((0.6, 0.4))
