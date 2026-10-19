"""Camera design trade study: CH4 single-pixel precision vs spectral resolution and SNR.

Compares a drone camera flying at 120 m with a satellite sensor viewing the
same scene. The drone sees less background absorption on the upward path,
which matters most in saturated bands at high spectral resolution.
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
from _common import DEMO_NOTE, FIG_DIR, demo_database

from plumefit import Geometry
from plumefit.design import single_pixel_precision, trade_study
from plumefit.instrument import uniform_instrument

warnings.filterwarnings("ignore", message="SpectralDatabase uses synthetic")

db = demo_database()
fwhm = np.array([1.5, 2.5, 4.0, 6.0, 8.0, 10.0, 12.0])
snr = np.array([75, 100, 150, 200, 300, 400, 600, 800])
cases = {"Drone, 120 m AGL": Geometry(40.0, 0.0, 120.0), "Satellite": Geometry(40.0, 0.0, None)}

fig, ax = plt.subplots(1, 3, figsize=(16, 4.4), constrained_layout=True)
grids = {}
for a, (name, geom) in zip(ax[:2], cases.items(), strict=True):
    g = trade_study(db, geom, fwhm, snr, wl_range=(2120.0, 2430.0))
    grids[name] = g
    im = a.pcolormesh(snr, fwhm, g, shading="nearest", cmap="viridis_r", vmin=0, vmax=600)
    cs = a.contour(snr, fwhm, g, levels=[50, 100, 200, 400], colors="w", linewidths=0.8)
    a.clabel(cs, fmt="%d")
    a.set_xscale("log")
    a.set_xlabel("SNR at reference radiance")
    a.set_ylabel("FWHM [nm]")
    a.set_title(name)
fig.colorbar(im, ax=ax[:2], label="1-sigma CH4 precision [ppm m]")

# Spectral window: how much does extending coverage help at fixed FWHM/SNR?
windows = [(2200, 2400), (2150, 2430), (2120, 2430)]
for name, geom in cases.items():
    vals = [
        single_pixel_precision(db, uniform_instrument(lo, hi, 3.2, 4.0, snr_ref=250), geom).sigma_ppmm
        for lo, hi in windows
    ]
    ax[2].plot([f"{lo}-{hi}" for lo, hi in windows], vals, "o-", label=name)
ax[2].set_ylabel("1-sigma CH4 precision [ppm m]")
ax[2].set_xlabel("Spectral window [nm] (FWHM 4 nm, SNR 250)")
ax[2].legend()
ax[2].set_title("Spectral coverage")
fig.suptitle(f"Camera trade study, SZA 40 deg, surface reflectance 0.25 ({DEMO_NOTE})", fontsize=10)
fig.savefig(FIG_DIR / "camera_trade_study.png", dpi=120)
for name, g in grids.items():
    print(name, "precision at FWHM 4 nm / SNR 200:", round(float(g[2, 3])), "ppm m")
print("saved", FIG_DIR / "camera_trade_study.png")
