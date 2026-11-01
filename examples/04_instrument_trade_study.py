"""Instrument trade study: what spectral resolution and SNR does a camera need?

Uses the linearised optimal-estimation posterior to map the precision of
water vapour (0.85-1.25 um window) and of a trace-gas column (CH4 scale
factor, 2.1-2.45 um window) against FWHM and SNR, for a satellite and for a
drone at 120 m. SNR is held fixed per band; in a real camera a narrower FWHM
lowers SNR, so compare designs along lines of equal throughput.
"""

import matplotlib.pyplot as plt
import numpy as np
from _common import DEMO_NOTE, FIG_DIR, demo_database

from hyperatmos import Geometry
from hyperatmos.design import trade_study

db = demo_database()
fwhm = np.array([2.0, 3.0, 5.0, 8.0, 12.0])
snr = np.array([100, 200, 400, 800])
platforms = {"Satellite": Geometry(40, 0, 0, None), "Drone, 120 m": Geometry(40, 0, 0, 120.0)}
studies = {
    "H2O [g cm$^{-2}$]": dict(quantity="H2O", wl_range=(850.0, 1250.0)),
    "CH4 column scale [%]": dict(quantity="scale_CH4", wl_range=(2100.0, 2450.0), gases=("CH4",)),
}

fig, ax = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
for i, (label, kw) in enumerate(studies.items()):
    grids = {}
    for name, geom in platforms.items():
        g = trade_study(db, geom, fwhm, snr, **kw)
        grids[name] = g * (100 if "CH4" in label else 1)
    vmax = max(v.max() for v in grids.values())
    for j, (name, g) in enumerate(grids.items()):
        a = ax[i, j]
        im = a.pcolormesh(snr, fwhm, g, shading="nearest", cmap="viridis_r", vmin=0, vmax=vmax)
        for (yy, xx), v in np.ndenumerate(g):
            a.text(snr[xx], fwhm[yy], f"{v:.2g}", ha="center", va="center", fontsize=7, color="w")
        a.set_xscale("log")
        a.set_xticks(snr, [str(s) for s in snr])
        a.xaxis.set_minor_formatter(plt.NullFormatter())
        a.set_xlabel("SNR at the median scene radiance")
        a.set_ylabel("FWHM [nm]")
        a.set_title(f"{name}: 1-sigma {label}", fontsize=10)
        print(f"{name:14s} {label:22s} FWHM 5 nm, SNR 400: {g[2, 2]:.3g}")
    fig.colorbar(im, ax=ax[i, :], shrink=0.85)
fig.suptitle(f"Single-pixel retrieval precision, SZA 40 deg, reflectance 0.25 ({DEMO_NOTE})", fontsize=10)
fig.savefig(FIG_DIR / "instrument_trade_study.png", dpi=110)
print("saved", FIG_DIR / "instrument_trade_study.png")
