"""Closed-loop atmospheric correction for a satellite and a drone imager.

Simulates radiance for a heterogeneous scene with spatially varying water
vapour, corrects it with dark-vegetation aerosol and per-pixel water vapour
retrievals, and compares the result with the known truth. The drone case
uses a VNIR camera at 120 m, where aerosol cannot be estimated from the
image and is supplied (e.g. from a sun photometer).
"""

import matplotlib.pyplot as plt
import numpy as np
from _common import DEMO_NOTE, FIG_DIR, demo_database

from hyperatmos import AtmosphericCorrection, Geometry, RadiativeTransferModel, preset
from hyperatmos.simulate import simulate_scene

db = demo_database()
AOD = 0.15
cases = {
    "EMIT-like satellite": (RadiativeTransferModel(db, preset("EMIT"), Geometry(35, 5, 30, None)), None),
    "VNIR drone camera, 120 m": (RadiativeTransferModel(db, preset("drone-vnir"), Geometry(35, 5, 30, 120.0)), AOD),
}

fig, ax = plt.subplots(2, 4, figsize=(18, 8), constrained_layout=True)
for row, (name, (rtm, aod_in)) in enumerate(cases.items()):
    scene = simulate_scene(rtm, (80, 80), h2o_mean=1.6, h2o_rel_std=0.15, aod_mean=AOD, seed=3)
    res = AtmosphericCorrection(rtm).correct(scene.radiance, aod550=aod_in)
    good = ~res.bad_bands
    err = (res.reflectance - scene.reflectance)[..., good]
    rmse = np.sqrt(np.mean(err**2, axis=(0, 1)))
    bias = np.mean(err, axis=(0, 1))
    dh = res.h2o_g_cm2 - scene.h2o_g_cm2
    print(
        f"{name}: AOD true {AOD:.2f} used {res.aod550:.3f} | H2O bias {dh.mean():+.3f}, "
        f"std {dh.std():.3f} g/cm2 | reflectance RMSE median {np.median(rmse):.4f}, max {rmse.max():.4f}"
    )

    vmin, vmax = np.percentile(scene.h2o_g_cm2, [1, 99])
    for a, img, t in ((ax[row, 0], scene.h2o_g_cm2, "true H2O"), (ax[row, 1], res.h2o_g_cm2, "retrieved H2O")):
        im = a.imshow(img, vmin=vmin, vmax=vmax, cmap="Blues")
        a.set_title(f"{name}\n{t}", fontsize=9)
        a.set_xticks([])
        a.set_yticks([])
    fig.colorbar(im, ax=ax[row, :2], label="g cm$^{-2}$", shrink=0.8)
    wl = res.wavelength_nm[good]
    ax[row, 2].plot(wl, rmse, ".", ms=3, label="RMSE")
    ax[row, 2].plot(wl, bias, ".", ms=3, label="bias")
    ax[row, 2].axhline(0, color="k", lw=0.5)
    ax[row, 2].set_title(f"Reflectance error vs truth (AOD used {res.aod550:.2f})", fontsize=9)
    ax[row, 2].set_xlabel("Wavelength [nm]")
    ax[row, 2].legend(fontsize=8)
    r, c = 20, 50
    ax[row, 3].plot(res.wavelength_nm, scene.reflectance[r, c], "k", lw=1.5, label="truth")
    ax[row, 3].plot(res.wavelength_nm, res.reflectance[r, c], "C1", lw=1, label="retrieved")
    ax[row, 3].plot(
        rtm.wavelength_nm, rtm.to_reflectance_factor(scene.radiance[r, c]), "C7", lw=0.8, label="top-of-atmosphere"
    )
    ax[row, 3].set_title("One pixel", fontsize=9)
    ax[row, 3].set_xlabel("Wavelength [nm]")
    ax[row, 3].legend(fontsize=8)
fig.suptitle(f"Closed-loop atmospheric correction ({DEMO_NOTE})", fontsize=10)
fig.savefig(FIG_DIR / "atmospheric_correction.png", dpi=110)
print("saved", FIG_DIR / "atmospheric_correction.png")
