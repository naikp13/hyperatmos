"""Closed-loop test: simulate an EMIT-like scene with a known CH4 source, retrieve, quantify.

Shows the matched filter's known under-estimation of strong plumes, how
optimal estimation with a surface reflectance prior removes it, and how the
IME and cross-sectional-flux methods recover the emission rate.
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
from _common import DEMO_NOTE, FIG_DIR, demo_database

from plumefit import ForwardModel, Geometry, preset
from plumefit import quantify as qf
from plumefit.retrieval import OptimalEstimation, matched_filter
from plumefit.simulate import simulate_scene

warnings.filterwarnings("ignore", message="SpectralDatabase uses synthetic")

Q_TRUE, WIND, PIX = 1500.0, 3.0, 60.0
db = demo_database()
fm = ForwardModel(db, preset("EMIT"), Geometry(35.0, 5.0))
scene = simulate_scene(fm, (300, 90), PIX, q_kg_h=Q_TRUE, wind_speed=WIND, wind_dir_deg=90.0, stability="B", seed=7)
truth = scene.true_enhancement

# 1) Matched filter on the whole scene
mf = matched_filter(scene.radiance, fm.unit_absorption_spectrum("CH4"))
bg_sigma = float(np.nanstd(mf.enhancement[truth < 20]))

# 2) Plume mask from the MF map, then OE on masked pixels with a reflectance prior
src = tuple(int(v) for v in scene.source_rc)
mask = qf.plume_mask(mf.enhancement, bg_sigma, src, threshold_sigma=2.0).mask
oe = OptimalEstimation(fm)
oe_map, oe_unc = oe.retrieve_cube(scene.radiance, mask=mask, reflectance_prior=scene.reflectance)

# 3) Emission rates
dist = np.arange(300.0, 2400.0, 300.0)
csf_mf, _ = qf.cross_sectional_flux(mf.enhancement, src, 90.0, WIND, PIX, dist, half_width_m=900)
oe_filled = np.where(mask, oe_map, mf.enhancement)
csf_oe, _ = qf.cross_sectional_flux(oe_filled, src, 90.0, WIND, PIX, dist, half_width_m=900)
print(f"Background MF noise: {bg_sigma:.0f} ppm m; plume mask: {mask.sum()} px")
print(f"True rate {Q_TRUE:.0f} kg/h | CSF from MF {csf_mf:.0f} kg/h | CSF from OE {csf_oe:.0f} kg/h")
sel = mask & np.isfinite(oe_map)
z = (oe_map[sel] - truth[sel]) / oe_unc[sel]
mf_ratio = mf.enhancement[sel].sum() / truth[sel].sum()
oe_ratio = oe_map[sel].sum() / truth[sel].sum()
print(f"Mass in plume mask, retrieved/true: MF {mf_ratio:.3f}, OE {oe_ratio:.3f}")
print(f"OE normalised error (retrieved - true) / sigma: mean {z.mean():+.2f}, std {z.std():.2f}")

# Figure
fig, ax = plt.subplots(1, 4, figsize=(17, 4.2), constrained_layout=True)
vmax = 3000
for a, img, title in zip(
    ax[:3],
    [truth, mf.enhancement, np.where(mask, oe_map, np.nan)],
    ["True enhancement", "Matched filter", "Optimal estimation (plume pixels)"],
    strict=True,
):
    im = a.imshow(img, vmin=-500, vmax=vmax, cmap="magma")
    a.set_title(title)
    a.set_xticks([])
    a.set_yticks([])
fig.colorbar(im, ax=ax[:3], label="CH4 enhancement [ppm m]", shrink=0.85)
ax[3].scatter(truth[sel], mf.enhancement[sel], s=8, alpha=0.6, label="Matched filter")
ax[3].scatter(truth[sel], oe_map[sel], s=8, alpha=0.6, label="OE + reflectance prior")
lim = [0, 8000]
ax[3].plot(lim, lim, "k--", lw=1)
ax[3].set_xlim(-300, lim[1])
ax[3].set_ylim(-1000, lim[1])
ax[3].set_xlabel("True [ppm m] (axes clipped at 8000)")
ax[3].set_ylabel("Retrieved [ppm m]")
ax[3].legend(loc="upper left")
ax[3].set_title(f"OE (retrieved-true)/sigma: mean {z.mean():+.2f}, std {z.std():.2f}", fontsize=9)
fig.suptitle(f"Closed-loop CH4 retrieval, EMIT-like sensor, 60 m pixels ({DEMO_NOTE})", fontsize=10)
fig.savefig(FIG_DIR / "closed_loop_ch4.png", dpi=120)
print("saved", FIG_DIR / "closed_loop_ch4.png")
