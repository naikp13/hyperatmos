"""Per-pixel optimal estimation of atmosphere and surface, with an error budget.

The fast correction of example 02 gives one answer per pixel. Optimal
estimation (as in ISOFIT) also gives the posterior uncertainty of water
vapour, aerosol and every reflectance band, plus the degrees of freedom
showing what the measurement actually constrains. Here the reported
uncertainties are checked against the known truth.
"""

import matplotlib.pyplot as plt
import numpy as np
from _common import DEMO_NOTE, FIG_DIR, demo_database

from hyperatmos import AtmosphereSurfaceModel, AtmosphericCorrection, Geometry, RadiativeTransferModel, preset
from hyperatmos.forward import state_prior, surface_guess
from hyperatmos.retrieval import OptimalEstimation
from hyperatmos.simulate import simulate_scene

db = demo_database()
rtm = RadiativeTransferModel(db, preset("EMIT"), Geometry(35, 5, 30, None))
scene = simulate_scene(rtm, (40, 40), h2o_mean=1.6, h2o_rel_std=0.2, aod_mean=0.15, seed=5)
fm = AtmosphereSurfaceModel(rtm)
oe = OptimalEstimation(fm)

# Fast correction as first guess and prior centre, as ISOFIT does with its heuristic initial guess.
h2o_fast = AtmosphericCorrection(rtm).estimate_h2o(scene.radiance, 0.1)
pixels = [(r, c) for r in range(2, 40, 6) for c in range(2, 40, 6)]
h2o, h2o_sig, aod, aod_sig, dofs, z_rho = [], [], [], [], [], []
for r, c in pixels:
    y = scene.radiance[r, c]
    xa, sa = state_prior(fm, surface_guess(fm, y, h2o_fast[r, c], 0.1), h2o=(h2o_fast[r, c], 0.5), aod=(0.1, 0.2))
    res = oe.retrieve(y, xa, sa)
    h2o.append(res.value("H2O"))
    h2o_sig.append(res.sigma("H2O"))
    aod.append(res.value("AOD550"))
    aod_sig.append(res.sigma("AOD550"))
    dofs.append(res.dofs(["H2O", "AOD550"]))
    rho_sig = np.sqrt(np.diag(res.s_hat))[fm.surface_index()]
    z_rho.append((res.x[fm.surface_index()] - scene.reflectance[r, c]) / rho_sig)
h2o_true = np.array([scene.h2o_g_cm2[p] for p in pixels])
h2o, h2o_sig, aod, aod_sig, z_rho = map(np.array, (h2o, h2o_sig, aod, aod_sig, z_rho))
z_h2o = (h2o - h2o_true) / h2o_sig
z_rho = z_rho[:, fm.measurement_mask]
print(
    f"{len(pixels)} pixels. H2O normalised error: mean {z_h2o.mean():+.2f}, std {z_h2o.std():.2f}; "
    f"median sigma {np.median(h2o_sig):.3f} g/cm2"
)
print(f"AOD {np.mean(aod):.3f} (true 0.15), median sigma {np.median(aod_sig):.3f}; DOFS(H2O, AOD) {np.mean(dofs):.2f}")
print(f"Reflectance normalised error: mean {z_rho.mean():+.2f}, std {z_rho.std():.2f}")

fig, ax = plt.subplots(1, 3, figsize=(15, 4.3), constrained_layout=True)
ax[0].errorbar(h2o_true, h2o, yerr=2 * h2o_sig, fmt="o", ms=3, lw=0.8)
lim = [h2o_true.min() - 0.1, h2o_true.max() + 0.1]
ax[0].plot(lim, lim, "k--", lw=1)
ax[0].set_xlabel("True H2O [g cm$^{-2}$]")
ax[0].set_ylabel("Retrieved H2O, $\\pm 2\\sigma$")
ax[0].set_title(f"Water vapour: normalised error std {z_h2o.std():.2f}")
r, c = pixels[10]
res = oe.retrieve(
    scene.radiance[r, c],
    *state_prior(fm, surface_guess(fm, scene.radiance[r, c], h2o_fast[r, c], 0.1), h2o=(h2o_fast[r, c], 0.5)),
)
sig = np.sqrt(np.diag(res.s_hat))[fm.surface_index()]
wl = fm.wavelength_nm
x = res.x[fm.surface_index()]
ax[1].fill_between(wl, x - 2 * sig, x + 2 * sig, color="C1", alpha=0.3, label="posterior $\\pm 2\\sigma$")
ax[1].plot(wl, scene.reflectance[r, c], "k", lw=1, label="truth")
ax[1].plot(wl, x, "C1", lw=1, label="retrieved")
ax[1].set_xlabel("Wavelength [nm]")
ax[1].set_title("Reflectance and its uncertainty, one pixel")
ax[1].legend(fontsize=8)
ax[2].hist(z_rho.ravel(), bins=np.linspace(-4, 4, 41), density=True, color="C2", alpha=0.7)
g = np.linspace(-4, 4, 200)
ax[2].plot(g, np.exp(-0.5 * g**2) / np.sqrt(2 * np.pi), "k", lw=1, label="N(0, 1)")
ax[2].set_xlabel("(retrieved - true) / posterior sigma, all bands")
ax[2].set_title("Reflectance error calibration")
ax[2].legend(fontsize=8)
fig.suptitle(f"Optimal estimation of H2O, AOD and reflectance in every band, EMIT-like ({DEMO_NOTE})", fontsize=10)
fig.savefig(FIG_DIR / "optimal_estimation.png", dpi=110)
print("saved", FIG_DIR / "optimal_estimation.png")
