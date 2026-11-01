"""Anatomy of the at-sensor signal from 400 to 2500 nm, satellite vs drone.

Shows the components of the radiative transfer equation used for atmospheric
correction: gas transmittance for several water vapour columns, path
reflectance for several aerosol loadings, total transmittance and spherical
albedo, and the resulting radiance over vegetation seen from orbit and from
a drone at 120 m.
"""

import matplotlib.pyplot as plt
import numpy as np
from _common import DEMO_NOTE, FIG_DIR, demo_database

from hyperatmos import Geometry, RadiativeTransferModel, uniform_instrument
from hyperatmos.simulate import endmember_spectra

db = demo_database()
inst = uniform_instrument(400.0, 2480.0, 5.0, 6.0, snr_ref=400)
sat = RadiativeTransferModel(db, inst, Geometry(35.0, 5.0, 30.0, None))
drone = RadiativeTransferModel(db, inst, Geometry(35.0, 5.0, 30.0, 120.0))
wl = sat.wavelength_nm

fig, ax = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
for w in (0.5, 1.5, 3.0):
    ax[0, 0].plot(wl, sat.h2o_transmittance(w), lw=1, label=f"H2O {w} g/cm$^2$")
ax[0, 0].set_title("Two-way gas transmittance (satellite)")
ax[0, 0].set_ylabel("$T_g$")
ax[0, 0].legend(fontsize=8, loc="lower center")
for name, a in (("O2 A", 762), ("H2O", 940), ("H2O", 1135), ("H2O", 1380), ("CO2", 2010), ("H2O", 1880)):
    ax[0, 0].annotate(name, (a, -0.07), ha="center", fontsize=7)
ax[0, 0].set_ylim(-0.1, 1.05)

for aod, ls in ((0.0, ":"), (0.1, "-"), (0.4, "--")):
    ax[0, 1].plot(wl, sat.scattering(aod)["rho_path"], "C0", ls=ls, label=f"satellite, AOD {aod}")
    ax[0, 1].plot(wl, drone.scattering(aod)["rho_path"], "C3", ls=ls, label=f"drone 120 m, AOD {aod}")
ax[0, 1].set_yscale("log")
ax[0, 1].set_title("Path reflectance")
ax[0, 1].set_ylabel(r"$\rho_{path}$")
ax[0, 1].legend(fontsize=7, ncol=2)

for rtm, col, lab in ((sat, "C0", "satellite"), (drone, "C3", "drone 120 m")):
    s = rtm.scattering(0.2)
    ax[1, 0].plot(wl, s["t_down"] * s["t_up"], col, label=f"{lab}: $T_\\downarrow T_\\uparrow$")
ax[1, 0].plot(wl, sat.scattering(0.2)["s_alb"], "k", lw=1, label="spherical albedo S")
ax[1, 0].set_title("Scattering transmittance and spherical albedo, AOD 0.2")
ax[1, 0].legend(fontsize=8)

veg = endmember_spectra(wl)["vegetation"]
ax[1, 1].plot(wl, sat.radiance(veg, 1.5, 0.2), "C0", lw=1, label="satellite")
ax[1, 1].plot(wl, drone.radiance(veg, 1.5, 0.2), "C3", lw=1, label="drone 120 m")
ax[1, 1].plot(wl, sat.geometry.mu0 * sat.e0 / np.pi * veg, "k:", lw=1, label="no atmosphere")
ax[1, 1].set_title("At-sensor radiance over vegetation (H2O 1.5 g/cm$^2$, AOD 0.2)")
ax[1, 1].set_ylabel(r"$\mu$W cm$^{-2}$ sr$^{-1}$ nm$^{-1}$")
ax[1, 1].legend(fontsize=8)
for a in ax.flat:
    a.set_xlabel("Wavelength [nm]")
fig.suptitle(f"Radiative transfer components, SZA 35 deg ({DEMO_NOTE})", fontsize=10)
fig.savefig(FIG_DIR / "radiative_transfer.png", dpi=110)

i = int(np.argmin(np.abs(wl - 550)))
print(
    f"Path reflectance at 550 nm, AOD 0.2: satellite {sat.scattering(0.2)['rho_path'][i]:.4f}, "
    f"drone {drone.scattering(0.2)['rho_path'][i]:.5f}"
)
print("saved", FIG_DIR / "radiative_transfer.png")
