"""Drone imaging DOAS of a ship NO2 plume, with a cross-sectional flux estimate.

A UV-visible push-broom camera at 100 m altitude images a ship exhaust plume
over sea. Each detector column uses the upwind lines as its plume-free
reference spectrum, which cancels solar Fraunhofer structure and most
instrument artefacts, as in airborne imaging DOAS.
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
from _common import DEMO_NOTE, FIG_DIR

from plumefit import Geometry
from plumefit.constants import molec_cm2_to_ppmm, ppmm_to_molec_cm2
from plumefit.instrument import uniform_instrument
from plumefit.quantify import cross_sectional_flux
from plumefit.retrieval import convolve_cross_section, doas_image
from plumefit.simulate import gaussian_plume_column
from plumefit.spectroscopy import demo_uvvis_cross_section

warnings.filterwarnings("ignore", message="SpectralDatabase uses synthetic")
rng = np.random.default_rng(1)

Q_NO2, WIND, PIX = 25.0, 5.0, 2.0  # kg/h, m/s, m
shape, source = (220, 160), (190, 80)
geom = Geometry(50.0, 10.0, 100.0)
inst = uniform_instrument(410.0, 490.0, 0.4, 0.5, snr_ref=600, radiance_ref=8.0)

# True NO2 vertical column [molec/cm^2] of a plume travelling "up" the image
vcd_true = ppmm_to_molec_cm2(gaussian_plume_column(shape, PIX, source, Q_NO2, WIND, 0.0, "D", gas="NO2"))
sigma = convolve_cross_section(demo_uvvis_cross_section("NO2"), inst)
wl = inst.wavelength_nm

# Sea-surface radiance with Fraunhofer-like structure and gentle spatial variation
fraunhofer = 1 - 0.25 * np.exp(-0.5 * ((wl[:, None] - np.array([430.8, 438.4, 486.1])) / 0.4) ** 2).sum(1)
spectrum = 6.0 * (1 - 0.002 * (wl - 450)) * fraunhofer
glint = 1 + 0.15 * np.sin(np.linspace(0, 3, shape[1]))[None, :] + 0.02 * rng.normal(size=shape)
rad = glint[..., None] * spectrum * np.exp(-sigma * (vcd_true * geom.plume_amf)[..., None])
rad += rng.normal(size=rad.shape) * inst.noise_sigma(rad)

scd, scd_err = doas_image(
    wl, rad, {"NO2": sigma}, slice(200, 220), gas="NO2", window=(425.0, 485.0), poly_order=3, fit_shift=False
)
vcd = scd / geom.plume_amf
q_est, _ = cross_sectional_flux(
    molec_cm2_to_ppmm(vcd), source, 0.0, WIND, PIX, np.arange(40.0, 340.0, 30.0), half_width_m=150.0, gas="NO2"
)
print(f"Median 1-sigma VCD error: {np.nanmedian(scd_err) / geom.plume_amf:.2e} molec/cm^2")
print(f"NO2 emission: true {Q_NO2:.1f} kg/h, CSF estimate {q_est:.1f} kg/h")

fig, ax = plt.subplots(1, 3, figsize=(14, 4.6), constrained_layout=True)
vmax = np.percentile(vcd_true, 99.5)
for a, img, t in zip(ax[:2], [vcd_true, vcd], ["True NO2 VCD", "DOAS retrieved VCD"], strict=True):
    im = a.imshow(img, vmin=-0.1 * vmax, vmax=vmax, cmap="cividis")
    a.set_title(t)
    a.set_xlabel("cross-track pixel (2 m)")
fig.colorbar(im, ax=ax[:2], label="molecules cm$^{-2}$", shrink=0.85)
r, c = source[0] - 40, source[1]
tau_meas = np.log(rad[200:, c].mean(0) / rad[r, c])
tau_model = sigma * scd[r, c]
ax[2].plot(wl, tau_meas - tau_meas.mean(), lw=1, label="measured ln(I0/I), mean removed")
ax[2].plot(wl, tau_model - tau_model.mean(), lw=1.5, label="fitted sigma x SCD, mean removed")
ax[2].set_xlabel("Wavelength [nm]")
ax[2].legend(fontsize=8)
ax[2].set_title(f"Pixel 80 m downwind. Q true {Q_NO2:.0f} kg/h, CSF estimate {q_est:.1f} kg/h", fontsize=9)
fig.suptitle(f"Drone imaging DOAS of a ship NO2 plume ({DEMO_NOTE})", fontsize=10)
fig.savefig(FIG_DIR / "ship_no2_doas.png", dpi=120)
print("saved", FIG_DIR / "ship_no2_doas.png")
