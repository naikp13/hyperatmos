"""NO2 column mapping with a UV-visible imager by DOAS.

A drone-borne push-broom spectrometer (410-490 nm, 0.5 nm FWHM) maps a
spatially varying tropospheric NO2 field over a city edge. Each detector
column uses the clean background lines as its reference, which cancels the
solar Fraunhofer structure and most instrument effects. The geometric
air-mass factor converts slant to vertical columns of the excess over the
reference.
"""

import matplotlib.pyplot as plt
import numpy as np
from _common import DEMO_NOTE, FIG_DIR

from hyperatmos import Geometry, uniform_instrument
from hyperatmos.retrieval import convolve_cross_section, doas_image
from hyperatmos.spectroscopy import demo_uvvis_cross_section

rng = np.random.default_rng(1)
shape = (160, 120)
geom = Geometry(50.0, 10.0, 0.0, 100.0)
inst = uniform_instrument(410.0, 490.0, 0.4, 0.5, snr_ref=800, radiance_ref=8.0)
wl = inst.wavelength_nm
sigma = convolve_cross_section(demo_uvvis_cross_section("NO2"), inst)

# NO2 excess column [molec/cm^2]: urban gradient plus a busy road, zero in the first 20 lines
rr, cc = np.indices(shape)
vcd_true = 1.2e16 * np.clip((rr - 20) / 140, 0, 1) ** 1.5 * (1 + 0.3 * np.sin(cc / 15))
vcd_true += 6e15 * np.exp(-0.5 * ((cc - 0.4 * rr - 40) / 3.0) ** 2) * (rr > 20)

fraunhofer = 1 - 0.25 * np.exp(-0.5 * ((wl[:, None] - np.array([430.8, 438.4, 486.1])) / 0.4) ** 2).sum(1)
spectrum = 6.0 * (1 - 0.002 * (wl - 450)) * fraunhofer
surface = 1 + 0.15 * np.sin(np.linspace(0, 3, shape[1]))[None, :] + 0.03 * rng.normal(size=shape)
rad = surface[..., None] * spectrum * np.exp(-sigma * (vcd_true * geom.airmass)[..., None])
rad += rng.normal(size=rad.shape) * inst.noise_sigma(rad)

scd, scd_err = doas_image(
    wl, rad, {"NO2": sigma}, slice(0, 20), gas="NO2", window=(425.0, 485.0), poly_order=3, fit_shift=False
)
vcd = scd / geom.airmass
err = (vcd - vcd_true)[20:]
print(
    f"VCD error: bias {err.mean():.2e}, std {err.std():.2e} molec/cm^2; "
    f"median reported 1-sigma {np.median(scd_err[20:]) / geom.airmass:.2e}"
)

fig, ax = plt.subplots(1, 3, figsize=(14, 4.6), constrained_layout=True)
vmax = np.percentile(vcd_true, 99.5)
for a, img, t in zip(ax[:2], [vcd_true, vcd], ["True NO2 excess VCD", "DOAS retrieved VCD"], strict=True):
    im = a.imshow(img, vmin=-0.1 * vmax, vmax=vmax, cmap="cividis")
    a.set_title(t)
    a.set_xlabel("cross-track pixel")
fig.colorbar(im, ax=ax[:2], label="molecules cm$^{-2}$", shrink=0.85)
ax[2].plot(vcd_true[20:].ravel()[::7], vcd[20:].ravel()[::7], ".", ms=2, alpha=0.5)
ax[2].plot([0, vmax], [0, vmax], "k--", lw=1)
ax[2].set_xlabel("True VCD")
ax[2].set_ylabel("Retrieved VCD")
ax[2].set_title(f"Error std {err.std():.1e} molec cm$^{{-2}}$")
fig.suptitle(f"Drone imaging DOAS of tropospheric NO2 ({DEMO_NOTE})", fontsize=10)
fig.savefig(FIG_DIR / "no2_doas.png", dpi=110)
print("saved", FIG_DIR / "no2_doas.png")
