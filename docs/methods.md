# Methods

## Forward model (SWIR)

For a Lambertian surface, with scattering neglected inside the absorption window, the radiance in band *b* is

$$
L_b = \frac{\mu_0}{\pi}\,\rho_b \int \mathrm{SRF}_b(\nu)\,E_0(\nu)\,
\exp\!\Big[-\tfrac{\tau_v(\nu)}{\mu_0} - \tfrac{\tau_{z<h}(\nu)}{\mu_v}
- A \sum_g \sigma_g(\nu)\,\Delta X_g\Big]\,d\nu ,
\qquad A = \frac{1}{\mu_0} + \frac{1}{\mu_v}
$$

- $\tau_v$ is the vertical optical depth of the background atmosphere. $\tau_{z<h}$ covers only the layers below the sensor altitude $h$; for a satellite it equals $\tau_v$.
- $\sigma_g$ is the cross section at plume-layer pressure and temperature. $\Delta X_g$ is the column enhancement, carried in ppm·m (1 atm, 288.15 K).
- Cross sections are Voigt line-by-line sums. Intensities are scaled with temperature (TIPS when `hitran-api` is installed, otherwise a power law), and widths with pressure and temperature.
- The monochromatic calculation is convolved with the Gaussian SRF *before* taking the band value, so line saturation is handled correctly.

The surface $\rho_b$ is either a polynomial in normalised wavelength (optionally multiplying a fixed reflectance spectrum) or one free reflectance per band.

## Matched filter

With the column background mean $\mu$, covariance $\Sigma$ (shrunk towards its diagonal), unit absorption spectrum $k = \partial \ln L / \partial \Delta X$, target $t = \mu \odot k$ and albedo factor $a = x^\top\mu / \mu^\top\mu$:

$$
\hat\alpha = \frac{(x-\mu)^\top \Sigma^{-1} t}{a\, t^\top \Sigma^{-1} t},\qquad
\sigma_{\hat\alpha} = \frac{1}{a\sqrt{t^\top \Sigma^{-1} t}} .
$$

Pixels above $2\sigma$ are removed from $\mu$ and $\Sigma$ and the filter is re-run. This step matters when a plume covers a noticeable fraction of a detector column: without it, the plume leaks into the background statistics and is under-estimated.

## Optimal estimation

The retrieval minimises

$$
\chi^2(x) = (y-F(x))^\top S_\epsilon^{-1}(y-F(x)) + (x-x_a)^\top S_a^{-1}(x-x_a)
$$

by Levenberg–Marquardt (Rodgers 2000, eq. 5.36) in the whitened state $z = L^{-1}(x-x_a)$, where $S_a = LL^\top$. It stops when the Gauss–Newton step is small compared with the posterior uncertainty (eq. 5.29). The outputs are $\hat S = (K^\top S_\epsilon^{-1}K + S_a^{-1})^{-1}$, $A = \hat S K^\top S_\epsilon^{-1} K$, DOFS and reduced $\chi^2$.

$S_\epsilon$ uses a shot plus read noise model, parameterised by SNR at a reference radiance.

With `surface="bands"`, $S_a$ contains the covariance of plume-free ISOFIT reflectance. The gas then competes with surface variability that is actually observed in the scene, not with an unconstrained polynomial.

## DOAS

$$
\ln\frac{I_0}{I} = \sum_g \sigma_g\,\mathrm{SCD}_g + \sum_k p_k \lambda^k \;[+\; s\,\partial_\lambda \ln I],
\qquad \mathrm{VCD} = \mathrm{SCD}/\mathrm{AMF}.
$$

$I_0$ is the mean of plume-free (upwind) lines in the same detector column. Cross sections are convolved to instrument resolution, optionally with the I0 correction.

## Emission rates

- IME: $Q = U_\text{eff}\,\mathrm{IME}/L$ with $L = \sqrt{\text{plume area}}$. Fit $U_\text{eff} = aU_{10}+b$ for each sensor with `calibrate_ueff`.
- Cross-sectional flux: $Q = U \int \Delta\Omega\,dy$ on transects perpendicular to the wind; the reported value is the median over downwind distances.
- Uncertainty: Monte Carlo over retrieval noise and wind speed (`monte_carlo_ime`).

## Known limitations

- Scattering is neglected in the SWIR forward model. This is acceptable for clear-sky plume windows, but it biases dark or hazy scenes; coupling to ISOFIT's radiative transfer is on the roadmap.
- The default atmosphere is a simple standard profile. Pass reanalysis profiles or the ISOFIT water vapour for real scenes.
- The demo spectroscopy is synthetic. Use HITRAN for any quantitative result.
