# Methods

This page states the equations hyperatmos implements and the approximations behind them.

## 1. Gas absorption

The layered atmosphere (`standard_atmosphere`, 15 layers, or 30 with `fine=True`) carries number densities of H2O, O2, CO2, CH4 and O3. For each line-by-line gas and each layer, the absorption cross section is a sum of Voigt lines:

σ(ν) = Σ S_i(T) · V(ν − ν_i − δ_i p; γ_D, γ_L)

- Line intensities are scaled to the layer temperature with the partition-function ratio and the lower-state energy.
- Lorentz widths scale with pressure and temperature; Doppler widths with temperature and molecular mass.
- The Voigt profile is computed from the Faddeeva function, cut at 15 cm⁻¹ from line centre.

O3, NO2 and SO2 use temperature-independent cross-section tables. The vertical optical depth τ_g(ν) = Σ_layers σ n Δz is stored on a fine wavenumber grid (default 0.05 cm⁻¹), together with the fraction of each gas column below every altitude.

Slant gas transmittance for a sensor at height h:

T_g(ν) = exp(−τ_g(ν) [1/μ0 + f_g(h)/μv])

Here f_g(h) is the fraction of the column below h (1 for a satellite). The transmittance is convolved to the instrument with a Gaussian spectral response, truncated at 4σ and stored as a sparse matrix. Because convolution and exponentiation do not commute, water vapour, the most variable absorber, is tabulated: band-averaged transmittance is precomputed on 48 log-spaced columns between 0.05 and 8 g cm⁻², and log T is interpolated. The table agrees with direct line-by-line convolution to about 0.3%.

## 2. Scattering

- **Rayleigh optical depth** follows Hansen and Travis (1974), scaled by surface pressure. The phase function is (3/4)(1 + cos²Θ).
- **Aerosol** optical depth follows an Ångström law, τ_a(λ) = τ_550 (λ/550)^−α. The model has a single-scattering albedo ω and a two-term Henyey–Greenstein phase function, with forward and backward asymmetry and a weight chosen to match the overall asymmetry g. A single-term HG phase function underestimates backscatter, which biases dark-target aerosol retrievals.

The sensor at height h sees fractions f_r = 1 − exp(−h/8 km) of the Rayleigh optical depth and f_a = 1 − exp(−h/H_a) of the aerosol optical depth (H_a = 2 km by default).

**Path reflectance** is single scattering, computed separately for each component and attenuated by the layer above the sensor:

ρ_path = [P_r ω_r (1 − e^{−f_r τ_r m}) + P_a ω (1 − e^{−f_a τ_a m})] / (4(μ0 + μv)) · A

Here m = 1/μ0 + 1/μv, and A is the direct transmittance of the air above the sensor. Computing the two components separately keeps path reflectance increasing with AOD; a combined formula saturates incorrectly.

**Total (direct + diffuse) transmittances** use effective optical depths that count forward-scattered light as transmitted:
- Rayleigh: τ_r/2.
- Aerosol: τ_a(1 − ω(1+g)/2).

T↓ uses the whole column along the solar path; T↑ uses only the fraction below the sensor.

**Spherical albedo** uses the two-stream form S = τ_d/(1 + τ_d), where τ_d = 0.75(τ_r + (1 − g) ω τ_a).

## 3. Radiative transfer equation

The top-of-atmosphere (or at-sensor) reflectance factor is:

ρ* = π L / (μ0 E0) = T_g [ρ_path + T↓ T↑ ρ / (1 − S ρ)]

This is the decoupled form of 6S and ATREM-style codes: gas absorption multiplies the scattering terms. For a Lambertian surface it inverts algebraically:

y = ρ*/T_g − ρ_path,  ρ = y / (T↓T↑ + S y)

## 4. Image-based atmospheric correction

**Aerosol (dark dense vegetation).** Pixels with NDVI > 0.7 and 0.01 < ρ*(2.13 µm) < 0.25 are selected. Bisection finds the AOD at which the median retrieved reflectances satisfy ρ_red = 0.5 ρ_2.13 and ρ_blue = 0.25 ρ_2.13 (Kaufman et al., 1997). On synthetic scenes this is biased high by 0.02 to 0.1, because the 2.13 µm band is not entirely aerosol-free.

**Water vapour (per pixel).** For each candidate column w on a log grid, the reflectance across the 0.94 µm (or 1.14 µm) window is inverted. A straight line is fitted through the two shoulders, and the cost is

J(w) = Σ_core ((ρ − ρ_line)/ρ_line)²

A parabola through the three best grid points refines the minimum. This is the band-depth principle of ATREM and continuum-interpolated band ratios, phrased as a smoothness constraint, which removes the need for a separate calibration curve.

**Inversion.** Section 3, per pixel with that pixel's water vapour. Bands with background gas transmittance below 0.1 (the 1.38 and 1.88 µm H2O bands) are set to NaN and flagged.

## 5. Optimal estimation

State vector:

x = [H2O, AOD550, s_g (column scales of other gases)…, ρ_1 … ρ_n]

The forward model is section 3 converted to radiance.
- **Jacobians:** analytic for the reflectances (∂L/∂ρ_i is diagonal); forward differences for the atmospheric elements.
- **Measurement noise:** S_e is diagonal, from the instrument noise model, plus an optional fractional model error.
- **Prior:** Gaussian. H2O, AOD and gas scales are independent. The surface reflectance has a squared-exponential spectral correlation (length 40 nm), a standard deviation of 10% of the mean plus 0.01, and a 1% uncorrelated nugget. The smooth correlation is what separates narrow absorption from surface features.

The cost

J = (y − F(x))ᵀ S_e⁻¹ (y − F(x)) + (x − x_a)ᵀ S_a⁻¹ (x − x_a)

is minimised by Levenberg–Marquardt in prior-whitened coordinates z = L⁻¹(x − x_a), with S_a = L Lᵀ. This keeps the problem well conditioned across state elements that differ by orders of magnitude. Convergence is declared when dᵢ² = gᵀ H⁻¹ g < 10⁻³ n (Rodgers eq. 5.29).

Outputs (Rodgers 2000, ch. 2–3):
- posterior covariance Ŝ = (KᵀS_e⁻¹K + S_a⁻¹)⁻¹;
- averaging kernel A = Ŝ KᵀS_e⁻¹K;
- DOFS, overall and for any subset of the state;
- reduced χ².

With a free reflectance in every band, a single pixel constrains AOD weakly. That is a property of the problem, not of the solver. Use the DDV estimate or a sun photometer as the AOD prior when available.

## 6. Instrument design

`retrieval_precision` evaluates Ŝ at the truth, the linear error analysis of Rodgers (2000), for any instrument, geometry and state. It needs no simulation and no iteration.

`trade_study` sweeps FWHM and SNR. SNR is defined at the median scene radiance in the analysed window, and band spacing is 0.8 × FWHM, so the band count changes with resolution. In a real camera SNR and FWHM trade against each other through throughput; the study keeps them independent so either constraint can be applied afterwards.

## 7. Trace-gas columns

**DOAS** (Platt and Stutz, 2008). The model is

ln(I_ref/I) = Σ_g σ_g SCD_g + polynomial (+ optional wavelength-shift term)

It is fitted by linear least squares. Cross sections are convolved with the instrument response, with the I0 correction (Aliwell et al., 2002). VCD = SCD / geometric air-mass factor. Scattering air-mass factors are not yet implemented.

**Matched filter.** α = (t − μ)ᵀ C⁻¹ s / sᵀ C⁻¹ s, where t is the pixel spectrum, μ and C the background mean and covariance, and s = μ ⊙ ∂ln L/∂c the target signature from `absorption_signature`. This maps absorber anomalies relative to the scene background, for any gas in the database.

## 8. Limitations

- Scattering is single scattering plus a two-stream spherical albedo. It is good to a few percent of path reflectance in the red and NIR at AOD ≲ 0.3, and worse in the blue and for thick aerosol. Lookup tables from a multiple-scattering code (libRadtran, 6S, sRTMnet) are the planned upgrade.
- The surface is Lambertian, with no adjacency effect and no topography.
- Gas absorption and scattering are decoupled, so absorption inside scattering layers is approximated.
- The default solar spectrum is a smooth blackbody. Supply a measured spectrum for real data.
- All published numbers use synthetic demo spectroscopy (`synthetic=True` in the database). HITRAN loading is implemented but was not exercised in this environment.

## References

See the [README](../README.md#references). Additional reference: Aliwell, S. R., et al. (2002). Analysis for BrO in zenith-sky spectra: an intercomparison exercise. *J. Geophys. Res.*, 107(D14), 4199.
