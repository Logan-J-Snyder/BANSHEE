# Method and Scientific Limitations

## Detection and Measurement

1. Read, quality-filter, clean, and normalize one product light curve.
2. Detrend by default, using the staged Ilin-inspired implementation.
3. Split at gaps; seed positive 3-sigma candidates and expand islands to 1 sigma.
4. Evaluate Bayesian grids for a linear baseline plus an exposure-averaged
   Tovar Mendoza flare, with background/reversed/artifact comparisons.
5. Refine measurements by bounded chi-square fitting, then classify and associate tails.

Seeding uses **segment-level** clipped robust scatter, not a protected rolling
3-sigma detector. This version does not enable experimental joint raw-flux
measurements, Gaussian processes, alternate templates, or modified flag policies.

The grid has **41 logarithmic widths**, and the union of **9** candidate-span peak
positions plus **9** around the observed peak, with duplicates removed. Eight
half-normal amplitude-prior scales are `0.0003, 0.001, 0.003, 0.01, 0.03, 0.1,
0.3, 1.0`. Amplitude is continuously marginalized, not eight fixed amplitudes.
`detailed_*` settings remain for compatibility but do not trigger refinement.

The continuous optimizer varies **peak time and log FWHM**. At each trial it solves
nonnegative amplitude and linear offset/slope by weighted least squares. This is
profile likelihood, not an updated Bayesian posterior. A local optimizer does not
guarantee the global minimum. `--no-chi2-fit` uses Bayesian grid measurements instead.

The template is analytically integrated over the exposure interval, not just
sampled at the timestamp. This accounts for temporal averaging but cannot uniquely
recover a peak not resolved by the observations. Intrinsic amplitude differs from
the largest recorded exposure-averaged sample, especially at long cadence.

## ED and Energy

```text
ED_seconds = 86400 * integral f_flare(t) dt_days
E_TESS_erg = ED_seconds * L_TESS_erg_per_second
```

`f_flare` is fractional excess above quiescence. ED is the **full model integral**,
not a trapezoidal integral of just visible positive samples. Inferred tails can
therefore bias ED/energy if width, amplitude, or baseline is wrong.

Use a supplied **TESS-band**, not bolometric, luminosity for one source:

```bash
python BANSHEE_Lite_Chi.py --input /path/to/known_Gaia_file.fits \
  --output-dir results/calibrated --tess-band-luminosity 1e30
```

This requires one known Gaia filename identity across inputs. Do not assign one
luminosity to multiple stars. Alternatively explicitly opt into approximation:

```bash
python BANSHEE_Lite_Chi.py --input /path/to/lightcurves \
  --output-dir results/approximate_energy \
  --stellar-catalogs /path/to/stars.parquet \
  --distance-column distance_pc --energy-method tmag-bandwidth
```

This uses TESS magnitude, distance, a 2416 Jy Vega-like zero point and default
bandwidth `1.2e14 Hz`. It is **not response-curve/SED integration** and does not
correct extinction/dilution or yield bolometric energy. The bandwidth is an
implementation approximation, not a calibration established by the method papers.
Missing magnitude/distance leaves energy missing rather than inventing a value.

## Limits

- Detrending can distort flare wings; detection does not guarantee parameter accuracy.
- Complex multi-component flares may not fit a single template.
- Correlated noise, eclipses, transients, gaps, and contamination can invalidate
  independent-Gaussian assumptions.
- Low reduced chi-square alone does not establish adequacy; inspect residuals.
- `parameter_usable=True` is an operational flag, not a physical-accuracy guarantee.
- Bayesian intervals are conditional on the grid/model and fixed luminosity;
  they are neither continuous-fit errors nor total energy errors.
- Unfitted overflow products cannot count as fully searched zero-flare exposure.
- Demo/tests use matching-template injections and simple noise, not a survey-wide
  sensitivity, false-positive, or catalogue-accuracy calibration.

## References

- [Tovar Mendoza et al. (2022), Llamaradas Estelares](https://arxiv.org/abs/2205.05706):
  continuous empirical flare-shape motivation.
- [Pitkin et al. (2014), A Bayesian method for detecting stellar flares](https://arxiv.org/abs/1406.1712):
  model-comparison motivation, not identical BANSHEE models/thresholds.
- [Ilin & Poppenhaeger (2022)](https://doi.org/10.1093/mnras/stac1232):
  staged detrending motivation, not validation of this implementation.
- [Kipping (2010), Binning is sinning](https://doi.org/10.1111/j.1365-2966.2010.17242.x):
  finite-exposure effects, not recovery of missing temporal information.
- [Davenport (2016)](https://arxiv.org/abs/1607.03494): injection/recovery context.
- [SciPy least_squares](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html):
  optimizer documentation.
- [TESS transient photometry documentation](https://tess.mit.edu/public/tesstransients/pages/readme.html#flux-calibration):
  zero-point source cited by the existing implementation.

Sources motivate components; none validates our specific code, selection
thresholds, flags, or the physical accuracy of an individual event.
