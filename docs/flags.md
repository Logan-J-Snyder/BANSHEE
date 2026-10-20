# Flags and Decisions: Lite Chi

These are **this version's** rules, not later Test/Updated policies. Flags overlap;
their counts must not be added as if mutually exclusive.

Accepted detections pass evidence/review checks but can retain measurement
warnings. Follow-up means review is needed, not necessarily a non-flare.

| Follow-up reason | Default trigger / interpretation |
|---|---|
| `insufficient_local_background` | Fewer than 8 retained local points; no fit |
| `fit_failed:<message>` | Characterization exception; no fit |
| `candidate_overflow_not_fitted` | Nonzero candidate cap exceeded; **none** fitted |
| `flare_direction_not_established` | Forward minus reversed ln evidence < ln(10) |
| `single_exposure_shape_ambiguous` | Flare minus impulse ln evidence < 1 |
| `raw_discontinuity_preferred` | Original-selected-flux step diagnostic > 2 |
| `peak_near_gap_or_edge` | Peak within 2 cadences of segment boundary |
| `fitted_peak_below_three_sigma` | Sampled exposure-averaged fitted peak < 3 local sigma |
| `possible_tail_of_followup_flare` | Possible tail of nearby follow-up event |

| Rejection reason | Interpretation |
|---|---|
| `log_bayes_factor_below_threshold` | ln Bayes factor < 5 |
| `seed_explained_by_stronger_flare` | Stronger accepted flare explains this seed |

## Measurement Warnings

Warnings do **not** automatically change tier. Any warning makes
`parameter_usable=False`. Only accepted, warning-free events have it True.

| Warning | Default trigger |
|---|---|
| `structured_positive_residual` | Two consecutive >3-sigma positive residuals in model-supported region |
| `cadence_limited` | Intrinsic FWHM < 1.5 exposure durations |
| `broad_width_posterior` | Bayesian width p84/p16 > 3 |
| `width_prior_boundary` | Width-boundary posterior mass > 0.1 |
| `extrapolated_model_area` | Observed model-area fraction < 0.9 |
| `chi2_not_converged` | Continuous optimizer did not succeed |
| `chi2_parameter_bound` | Continuous fit near parameter boundary |

`chi2_fit_status` / `chi2_fit_flags` retain optimizer details. Exceptions or a
zero-amplitude continuous solution may retain Bayesian measurements with
`measurement_method=bayesian_fallback`. `cadence_limited` is a warning, not a
dedicated follow-up reason in Lite Chi. Changing these rules is a science change.

## Other Diagnostics

Detrending warnings are stored in portion summaries, not automatically as event
cuts: `short_portion`, `cosine_fit_failed:<type>`, `savgol_window_too_short`,
`insufficient_savgol_background`, `insufficient_fringe_background`,
`invalid_fringe_span`, `fringe_fit_failed:<type>`.

Energy status distinguishes supplied-luminosity energy, approximate TESS magnitude
calibration, missing/invalid calibration, and unfitted events. A warning does not
itself prevent a numerical energy being written. Check detection, measurement,
and calibration together; unfitted or missing energy is not zero energy.
