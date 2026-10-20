# Inputs and Outputs

## Input Contract

Supply **one target/product light curve per file**, not a stellar catalogue with
one row per star. Do not concatenate multiple stars: the reader does not validate
all rows' identities. FITS must have one-dimensional time/flux table columns,
not a target-pixel cube or TESSCut image product. CSV and Parquet are also supported.

| Quantity | Recognized names / convention |
|---|---|
| Time | `TIME`, `time`, `BTJD`, `time_btjd`, `bjd`; **days** |
| Default flux | `PDCSAP_FLUX`, `KSPSAP_FLUX`, `SAP_FLUX`, `raw_flux`, `flux` |
| Pre-detrended flux | `detrended_residual`, `detrended_flux`, `det_flux`, `residual_flux`, `relative_flux` |
| Explicit selection | `--flux-column NAME`, case-insensitive |
| Errors | Selected flux name + `_err`, or standard pipeline errors |
| Quality | `QUALITY` / `quality`; **only QUALITY == 0 survives**, when supplied |
| Table exposure | `exposure_duration_days`, `exposure_days`, `timedel`, in days |
| FITS exposure | Header metadata; otherwise cadence fallback |
| Table identity | `gdr3_id`, `gaia_dr3_id`, `source_id`, `member_id` |
| Other metadata | `sector`, `mission` / `lc_type`, optional TESS magnitude/distance |

Normalized flux near one: `--already-detrended`, without `--residual`.
Zero-centered residuals: **both** `--already-detrended --residual`.
Default preprocessing normalizes and detrends. T16 magnitudes have separate
IRM/EPD/TFA handling; check the selected column in the summary.

Absolute BJD is recognized heuristically and converted to `BTJD = BJD - 2457000`.
**MJD is not automatically converted.** Convert to the intended BJD/BTJD system,
including barycentric/time-scale corrections where appropriate, before ingestion.
Do not simply relabel MJD. Use consistent time systems and days throughout.

Keep Gaia IDs exact strings/integers, never floating-point catalogue IDs.
Gaia-matched energy calibration requires known Gaia identity, for example
`<GaiaDR3_ID>_S01_SPOC_LC.fits`. Standard SPOC filenames carry a TIC identity,
which is not automatically crossmatched to Gaia by this code.

## Ordinary Output

```text
results/run/
  manifest.json
  runtimes.csv
  accepted_flares.csv
  followup_flares.csv
  rejected_candidates.csv
  <input-stem>_<path-hash>/
    all_events.csv
    accepted_flares.csv
    followup_flares.csv
    rejected_candidates.csv
    summary.json
    progress.json          # Checkpoint, not final completion
    partial_events.csv     # Periodic candidate checkpoint, when written
    prepared.npz           # Only with --plots
    overview.png           # Only with --plots
    accepted_001.png       # Only with --plots and events in this tier
```

The folder suffix hashes the input **path**, not contents. `manifest.json`
separately records input/code content hashes and configuration. Ordinary mode is
not a resume cache: choose a fresh output directory. Do not run two jobs into one
output folder. Tree/server mode uses different manifests and resumable checks.

## Main Fields

| Column | Meaning / unit |
|---|---|
| `tier` | Accepted, followup, rejected; full per-file output |
| `candidate_id` | Local index, not globally unique |
| `gdr3_id`, `mission`, `sector` | Source/product metadata |
| `source_path` | Local input provenance; sanitize private paths before sharing |
| `tpeak`, `tstart`, `tend` | BTJD days; bounds enclose 99% of fitted area |
| `fwhm_days` | Intrinsic fitted template FWHM, days |
| `duration_days` | Difference of area bounds, **not FWHM** |
| `amplitude` | Intrinsic fractional-flux amplitude |
| `sampled_model_amplitude` | Largest exposure-averaged model sample |
| `equivalent_duration_s` | Full fitted fractional-flux integral, seconds |
| `flare_energy_erg` | ED times band luminosity, if calibrated |
| `measurement_method` | Continuous chi-square, Bayesian grid, or explicit fallback |
| `chi2_red` | Local reduced chi-square; alone not an adequacy guarantee |
| `log_bayes_factor` | Natural-log evidence ratio, not log10 |
| `fitted_peak_snr` | Sampled fitted peak above baseline / local sigma |
| `reason` | Semicolon-separated decision reasons |
| `warnings` | Semicolon-separated measurement warnings |
| `parameter_usable` | Accepted without warnings, not proof of accuracy |
| `*_bayesian*` | Separate grid estimates and conditional percentiles |

Early failures/overflow may have no measurements. Missing energy is not zero.
Combined tables are compact; `reason` is included only in combined follow-up and
rejected tables. Read full `all_events.csv` for tier, IDs, paths, and diagnostics.
No candidate means no 3-sigma seed, not a rejected event. Check `runtimes.csv`
and product summaries before interpreting an empty output as successful processing.

```python
import pandas as pd

accepted = pd.read_csv("results/run/accepted_flares.csv", dtype={"gdr3_id": "string"})
usable = accepted.loc[accepted["parameter_usable"].astype("string").str.lower().eq("true")]
energetic = usable.loc[usable["flare_energy_erg"].gt(0)]
```

Also check `energy_status`. For rates, count valid exposure from zero-flare
products too: flare-containing sectors alone do not define observing time.
