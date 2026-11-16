BANSHEE FINAL
Release: 2026.11.10.cutoffs1

Local stellar-flare screening for TESS light curves. The standalone detector is
BANSHEE_Final.py. It reads FITS, CSV or Parquet light curves and writes accepted,
follow-up and rejected candidate tables. The optional precheck reports changing
noise and unusual features without changing detection decisions.

This preserves the November 9 scientific functions and numerical SAP cuts,
with the November 10 fixed-cut confirmation recorded in the embedded calibration. "Final" names this release;
it is not a claim of perfect completeness or zero false detections.

1. INSTALL ONCE

Use Linux or macOS, with Python 3.11-3.13. Windows requires WSL because the
runner uses POSIX file locking. This release was tested with Python 3.13.9.
Open a terminal in this extracted folder, then run:

  python3 -m venv .venv
  source .venv/bin/activate
  python -m pip install -r requirements.txt
  python BANSHEE_Final.py --version

The requirements file pins the tested versions. Lightkurve brings additional
dependencies. These are installed once, not downloaded for each light curve.
The detector reads local inputs; it does not fetch observations or log into a
server. It can run offline after dependencies have been installed.

2. CHECK THE INSTALLATION WITH SYNTHETIC DATA

  python make_example_lightcurves.py --output ./example_lightcurves
  python -u BANSHEE_Final.py --input ./example_lightcurves --output-dir ./example_results --workers 2 --plots

The example generator creates five artificial SAP light curves, one per cadence,
with known injected flares. IDs are synthetic placeholders, not catalog targets.
This demonstrates file loading, fitting and exports; it is not a completeness
measurement. The 1800-second example now uses the confirmed default SAP calibration.

3. RUN YOUR LIGHT CURVES

Put light curves in a separate folder, then replace the example paths below:

  python -u BANSHEE_Final.py --input /path/to/lightcurves --output-dir /path/to/results --workers 4

--input accepts one or more individual files or folders and searches folders
recursively. Each table file must hold ONE star, sector and flux stream.
Do not pass a per-star metadata catalog as a light curve. Input and output
directories must be separate and non-nested. Quote paths containing spaces.

For a single light curve with review plots:

  python -u BANSHEE_Final.py --input /path/to/lightcurve.fits --output-dir /path/to/single_result --workers 1 --plots

For the optional OC/FS folder convention:

  lightcurves/
    open_clusters/cluster_name/<gaia_dr3_id>_S01_SPOC_LC.fits
    field_stars/<gaia_dr3_id>_S01_QLP_LC.fits

  python -u BANSHEE_Final.py --server --input-root /path/to/lightcurves --samples open_clusters field_stars --output-root /path/to/results --workers 4

--server selects this folder convention; it does not connect to a server.
Its default input root is ./lightcurves. Filenames in this mode must contain a
Gaia DR3 ID in a supported pattern. Use --input for other folder conventions.
Native SPOC filenames are supported in --input mode; without a GAIA_DR3 header
or another supported ID source, Gaia ID is UNKNOWN, never replaced with TIC ID.

FITS time columns: TIME, BTJD, time_btjd or TMID_BJD (case-insensitive).
CSV/Parquet time columns: TIME, BTJD, time_btjd or bjd. Metadata columns include
gaia_dr3_id (also gdr3_id, source_id or member_id), sector and mission (or lc_type).
Preserve Gaia IDs as strings or integers; floating-point IDs are refused.
Positive, constant exposure duration is required when explicitly supplied in a
table's exposure_days column. Missing exposure metadata falls back to cadence,
which may NOT match a built-in calibration group.

Automatic flux priority:
  SPOC: SAP_FLUX, then PDCSAP_FLUX, then raw_flux/flux.
  QLP:  SAP_FLUX, then KSPSAP_FLUX, then raw_flux/flux.
  Other supported products retain their declared product-specific handling.
To require SAP rather than allow a fallback, add --flux-column SAP_FLUX.
Only the selected stream's matching uncertainty column is used. QUALITY == 0
is applied when that column exists; the presence/application is recorded.

4. KEEP A REMOTE RUN ALIVE WITH TMUX (OPTIONAL)

Copy this folder to a computer/account you control. Generic upload example:

  scp -r BANSHEE_Final username@hostname:~/
  ssh username@hostname
  cd ~/BANSHEE_Final

Set up/activate the environment on that computer using step 1, then:

  tmux new -s banshee
  source .venv/bin/activate
  python -u BANSHEE_Final.py --input /path/to/lightcurves --output-dir /path/to/results --workers 4

Detach: press Ctrl+b, release both keys, then d. Reconnect with:

  tmux attach -t banshee

The progress bar shows elapsed time, estimated remaining time and processed
jobs. A processed job can succeed or fail; it is not a count of flare detections.
Startup reports cached and remaining inputs. run_summary.json separates complete
and incomplete products. Per-light-curve JSON is saved, not printed repeatedly.

A bad light curve is recorded while other usable inputs continue. After the
batch finishes, any remaining failures produce a nonzero exit status and keep
the successful results. Invalid configuration, changed inputs, or a conflicting
output folder can stop the run; the program does not hide these conditions.

5. DEFAULT CALIBRATION

No calibration file or new calibration run is needed for matching default SAP
settings. The actual cuts are embedded in BANSHEE_Final.py:

Cadence  Product Exposure  Ordinary lnBF  High-SNR lnBF  Saved status
  20 s   SPOC     19.80 s    4.172180       1.386294     validated
 120 s   SPOC    118.80 s    1.422173       2.878134     validated
 200 s   QLP     200.00 s    1.386294       1.386294     validated
 600 s   QLP     600.00 s    1.386294       1.386294     validated
1800 s   QLP    1800.00 s    1.386294       1.386294     validated

High-SNR means fitted, exposure-averaged peak SNR >= 8, not injected truth.
These cuts apply to detection_log_bayes_factor with detection_statistic set to
alternatives: flare versus an equal-weight mixture of no flare, reversed flare,
symmetric bump and impulse. The separate log_bayes_factor column measures the
flare against the affine background and is NOT the default operating score.
Raw-flux checks, morphology, local three-sigma criteria and other eligibility
checks still apply; passing one lnBF cut alone does not establish acceptance.

The November 10 confirmation passed at the unchanged 600s and 1800s cuts.
The nominal 95% upper false-window bounds were 1.250% (8/1152) and 2.917%
(24/1152), respectively. Otherwise passing 1800s events can now be accepted.
The prior 1800s failed check is retained in the calibration history.
Other products, exposures, flux streams and changed scientific settings also
do not inherit SAP validation. Inspect calibration_status and calibration_group.

The 5% target is a probability per tested control window under the stated
simulation/background selection; it is NOT 5% contamination of a flare catalog
or a per-sector false-alarm rate. Only a small number of background stars was
used; some had appeared during development. Embedded provenance retains these
limitations. The short-cadence follow-up workload remains substantial.

--calibrate, --auto-calibrate and --calibration remain research options.
External calibration JSON must match this exact code, version and settings;
old JSON from another file is deliberately refused. The optional calibration
inventory can stop on unusable inputs. The normal default run bypasses it.
--allow-uncalibrated explicitly permits provisional research acceptance; it is
not needed or recommended for the ordinary examples above.

6. EDGE AND GAP HANDLING

All candidate seeds are eligible for fitting. A fitted peak in the first or
last 1% of the full cleaned time span carries edge_case=True. If the fit
returns no peak, the flag uses the seed time. The tier and diagnostic reasons
reflect the usual morphology, background and evidence checks.

The existing --edge-fraction and --edge-scope settings define the calibration
and searchable exposure region. The default excludes the outer 5% at each
end from those calculations and isolates that region from tail association.
The saved false-alarm bounds apply to the calibrated interior. The added edge
detections use the existing evidence thresholds and have no separate edge
calibration. The two-cadence check at gaps still applies.

Backfill previously skipped candidates from an existing run:

  python -u backfill_edge_candidates.py --results /path/to/old/results --output /path/to/merged --workers 4

Run where the original light curves and completed run folder are available.
The script uses the detector snapshot, calibration and candidate IDs from the
old run. It replaces the skipped rows in four merged Parquets in the output
folder. Existing fitted rows keep their values. The edge_refit column marks
the new results and edge_case is populated for refitted candidates. Rerunning
the same command reuses completed fits and retries missing or failed inputs.
Check backfill_summary.json for the number processed and any failures.

7. OUTPUTS AND RESUMING

The output folder contains CSV and Parquet versions of:
  accepted_flares       Passing candidates, not externally confirmed flares.
  followup_flares       Ambiguous, unvalidated or otherwise review-only events.
  rejected_candidates  Candidates that failed the detector's checks.
  all_events           Every candidate, including detailed diagnostics.

Also saved: run_manifest.json, run_summary.json, run_log.jsonl,
calibration_used.json, BANSHEE_Final_snapshot.py and per-light-curve products.
--plots adds overview/contact-sheet images and prepared arrays; this can consume
considerable disk space for a large collection.

Useful columns: gaia_dr3_id (exact string), sector, sample, tier, reason,
warnings, tpeak (BTJD), fwhm_days, amplitude (relative instantaneous model peak),
sampled_model_amplitude, fitted_peak_snr, equivalent_duration_s,
detection_log_bayes_factor, log_bayes_factor, logbf_threshold_used,
operating_snr_class, calibration_status, parameter_usable and edge_case.
Detailed diagnostics are in all_events. Bayesian credible intervals are distinct
from continuous chi-square point estimates, not uncertainties on those fits.
Accepted warnings display only cadence_limited when applicable; always check
parameter_usable before using fitted quantities for population measurements.

Absolute flare energy requires a supplied TESS-band luminosity or the explicit
approximate TESSMAG/distance conversion. Missing metadata leaves energy null.
Equivalent duration remains available when a fit is available. Energy is not
automatically bolometric, and does not include an automatic dilution correction.

Repeat the SAME command with unchanged inputs, code, environment and settings
to reuse complete products and retry failures. Use a new output folder when
any of these change, including switching from Perfected to Final. Never merge
new results into an older run by bypassing the manifest checks.
Run outputs record your actual paths for provenance; review those outputs
separately before sharing them. This distribution contains no observed light
curves, private catalogs, account names or host-specific upload scripts.

8. OPTIONAL ADVISORY PRECHECK

  python -u BANSHEE_Final_With_Precheck.py --input /path/to/lightcurves --output-dir /path/to/results --workers 4

This checks all selected light curves first, then runs BANSHEE_Final.py. Its
reports go in the sibling results_precheck folder. Add --precheck-only to skip
fitting, or --plots to include diagnostics. Noise changes, broad/symmetric bumps,
steps, impulses, dips and unusual curves are screening flags, not astrophysical
classifications. The precheck does not mask data or automatically reject flares.

9. TESTS AND CHANGE HISTORY

  python -m unittest discover -s tests -v

See VALIDATION.txt for the checks actually run for this release, and
CHANGES_SINCE_LITE_CHI.txt for the detailed comparison. RELEASE_MANIFEST.json
records file hashes, dependency versions and the compared source hashes.
See CUTOFFS_FINAL.txt/JSON for the fixed-cut confirmation and its scope.

Method references retained in the source:
  Pitkin et al. (2014), https://doi.org/10.1093/mnras/stu1889
  Tovar Mendoza et al. (2022), https://arxiv.org/abs/2205.05706
  Ilin & Poppenhaeger (2022), https://doi.org/10.1093/mnras/stac1232
This implementation contains additional empirical choices and is not an exact
reproduction of any one of these papers.
