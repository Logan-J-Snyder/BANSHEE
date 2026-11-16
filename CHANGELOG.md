# Changelog

## BANSHEE Final 2026.11.16.edge1 - 2026-11-16

- Fit edge candidates and keep their detector tier. Flag peaks within the outer
  1% of the cleaned light-curve time span in every flare table.
- Preserve the interior calibration protocol and prevent candidates in the
  previous edge exclusion zone from changing interior tail decisions.
- Refit legacy skipped edge candidates using the original detector snapshot
  and merge replacements by input path and candidate ID.

## BANSHEE Final 2026.11.10.cutoffs1 - 2026-11-10

- Finalize the existing numerical cadence/SNR cuts without retuning them.
- Record a predeclared 3,064-trial confirmation at 600s and 1800s, with no failed
  trials. Upper false-window bounds are 1.25% and 2.92%, respectively.
- Update 1800s calibration status to validated, allowing otherwise-passing
  candidates to be accepted. Preserve the previous failed check in the history.
- Include the cutoff table, plot, control-kind counts, uncertainty, protocol
  hashes and limitations of reused finite backgrounds. The 5% target is for
  the tested control mixture, not catalog contamination or every artifact type.
- Preserve all scientific function definitions and the 20s/120s/200s calibration
  records. Local validation: 89 tests and the five-cadence CLI example passed.

## BANSHEE Final 2026.11.09.share1 - 2026-11-09

- Add the self-contained `BANSHEE_Final/` release with the standalone detector,
  optional advisory precheck, pinned dependencies, synthetic generator and tests.
- Include the full Lite_Chi comparison, validation record and file-hash manifest.
- Make Final the primary README workflow; preserve the legacy guide and code.
- Extend CI to run both the legacy tests and the 89 Final regression tests.
- Preserve the shared release's scientific settings and embedded calibration,
  including the unvalidated 1800-second follow-up policy. No new calibration or
  recovery benchmark was performed as part of this repository update.

## Rights Notice - 2026-10-20

- Add an all-rights-reserved notice credited to Logan Snyder, as requested.
- Require prior written permission, subject to the exceptions in LICENSE;
  preserve third-party terms and update documentation/package metadata.
- No scientific implementation or fitting behavior changed.

## 0.1.0.dev0 - 2026-10-20

- Package the existing Lite Chi implementation with installation metadata,
  synthetic examples, offline tests, CI, and documentation.
- Replace owner-specific server input/output defaults with the current user's
  home directory; preserve fitting and classification logic.
- Correct introductory script names and the plotted Tovar Mendoza name.
- Leave the original working source outside this repository unchanged.

This is not a claim of a new scientifically validated release or PyPI publication.
