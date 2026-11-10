# Changelog

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
