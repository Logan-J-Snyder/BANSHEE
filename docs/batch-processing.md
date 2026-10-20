# Batch Processing

## Ordinary Folders

```bash
python BANSHEE_Lite_Chi.py --input /path/to/lightcurves \
  --output-dir results/run_001 --workers 4 --candidate-workers 1
```

File workers times candidate workers determines the approximate process load.
Avoid large nested pools. Plots add runtime and disk use. Use a fresh output
directory for changed code, input lists, or settings. Ordinary mode is not resumable.

## Audited OC/FS Tree

`--server` is a local batch interface, not a remote login. It expects a tree like:

```text
input_root/
  open_clusters/
    <GaiaDR3_ID>/
      <GaiaDR3_ID>_S01_SPOC_LC.fits
  field_stars/
    <GaiaDR3_ID>/
      <GaiaDR3_ID>_S01_SPOC_LC.fits
```

Subdirectories are optional; Gaia-prefixed FITS filenames are required.
Read `discover()` for the exact rules. Pilot and verify discovered counts first:

```bash
python BANSHEE_Lite_Chi.py --server \
  --input-root /path/to/input_root --output-root results/tree_pilot \
  --samples open_clusters field_stars --workers 4 --limit 2
```

Use a different output for the full run:

```bash
python BANSHEE_Lite_Chi.py --server \
  --input-root /path/to/input_root --output-root results/tree_full \
  --workers 4 --max-candidates 0
```

Tree mode uses locks, manifests, per-product completion checks, progress/ETA,
and resume. Resume with the **same command, code, inputs, and configuration**.
Do not remove a live lock or edit running code. Read `run_log.jsonl`,
`run_manifest.json`, and `run_summary.json` for failures/completion.

Ordinary mode defaults to an unlimited candidate count (`--max-candidates 0`).
Tree mode defaults to **300**. Above a nonzero limit **all** candidates become
unfitted follow-up, not just those beyond the first 300. Set zero to fit all,
allowing for substantial runtime. This preserves the Chi-version policy.

The inherited tree default is **20 workers** if omitted. Examples explicitly use
four; pick a safe count for the machine. Default paths are under the current
user's home, not a developer account. Explicit absolute paths are preferred.

### Approximate Energy Calibration

```bash
python BANSHEE_Lite_Chi.py --server \
  --input-root /path/to/input_root --output-root results/tree_energy \
  --workers 4 --max-candidates 0 \
  --oc-catalog /path/to/OC.parquet --fs-catalog /path/to/FS.parquet \
  --distance-column distance_pc --energy-method tmag-bandwidth
```

Catalogues are read-only and matched by exact Gaia identity, not queried from a
server. Each sample uses its own catalogue. See [energy limitations](method.md).
