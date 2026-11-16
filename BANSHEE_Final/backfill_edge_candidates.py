#!/usr/bin/env python3
"""Refit skipped edge candidates and write a separate set of merged Parquets."""
import argparse
from contextlib import ExitStack
from dataclasses import replace
from functools import lru_cache
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import sqlite3
import sys
import zlib

import BANSHEE_Final as current
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from tqdm import tqdm


@lru_cache(maxsize=1)
def load_snapshot(path, digest):
    if current.file_sha256(path) != digest:
        raise ValueError("Detector snapshot does not match run_manifest.json")
    spec = importlib.util.spec_from_file_location("_banshee_saved_detector", path)
    detector = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = detector
    spec.loader.exec_module(detector)
    return detector


def refit_job(record, seeds, snapshot_path, manifest_hash, config, calibration_path, results, run_id):
    detector = load_snapshot(snapshot_path, manifest_hash)
    source = record["path"]
    try:
        if detector.fingerprint(source) != record["input"]:
            raise ValueError("Original light-curve fingerprint differs from the manifest")
        settings = detector.Config(**{**config, "ilin": detector.IlinConfig(**config["ilin"])})
        if settings.calibration_file:
            settings = replace(settings, calibration_file=calibration_path)
        # Use the saved energy calibration and resolve photometry with the original
        # loader. The stellar catalog is not needed to reproduce the cleaned flux.
        settings = replace(settings, stellar_catalog_paths=(), candidate_workers=1)
        folder = detector.product_folder(record, results)
        saved = json.loads((folder / "summary.json").read_text())
        receipt = json.loads((folder / "complete.json").read_text())
        if (saved["status"] != "complete" or saved["source"] != source
                or receipt["record"] != record or receipt["run_id"] != run_id
                or receipt["outputs"]["summary.json"] != (folder / "summary.json").stat().st_size):
            raise ValueError("Saved light-curve summary differs from its completion receipt")
        energy = saved["energy_calibration"]
        settings = replace(settings, tess_band_luminosity_erg_s=energy["tess_band_luminosity_erg_s"])
        lc = detector.load_lightcurve(Path(source), settings)
        settings, status = detector.calibrated_config(lc, settings)
        if status != saved["calibration_status"] or detector.calibration_group(lc) != saved["calibration_group"]:
            raise ValueError("Original calibration group or status could not be reproduced")
        prepared, _ = detector.prepare(lc, settings)
        candidates = detector.find_candidates(prepared, settings)
        if len(candidates) != saved["candidates"]:
            raise ValueError("Original candidate count could not be reproduced")
        updates = []
        for index, seed in seeds:
            candidate = candidates[index]
            if not (np.isclose(lc.time[candidate.peak_index], seed, rtol=0, atol=1e-9)
                    and prepared.detrend_edge_mask[candidate.peak_index]):
                raise ValueError(f"Original edge seed could not be reproduced for candidate {index}")
            row, _ = current._fit_one(prepared, candidates, index, settings, fitter=detector.characterize)
            nearest = int(np.argmin(abs(lc.time - row.get("tpeak", row["seed_time"]))))
            current.apply_candidate_context(row, prepared, nearest, status, settings)
            current.score_candidate(row, settings)
            current.flag_edge_case(row, lc.time)
            current.concise_flags(row)
            row.update(candidate_id=index, seed_time=seed, edge_review_only=False, edge_refit=True,
                       parameter_usable=bool(row.get("parameter_usable", False)))
            updates.append(row)
        frame = pd.DataFrame(updates)
        detector.annotate_energy(frame, energy)
        if detector.fingerprint(source) != record["input"]:
            raise ValueError("Original light curve changed during the refit")
        rows = current.json_safe(frame.to_dict("records"))
        return source, rows, None
    except (OSError, ValueError, KeyError, IndexError) as error:
        return source, None, str(error)


def find_skipped(results, summary):
    path = results / "followup_flares.parquet"
    if pq.ParquetFile(path).metadata.num_rows != summary["counts"]["followup"]:
        raise ValueError("Follow-up table and original run summary disagree")
    fields = ["source_path", "candidate_id", "seed_time", "tpeak"]
    schema = pq.read_schema(path)
    edge_rows = pc.field("reason") == "edge_region" if "reason" in schema.names else None
    if edge_rows is None:
        raise ValueError("Follow-up table has no reason column")
    if "fwhm_days" in schema.names:
        edge_rows &= pc.field("fwhm_days").is_null()
    table = pq.read_table(path, columns=fields, filters=edge_rows)
    seeds = {}
    for row in table.to_pylist():
        source, index, seed = row["source_path"], float(row["candidate_id"]), float(row["seed_time"])
        if not (source and index is not None and index >= 0 and index == int(index)
                and np.isfinite(seed) and float(row["tpeak"]) == seed):
            raise ValueError(f"Invalid skipped edge candidate: {row}")
        group = seeds.setdefault(source, {})
        if int(index) in group:
            raise ValueError(f"Duplicate skipped edge candidate: {source} #{index}")
        group[int(index)] = seed
    return seeds


def merge_tables(results, destination, cache, seeds, summary, new_fields):
    original = pq.ParquetFile(results / "all_events.parquet")
    schema = original.schema_arrow
    # Regenerate the schema metadata after adding columns. Original data types
    # and full/compact column order are preserved.
    full_schema = pa.schema(list(schema) + [
        pa.field(name, dtype) for name, dtype in new_fields.items() if name not in schema.names
    ])
    compact_schema = pa.schema([field for field in full_schema
                                if field.name not in current.COMPACT_DIAGNOSTIC_FIELDS])
    tables = {"all": "all_events", **current.TABLES}
    counts = dict.fromkeys(current.TABLES, 0)
    changed, flagged = 0, 0
    seen = set()
    original_counts = dict.fromkeys(current.TABLES, 0)
    with ExitStack() as stack:
        writers = {tier: stack.enter_context(pq.ParquetWriter(
            destination / (name + ".parquet.tmp"),
            full_schema if tier == "all" else compact_schema, compression="zstd"))
            for tier, name in tables.items()}
        for batch in tqdm(original.iter_batches(batch_size=32768),
                          desc="Merging updated flare tables", unit="batch"):
            patches, positions = [], []
            source_paths = batch.column(schema.get_field_index("source_path")).to_pylist()
            indices = pc.cast(batch.column(schema.get_field_index("candidate_id")), pa.float64()).to_pylist()
            reasons = batch.column(schema.get_field_index("diagnostic_reasons")).to_pylist()
            original_seeds = pc.cast(batch.column(schema.get_field_index("seed_time")), pa.float64()).to_pylist()
            tiers = batch.column(schema.get_field_index("tier")).to_pylist()
            old_tiers = pd.Series(tiers).value_counts()
            for tier in original_counts:
                original_counts[tier] += int(old_tiers.get(tier, 0))
            by_source = {}
            mask = np.zeros(batch.num_rows, dtype=bool)
            for i, (source, index, reason, tier) in enumerate(zip(source_paths, indices, reasons, tiers)):
                if source not in seeds or index not in seeds[source]:
                    continue
                if (reason != "edge_fraction_cut" or tier != "followup" or (source, index) in seen
                        or original_seeds[i] != seeds[source][index]):
                    raise ValueError(f"Unexpected original candidate {source} #{index}")
                seen.add((source, index))
                if source not in by_source:
                    result = cache.execute("SELECT payload FROM fits WHERE source=?", (source,)).fetchone()
                    by_source[source] = {
                        row["candidate_id"]: row for row in json.loads(zlib.decompress(result[0]))
                    } if result else {}
                row = by_source[source].get(index)
                if row is not None:
                    mask[i] = True
                    positions.append(i)
                    patches.append(row)
                    changed += 1
                    flagged += row["edge_case"]
            arrow_mask = pa.array(mask)
            position_indices = pa.array(positions, type=pa.int64())
            columns = []
            keys = set().union(*(row.keys() for row in patches)) - {"candidate_id", "seed_time"}
            for field in full_schema:
                name = field.name
                values = (batch.column(schema.get_field_index(name)) if name in schema.names
                          else pa.nulls(batch.num_rows, type=field.type))
                if name in keys:
                    old = pc.take(values, position_indices).to_pylist()
                    replacement = [patch.get(name, value) for patch, value in zip(patches, old)]
                    if pa.types.is_string(field.type) or pa.types.is_large_string(field.type):
                        replacement = [str(value) if value is not None and value != "" else None for value in replacement]
                    new = pa.array(replacement, type=field.type)
                    values = pc.replace_with_mask(values, arrow_mask, new)
                columns.append(values)
            updated = pa.Table.from_arrays(columns, schema=full_schema)
            writers["all"].write_table(updated)
            for tier, count in counts.items():
                selected = updated.filter(pc.equal(updated["tier"], tier)).select(compact_schema.names)
                counts[tier] = count + len(selected)
                if len(selected):
                    writers[tier].write_table(selected)
        if original_counts != summary["counts"] or len(seen) != sum(map(len, seeds.values())):
            raise ValueError("Original tables disagree about candidate counts")
        assert sum(counts.values()) == original.metadata.num_rows
    for tier, name in tables.items():
        pending = destination / (name + ".parquet.tmp")
        expected = original.metadata.num_rows if tier == "all" else counts[tier]
        if pq.ParquetFile(pending).metadata.num_rows != expected:
            raise ValueError(f"Output row count differs: {name}")
    return counts, changed, flagged


def run(args, out):
    results = args.results.resolve(strict=True)
    manifest_path = results / "run_manifest.json"
    summary_path = results / "run_summary.json"
    with (results / ".runner.lock").open("rb") as run_lock:
        fcntl.flock(run_lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        manifest = json.loads(manifest_path.read_text())
        summary = json.loads(summary_path.read_text())
        config = manifest["config"]
        snapshots = sorted(results.glob("BANSHEE_*_snapshot.py"))
        if len(snapshots) != 1:
            raise ValueError("Expected exactly one original detector snapshot in the results folder")
        snapshot = snapshots[0]
        load_snapshot(snapshot, manifest["code_sha256"])
        calibration_path = results / "calibration_used.json"
        if config.get("calibration_file") and current.file_sha256(calibration_path) != manifest["calibration_sha256"]:
            raise ValueError("Original calibration does not match the run manifest")
        paths = [results / name for name in ("all_events.parquet", "followup_flares.parquet",
                                             "run_summary.json", "run_manifest.json")]
        provenance = current.json_safe(dict(
            source=str(results), original_manifest_sha256=current.file_sha256(manifest_path),
            inputs=[current.fingerprint(path) for path in paths], version=current.DETECTOR_VERSION,
            backfill_sha256=current.file_sha256(__file__),
            implementation_sha256=current.file_sha256(current.__file__)))
        marker = out / "backfill_manifest.json"
        if marker.exists():
            if json.loads(marker.read_text()) != provenance:
                raise ValueError("Inputs or implementation changed. Use a new output folder.")
        else:
            if any(path.name != ".backfill.lock" for path in out.iterdir()):
                raise ValueError("Output folder already contains unrelated files.")
            current.atomic_json(marker, provenance)

        seeds = find_skipped(results, summary)
        records = {row["path"]: row for row in manifest["inputs"]}
        run_id = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
        with sqlite3.connect(out / "edge_refits.sqlite") as cache:
            cache.execute("CREATE TABLE IF NOT EXISTS fits(source TEXT PRIMARY KEY, payload BLOB)")
            complete = {row[0] for row in cache.execute("SELECT source FROM fits")}
            jobs = [(records[source], list(group.items()), str(snapshot), manifest["code_sha256"],
                     config, str(calibration_path), results, run_id)
                    for source, group in seeds.items() if source not in complete]
            failures = {}
            for i, (source, rows, error) in enumerate(current.bounded_jobs(
                    refit_job, jobs, args.workers, "Refitting skipped edge candidates"), 1):
                if error is not None:
                    failures[source] = error
                    continue
                cache.execute("INSERT INTO fits VALUES (?, ?)", (source, zlib.compress(json.dumps(rows).encode())))
                if i % 100 == 0:
                    cache.commit()
            cache.commit()
            fields = {"edge_case": pa.bool_(), "edge_refit": pa.bool_()}
            for (payload,) in cache.execute("SELECT payload FROM fits ORDER BY source"):
                for row in json.loads(zlib.decompress(payload)):
                    for name, value in row.items():
                        if name not in fields and value is not None:
                            fields[name] = pa.scalar(value).type
            counts, replaced, flagged = merge_tables(results, out, cache, seeds, summary, fields)

        if [current.fingerprint(path) for path in paths] != provenance["inputs"]:
            raise ValueError("Original tables changed during the backfill")
        report = dict(counts=counts, original_counts=summary["counts"], candidates_to_refit=sum(map(len, seeds.values())),
                      refit_candidates=replaced, edge_cases=flagged, failed_lightcurves=failures,
                      complete=not failures, edge_case_fraction=current.EDGE_CASE_FRACTION)
        if report["candidates_to_refit"] - replaced != sum(len(seeds[source]) for source in failures):
            raise ValueError("Merged replacement count disagrees with the refit audit")
        for name in ("all_events", *current.TABLES.values()):
            (out / (name + ".parquet.tmp")).replace(out / (name + ".parquet"))
        current.atomic_json(out / "backfill_summary.json", report)
        print(json.dumps({k: v for k, v in report.items() if k != "failed_lightcurves"}, indent=2))
        if failures:
            raise SystemExit("Some light curves could not be refit. See backfill_summary.json and rerun to retry.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, type=Path, help="Completed original BANSHEE run folder")
    parser.add_argument("--output", required=True, type=Path, help="Separate folder for the merged Parquets")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    out = args.output.resolve()
    results = args.results.resolve()
    if args.workers < 1 or results == out or results in out.parents or out in results.parents:
        parser.error("Use positive workers and separate, non-nested results and output folders.")
    out.mkdir(parents=True, exist_ok=True)
    with (out / ".backfill.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args, out)


if __name__ == "__main__":
    main()
