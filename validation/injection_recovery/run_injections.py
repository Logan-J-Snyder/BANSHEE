"""Reproduce the frozen paired pre/post-detrending SAP injection experiment."""
import argparse
from dataclasses import replace
from functools import lru_cache
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "BANSHEE_Final"))
import BANSHEE_Final as b

CONFIG = b.Config(flux_column="SAP_FLUX")
N_PROFILES = 32
SNRS = tuple(range(31))


def write_json(path, obj):
    b.atomic_json(path, b.json_safe(obj))


@lru_cache(maxsize=4)
def load_source(path):
    return b.load_lightcurve(Path(path), CONFIG)


def detect(lc, bounds, prepared=None, diagnostics=None):
    if prepared is None:
        frame, _, _, diag = b.analyze(lc, CONFIG, search_bounds=bounds)
    else:
        with patch.object(b, "prepare", return_value=(prepared, diagnostics.copy())):
            frame, _, _, diag = b.analyze(lc, CONFIG, search_bounds=bounds)
    if diag["calibration_status"] != "validated":
        raise ValueError("Injected light curve did not use its validated SAP cutoffs")
    return frame


def identify(frame, peak, tolerance):
    if frame.empty:
        return "missed", None
    fitted = frame["tpeak"] if "tpeak" in frame else frame["seed_time"] * np.nan
    separation = abs(fitted.fillna(frame["seed_time"]) - peak)
    for tier in ("accepted", "followup", "rejected"):
        near = separation.loc[frame.tier.eq(tier) & separation.le(tolerance)]
        if len(near):
            return tier, frame.loc[near.idxmin()]
    return "missed", None


@lru_cache(maxsize=None)
def verified_source(path, expected):
    if b.file_sha256(path) != expected:
        raise ValueError(f"Source hash changed: {path}")
    return True


def run_case(case, output):
    path = output / "trials" / (case["case_id"] + ".json")
    if path.exists():
        return len(json.loads(path.read_text()))
    verified_source(case["path"], case["source_sha256"])
    source = load_source(case["path"])
    lc = b.slice_lightcurve(source, np.arange(case["start"], case["stop"]))
    original, diag = b.prepare(lc, CONFIG)
    peak, width = case["peak"], case["fwhm_minutes"] / 1440
    dt, exp = case["cadence_seconds"] / 86400, case["exposure_seconds"] / 86400
    unit = b.flare_model(lc.time, peak, width, 1., exp, case["model"])
    pk = int(unit.argmax())
    norm = float(np.median(lc.flux))
    tolerance = max(2 * dt, width)
    # Fitting sees the whole prepared window as context. Only nearby seeds
    # are fit, and real background matches are measured at SNR zero.
    span = max(4 * dt, 3 * width)
    bounds = (peak - span, peak + span)
    control = detect(lc, bounds)
    control_tier, _ = identify(control, peak, tolerance)
    results = []
    for snr in SNRS:
        for stage in ("pre", "post"):
            post_amp = snr * original.segment_sigma[pk] / unit[pk]
            pre_amp = post_amp / original.detrend_error_scale[pk]
            raw_signal = pre_amp * unit
            injected = replace(lc, flux=lc.flux + norm * raw_signal)
            if snr == 0:
                frame = control
            elif stage == "pre":
                frame = detect(injected, bounds)
            else:
                # Raw-flux vetoes see the same injected raw data in both arms.
                # The post arm adds its calibrated template to the frozen residual.
                post = replace(original, lightcurve=injected,
                               normalized_flux=original.normalized_flux + raw_signal,
                               residual=original.residual + post_amp * unit)
                frame = detect(injected, bounds, post, diag)
            amplitude = pre_amp if stage == "pre" else post_amp
            actual_snr = amplitude * unit[pk] / original.segment_sigma[pk]
            if stage == "pre":
                actual_snr *= original.detrend_error_scale[pk]
            assert np.isclose(actual_snr, snr, rtol=1e-12, atol=1e-12)
            tier, row = identify(frame, peak, tolerance)
            rec = dict(case_id=case["case_id"], star=case["star"], cadence=case["cadence"],
                       profile=case["profile"], stage=stage, snr=snr, realized_snr=float(actual_snr),
                       tier=tier, control_tier=control_tier, model=case["model"],
                       truth_amplitude=amplitude, truth_fwhm_minutes=case["fwhm_minutes"],
                       truth_ed_seconds=b.full_equivalent_duration_days(width, amplitude, case["model"]) * 86400,
                       truth_peak_phase_minutes=(peak - lc.time[pk]) * 1440,
                       candidate_count=len(frame))
            if row is not None:
                for key in ("reason", "diagnostic_reasons", "fitted_peak_snr", "parameter_usable", "edge_case",
                            "detection_log_bayes_factor", "logbf_threshold_used", "measurement_method"):
                    rec[key] = row.get(key)
                rec.update(fit_amplitude=row.get("amplitude"),
                           fit_fwhm_minutes=row.get("fwhm_days", np.nan) * 1440,
                           fit_ed_seconds=row.get("equivalent_duration_s"),
                           fit_peak_phase_minutes=(row.get("tpeak", np.nan) - lc.time[pk]) * 1440)
            results.append(rec)
    assert len(results) == 2 * len(SNRS)
    write_json(path, results)
    return len(results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True, help="Directory containing the exact source FITS")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    doc = json.loads((HERE / "plan.json").read_text())
    if b.file_sha256(b.__file__) != doc["detector_sha256"]:
        raise ValueError("Use the detector version and hash recorded in plan.json")

    sources = {path.name: path for path in args.inputs.glob("*.fits")}
    required = {case["source_file"] for case in doc["cases"]}
    if missing := required - sources.keys():
        raise FileNotFoundError("Missing injection inputs: " + ", ".join(sorted(missing)))
    cases = [dict(case, path=str(sources[case["source_file"]])) for case in doc["cases"]]
    args.output.mkdir(parents=True, exist_ok=True)
    marker = args.output / "plan.json"
    if marker.is_file() and json.loads(marker.read_text()) != doc:
        raise ValueError("Use a new output directory for this frozen plan")
    write_json(marker, doc)
    (args.output / "trials").mkdir(exist_ok=True)
    jobs = [(case, args.output) for case in cases]
    total = sum(b.bounded_jobs(run_case, jobs, args.workers, "Paired SNR sweeps"))
    assert total == len(cases) * 2 * len(SNRS)
    rows = [row for case in cases
            for row in json.loads((args.output / "trials" / (case["case_id"] + ".json")).read_text())]
    frame = pd.DataFrame(rows)
    assert not frame.duplicated(["case_id", "stage", "snr"]).any()
    assert frame.groupby(["cadence", "stage", "snr"]).size().eq(N_PROFILES).all()
    b.atomic_frame(frame, args.output / "injection_results.parquet")
    print(f"Verified {len(frame):,} trials")


if __name__ == "__main__":
    main()
