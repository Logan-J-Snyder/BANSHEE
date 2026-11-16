"""Run BANSHEE on native 120 s sectors and score Günther et al. Table 1."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import re
import shutil
import sys
import time
from urllib.error import URLError
from urllib.request import urlopen

import numpy as np
import pandas as pd
from astropy.io import fits
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "BANSHEE_Final"))
import BANSHEE_Final as detector

CONFIG = detector.Config(flux_column="SAP_FLUX")
TIERS = ["accepted", "followup", "rejected", "no_peak_match"]
FITS_NAME = re.compile(r"tess\d+-s(000[12])-(\d{16})-\d+-s_lc\.fits")
TOLERANCES = (2, 5, 10)


def catalog(path):
    keys = ["TIC", "Sec", "Outbst", "Flare"]
    table = pd.read_csv(path, sep="\t", comment="#", dtype=str)
    table = table.apply(pd.to_numeric, errors="coerce").dropna(subset=[*keys, "tpeak"]).copy()
    table[keys] = table[keys].astype(int)
    if len(table) != 8695 or table.duplicated(keys).any():
        raise ValueError("Provide the full 8,695-row VizieR Table 1 TSV")
    table["simple"] = table.groupby(keys[:3]).Flare.transform("size").eq(1)
    table["catalog_peak_btjd"] = table.tpeak - 2457000.
    table["catalog_event_id"] = [f"TIC{t}_S{s:02d}_O{o}_F{f}" for t,s,o,f in table[keys].itertuples(index=False)]
    return table


def indexed_products(index_paths):
    names = {m[0] for path in index_paths for m in FITS_NAME.finditer(path.read_text())}
    result = {}
    for name in sorted(names):
        sector, tic = map(int, FITS_NAME.fullmatch(name).groups())
        if (tic, sector) in result:
            raise ValueError(f"Ambiguous product for TIC {tic}, sector {sector}")
        result[tic, sector] = name
    return result


def download(path, tic, sector):
    # Public S3 location that the indexed MAST product URI redirects to.
    digits = f"{tic:016d}"
    blocks = "/".join(digits[i:i+4] for i in range(0,16,4))
    url = f"https://stpubdata.s3.us-east-1.amazonaws.com/tess/public/tid/s{sector:04d}/{blocks}/{path.name}"
    temp = path.with_suffix(".part")
    for attempt in range(4):
        try:
            with urlopen(url, timeout=35) as response, temp.open("wb") as handle:
                expected = int(response.headers["Content-Length"])
                shutil.copyfileobj(response, handle)
            if temp.stat().st_size != expected:
                raise OSError("Partial archive download")
            with fits.open(temp, memmap=True) as hdul:
                if (hdul[0].header["TICID"], hdul[0].header["SECTOR"]) != (tic, sector):
                    raise ValueError("Downloaded FITS does not match target and sector")
            temp.replace(path)
            return
        except (OSError, URLError):
            temp.unlink(missing_ok=True)
            if attempt == 3:
                raise
            time.sleep(2**attempt)


def match(labels, events, minutes):
    """Maximize one-to-one matches per tier, then minimize peak-time offsets."""
    result = labels.copy()
    result["recovery"] = "no_peak_match"
    result["candidate_id"] = pd.Series(pd.NA, index=result.index, dtype="Int64")
    for tier in TIERS[:3]:
        remaining = result.loc[result.candidate_id.isna()]
        eligible = events.loc[events.tier.eq(tier)]
        if remaining.empty or eligible.empty:
            continue
        distance = abs(remaining.catalog_peak_btjd.to_numpy()[:, None] -
                       eligible.match_peak.to_numpy()[None, :]) * 1440
        penalty = len(remaining) + 1
        cost = np.full((len(remaining), len(eligible) + len(remaining)), penalty, dtype=float)
        cost[:, :len(eligible)] = np.where(distance <= minutes, distance / minutes, penalty * 2)
        row, col = linear_sum_assignment(cost)
        ok = col < len(eligible)
        row, col = row[ok], col[ok]
        valid = distance[row, col] <= minutes
        selected = remaining.index[row[valid]]
        result.loc[selected, "recovery"] = tier
        result.loc[selected, "candidate_id"] = eligible.candidate_id.to_numpy()[col[valid]]
    assert not result.candidate_id.dropna().duplicated().any()
    columns = ["candidate_id", "match_peak", "tpeak", "seed_time", "amplitude", "fwhm_days",
               "equivalent_duration_s", "diagnostic_reasons", "edge_case", "parameter_usable",
               "detection_log_bayes_factor", "logbf_threshold_used"]
    related = events.reindex(columns=columns).set_index("candidate_id").add_prefix("banshee_")
    result = result.join(related, on="candidate_id", validate="many_to_one")
    result["peak_offset_minutes"] = (result.banshee_match_peak - result.catalog_peak_btjd) * 1440
    return result


def run_sector(tic, sector, name, file, fetch, labels, out):
    result_path = out / "sectors" / f"TIC{tic}_S{sector:02d}.json"
    if result_path.exists():
        return json.loads(result_path.read_text())["summary"]
    if not file.is_file():
        if not fetch:
            raise FileNotFoundError(file)
        download(file, tic, sector)
    header = fits.getheader(file, 0)
    if (header["TICID"], header["SECTOR"]) != (tic, sector):
        raise ValueError(f"Misidentified archive file: {name}")
    lc = detector.load_lightcurve(file, CONFIG)
    cadence = detector.cadence_days(lc.time) * 86400
    if abs(cadence - 120) > 1 or lc.selected_flux_column != CONFIG.flux_column:
        raise ValueError(f"Unexpected cadence or flux column: {name}")
    events, prepared, _, diag = detector.analyze(lc, CONFIG)
    if diag["calibration_status"] != "validated":
        raise ValueError(f"No validated SAP operating point: {name}")

    # Published times enter only after the detector completes the whole sector.
    t = np.asarray(prepared.lightcurve.time)
    peaks = labels.catalog_peak_btjd.to_numpy()
    distance = np.abs(t[None, :] - peaks[:, None])
    nearest = distance.argmin(axis=1)
    cadence_days = cadence / 86400
    labels["nearest_retained_minutes"] = distance.min(axis=1) * 1440
    labels["covered"] = ((distance.min(axis=1) <= .75 * cadence_days)
                         & (prepared.segment_id[nearest] >= 0))
    labels["edge"] = prepared.detrend_edge_mask[nearest]
    labels["uncertain"] = prepared.background_uncertain_mask[nearest]
    labels["interior"] = labels.covered & ~labels.edge & ~labels.uncertain
    snr = np.divide(prepared.residual - prepared.local_baseline, prepared.segment_sigma,
                    out=np.full_like(t, np.nan), where=prepared.segment_sigma > 0)
    labels["observed_peak_snr"] = [
        np.max(nearby) if len(nearby) else np.nan
        for delta in distance
        for nearby in [snr[(delta <= cadence_days) & np.isfinite(snr)]]
    ]
    events["match_peak"] = events.tpeak.fillna(events.seed_time)
    matches = {str(m): match(labels, events, m).to_dict("records") for m in TOLERANCES}
    summary = dict(tic=tic, sector=sector, filename=name, sha256=detector.file_sha256(file),
                   cadence_seconds=cadence, retained_samples=len(t),
                   calibration_group=diag["calibration_group"],
                   logbf_threshold=diag["logbf_threshold_used"], counts=events.tier.value_counts().to_dict())
    detector.atomic_json(result_path, detector.json_safe(dict(summary=summary, matches=matches)))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", required=True, type=Path, help="Full VizieR Table 1 TSV")
    parser.add_argument("--indices", required=True, nargs=2, type=Path, help="MAST S1 and S2 light-curve indexes")
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--search-root", action="append", type=Path, default=[])
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    out = args.output
    (out / "sectors").mkdir(parents=True, exist_ok=True)
    args.cache.mkdir(parents=True, exist_ok=True)
    labels = catalog(args.catalog)
    keys = set(labels.loc[labels.simple, ["TIC", "Sec"]].itertuples(index=False, name=None))
    products = indexed_products(args.indices)
    if not keys <= products.keys():
        raise ValueError("Some target–sector pairs are absent from the supplied indexes")
    local = {p.name: p for root in args.search_root for p in root.rglob("*_lc.fits") if FITS_NAME.fullmatch(p.name)}
    wanted = []
    for key, group in labels.groupby(["TIC", "Sec"]):
        if key not in keys:
            continue
        tic, sector = key
        name = products[key]
        file = local.get(name, args.cache / name)
        wanted.append((tic, sector, name, file, args.download, group.copy(), out))
    wanted.sort(key=lambda job: (not job[3].is_file(), job[1], job[0]))
    protocol = dict(detector=detector.DETECTOR_VERSION, detector_sha256=detector.file_sha256(detector.__file__),
                    runner_sha256=detector.file_sha256(__file__), catalog_sha256=detector.file_sha256(args.catalog),
                    index_sha256=[detector.file_sha256(p) for p in args.indices], config=asdict(CONFIG),
                    pairs=len(wanted), single_peaks=int(labels.simple.sum()), tolerances_minutes=TOLERANCES)
    marker = out / "protocol.json"
    safe_protocol = detector.json_safe(protocol)
    if marker.exists() and json.loads(marker.read_text()) != safe_protocol:
        raise ValueError("Use a new output directory after changing the detector, runner or inputs")
    detector.atomic_json(marker, safe_protocol)
    print(f"{len(wanted)} target-sector pairs, {labels.simple.sum()} single peaks, "
          f"{sum(job[3].is_file() for job in wanted)} local files", flush=True)
    progress = []
    for step, summary in enumerate(detector.bounded_jobs(run_sector, wanted, args.workers, "Native 120 s sectors"), 1):
        progress.append(summary)
        if step % 10 == 0 or step == len(wanted):
            detector.atomic_json(out / "progress.json", {"completed": step, "planned": len(wanted)})
    assert len(progress) == len(wanted)
    manifest = sorted(progress, key=lambda s: (s["sector"], s["tic"]))
    detector.atomic_json(out / "source_manifest.json", detector.json_safe(manifest))
    completed = [json.loads((out / "sectors" / f"TIC{job[0]}_S{job[1]:02d}.json").read_text())
                 for job in wanted]
    for tolerance in TOLERANCES:
        rows = [row for case in completed for row in case["matches"][str(tolerance)]]
        matched = pd.DataFrame(rows).sort_values(["Sec", "TIC", "Outbst", "Flare"])
        assert matched.simple.sum() == labels.simple.sum()
        assert not matched.duplicated("catalog_event_id").any()
        detector.atomic_frame(matched, out / f"catalog_matches_{tolerance}min.parquet")
    print(f"Scored {labels.simple.sum()} single peaks across {len(progress)} full sectors", flush=True)


if __name__ == "__main__":
    main()
