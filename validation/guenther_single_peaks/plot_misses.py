"""Plot every rejected or unmatched single peak from the five-minute benchmark."""
import argparse
from io import BytesIO
import json
from pathlib import Path
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from run_catalog import detector, CONFIG

MISSES = {"rejected": "Rejected match", "no_peak_match": "No peak match"}
COLORS = {"catalog": "#1f659e", "flux": "#263d50", "trend": "#26847c",
          "threshold": "#715397", "candidate": "#ab3c59"}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
    "axes.spines.right": False, "axes.edgecolor": "#9eacb7", "axes.labelcolor": "#243645",
    "text.color": "#1c2e3c", "grid.color": "#dce4ea",
})


def show_series(ax, x, y, segment, cadence_min, color, label, points=False):
    """Leave gaps unconnected, including runs of removed quality cadences."""
    if points:
        ax.scatter(x, y, color=color, s=7, zorder=3, label=label)
    split = np.flatnonzero((np.diff(x) > 1.55 * cadence_min) | (np.diff(segment) != 0)) + 1
    for i, (time, value) in enumerate(zip(np.split(x, split), np.split(y, split))):
        ax.plot(time, value, "-", color=color, lw=.78, alpha=.82,
                label=label if i == 0 and not points else None)


def plot_peak(row, prepared, cadence_min, neighbors, filename):
    peak = row.catalog_peak_btjd
    offset = (prepared.lightcurve.time - peak) * 1440
    span = max(30, min(180, 3 * row.FWHMdt * 1440))
    if np.isfinite(row.nearest_retained_minutes):
        span = max(span, row.nearest_retained_minutes + 10)
    show = np.abs(offset) <= span
    time, segment = offset[show], prepared.segment_id[show]
    model_peak = row.banshee_tpeak
    has_peak = np.isfinite(model_peak)
    seed = row.banshee_seed_time
    has_seed = np.isfinite(seed)

    fig, axes = plt.subplots(2, 1, figsize=(9.3, 5.6), sharex=True)
    fig.subplots_adjust(left=.11, bottom=.104, right=.973, top=.742, hspace=.25)
    fig.text(.11, .95, row.catalog_event_id + "   ·   " + MISSES[row.recovery], fontsize=16,
             weight="semibold", va="top")
    snr = f"{row.observed_peak_snr:.1f}" if np.isfinite(row.observed_peak_snr) else "unavailable"
    fig.text(.11, .891, f"Published peak {peak:.6f} BTJD   ·   120 s SAP   ·   Observed peak SNR {snr}",
             fontsize=10, va="top")
    if row.recovery == "rejected":
        reason = row.banshee_diagnostic_reasons
        note = str(reason).replace("_", " ").replace(";", " · ") if pd.notna(reason) else "Rejected by BANSHEE"
    else:
        note = ("No one-to-one candidate match within ±5 min" if row.covered else
                f"Published peak has no valid nearby cadence · closest retained point {row.nearest_retained_minutes:.1f} min away")
    fig.text(.11, .846, textwrap.fill(note, width=104, max_lines=2, placeholder=" …"),
             fontsize=9.3, va="top", color="#445c6c")

    raw, residual = axes
    center = float(np.nanmedian(prepared.normalized_flux[show]))
    # Shift only the display zero point of the raw photometry and its trend.
    show_series(raw, time, 100 * (prepared.normalized_flux[show] - center), segment,
                cadence_min, COLORS["flux"], "SAP flux", points=True)
    show_series(raw, time, 100 * (prepared.trend[show] - center), segment,
                cadence_min, COLORS["trend"], "Trend")
    show_series(residual, time, 100 * (prepared.residual[show] - prepared.local_baseline[show]),
                segment, cadence_min, COLORS["flux"], "Detrended SAP", points=True)
    show_series(residual, time, 300 * prepared.segment_sigma[show], segment,
                cadence_min, COLORS["threshold"], "Local 3σ height")

    for i, ax in enumerate(axes):
        ax.axvspan(-5, 5, color=COLORS["catalog"], alpha=.08, lw=0)
        ax.axvline(0, color=COLORS["catalog"], ls="--", lw=1.15, label="Published peak")
        if has_peak:
            ax.axvline((model_peak - peak) * 1440, color=COLORS["candidate"], ls="-.",
                       lw=1.15, label="Matched peak")
        if has_seed and (not has_peak or abs(seed-model_peak) * 1440 > .4):
            ax.axvline((seed - peak) * 1440, color="#955a2e", ls=":", lw=1.1, label="Matched seed")
        for other in (neighbors - peak) * 1440:
            if 0 < abs(other) < span:
                ax.axvline(other, color=COLORS["catalog"], ls=":", lw=.65, alpha=.42)
        ax.set_xlim(-span, span)
        ax.grid(axis="y", alpha=.6)
        ax.legend(loc="lower left", bbox_to_anchor=(0, 1.015), ncol=5,
                  frameon=False, borderaxespad=0, handlelength=1.55, fontsize=8)

    residual.axhline(0, color="#82939f", lw=.65, alpha=.7)
    raw.set_ylabel("SAP flux [%]\nrelative to window")
    residual.set_ylabel("Detrended SAP [%]")
    residual.set_xlabel("Minutes from published peak")
    filename.parent.mkdir(parents=True, exist_ok=True)
    with BytesIO() as buffer:
        fig.savefig(buffer, format="png", dpi=103, facecolor="white")
        plt.close(fig)
        buffer.seek(0)
        with Image.open(buffer) as image:
            optimized = image.convert("RGB").quantize(colors=112)
            temp = filename.with_suffix(".part")
            optimized.save(temp, format="PNG", optimize=True)
            temp.replace(filename)


def render_sector(record, matches, neighbor_peaks, sources, out):
    candidates = sources[record["filename"]]
    for path in candidates:
        if detector.file_sha256(path) == record["sha256"]:
            break
    else:
        raise ValueError(f"No original FITS with the recorded hash: {record['filename']}")
    lightcurve = detector.load_lightcurve(path, CONFIG)
    config, status = detector.calibrated_config(lightcurve, CONFIG)
    if status != "validated":
        raise ValueError(f"No validated SAP operating point for {record['filename']}")
    prepared, _ = detector.prepare(lightcurve, config)
    for row in matches.itertuples(index=False):
        destination = out / f"sector_{row.Sec:02d}" / row.recovery / f"{row.catalog_event_id}.png"
        plot_peak(row, prepared, record["cadence_seconds"] / 60, neighbor_peaks, destination)
    return len(matches)


def indexes(out, missed):
    overview = [
        "# Missed Günther single peaks", "",
        "One image per rejected match or published peak without a one-to-one match within five minutes.",
        "Each image shows quality-filtered SAP photometry and BANSHEE's detrended residual.",
        "The shaded area is the ±5 minute matching window. Dotted blue lines mark neighboring catalog components.",
        "", "| Sector | Rejected match | No peak match |", "| --- | ---: | ---: |",
    ]
    for sector, group in missed.groupby("Sec"):
        links = []
        for tier, label in MISSES.items():
            subset = group.loc[group.recovery.eq(tier)].sort_values(
                ["observed_peak_snr", "TIC", "Outbst"], ascending=[False, True, True])
            folder = Path(f"sector_{sector:02d}") / tier
            links.append(f"[{len(subset)}]({folder.as_posix()}/README.md)")
            lines = [f"# Sector {sector} · {label}", "",
                     f"{len(subset)} images, ordered by observed peak SNR.", "",
                     "| Image | Observed peak SNR | Published peak (BTJD) |",
                     "| --- | ---: | ---: |"]
            for row in subset.itertuples(index=False):
                key = row.catalog_event_id
                snr = f"{row.observed_peak_snr:.1f}" if np.isfinite(row.observed_peak_snr) else "—"
                lines.append(f"| [{key}]({key}.png) | {snr} | {row.catalog_peak_btjd:.6f} |")
            (out / folder / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        overview.append(f"| {sector} | {' | '.join(links)} |")
    overview += ["", f"Total images: **{len(missed):,}**.", ""]
    (out / "README.md").write_text("\n".join(overview), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--search-root", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    run, out = args.run_directory, args.output
    protocol = json.loads((run / "protocol.json").read_text())
    if detector.file_sha256(detector.__file__) != protocol["detector_sha256"]:
        raise ValueError("Use the detector version recorded in protocol.json")
    table = pd.read_parquet(run / "catalog_matches_5min.parquet")
    missed = table.loc[table.simple & table.recovery.isin(MISSES)].copy()
    assert int(table.simple.sum()) == protocol["single_peaks"]
    assert not missed.catalog_event_id.duplicated().any()
    summary = json.loads((run / "summary.json").read_text())["cohorts"]["All peaks"]
    assert missed.recovery.value_counts().to_dict() == {tier: summary[tier] for tier in MISSES}
    catalog_peaks = table.groupby(["TIC", "Sec"]).catalog_peak_btjd.apply(np.asarray)
    manifest = {(record["tic"], record["sector"]): record
                for record in json.loads((run / "source_manifest.json").read_text())}
    groups = dict(tuple(missed.groupby(["TIC", "Sec"])))
    names = {manifest[key]["filename"] for key in groups}
    sources = {name: [] for name in names}
    for root in args.search_root:
        for path in root.rglob("*_lc.fits"):
            if path.name in sources:
                sources[path.name].append(path)
    if missing := [name for name, paths in sources.items() if not paths]:
        raise FileNotFoundError(f"{len(missing)} source FITS files missing, including {missing[0]}")
    jobs = [(manifest[key], group, catalog_peaks[key], sources, out) for key, group in groups.items()]
    assert sum(detector.bounded_jobs(render_sector, jobs, args.workers, "Plotting missed peaks")) == len(missed)
    expected = {f"sector_{row.Sec:02d}/{row.recovery}/{row.catalog_event_id}.png"
                for row in missed.itertuples(index=False)}
    actual = {path.relative_to(out).as_posix() for path in out.rglob("*.png")}
    assert actual == expected
    indexes(out, missed)
    print(f"Verified {len(actual):,} plots across {len(groups):,} target-sector pairs")


if __name__ == "__main__":
    main()
