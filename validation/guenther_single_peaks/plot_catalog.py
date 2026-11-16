"""Plot single-peak catalog recovery and matched-parameter agreement."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

TIERS = ("accepted", "followup", "rejected", "no_peak_match")
COLORS = ("#117d78", "#d18a3b", "#b1667c", "#d9e2ea")
LABELS = ("Accepted", "Follow-up", "Rejected", "No peak match")
BINS = [-np.inf, 3, 5, 8, 12, 20, np.inf]
SNR_LABELS = ["<3", "3–5", "5–8", "8–12", "12–20", "≥20"]


def counts(frame):
    values = frame.recovery.value_counts().reindex(TIERS, fill_value=0)
    return {"n": len(frame), **{key: int(value) for key, value in values.items()}}


def summary_groups(frame):
    return {
        "All peaks": frame,
        "Covered": frame.loc[frame.covered],
        "Interior": frame.loc[frame.interior],
    }


def outcome_bars(ax, subsets):
    rows = pd.DataFrame([counts(sample) for sample in subsets.values()])
    positions = np.arange(len(rows))
    bottom = np.zeros(len(rows))
    for tier, color in zip(TIERS, COLORS):
        fraction = (rows[tier] / rows.n.where(rows.n > 0)).fillna(0).to_numpy()
        ax.bar(positions, fraction, bottom=bottom, color=color, width=.69, zorder=3)
        bottom += fraction
    for x, n in zip(positions, rows.n):
        ax.text(x, 1.023, f"n={n:,}", ha="center", fontsize=9, color="#50616e")
    ax.set(xticks=positions, xticklabels=subsets.keys(), ylim=(0, 1.09))
    ax.set_yticks(np.linspace(0, 1, 5))
    ax.grid(axis="y", alpha=.55)
    ax.set_axisbelow(True)
    ax.yaxis.set_major_formatter(PercentFormatter(1))


def parameter_agreement(ax, frame, catalog_key, fit_key, label, scale):
    usable = frame.loc[frame.recovery.isin(TIERS[:2])].copy()
    usable["x"] = usable[catalog_key] * scale
    usable["y"] = usable[fit_key] * scale
    usable = usable[np.isfinite(usable.x) & np.isfinite(usable.y) & usable.x.gt(0) & usable.y.gt(0)]
    if len(usable):
        low, high = np.log10([usable[["x", "y"]].min().min(), usable[["x", "y"]].max().max()])
        pad = max(.08, (high-low)*.03)
        lim = (10**(low-pad), 10**(high+pad))
    else:
        lim = (.1, 10)
    ax.plot(lim, lim, "--", color="#627682", lw=1.1, zorder=2)
    for tier, color, marker, size, alpha in zip(TIERS[:2], COLORS, ["o", "x"], [13, 23], [.39, .8]):
        rows = usable.loc[usable.recovery.eq(tier)]
        ax.scatter(rows.x, rows.y, s=size, marker=marker, color=color,
                   alpha=alpha, linewidths=0 if marker == "o" else .85, rasterized=True, zorder=3)
    ax.set(xscale="log", yscale="log", xlim=lim, ylim=lim,
           xlabel="Published " + label, ylabel="BANSHEE " + label)
    ax.grid(which="major", alpha=.5)
    ax.set_axisbelow(True)
    ax.text(.98, .035, f"n = {len(usable):,}", transform=ax.transAxes, ha="right", va="bottom",
            color="#50616e", bbox=dict(facecolor="white", edgecolor="none", alpha=.85, pad=2))
    stats = {}
    for tier in TIERS[:2]:
        sel = usable.loc[usable.recovery.eq(tier)]
        ratios = sel.y / sel.x
        stats[tier] = {"n": len(sel), **{
            f"ratio_p{q}": float(ratios.quantile(q/100)) if len(sel) else None for q in (16, 50, 84)
        }}
    return stats


def read_single(run, minutes):
    table = pd.read_parquet(run / f"catalog_matches_{minutes}min.parquet")
    return table.loc[table.simple].copy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--plot-directory", type=Path)
    args = parser.parse_args()
    out = args.plot_directory or args.run_directory
    out.mkdir(parents=True, exist_ok=True)
    frame = read_single(args.run_directory, 5)
    protocol = json.loads((args.run_directory / "protocol.json").read_text())
    assert len(frame) == protocol["single_peaks"] and not frame.catalog_event_id.duplicated().any()
    assert frame.recovery.isin(TIERS).all()
    selected = frame[frame.candidate_id.notna()]
    assert not selected.duplicated(["TIC", "Sec", "candidate_id"]).any()
    assert selected.peak_offset_minutes.abs().le(5 + 1e-8).all()
    interior = frame.loc[frame.interior]
    snr = pd.cut(interior.observed_peak_snr, bins=BINS, labels=SNR_LABELS, right=False)
    by_snr = {label: interior.loc[snr.eq(label)] for label in SNR_LABELS}
    if snr.isna().any():
        by_snr["Unavailable"] = interior.loc[snr.isna()]
    assert sum(map(len, by_snr.values())) == len(interior)

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False,
        "axes.spines.right": False, "axes.edgecolor": "#a7b1b9", "axes.labelcolor": "#26343f",
        "text.color": "#182733", "grid.color": "#dce5ea", "pdf.fonttype": 42,
    })
    fig, panels = plt.subplots(2, 2, figsize=(12.4, 8.6))
    fig.subplots_adjust(left=.09, right=.975, top=.79, bottom=.11, wspace=.24, hspace=.6)
    fig.text(.08, .95, "Günther single-peak flares", fontsize=22, weight="semibold")
    n_curves = frame[["TIC", "Sec"]].drop_duplicates().shape[0]
    fig.text(.08, .912, f"{len(frame):,} catalog peaks  |  {n_curves:,} TESS light curves  |  120 s SAP  |  Sectors 1 and 2",
             fontsize=11, color="#50616e")
    handles = [Line2D([], [], linestyle="", marker="s", color=color, label=label)
               for color, label in zip(COLORS, LABELS)]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.071, .899), frameon=False, ncol=4)
    top, bottom = panels
    outcome_bars(top[0], summary_groups(frame))
    outcome_bars(top[1], by_snr)
    top[0].set_title("One-to-one matches within five minutes", loc="left", pad=12)
    top[1].set_title("Recovery versus observed SNR", loc="left", pad=12)
    top[0].set_ylabel("Fraction of catalog peaks")
    top[1].set_xlabel("Local peak SNR in covered, usable interiors")
    parameters = {
        "amplitude": parameter_agreement(bottom[0], frame, "Amp", "banshee_amplitude", "peak amplitude (%)", 100),
        "fwhm": parameter_agreement(bottom[1], frame, "FWHMdt", "banshee_fwhm_days", "FWHM (min)", 1440),
    }
    bottom[0].set_title("Peak amplitude agreement", loc="left", pad=10)
    bottom[1].set_title("Width agreement", loc="left", pad=10)
    fig.text(.08, .039, "Parameter panels show matched accepted (dots) and follow-up (crosses) fits. "
             "The dashed line marks agreement with the published fit.", fontsize=9.5, color="#50616e")
    fig.savefig(out / "guenther_single_peaks.png", dpi=170)
    plt.close(fig)

    stats = dict(n_light_curves=n_curves, n_stars=int(frame.TIC.nunique()),
                 cohorts={key: counts(group) for key, group in summary_groups(frame).items()},
                 interior_by_observed_snr={key: counts(group) for key, group in by_snr.items()},
                 timing_tolerance_sensitivity={str(minutes): counts(read_single(args.run_directory, minutes))
                                               for minutes in (2, 5, 10)},
                 edge=counts(frame.loc[frame.covered & frame.edge]),
                 parameters=parameters)
    (out / "summary.json").write_text(json.dumps(stats, indent=2) + "\n")
    pd.DataFrame.from_dict(stats["interior_by_observed_snr"], orient="index").to_csv(out / "recovery_by_snr.csv",
                                                                                    index_label="observed_snr")
    frame.to_csv(out / "single_peak_results.csv", index=False)
    print(json.dumps(stats["cohorts"], indent=2))


if __name__ == "__main__":
    main()
