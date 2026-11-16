"""Render paired parameter recovery and candidate efficiency from the frozen run."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
from scipy.ndimage import gaussian_filter1d

CADENCES = (20, 120, 200, 600, 1800)
STAGES = {"pre": "Injected before detrending", "post": "Injected after detrending"}
COLORS = {"pre": "#167580", "post": "#ac4f2e", "accepted": "#126c78", "followup": "#c56232"}
PARAMETERS = (
    ("amplitude", "Peak amplitude (%)", 100.),
    ("fwhm_minutes", "FWHM (min)", 1.),
    ("ed_seconds", "Equivalent duration (s)", 1.),
)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10.5, "axes.spines.top": False,
    "axes.spines.right": False, "axes.titleweight": "medium", "axes.edgecolor": "#93a1aa",
    "axes.labelcolor": "#26343f", "text.color": "#182733", "grid.color": "#dde4e8",
    "xtick.color": "#26343f", "ytick.color": "#26343f", "pdf.fonttype": 42,
})


def is_plottable(df, key):
    return (df.snr.gt(0) & df.tier.isin(["accepted", "followup"])
            & ~df.control_tier.isin(["accepted", "followup"])
            & np.isfinite(df["truth_" + key]) & np.isfinite(df["fit_" + key])
            & df["truth_" + key].gt(0) & df["fit_" + key].gt(0))


def log_limits(frame, key, scale):
    a = frame.loc[is_plottable(frame, key), ["truth_" + key, "fit_" + key]].to_numpy() * scale
    if not len(a):
        return .1, 10
    lo, hi = np.log10([np.min(a), np.max(a)])
    pad = max(.08, .038 * (hi-lo))
    return 10**(lo-pad), 10**(hi+pad)


def parameter_page(frame, cadence, exposure):
    page = frame[frame.cadence.eq(cadence)]
    fig, panels = plt.subplots(2, 3, figsize=(12.8, 8.45))
    fig.subplots_adjust(left=.098, bottom=.115, right=.98, top=.802, hspace=.36, wspace=.28)
    fig.text(.075, .95, f"Injected vs fitted flare parameters  |  {cadence} s",
             fontsize=20, weight="semibold")
    fig.text(.075, .909, "Single-flare injections on real TESS SAP light-curve windows",
             fontsize=11, color="#50616e")
    legend = [
        Line2D([], [], marker="o", ls="", color=COLORS["accepted"], label="Accepted", markersize=6),
        Line2D([], [], marker="x", ls="", color=COLORS["followup"], label="Follow-up", markersize=6),
        Line2D([], [], ls="--", color="#64717c", label="Perfect recovery (1:1)", lw=1),
    ]
    fig.legend(handles=legend, loc="upper left", bbox_to_anchor=(.066, .891),
               ncol=3, frameon=False, borderaxespad=0.35)
    for i, (stage, stage_label) in enumerate(STAGES.items()):
        rows = page[page.stage.eq(stage)]
        fig.text(.075, .828 if i == 0 else .437, stage_label, fontsize=11.5, weight="semibold")
        for j, (key, label, scale) in enumerate(PARAMETERS):
            ax = panels[i, j]
            valid = rows[is_plottable(rows, key)]
            x, y = "truth_" + key, "fit_" + key
            lim = log_limits(page, key, scale)
            ax.plot(lim, lim, "--", color="#64717c", lw=1, zorder=1)
            for tier, marker in (("accepted", "o"), ("followup", "x")):
                q = valid[valid.tier.eq(tier)]
                ax.scatter(q[x]*scale, q[y]*scale, marker=marker, s=14 if tier == "accepted" else 23,
                           color=COLORS[tier], linewidths=0 if tier == "accepted" else .85,
                           alpha=.34 if tier == "accepted" else .84, rasterized=True, zorder=3)
            if key == "fwhm_minutes":
                ax.axvline(1.5 * exposure/60, color="#77849f", linestyle=":", lw=1, zorder=2)
            ax.set(xscale="log", yscale="log", xlim=lim, ylim=lim,
                   xlabel="Injected " + label, ylabel="Fitted " + label)
            ax.grid(alpha=.5, which="major")
            ax.set_axisbelow(True)
            ax.text(.985, .035, f"n = {len(valid):,}", transform=ax.transAxes,
                    va="bottom", ha="right", fontsize=9, color="#4b5c69",
                    bbox={"facecolor":"white","edgecolor":"none","alpha":.86,"pad":2})
    fig.text(.075, .04, "Dotted FWHM line = 1.5 exposures. Only matched fits with positive fitted values appear here.",
             fontsize=9, color="#50616e")
    return fig


def efficiency_page(positive):
    fig, axes = plt.subplots(2, 3, figsize=(12.8, 8.4), sharey=True)
    fig.subplots_adjust(left=.083, bottom=.10, right=.98, top=.814, hspace=.40, wspace=.23)
    fig.text(.07, .95, "Single-flare recovery vs injected peak SNR", fontsize=20, weight="semibold")
    fig.text(.07, .908, "32 paired profiles per cadence and SNR  |  BANSHEE Final 2026.11.16.edge1",
             fontsize=11, color="#50616e")
    legend = [Line2D([], [], color=COLORS[k], lw=2, label=v) for k, v in STAGES.items()]
    legend += [Line2D([], [], color="#697681", lw=1.4, ls=s, label=label)
               for s, label in [("-", "Accepted"), ("--", "Accepted + follow-up")]]
    fig.legend(handles=legend, loc="upper left", ncol=4, frameon=False, bbox_to_anchor=(.064, .887))
    for cadence, ax in zip(CADENCES, axes.flat):
        sub = positive[positive.cadence.eq(cadence)]
        for stage in STAGES:
            rows = sub[sub.stage.eq(stage)].copy()
            rows["accepted"] = rows.tier.eq("accepted")
            rows["returned"] = rows.tier.isin(["accepted", "followup"])
            rates = rows.groupby("snr")[["accepted", "returned"]].mean()
            for field, style in [("accepted", "-"), ("returned", "--")]:
                ax.plot(rates.index, gaussian_filter1d(rates[field], .95, mode="nearest"),
                        style, color=COLORS[stage], lw=1.8)
                if field == "accepted":
                    ax.scatter(rates.index, rates[field], color=COLORS[stage], s=8, alpha=.42)
        ax.set(xlim=(0, 30), ylim=(-.02, 1.04), xlabel="Injected peak SNR", title=f"{cadence} s")
        ax.set_yticks([0,.25,.5,.75,1])
        ax.grid(axis="y", alpha=.8)
        ax.yaxis.set_major_formatter(PercentFormatter(1))
    for ax in axes[:, 0]:
        ax.set_ylabel("Recovered injections")
    axes.flat[-1].axis("off")
    axes.flat[-1].text(.04,.82,"How to read this page",fontsize=13,weight="semibold",transform=axes.flat[-1].transAxes)
    axes.flat[-1].text(.04,.67,"Dots show the measured accepted fractions.\n"
                                "Lines smooth across adjacent SNR levels.\n\n"
                                "The post arm holds the pre-injection trend\n"
                                "and noise fixed. Both arms use the same\n"
                                "raw-flux checks and detection thresholds.",
                       transform=axes.flat[-1].transAxes, va="top", fontsize=11, linespacing=1.42)
    fig.text(.07, .037, "Peak SNR uses the sampled signal above the local pre-injection noise. SNR = 0 controls are tabulated separately.",
             fontsize=9, color="#50616e")
    return fig


def save_summaries(frame, out):
    positive = frame[frame.snr.gt(0)].copy()
    records = []
    for (cadence, stage), group in positive.groupby(["cadence", "stage"]):
        for low, high in [(1,30), (3,30), (10,30), (20,30), (30,30)]:
            rows = group[group.snr.between(low, high)]
            record = dict(cadence=cadence, stage=stage, min_snr=low, max_snr=high, n=len(rows))
            for tier in ("accepted", "followup", "rejected", "missed"):
                record[tier] = int(rows.tier.eq(tier).sum())
            for key, _, _ in PARAMETERS:
                available = rows[is_plottable(rows, key)]
                ratio = available["fit_" + key] / available["truth_" + key]
                record[key+"_plotted"] = len(ratio)
                for quantile in [.16, .50, .84]:
                    record[key+f"_fit_over_injected_p{round(quantile*100):02}"] = ratio.quantile(quantile)
            eligible = rows[is_plottable(rows, "amplitude")]
            record["median_absolute_peak_error_seconds"] = (
                (eligible.fit_peak_phase_minutes - eligible.truth_peak_phase_minutes).abs().median() * 60)
            records.append(record)
    summary = pd.DataFrame(records)
    summary.to_csv(out / "recovery_summary.csv", index=False)
    zero = frame[frame.snr.eq(0)].groupby(["cadence", "stage", "tier"]).size().rename("n").reset_index()
    zero.to_csv(out / "zero_snr_controls.csv", index=False)
    grid = positive.groupby(["cadence", "stage", "snr", "tier"]).size().unstack(fill_value=0)
    grid.to_csv(out / "recovery_by_snr.csv")
    return positive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path, help="Results file and frozen plan directory")
    out = parser.parse_args().run_directory
    frame = pd.read_parquet(out / "injection_results.parquet")
    assert len(frame) == 9920 and not frame.duplicated(["case_id", "stage", "snr"]).any()
    assert frame.groupby(["cadence", "stage", "snr"]).size().eq(32).all()
    assert np.allclose(frame.snr, frame.realized_snr, rtol=0, atol=1e-10)
    assert not frame.edge_case.eq(True).any()
    doc = json.loads((out / "plan.json").read_text())
    cases = pd.DataFrame(doc["cases"])
    assert len(cases) == 160 and cases.placement_fluence.min() >= .996
    positive = save_summaries(frame, out)
    output = out / "injection_recovery_1to1.pdf"
    with PdfPages(output, metadata={"Title": "BANSHEE paired injection recovery",
                                    "Subject": "Before and after detrending at five TESS cadences"}) as pdf:
        fig = efficiency_page(positive)
        fig.savefig(out / "efficiency_by_cadence.png", dpi=170)
        pdf.savefig(fig)
        plt.close(fig)
        for cadence in CADENCES:
            exposure = cases[cases.cadence.eq(cadence)].exposure_seconds.median()
            fig = parameter_page(frame, cadence, exposure)
            fig.savefig(out / f"one_to_one_{cadence}s.png", dpi=175)
            pdf.savefig(fig)
            plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
