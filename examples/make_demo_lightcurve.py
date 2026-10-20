"""Generate local matching-template data; never download or use a real source."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import BANSHEE_Lite_Chi as banshee


def generate(folder, cadence_seconds=120, seed=20261020):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "0000000000000000000_S01_SPOC_demo.csv"
    truth_path = folder / "injection_truth.json"
    if path.exists() or truth_path.exists():
        raise FileExistsError("Demo files already exist; choose a fresh output directory.")
    exposure = cadence_seconds / 86400
    t = 2000 + np.arange(0, 2, exposure)
    peak, width, amplitude, sigma = 2001.0, 20 / 1440, 0.03, 0.0005
    flux = 1 + banshee.flare_model(t, peak, width, amplitude, exposure)
    flux += np.random.default_rng(seed).normal(0, sigma, len(t))
    pd.DataFrame({
        "time": t, "flux": flux, "flux_err": sigma,
        "quality": 0, "exposure_days": exposure,
    }).to_csv(path, index=False)
    truth = dict(
        synthetic=True, source_id_is_placeholder=True, seed=seed,
        cadence_seconds=cadence_seconds, tpeak_btjd=peak, fwhm_days=width,
        amplitude=amplitude, sigma=sigma, template="tovar2022",
        equivalent_duration_s=86400 * banshee.full_equivalent_duration_days(width, amplitude, "tovar2022"),
        note="Matching-template demonstration, not an independent completeness test.",
    )
    truth_path.write_text(json.dumps(truth, indent=2) + "\n", encoding="utf-8")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("demo_data"))
    parser.add_argument("--cadence-seconds", type=int, choices=(20, 120, 200, 600, 1800), default=120)
    parser.add_argument("--seed", type=int, default=20261020)
    args = parser.parse_args()
    print(generate(args.output_dir, args.cadence_seconds, args.seed))


if __name__ == "__main__":
    main()
