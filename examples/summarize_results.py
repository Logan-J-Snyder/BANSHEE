"""Summarize ordinary-CLI outputs without changing files or hiding failures."""

import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", type=Path)
    root = parser.parse_args().results_dir
    runtime = pd.read_csv(root / "runtimes.csv")
    print("Input processing status:")
    print(runtime["status"].value_counts(dropna=False).to_string())
    for filename in ("accepted_flares.csv", "followup_flares.csv", "rejected_candidates.csv"):
        frame = pd.read_csv(root / filename, dtype={"gdr3_id": "string"})
        usable = frame["parameter_usable"].astype("string").str.lower().eq("true").sum()
        print(f"{filename}: {len(frame)} rows, {usable} parameter-usable")
    if runtime["status"].ne("complete").any():
        print("WARNING: Incomplete inputs remain. Inspect runtimes.csv and product summaries.")


if __name__ == "__main__":
    main()
