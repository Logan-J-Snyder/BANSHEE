#!/usr/bin/env python3
"""BANSHEE_Final: local-noise flare screening and empirical calibration.

Standalone revision of BANSHEE_Lite_Chi with exposure-integrated Tovar models,
proper amplitude-marginalized grid evidence, independent continuous fits,
cadence/product-specific injection controls, and audited resumable outputs.
Saved SAP cadence thresholds are built in and used by default; no calibration
step or separate calibration file is required for matching settings/groups.
Candidates excluded only by the edge-fraction cut are retained for follow-up.
Portable distribution of the existing detector; no new recovery guarantee.
See README.txt, CHANGES_SINCE_LITE_CHI.txt, and --help.

Examples (paths are relative to your current working directory):
  python BANSHEE_Final.py --input ./lightcurves --output-dir ./results
  python BANSHEE_Final.py --server --input-root ./lightcurves --output-root ./results
Linux and macOS are supported; Windows users should use WSL.

References: Pitkin et al. 2014, doi:10.1093/mnras/stu1889;
Tovar Mendoza et al. 2022, arXiv:2205.05706;
Ilin & Poppenhaeger 2022, doi:10.1093/mnras/stac1232.
"""

from __future__ import annotations

# Limit numerical-library threads before importing numpy; parallelize independent
# light curves/candidates with the explicitly requested process counts instead.
import os
import sys
import tempfile
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "banshee_mpl"))
for _thread_env in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_thread_env] = "1"


# ======================================================================
# CORE
# ======================================================================
import re
from dataclasses import dataclass, replace
from decimal import Decimal, InvalidOperation
from functools import lru_cache
from pathlib import Path
from typing import Iterable
import numpy as np
import pandas as pd
from astropy.io import fits
from scipy import special
from scipy.optimize import least_squares



@dataclass
class LightCurve:
    """One cleaned light curve and the metadata needed by the detector."""

    time: np.ndarray
    flux: np.ndarray
    flux_err: np.ndarray | None
    gdr3_id: str
    sector: int | None
    mission: str
    source_path: str
    selected_flux_column: str
    input_mode: str
    exposure_days: float | None = None
    tess_magnitude: float | None = None
    distance_pc: float | None = None
    stellar_catalog_source: str = ""
    raw_flux: np.ndarray | None = None
    quality_filter_applied: bool = False
    n_input: int = 0
    n_quality_rejected: int = 0
    n_duplicates_merged: int = 0

@dataclass
class PreparedLightCurve:
    """Normalized data, in-memory trend, residual, and per-segment scatter."""

    lightcurve: LightCurve
    normalized_flux: np.ndarray
    trend: np.ndarray
    residual: np.ndarray
    relative_flux_err: np.ndarray | None
    segment_id: np.ndarray
    segment_sigma: np.ndarray
    detrend_edge_mask: np.ndarray
    protection_mask: np.ndarray | None = None
    background_uncertain_mask: np.ndarray | None = None
    detrend_error_scale: np.ndarray | None = None
    local_baseline: np.ndarray | None = None

@dataclass(frozen=True)
class Candidate:
    """A three-sigma seed and its one-sigma expanded observed island."""

    seed_indices: np.ndarray
    event_indices: np.ndarray
    peak_index: int
    segment: int
    width_hint: float | None = None

@dataclass
class GridResult:
    """Bayesian grid values for each (tpeak, FWHM, amplitude-scale) cell."""

    tpeaks: np.ndarray
    fwhms: np.ndarray
    logbf: np.ndarray
    amplitude_mean: np.ndarray
    amplitude_sd: np.ndarray
    theta_prior_mass: np.ndarray
    amplitude_prior_mass: np.ndarray
    projected_d: np.ndarray
    projected_m: np.ndarray
    model_names: np.ndarray | None = None

def clean_id(value) -> str:
    """Return an integer-like identifier without decimal/exponent formatting."""
    if isinstance(value, (float, np.floating)) and np.isfinite(value):
        raise ValueError("Gaia identifiers must be exact strings or integers, not floats")
    if value is None:
        return ""
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "<na>", "nat"}:
        return ""
    try:
        exact = Decimal(text)
        if exact.is_finite() and exact == exact.to_integral_value():
            return str(int(exact))
    except (InvalidOperation, TypeError, ValueError, OverflowError):
        pass
    if re.fullmatch(r"[+-]?\d+", text):
        try:
            return str(int(text))
        except (TypeError, ValueError, OverflowError):
            pass
    else:
        return text
    return text

def standardize_mission(value) -> str:
    """Normalize common product labels to SPOC, QLP, or T16."""
    text = str(value or "").upper()
    for mission in ("SPOC", "QLP", "T16"):
        if mission in text:
            return mission
    if re.search(r"^TESS\d{13}-S\d{4}-\d{16}-\d{4}-S_LC", text):
        return "SPOC"
    return text or "UNKNOWN"

FILENAME_PATTERNS = (
    re.compile(
        r"^tess\d{13}-s(?P<sector>\d{4})-\d{16}-\d{4}-s_lc\.fits$",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?P<gdr3>\d+)_S(?P<sector>\d+)(?:_[^.]*)?_(?P<mission>QLP|SPOC|T16)(?=[_.])",
        re.IGNORECASE,
    ),
    re.compile(r"Gaia_DR3_(?P<gdr3>\d+)_(?P<sector>\d{4})", re.IGNORECASE),
)

def metadata_from_path(path: Path) -> tuple[str, int | None, str]:
    """Parse Gaia DR3 ID, sector, and mission from established file names."""
    for pattern in FILENAME_PATTERNS:
        match = pattern.search(path.name)
        if match is None:
            continue
        values = match.groupdict()
        mission = values.get("mission") or standardize_mission(path.name)
        return (
            clean_id(values.get("gdr3")),
            int(values["sector"]) if values.get("sector") else None,
            standardize_mission(mission),
        )
    sector_match = re.search(
        r"(?:^|[_-])(?:s|sector[_ -]?)(\d{1,4})(?=[_.-]|$)",
        path.name,
        re.IGNORECASE,
    )
    id_match = re.search(r"^(\d{15,20})_S\d+", path.name, re.IGNORECASE)
    return (
        id_match.group(1) if id_match else "UNKNOWN",
        int(sector_match.group(1)) if sector_match else None,
        standardize_mission(path.name),
    )

def case_column(columns: Iterable[str], choices: Iterable[str]) -> str | None:
    """Find the first case-insensitive column name in a declared priority list."""
    lookup = {str(column).lower(): str(column) for column in columns}
    for choice in choices:
        if choice and str(choice).lower() in lookup:
            return lookup[str(choice).lower()]
    return None

def _finite_float(value) -> float | None:
    """Return a finite scalar float or None for missing/non-numeric metadata."""
    try:
        output = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return output if np.isfinite(output) else None

def _first_finite_value(values) -> float | None:
    """Return the first finite value from a table column."""
    for value in values:
        output = _finite_float(value)
        if output is not None:
            return output
    return None

@lru_cache(maxsize=8)
def stellar_catalog_lookup(
    catalog_paths: tuple[str, ...], distance_column_name: str
) -> dict[str, dict]:
    """Load Gaia-keyed distance/TESS-magnitude metadata once per worker.

    Distance is read from ``distance_pc`` when available and otherwise derived
    as 1000/parallax for a positive parallax in milliarcseconds. Earlier paths
    have priority, while a later table may fill a missing value.
    """
    lookup: dict[str, dict] = {}
    for path_text in catalog_paths:
        path = Path(path_text).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Stellar catalog not found: {path}")
        frame = pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path, dtype=str)
        id_column = case_column(
            frame.columns, ["gdr3_id", "gaia_dr3_id", "source_id", "member_id"]
        )
        if id_column is None:
            raise ValueError(f"Stellar catalog has no Gaia DR3 ID column: {path}")
        if pd.api.types.is_float_dtype(frame[id_column].dtype):
            raise ValueError(f"Gaia IDs must be strings or integers, never floats: {path}")
        identifiers = frame[id_column].map(clean_id)
        if not identifiers.str.fullmatch(r"[1-9][0-9]{14,19}").all() or identifiers.duplicated().any():
            raise ValueError(f"Stellar catalog requires unique, valid Gaia DR3 IDs: {path}")
        distance_column = case_column(frame.columns, [distance_column_name])
        parallax_column = case_column(frame.columns, ["parallax"])
        tmag_column = case_column(
            frame.columns, ["TESSMAG", "tess_mag", "tmag", "Tmag"]
        )
        selected = [id_column]
        for column in (distance_column, parallax_column, tmag_column):
            if column is not None and column not in selected:
                selected.append(column)
        for values in frame[selected].itertuples(index=False, name=None):
            row = dict(zip(selected, values))
            identifier = clean_id(row[id_column])
            if not identifier:
                continue
            entry = lookup.setdefault(
                identifier,
                {"distance_pc": None, "tess_magnitude": None, "source": ""},
            )
            if entry["distance_pc"] is None:
                distance = _finite_float(row.get(distance_column)) if distance_column else None
                if distance is None and parallax_column:
                    parallax = _finite_float(row.get(parallax_column))
                    if parallax is not None and parallax > 0:
                        distance = 1000.0 / parallax
                if distance is not None and distance > 0:
                    entry["distance_pc"] = distance
                    entry["source"] = str(path)
            if entry["tess_magnitude"] is None and tmag_column:
                tmag = _finite_float(row.get(tmag_column))
                if tmag is not None:
                    entry["tess_magnitude"] = tmag
                    entry["source"] = entry["source"] or str(path)
    return lookup

def stellar_metadata(identifier: str, config: Config) -> dict:
    """Return catalog metadata for one Gaia DR3 ID without fuzzy matching."""
    return stellar_catalog_lookup(
        tuple(config.stellar_catalog_paths), config.distance_column
    ).get(
        clean_id(identifier),
        {"distance_pc": None, "tess_magnitude": None, "source": ""},
    )

def _header_value(headers, key: str):
    for header in headers:
        if key in header:
            return header[key]
    return None

def exposure_from_headers(headers) -> float | None:
    """Read one live integration duration, never the full T16 time span."""
    try:
        seconds = float(_header_value(headers, "INT_TIME")) * float(
            _header_value(headers, "NUM_FRM")
        )
        if np.isfinite(seconds) and seconds > 0:
            return seconds / 86400.0
    except (TypeError, ValueError):
        pass
    try:
        seconds = float(_header_value(headers, "XPOSURE"))
        if np.isfinite(seconds) and seconds > 0:
            return seconds / 86400.0
    except (TypeError, ValueError):
        pass
    if str(_header_value(headers, "HLSPID") or "").upper() == "T16":
        return None
    try:
        value = float(_header_value(headers, "TIMEDEL"))
        unit = str(_header_value(headers, "TIMEUNIT") or "d").lower()
        if unit.startswith("s"):
            value /= 86400.0
        return value if np.isfinite(value) and value > 0 else None
    except (TypeError, ValueError):
        return None

def _find_table_hdu(hdul):
    if "LIGHTCURVE" in hdul:
        return hdul["LIGHTCURVE"]
    for hdu in hdul:
        if isinstance(hdu, fits.BinTableHDU) and hdu.data is not None:
            return hdu
    raise ValueError("No binary-table light-curve HDU was found")

def _native(values) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype.byteorder not in {"=", "|"}:
        array = array.byteswap().view(array.dtype.newbyteorder("="))
    return array

def to_btjd(time_values) -> np.ndarray:
    """Normalize numeric BJD-like timestamps to BTJD = BJD - 2457000.

    SPOC and QLP products normally store BTJD directly, while T16 stores BJD.
    The conversion keeps every output time column on one declared system.
    """
    values = np.asarray(time_values, dtype=float)
    median = float(np.nanmedian(values))
    if np.isfinite(median) and median > 2_400_000.0:
        values = values - 2_457_000.0
    return values

def _select_flux(columns, config: Config, mission: str) -> tuple[str, str]:
    if config.flux_column:
        found = case_column(columns, [config.flux_column])
        if found is None:
            raise ValueError(f"Requested flux column {config.flux_column!r} was not found")
        return found, "explicit"
    if mission == "T16":
        choices = (
            ["TFA1", "TFA2", "TFA3", "EPD1", "EPD2", "EPD3"]
            if config.already_detrended
            else ["IRM1", "IRM2", "IRM3"]
        )
    elif config.already_detrended:
        choices = [
            "detrended_residual",
            "detrended_flux",
            "det_flux",
            "residual_flux",
            "relative_flux",
        ]
    else:
        choices = (["SAP_FLUX", "KSPSAP_FLUX", "raw_flux", "flux"] if mission == "QLP"
                   else ["SAP_FLUX", "PDCSAP_FLUX", "raw_flux", "flux"] if mission == "SPOC"
                   else ["SAP_FLUX", "PDCSAP_FLUX", "KSPSAP_FLUX", "raw_flux", "flux"])
    found = case_column(columns, choices)
    if found is None:
        requested = "detrended" if config.already_detrended else "raw/PDC"
        raise ValueError(f"No usable {requested} flux column. Columns: {list(columns)}")
    return found, "detrended" if config.already_detrended else "raw"

def _magnitudes_to_relative(magnitude, magnitude_err=None):
    magnitude = np.asarray(magnitude, dtype=float)
    reference = float(np.nanmedian(magnitude))
    flux = 10.0 ** (-0.4 * (magnitude - reference))
    if magnitude_err is None:
        return flux, None
    error = np.asarray(magnitude_err, dtype=float)
    return flux, np.log(10.0) * 0.4 * flux * error

def load_fits(path: Path, config: Config) -> LightCurve:
    """Load SPOC, QLP, or T16 FITS data under one explicit flux contract."""
    gdr3, sector, mission = metadata_from_path(path)
    with fits.open(path, memmap=False) as hdul:
        hdu = _find_table_hdu(hdul)
        columns = list(hdu.columns.names)
        time_name = case_column(
            columns, ["TIME", "time", "BTJD", "time_btjd", "TMID_BJD"]
        )
        if time_name is None:
            raise ValueError("No time column")
        if case_column(columns, ["TMID_BJD"]) is not None:
            mission = "T16"
        flux_name, input_mode = _select_flux(columns, config, mission)
        suffix = re.search(r"([123])$", flux_name)
        err_choices = [
            f"IRE{suffix.group(1)}" if mission == "T16" and suffix else "",
            f"{flux_name}_ERR",
        ]
        err_name = case_column(columns, err_choices)
        quality_name = case_column(columns, ["QUALITY", "quality"])
        time_values = to_btjd(_native(hdu.data[time_name]))
        flux_values = _native(hdu.data[flux_name]).astype(float)
        error_values = (
            _native(hdu.data[err_name]).astype(float) if err_name is not None else None
        )
        if mission == "T16" and flux_name.upper().startswith(("IRM", "EPD", "TFA")):
            flux_values, error_values = _magnitudes_to_relative(flux_values, error_values)
            flux_name = f"{flux_name}_relative_flux"
        quality = (
            _native(hdu.data[quality_name]) if quality_name is not None else None
        )
        headers = [hdu.header, hdul[0].header]
        try:
            header_sector = int(_header_value(headers, "SECTOR"))
            if header_sector >= 0:
                sector = header_sector
        except (TypeError, ValueError):
            pass
        # Generic GAIA_ID headers may be DR2. Never replace a DR3 filename ID
        # with one, and never substitute a TIC identifier for a Gaia ID.
        if gdr3 in {"", "UNKNOWN"}:
            gdr3 = clean_id(_header_value(headers, "GAIA_DR3")) or "UNKNOWN"
        if case_column(columns, ["PDCSAP_FLUX"]) is not None:
            mission = "SPOC"
        exposure = exposure_from_headers(headers)
        tess_magnitude = _finite_float(_header_value(headers, "TESSMAG"))
        if tess_magnitude is None:
            tess_magnitude = _finite_float(_header_value(headers, "TMAG"))
        catalog = stellar_metadata(gdr3, config)
        if tess_magnitude is None:
            tess_magnitude = catalog["tess_magnitude"]
        distance_pc = catalog["distance_pc"]
    return clean_lightcurve(
        time_values,
        flux_values,
        error_values,
        quality,
        path,
        config,
        gdr3,
        sector,
        mission,
        flux_name,
        input_mode,
        exposure,
        tess_magnitude,
        distance_pc,
        catalog["source"],
    )

def load_table(path: Path, config: Config) -> LightCurve:
    """Load CSV/parquet light curves using the same contract as FITS input."""
    frame = pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path, dtype=str)
    gdr3, sector, mission = metadata_from_path(path)
    gcol = case_column(frame.columns, ["gdr3_id", "gaia_dr3_id", "source_id", "member_id"])
    scol = case_column(frame.columns, ["sector"])
    mcol = case_column(frame.columns, ["mission", "lc_type"])
    for column in (gcol, scol, mcol):
        if column and frame[column].dropna().nunique() > 1:
            raise ValueError(f"Mixed products in {column}; supply one star/sector/stream")
    if gcol and pd.api.types.is_float_dtype(frame[gcol].dtype):
        raise ValueError("Gaia identifiers must be exact strings or integers, not floats")
    if gcol and len(frame[gcol].dropna()):
        gdr3 = clean_id(frame[gcol].dropna().iloc[0])
    if scol and len(frame[scol].dropna()):
        try:
            sector = int(frame[scol].dropna().iloc[0])
        except (TypeError, ValueError):
            pass
    if mcol and len(frame[mcol].dropna()):
        mission = standardize_mission(frame[mcol].dropna().iloc[0])
    time_name = case_column(frame.columns, ["TIME", "time", "BTJD", "time_btjd", "bjd"])
    if time_name is None:
        raise ValueError("No time column")
    flux_name, input_mode = _select_flux(frame.columns, config, mission)
    err_name = case_column(
        frame.columns,
        [f"{flux_name}_err"],
    )
    quality_name = case_column(frame.columns, ["QUALITY", "quality"])
    exposure_name = case_column(
        frame.columns, ["exposure_duration_days", "exposure_days", "timedel"]
    )
    exposure = None
    if exposure_name and len(frame[exposure_name].dropna()):
        values = pd.to_numeric(frame[exposure_name], errors="coerce").to_numpy(float)
        if not np.all(np.isfinite(values)) or np.any(values <= 0):
            raise ValueError("Exposure durations must be finite and positive for every row")
        if not np.allclose(values, values[0], rtol=1e-10, atol=0):
            raise ValueError("Mixed exposures are unsupported; split into separate streams")
        exposure = float(values[0])
    tmag_name = case_column(frame.columns, ["TESSMAG", "tess_mag", "tmag", "Tmag"])
    distance_name = case_column(frame.columns, [config.distance_column])
    tess_magnitude = (
        _first_finite_value(frame[tmag_name]) if tmag_name is not None else None
    )
    distance_pc = (
        _first_finite_value(frame[distance_name]) if distance_name is not None else None
    )
    catalog = stellar_metadata(gdr3, config)
    if tess_magnitude is None:
        tess_magnitude = catalog["tess_magnitude"]
    if distance_pc is None:
        distance_pc = catalog["distance_pc"]
    return clean_lightcurve(
        to_btjd(frame[time_name].to_numpy(float)),
        frame[flux_name].to_numpy(float),
        frame[err_name].to_numpy(float) if err_name else None,
        frame[quality_name].to_numpy() if quality_name else None,
        path,
        config,
        gdr3,
        sector,
        mission,
        flux_name,
        input_mode,
        exposure,
        tess_magnitude,
        distance_pc,
        catalog["source"],
    )

def clean_lightcurve(
    time_values,
    flux_values,
    flux_errors,
    quality,
    path: Path,
    config: Config,
    gdr3: str,
    sector: int | None,
    mission: str,
    flux_name: str,
    input_mode: str,
    exposure: float | None,
    tess_magnitude: float | None,
    distance_pc: float | None,
    stellar_catalog_source: str,
) -> LightCurve:
    """Always require QUALITY==0 when supplied, then sort and merge times."""
    time_values = np.asarray(time_values, dtype=float)
    flux_values = np.asarray(flux_values, dtype=float)
    flux_errors = None if flux_errors is None else np.asarray(flux_errors, dtype=float)
    if exposure is not None and (not np.isfinite(exposure) or exposure <= 0):
        raise ValueError("Exposure duration must be finite and positive")
    if time_values.ndim != 1 or time_values.shape != flux_values.shape:
        raise ValueError("Time and flux must be matching one-dimensional arrays")
    if flux_errors is not None and flux_errors.shape != time_values.shape:
        raise ValueError("Uncertainties must match the selected flux")
    n_input = len(time_values)
    good = np.isfinite(time_values) & np.isfinite(flux_values)
    quality_applied = quality is not None
    n_quality = 0
    if quality_applied:
        quality = pd.to_numeric(np.asarray(quality), errors="coerce")
        if quality.shape != time_values.shape:
            raise ValueError("QUALITY must have one value per observation")
        quality_good = quality == 0
        n_quality = int(np.sum(good & ~quality_good))
        good &= quality_good
    time_values = time_values[good]
    flux_values = flux_values[good]
    if flux_errors is not None and flux_errors.shape == good.shape:
        flux_errors = flux_errors[good]
    else:
        flux_errors = None
    if len(time_values) < 5:
        raise ValueError("Fewer than five clean cadences")
    order = np.argsort(time_values, kind="mergesort")
    time_values, flux_values = time_values[order], flux_values[order]
    if flux_errors is not None:
        flux_errors = flux_errors[order]
    unique, inverse, counts = np.unique(time_values, return_inverse=True, return_counts=True)
    duplicates = int(len(time_values) - len(unique))
    if duplicates:
        merged_flux = np.empty(len(unique))
        merged_err = np.empty(len(unique)) if flux_errors is not None else None
        for index in range(len(unique)):
            members = inverse == index
            if flux_errors is not None:
                valid = members & np.isfinite(flux_errors) & (flux_errors > 0)
                if np.any(valid):
                    weight = 1.0 / np.square(flux_errors[valid])
                    merged_flux[index] = np.sum(weight * flux_values[valid]) / np.sum(weight)
                    merged_err[index] = np.sqrt(1.0 / np.sum(weight))
                    continue
            merged_flux[index] = np.nanmedian(flux_values[members])
            if merged_err is not None:
                merged_err[index] = np.nan
        time_values, flux_values, flux_errors = unique, merged_flux, merged_err
    if len(time_values) < 5:
        raise ValueError("Fewer than five unique clean cadences")
    return LightCurve(
        time=time_values,
        flux=flux_values,
        flux_err=flux_errors,
        gdr3_id=clean_id(gdr3) or "UNKNOWN",
        sector=sector,
        mission=standardize_mission(mission),
        source_path=str(path),
        selected_flux_column=flux_name,
        input_mode=input_mode,
        exposure_days=exposure,
        tess_magnitude=tess_magnitude,
        distance_pc=distance_pc,
        stellar_catalog_source=stellar_catalog_source,
        raw_flux=flux_values.copy(),
        quality_filter_applied=quality_applied,
        n_input=n_input,
        n_quality_rejected=n_quality,
        n_duplicates_merged=duplicates,
    )

def load_lightcurve(path: Path, config: Config) -> LightCurve:
    """Dispatch a supported FITS, CSV, or parquet light curve."""
    suffix = path.suffix.lower()
    if suffix in {".fits", ".fit", ".fts"}:
        return load_fits(path, config)
    if suffix in {".csv", ".parquet"}:
        return load_table(path, config)
    raise ValueError(f"Unsupported light-curve type: {path}")

def robust_std(values) -> float:
    """Gaussian-equivalent MAD with a standard-deviation fallback."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan
    median = np.median(values)
    sigma = 1.482602218505602 * np.median(np.abs(values - median))
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = np.std(values, ddof=1) if values.size > 1 else np.nan
    return float(sigma)

def sigma_clip_location_scale(
    values, clip_sigma: float = 3.0, max_iter: int = 10
) -> tuple[float, float, np.ndarray]:
    """Iteratively estimate median and MAD scatter after symmetric clipping."""
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    keep = finite.copy()
    for _ in range(max_iter):
        sample = values[keep]
        if sample.size < 5:
            break
        center = float(np.median(sample))
        sigma = robust_std(sample - center)
        if not np.isfinite(sigma) or sigma <= 0:
            break
        updated = finite & (np.abs(values - center) <= clip_sigma * sigma)
        if np.array_equal(updated, keep):
            break
        keep = updated
    sample = values[keep]
    if sample.size == 0:
        return np.nan, np.nan, keep
    center = float(np.median(sample))
    return center, robust_std(sample - center), keep

def cadence_days(time_values) -> float:
    """Median positive adjacent cadence in days."""
    differences = np.diff(np.asarray(time_values, dtype=float))
    differences = differences[np.isfinite(differences) & (differences > 0)]
    return float(np.median(differences)) if differences.size else np.nan

def split_segments(time_values, gap_factor: float = 5.0) -> list[np.ndarray]:
    """Split at gaps larger than ``gap_factor`` times the native cadence."""
    time_values = np.asarray(time_values, dtype=float)
    cadence = cadence_days(time_values)
    threshold = gap_factor * cadence if np.isfinite(cadence) else 2.0 / 24.0
    boundaries = np.flatnonzero(np.diff(time_values) > threshold) + 1
    return [part for part in np.split(np.arange(len(time_values)), boundaries) if len(part)]

TOVAR_PEAK = 0.9501548980476425

TOVAR_PEAK_SHIFT = -0.0025694354958475146

TOVAR_DIMENSIONLESS_FWHM = 1.0866563463391445

def tovar_template(t, amplitude: float = 1.0):
    """Continuous Tovar Mendoza et al. (2022) dimensionless flare template."""
    raw_t = np.asarray(t, dtype=float) + TOVAR_PEAK_SHIFT
    A, B, C = 0.9687734504375167, -0.251299705922117, 0.22675974948468916
    rates = (0.15551880775110513, 1.2150539528490194)
    fractions = (0.12695865022878844, 1.0 - 0.12695865022878844)
    result = np.zeros_like(raw_t)
    for rate, fraction in zip(rates, fractions):
        z = (B - raw_t) / C + C * rate / 2.0
        log_term = (-rate * raw_t + ((B / C) + rate * C / 2.0) ** 2
                    + special.log_ndtr(-np.sqrt(2.0) * z))
        result += np.sqrt(np.pi) * A * C * fraction * np.exp(log_term)
    return result / TOVAR_PEAK * amplitude

def _exgaussian_cdf(x, rate: float):
    """Stable analytic CDF with binary64-saturated Gaussian tails.

    Above 9 sigma the omitted Gaussian survival is <1.2e-19; below -39
    sigma it underflows binary64. Paired tests bound the CDF change to one
    machine epsilon; no temporal integration or model grid is approximated.
    """
    x = np.asarray(x, dtype=float)
    center = -0.251299705922117
    width = 0.22675974948468916 / np.sqrt(2.0)
    standardized = (x - center) / width
    cdf = np.zeros_like(x)
    late = standardized-rate*width >= 9.0
    cdf[late] = -np.expm1(-rate*(x[late]-center)+0.5*(rate*width)**2)
    middle = (~late) & (standardized > -39.0)
    z = standardized[middle]
    log_first = special.log_ndtr(z)
    log_second = (-rate*(x[middle]-center)+0.5*(rate*width)**2
                  +special.log_ndtr(z-rate*width))
    cdf[middle] = np.exp(log_first)*(-np.expm1(np.minimum(log_second-log_first,0.0)))
    return np.clip(cdf,0.0,1.0)

def tovar_antiderivative(t, amplitude: float = 1.0):
    """Cumulative integral of the normalized Tovar template."""
    raw_t = np.asarray(t, dtype=float) + TOVAR_PEAK_SHIFT
    A, B, C = 0.9687734504375167, -0.251299705922117, 0.22675974948468916
    rates = (0.15551880775110513, 1.2150539528490194)
    fractions = (0.12695865022878844, 1.0 - 0.12695865022878844)
    result = np.zeros_like(raw_t)
    common = np.sqrt(np.pi) * A * C * np.exp((B / C) ** 2)
    for rate, fraction in zip(rates, fractions):
        result += common * fraction / rate * _exgaussian_cdf(raw_t, rate)
    return np.nan_to_num(result * amplitude / TOVAR_PEAK, nan=0.0, posinf=0.0)

def flare_model(
    time_values,
    tpeak: float,
    fwhm_days: float,
    amplitude: float = 1.0,
    exposure_days: float | None = None,
    model_name: str = "tovar2022",
):
    """Evaluate the exact exposure-averaged analytic flare model."""
    time_values = np.asarray(time_values, dtype=float)
    function, antiderivative, dimensionless_fwhm = template_functions(model_name)
    scale = float(fwhm_days) / dimensionless_fwhm
    if not np.isfinite(scale) or scale <= 0:
        return np.zeros_like(time_values)
    if exposure_days is None or not np.isfinite(exposure_days) or exposure_days <= 0:
        return function((time_values - tpeak) / scale, amplitude)
    lower = (time_values - exposure_days / 2 - tpeak) / scale
    upper = (time_values + exposure_days / 2 - tpeak) / scale
    result = scale * (antiderivative(upper, amplitude) - antiderivative(lower, amplitude)) / exposure_days
    return np.maximum(np.nan_to_num(result), 0.0)

def flare_model_batch(
    time_values,
    tpeaks,
    fwhms,
    exposure_days: float | None,
    model_name: str,
):
    """Broadcasted exact exposure integration for paired peak/width cells."""
    time_values = np.asarray(time_values, dtype=float)
    tpeaks = np.asarray(tpeaks, dtype=float)
    fwhms = np.asarray(fwhms, dtype=float)
    function, antiderivative, dimensionless_fwhm = template_functions(model_name)
    scales = fwhms / dimensionless_fwhm
    if exposure_days is None or not np.isfinite(exposure_days) or exposure_days <= 0:
        return function((time_values[:, None] - tpeaks[None, :]) / scales[None, :])
    lower = (time_values[:, None] - exposure_days / 2 - tpeaks[None, :]) / scales[None, :]
    upper = (time_values[:, None] + exposure_days / 2 - tpeaks[None, :]) / scales[None, :]
    result = scales[None, :] * (antiderivative(upper) - antiderivative(lower)) / exposure_days
    return np.maximum(np.nan_to_num(result), 0.0)

@lru_cache(maxsize=8)
def template_total_integral(model_name: str) -> float:
    """Dimensionless area of the selected unit-amplitude template."""
    _, antiderivative, _ = template_functions(model_name)
    return float(antiderivative(np.array([1e4]))[0] - antiderivative(np.array([-1e4]))[0])

def full_equivalent_duration_days(fwhm_days: float, amplitude: float, model_name: str) -> float:
    """Analytic full-template equivalent duration in days."""
    _, _, dimensionless_fwhm = template_functions(model_name)
    return float(amplitude * fwhm_days / dimensionless_fwhm * template_total_integral(model_name))

@lru_cache(maxsize=32)
def template_quantile_time(quantile: float, model_name: str) -> float:
    """Dimensionless time at a requested cumulative template-energy quantile."""
    _, antiderivative, _ = template_functions(model_name)
    low, high = -20.0, 200.0
    floor = float(antiderivative(np.array([low]))[0])
    total = template_total_integral(model_name)
    target = floor + quantile * total
    for _ in range(100):
        middle = 0.5 * (low + high)
        if float(antiderivative(np.array([middle]))[0]) < target:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)

def template_energy_bounds(tpeak: float, fwhm_days: float, model_name: str, fraction=0.99):
    """Central template-energy interval used for coverage and duration."""
    _, _, dimensionless_fwhm = template_functions(model_name)
    scale = fwhm_days / dimensionless_fwhm
    tail = (1.0 - fraction) / 2.0
    return (
        tpeak + scale * template_quantile_time(tail, model_name),
        tpeak + scale * template_quantile_time(1.0 - tail, model_name),
    )

def _contiguous_groups(indices: np.ndarray, time_values: np.ndarray, cadence: float, gap_factor: float):
    """Split sorted indices at missing rows or non-contiguous timestamps."""
    indices = np.asarray(indices, dtype=int)
    if not len(indices):
        return []
    breaks = []
    for position in range(1, len(indices)):
        previous, current = indices[position - 1], indices[position]
        if current != previous + 1:
            breaks.append(position)
            continue
        if np.isfinite(cadence) and time_values[current] - time_values[previous] > gap_factor * cadence:
            breaks.append(position)
    return [part for part in np.split(indices, breaks) if len(part)]

def find_candidates(prepared: PreparedLightCurve, config: Config) -> list[Candidate]:
    """Find significant local peaks without joining hours of positive background.

    Both local prominence and height must reach the same rolling 3-sigma
    noise. Valleys partition adjacent peaks; observations are never altered.
    Exposure-limited peaks remain candidates for the impulse-model comparison.
    """
    t=prepared.lightcurve.time; y=prepared.residual.copy(); sigma=prepared.segment_sigma
    if prepared.local_baseline is not None:y-=prepared.local_baseline
    dt=cadence_days(t); output=[]
    for segment in np.unique(prepared.segment_id[prepared.segment_id>=0]):
        ii=np.flatnonzero(prepared.segment_id==segment)
        if len(ii)<3:continue
        yy=y[ii]; ss=sigma[ii]
        peaks,props=find_peaks(yy,height=config.candidate_sigma*ss,
                              prominence=config.candidate_sigma*ss)
        if not len(peaks):continue
        _,_,left_half,right_half=peak_widths(yy,peaks,rel_height=.5,
            prominence_data=(props['prominences'],props['left_bases'],props['right_bases']))
        # The valley is a boundary, not a new observation or an interpolated LC.
        splits=[0]+[int(a+np.argmin(yy[a:b+1])) for a,b in zip(peaks[:-1],peaks[1:])]+[len(ii)-1]
        for j,pk in enumerate(peaks):
            baseline=yy[pk]-props['prominences'][j]
            low=high=int(pk)
            while low>splits[j] and yy[low-1]>baseline+config.expand_sigma*ss[low-1]:low-=1
            while high<splits[j+1] and yy[high+1]>baseline+config.expand_sigma*ss[high+1]:high+=1
            hint=max(.5*dt,float(np.interp(right_half[j],np.arange(len(ii)),t[ii])-np.interp(left_half[j],np.arange(len(ii)),t[ii])))
            low=max(low,int(np.searchsorted(t[ii],t[ii[pk]]-max(3*dt,2*hint))))
            high=min(high,int(np.searchsorted(t[ii],t[ii[pk]]+max(3*dt,5*hint),side="right")-1))
            event=ii[low:high+1]
            seed=event[y[event]-baseline>=config.candidate_sigma*sigma[event]]
            if not len(seed):seed=ii[[pk]]
            output.append(Candidate(seed,event,int(ii[pk]),int(segment),hint))
    return output


def group_overlapping_fit_regions(candidates, time_values, residual, cadence, sigma, expand_sigma):
    """Fit together islands that would otherwise mask each other's data.

    Use the existing neighbor-exclusion padding, not a new time threshold.
    Connected groups stay within a data segment. Preserve every seed and every
    observation between their boundaries; do not smooth or alter the flux.
    Keep events separate when the intervening mean is below the existing
    per-observation expansion threshold. Do not divide this threshold by
    sqrt(N): a long, faint tail is not a reason to join separate outbursts.
    Grouping is an event-association heuristic, not proof of a simple flare.
    """
    groups = []
    for candidate in candidates:
        event = candidate.event_indices
        lo, hi = time_values[event[[0, -1]]]
        padding = max(cadence, hi-lo)
        current = [candidate, lo, hi, lo-padding, hi+2*padding]
        while groups:
            previous = groups[-1]
            if previous[0].segment != candidate.segment:
                break
            if current[1] > previous[4] and current[3] > previous[2]:
                break
            gap = np.arange(previous[0].event_indices[-1]+1, current[0].event_indices[0])
            if len(gap) > 1:
                if np.mean(residual[gap]) <= expand_sigma*np.mean(np.asarray(sigma)[gap]):
                    break
            groups.pop()
            event = np.arange(previous[0].event_indices[0], current[0].event_indices[-1]+1)
            seeds = np.union1d(previous[0].seed_indices, current[0].seed_indices)
            peak = int(event[np.nanargmax(residual[event])])
            current = [Candidate(seeds, event, peak, candidate.segment),
                       previous[1], current[2], min(previous[3], current[3]),
                       max(previous[4], current[4])]
        groups.append(current)
    return [group[0] for group in groups]


def candidate_peak_grid(time_values, candidate, cadence, config):
    """Symmetric local peak prior tied to observed prominence width, not tails."""
    peak=time_values[candidate.peak_index]
    hint=candidate.width_hint or max(cadence,np.ptp(time_values[candidate.event_indices])/4)
    half=max(cadence,.5*hint)
    return np.unique(np.r_[np.linspace(peak-half,peak+half,config.broad_tpeak_points),
                           np.linspace(peak-cadence,peak+cadence,config.broad_tpeak_points)])


def prior_cell_masses(values, prior: str) -> np.ndarray:
    """Trapezoidal masses for a uniform or log-uniform continuous prior."""
    values = np.asarray(values, dtype=float)
    transformed = np.log(values) if prior == "log_uniform" else values
    if values.size == 1:
        return np.ones(1)
    widths = np.empty_like(transformed)
    widths[0] = 0.5 * (transformed[1] - transformed[0])
    widths[-1] = 0.5 * (transformed[-1] - transformed[-2])
    widths[1:-1] = 0.5 * (transformed[2:] - transformed[:-2])
    if np.any(widths <= 0):
        raise ValueError("Grid must be strictly increasing")
    return widths / np.sum(widths)

def logsumexp(values) -> float:
    values = np.asarray(values, dtype=float)
    return float(special.logsumexp(values)) if np.any(np.isfinite(values)) else -np.inf

def _baseline_projection(time_values, values, variance, degree=1):
    """Prepare weighted projection integrating a shared polynomial baseline out.

    The same polynomial coefficients occur under H0 and H1. With the same
    broad (improper) prior their normalization cancels in the Bayes factor;
    projecting them out is the compact form of the analytic marginalization in
    Pitkin et al. (2014, Appendix A).
    """
    time_values = np.asarray(time_values, dtype=float)
    values = np.asarray(values, dtype=float)
    variance = np.asarray(variance, dtype=float)
    span = max(float(np.ptp(time_values)), np.finfo(float).eps)
    x = 2.0 * (time_values - np.median(time_values)) / span
    baseline = np.column_stack([x**j for j in range(degree+1)])
    weight = 1.0 / variance
    gram = baseline.T @ (weight[:, None] * baseline)
    inverse = np.linalg.pinv(gram, rcond=1e-12)
    bwy = baseline.T @ (weight * values)
    return baseline, weight, inverse, bwy

def _projected_template_stats(templates, values, baseline, weight, inverse, bwy):
    """Return D=f'Py and M=f'Pf for many flare templates."""
    templates = np.asarray(templates, dtype=float)
    weighted_templates = templates * weight[:, None]
    fw_baseline = weighted_templates.T @ baseline
    fw_y = weighted_templates.T @ values
    fw_f = np.sum(weighted_templates * templates, axis=0)
    d_value = fw_y - np.einsum("ki,ij,j->k", fw_baseline, inverse, bwy)
    m_value = fw_f - np.einsum("ki,ij,kj->k", fw_baseline, inverse, fw_baseline)
    return d_value, np.maximum(m_value, np.finfo(float).tiny)

def positive_amplitude_logbf(d_value, m_value, scales):
    """Bayes factors and posterior moments for A>=0 half-normal priors.

    For each template, the likelihood ratio is

        exp(A D - A^2 M / 2).

    Multiplication by a proper half-normal prior and integration from zero to
    infinity gives the expression below. Amplitude is therefore fully
    marginalized, not optimized or omitted from the Bayesian calculation.
    """
    d_value = np.asarray(d_value, dtype=float)[:, None]
    m_value = np.asarray(m_value, dtype=float)[:, None]
    scales = np.asarray(scales, dtype=float)[None, :]
    precision = m_value + 1.0 / np.square(scales)
    mean = d_value / precision
    sd = 1.0 / np.sqrt(precision)
    z = d_value / np.sqrt(precision)
    logbf = (
        np.log(2.0)
        - np.log(scales)
        - 0.5 * np.log(precision)
        + 0.5 * np.square(d_value) / precision
        + special.log_ndtr(z)
    )
    return logbf, mean, sd

def noise_variance(relative_errors, sigma: float, indices: np.ndarray,
                   calibration_indices: np.ndarray | None = None) -> np.ndarray:
    """Apply the fixed local error calibration to the requested fit points."""
    if relative_errors is None:
        return np.full(len(indices), sigma**2)
    error = np.asarray(relative_errors, dtype=float)[indices]
    valid = np.isfinite(error) & (error > 0)
    reference = error if calibration_indices is None else np.asarray(relative_errors, dtype=float)[calibration_indices]
    reference_valid = np.isfinite(reference) & (reference > 0)
    if np.sum(reference_valid) < 5:
        return np.full(len(indices), sigma**2)
    median_variance = float(np.median(np.square(reference[reference_valid])))
    if median_variance >= sigma**2:
        scale = sigma**2 / median_variance
        variance = np.square(error) * scale
    else:
        variance = np.square(error) + (sigma**2 - median_variance)
    variance[~valid] = sigma**2
    return np.maximum(variance, np.finfo(float).tiny)

def evaluate_grid(
    time_values,
    values,
    variance,
    tpeak_grid,
    fwhm_grid,
    exposure_days,
    config: Config,
    batch_size: int = 512,
    baseline_degree: int = 1,
    model_name_override: str | None = None,
) -> GridResult:
    """Integrate peak, physical width, amplitude scale and flare-family label."""
    if model_name_override is None:
        names=[config.flare_model if factor==1 else config.flare_model+"_rise"+str(float(factor))
               for factor in config.rise_factors]
        if len(names)>1:
            parts=[evaluate_grid(time_values,values,variance,tpeak_grid,fwhm_grid,
                exposure_days,config,batch_size,baseline_degree,name) for name in names]
            fields=("tpeaks","fwhms","logbf","amplitude_mean","amplitude_sd","projected_d","projected_m","model_names")
            joined={k:np.concatenate([getattr(g,k) for g in parts]) for k in fields}
            return GridResult(**joined,theta_prior_mass=np.concatenate([g.theta_prior_mass/len(parts) for g in parts]),
                amplitude_prior_mass=parts[0].amplitude_prior_mass)
        model_name_override=names[0]
    tpeak_grid = np.asarray(tpeak_grid, dtype=float)
    fwhm_grid = np.asarray(fwhm_grid, dtype=float)
    tmesh, fmesh = np.meshgrid(tpeak_grid, fwhm_grid, indexing="xy")
    tflat, fflat = tmesh.ravel(), fmesh.ravel()
    t_mass = prior_cell_masses(tpeak_grid, "uniform")
    f_mass = prior_cell_masses(fwhm_grid, "log_uniform")
    theta_mass = np.outer(f_mass, t_mass).ravel()
    amp_scales = np.asarray(config.amplitude_prior_scales, dtype=float)
    amp_mass = np.full(len(amp_scales), 1.0 / len(amp_scales))
    baseline, weight, inverse, bwy = _baseline_projection(time_values, values, variance, baseline_degree)
    shape = (len(tflat), len(amp_scales))
    all_logbf = np.empty(shape)
    all_mean = np.empty(shape)
    all_sd = np.empty(shape)
    all_d, all_m = np.empty(len(tflat)), np.empty(len(tflat))
    for start in range(0, len(tflat), batch_size):
        stop = min(start + batch_size, len(tflat))
        templates = flare_model_batch(
            time_values,
            tflat[start:stop],
            fflat[start:stop],
            exposure_days,
            model_name_override,
        )
        d_value, m_value = _projected_template_stats(
            templates, values, baseline, weight, inverse, bwy
        )
        logbf, mean, sd = positive_amplitude_logbf(d_value, m_value, amp_scales)
        all_logbf[start:stop] = logbf
        all_mean[start:stop] = mean
        all_sd[start:stop] = sd
        all_d[start:stop], all_m[start:stop] = d_value, m_value
    return GridResult(
        tpeaks=tflat,
        fwhms=fflat,
        logbf=all_logbf,
        amplitude_mean=all_mean,
        amplitude_sd=all_sd,
        theta_prior_mass=theta_mass,
        amplitude_prior_mass=amp_mass,
        projected_d=all_d,
        projected_m=all_m,
        model_names=np.full(len(tflat),model_name_override),
    )

def grid_log_evidence(grid: GridResult) -> float:
    """Integrate conditional Bayes factors over every proper grid prior."""
    log_weight = (
        np.log(grid.theta_prior_mass)[:, None]
        + np.log(grid.amplitude_prior_mass)[None, :]
    )
    return logsumexp(grid.logbf + log_weight)

def grid_component_weights(grid: GridResult) -> np.ndarray:
    """Normalized posterior component masses conditional on the flare model."""
    log_weight = (
        grid.logbf
        + np.log(grid.theta_prior_mass)[:, None]
        + np.log(grid.amplitude_prior_mass)[None, :]
    )
    normalizer = logsumexp(log_weight)
    return np.exp(log_weight - normalizer)

def truncated_normal_quantile(mean, sd, quantile: float):
    """Quantile of N(mean,sd) conditioned on amplitude >= 0."""
    mean = np.asarray(mean, dtype=float)
    sd = np.asarray(sd, dtype=float)
    log_survival = np.log1p(-quantile) + special.log_ndtr(mean / sd)
    return np.maximum(0.0, mean - sd * special.ndtri_exp(log_survival))

def amplitude_mixture_cdf(value: float, means, sds, weights, *, log_normalizer=None) -> float:
    """CDF of a mixture of positive-truncated normal amplitude posteriors."""
    means = np.asarray(means, dtype=float)
    sds = np.asarray(sds, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if log_normalizer is None:
        log_normalizer = special.log_ndtr(means / sds)
    log_ratio = special.log_ndtr((means - value) / sds) - log_normalizer
    cdf = -np.expm1(np.minimum(log_ratio, 0.0))
    return float(np.sum(weights * cdf))

def amplitude_mixture_quantile(means, sds, weights, quantile: float) -> float:
    """Return one quantile with the same mixture and bisection as batch queries."""
    return float(amplitude_mixture_quantiles(means, sds, weights, [quantile])[0])

def amplitude_mixture_quantiles(means, sds, weights, quantiles) -> np.ndarray:
    """Reuse mixture setup while preserving each original 48-step bisection."""
    quantiles = np.asarray(quantiles, dtype=float).ravel()
    means = np.asarray(means, dtype=float).ravel()
    sds = np.asarray(sds, dtype=float).ravel()
    weights = np.asarray(weights, dtype=float).ravel()
    finite = np.isfinite(means) & np.isfinite(sds) & np.isfinite(weights) & (sds > 0) & (weights > 0)
    means, sds, weights = means[finite], sds[finite], weights[finite]
    if not len(weights):
        return np.full(len(quantiles), np.nan)
    order = np.argsort(weights)[::-1]
    cumulative = np.cumsum(weights[order]) / np.sum(weights)
    keep_count = int(np.searchsorted(cumulative, 1.0 - 1e-8, side="left") + 1)
    keep = order[: max(keep_count, 1)]
    means, sds, weights = means[keep], sds[keep], weights[keep]
    weights = weights / np.sum(weights)
    high = float(np.nanmax(np.maximum(means, 0.0) + 10.0 * sds))
    high = max(high, np.finfo(float).eps)
    # The denominator depends on the mixture, not on the trial amplitude.
    log_normalizer = special.log_ndtr(means / sds)
    results = []
    for quantile in quantiles:
        low, upper = 0.0, high
        for _ in range(48):
            middle = 0.5 * (low + upper)
            if amplitude_mixture_cdf(middle, means, sds, weights,
                                     log_normalizer=log_normalizer) < quantile:
                low = middle
            else:
                upper = middle
        results.append(0.5 * (low + upper))
    return np.asarray(results, dtype=float)

def weighted_quantile(values, weights, quantiles):
    """Weighted empirical quantiles for one-dimensional grid coordinates."""
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    cumulative = np.cumsum(weights)
    cumulative /= cumulative[-1]
    return np.interp(np.asarray(quantiles), cumulative, values)

def detailed_grid_bounds(broad: GridResult, retained_mass: float):
    """Bound the detailed grid around the smallest broad posterior mass set."""
    weights = np.sum(grid_component_weights(broad), axis=1)
    order = np.argsort(weights)[::-1]
    selected = order[np.cumsum(weights[order]) <= retained_mass]
    if selected.size == 0:
        selected = order[:1]
    # Include the first cell that crosses the requested retained mass.
    crossing = min(len(order) - 1, len(selected))
    selected = np.unique(np.r_[selected, order[crossing]])
    return (
        float(np.min(broad.tpeaks[selected])),
        float(np.max(broad.tpeaks[selected])),
        float(np.min(broad.fwhms[selected])),
        float(np.max(broad.fwhms[selected])),
        float(np.sum(weights[selected])),
    )

@lru_cache(maxsize=16)
def template_functions(model_name="tovar2022"):
    """Continuous Tovar family with explicit, finite rise-stretch choices.

    Peak amplitude and physical FWHM retain the same definitions. The rise
    stretch is a discrete model label with a normalized prior, not an extra
    best-fit template silently substituted into a Bayes factor.
    """
    if model_name == "tovar2022":
        return tovar_template, tovar_antiderivative, TOVAR_DIMENSIONLESS_FWHM
    if not model_name.startswith("tovar2022_rise"):
        raise ValueError("Unsupported flare family")
    stretch=float(model_name.split("_rise")[-1])
    if stretch not in (.5,1.,2.,4.):raise ValueError("Unsupported rise stretch")
    zero=float(tovar_antiderivative(np.array([0.]))[0])
    def function(x,amplitude=1.):
        x=np.asarray(x,float)
        return tovar_template(np.where(x<0,x/stretch,x),amplitude)
    def integral(x,amplitude=1.):
        x=np.asarray(x,float)
        return amplitude*np.where(x<0,stretch*tovar_antiderivative(x/stretch),
            tovar_antiderivative(x)+(stretch-1)*zero)
    return function,integral,.7981444440820005+stretch*.28851193331160346


# ======================================================================
# ILIN
# ======================================================================
import warnings
from scipy.interpolate import CubicSpline, PchipInterpolator
from scipy.signal import savgol_filter, find_peaks, peak_widths
from scipy.ndimage import minimum_filter1d, maximum_filter1d
from scipy.optimize import curve_fit, OptimizeWarning
from lightkurve import LightCurve as LightkurveLightCurve




@dataclass(frozen=True)
class IlinConfig:
    """Physical windows from the AU Mic paper; no cadence-specific tuning."""
    spline_hours: float = 6.0
    spline_min_days: float = 5.0
    gap_hours: float = 2.0
    clip_sigma: float = 3.5
    cosine_iterations: int = 5
    savgol_hours: tuple = (6.0, 3.0)
    noise_hours: float = 2.0
    noise_clip_sigma: float = 1.5


def padded_clip(flux, threshold=3.5):
    """Iterative MAD clipping, with sqrt(n)/2sqrt(n) padding of positive runs.

    Single outliers are masked without padding. Negative outliers are clipped
    but not expanded. Boundary indices are clamped, never wrapped.
    """
    values = np.asarray(flux, float)
    finite = np.isfinite(values)
    keep = finite.copy()
    for _ in range(10):
        if keep.sum() < 5:
            break
        center, scatter = np.median(values[keep]), robust_std(values[keep])
        if not np.isfinite(scatter) or scatter <= 0:
            break
        positive = finite & (values > center + threshold * scatter)
        bad = positive | (finite & (values < center - threshold * scatter))
        indices = np.flatnonzero(positive)
        for run in np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1):
            if len(run) > 1:
                left, right = round(np.sqrt(len(run))), round(2 * np.sqrt(len(run)))
                bad[max(0, run[0]-left):min(len(values), run[-1]+right+1)] = True
        updated = keep & ~bad
        if np.array_equal(updated, keep):
            break
        keep = updated
    return keep


def background_keep(flux, config, protected=None):
    """Clip quiet observations without ever putting protected points back."""
    keep = padded_clip(flux, config.clip_sigma)
    if protected is not None:
        keep &= ~protected
    return keep


def spline_stage(t, flux, config, protected=None):
    """Subtract a shape-preserving spline through quiet 6h bin medians.

    Median bin flux follows the archived authors' implementation of the
    paper's 'averaged' point. A portion shorter than five days is unchanged.
    """
    if np.ptp(t) < config.spline_min_days:
        return flux.copy()
    bins = np.floor((t - t[0]) / (config.spline_hours / 24)).astype(int)
    keep = background_keep(flux, config, protected)
    if keep.sum() < 10:
        return flux.copy()
    bt, bf = [], []
    for label in np.unique(bins):
        inside = (bins == label) & keep
        if inside.sum() < 3:
            continue
        mid = float(np.mean(t[inside]))
        if t[0] < mid < t[-1]:
            bt.append(mid)
            bf.append(float(np.median(flux[inside])))
    if len(bt) < 4:
        return flux.copy()
    # Quiet bin centers constrain the spline; do not anchor it to flare peaks
    # at either endpoint or extrapolate an unconstrained cubic outside support.
    trend = PchipInterpolator(bt, bf)(np.clip(t, bt[0], bt[-1]))
    return flux - trend + np.median(flux[keep])


def cosine_stage(t, flux, config, protected=None):
    """Remove up to five significant 2h-to-5d cosine components.

    The periodogram SNR is Lightkurve's flattened maximum power, as in the
    archived analysis. All four cosine coefficients are refined by least
    squares. As in the archived implementation, the current component is
    removed before its SNR determines whether to attempt another iteration.
    """
    values = flux.copy()
    frequencies, flags = [], []
    x = t - t[0]
    for _ in range(config.cosine_iterations):
        keep = background_keep(values, config, protected)
        if keep.sum() < 10 or np.ptp(t[keep]) < 2/24:
            break
        pg = LightkurveLightCurve(time=x[keep], flux=values[keep]).to_periodogram(
            minimum_frequency=0.2, maximum_frequency=12.0)
        snr = float(pg.flatten().max_power.value)
        frequency = float(pg.frequency_at_max_power.value)

        def cosine(xx, amplitude, omega, phase, offset):
            return amplitude * np.cos(omega * xx + phase) + offset

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", OptimizeWarning)
                fitted, _ = curve_fit(
                    cosine, x[keep], values[keep],
                    p0=[np.std(values[keep]), 2*np.pi*frequency, 0, np.mean(values[keep])],
                    bounds=([-np.inf, 2*np.pi*0.2, -np.inf, -np.inf],
                            [np.inf, 2*np.pi*12.0, np.inf, np.inf]),
                    ftol=1e-6, maxfev=10000)
            correction = cosine(x, *fitted) - fitted[3]
            if not np.all(np.isfinite(correction)):
                raise ValueError("nonfinite cosine")
            limit=max(2*np.ptp(values[keep]),10*robust_std(values[keep]))
            if np.max(np.abs(correction)) > limit:
                flags.append("cosine_correction_unbounded")
                break
            before=values[keep]-np.median(values[keep])
            after=values[keep]-correction[keep]
            after-=np.median(after)
            if np.dot(after,after)>np.dot(before,before):
                flags.append("cosine_correction_no_improvement")
                break
            values -= correction
            frequencies.append(float(fitted[1] / (2*np.pi)))
            if snr <= 1.0:
                break
        except (RuntimeError, ValueError, OptimizeWarning) as error:
            flags.append("cosine_fit_failed:" + type(error).__name__)
            break
    return values, frequencies, flags


def fringe_stage(t, flux, config, protected=None):
    """Fit decaying exponentials only at >1-sigma outlying portion endpoints.

    Clipped flares do not constrain the fit. Positive decay rates are bounded
    by construction to prevent numerical exponential growth. All observed
    points receive the same fitted background correction, including flares.
    """
    keep = background_keep(flux, config, protected)
    if keep.sum() < 10:
        return flux.copy(), "insufficient_fringe_background"
    center, sigma = np.median(flux[keep]), robust_std(flux[keep])
    left, right = abs(flux[0]-center) > sigma, abs(flux[-1]-center) > sigma
    if not (left or right):
        return flux.copy(), ""
    x = t - t[0]
    span = x[-1]
    if span <= 0:
        return flux.copy(), "invalid_fringe_span"

    def exponential(xx, *pars):
        result = np.full_like(xx, pars[-1])
        position = 0
        if left:
            result += pars[position]*np.exp(-pars[position+1]*xx)
            position += 2
        if right:
            result += pars[position]*np.exp(-pars[position+1]*(span-xx))
        return result

    p0, lower, upper = [], [], []
    for active, height in [(left, flux[0]-center), (right, flux[-1]-center)]:
        if active:
            p0.extend([height, 10.0]); lower.extend([-np.inf, 0]); upper.extend([np.inf, np.inf])
    p0.append(center); lower.append(-np.inf); upper.append(np.inf)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", OptimizeWarning)
            pars, _ = curve_fit(exponential, x[keep], flux[keep], p0=p0,
                                bounds=(lower, upper), maxfev=10000, ftol=1e-6)
        correction = exponential(x, *pars)-pars[-1]
        # Clipping can leave an edge unconstrained. A formally converged
        # exponential may then extrapolate by millions of relative-flux units.
        # Reject that correction, keeping the preceding valid detrending stage.
        limit = max(10*sigma, 2*np.ptp(flux[keep]),
                    2*np.max(np.abs(flux[[0,-1]]-center)))
        if np.any(~np.isfinite(correction)) or np.max(np.abs(correction)) > limit:
            return flux.copy(), "fringe_correction_unbounded"
        proposed = flux-correction
        before = flux[keep]-np.median(flux[keep])
        after = proposed[keep]-np.median(proposed[keep])
        if np.dot(after,after) > np.dot(before,before):
            return flux.copy(), "fringe_correction_no_improvement"
        return proposed, ""
    except (RuntimeError, ValueError, OptimizeWarning) as error:
        return flux.copy(), "fringe_fit_failed:" + type(error).__name__


def rolling_noise(t, flux, hours, clip):
    """Rolling sample scatter after padded clipping; interpolate noise only."""
    keep = padded_clip(flux, clip)
    masked = np.where(keep, flux, np.nan)
    n = max(3, int(round(hours / 24 / cadence_days(t))))
    noise = pd.Series(masked).rolling(n, center=True, min_periods=3).std()
    noise = noise.interpolate(limit_direction="both").to_numpy()
    fallback = robust_std(flux[keep])
    noise[~np.isfinite(noise) | (noise <= 0)] = fallback
    return noise


def protected_savgol_background(t, flux, keep, window):
    """Smooth a quiet-only background on the original observed time grid.

    Interpolated values belong only to the background model, never to the
    observations. Endpoint values are held constant outside quiet support.
    Bound the polynomial by the local range of that background model, avoiding
    extrapolation through large protected holes at coarse cadence.
    """
    if np.count_nonzero(keep) < 5:
        raise ValueError("Insufficient quiet support for Savitzky-Golay background")
    filled = np.interp(t, t[keep], flux[keep])
    baseline = savgol_filter(filled, window, 3, mode="nearest")
    return np.clip(baseline, minimum_filter1d(filled, size=window, mode="nearest"),
                   maximum_filter1d(filled, size=window, mode="nearest"))


def detrend(t, flux, config=IlinConfig(), protected=None):
    """Return residual, noise, error scale, intermediate stages, and warnings.

    Splits at >2h; never adds observations or deletes masked flare points.
    Savitzky-Golay uses order 3 and explicit 6h/3h windows on a quiet-only
    bounded background. No detrending uncertainty is invented: error scaling
    propagates only multiplicative normalization factors.
    """
    t, flux = np.asarray(t, float), np.asarray(flux, float)
    protected = np.zeros(len(t), bool) if protected is None else np.asarray(protected, bool)
    if protected.shape != t.shape:
        raise ValueError("Protection mask must match observations")
    if len(t) < 5 or np.any(~np.isfinite(flux)) or np.any(np.diff(t) <= 0):
        raise ValueError("Ilin detrending requires sorted, unique finite observations")
    stages = {key: np.empty_like(flux) for key in ("spline", "cosine", "savgol6", "savgol3", "final")}
    scales, noise = np.ones_like(flux), np.empty_like(flux)
    metadata = []
    for indices in np.split(np.arange(len(t)), np.flatnonzero(np.diff(t) > config.gap_hours/24)+1):
        if len(indices) < 5:
            for array in stages.values():
                array[indices] = flux[indices]
            stages["final"][indices] = flux[indices]-np.median(flux[indices])
            noise[indices] = np.nan
            metadata.append({"start": float(t[indices[0]]), "warnings": ["short_portion"]})
            continue
        tt, yy = t[indices], flux[indices]
        pp = protected[indices]
        yy = spline_stage(tt, yy, config, pp)
        stages["spline"][indices] = yy
        yy, frequencies, flags = cosine_stage(tt, yy, config, pp)
        if (~pp).sum() < max(15, int(.2*len(pp))):
            flags.append("insufficient_protected_background")
        stages["cosine"][indices] = yy
        for hours, name in zip(config.savgol_hours, ("savgol6", "savgol3")):
            window = int(round(hours / 24 / cadence_days(tt)))
            window += 1 - window % 2
            window = min(window, len(tt) if len(tt) % 2 else len(tt)-1)
            if window < 5:
                flags.append("savgol_window_too_short")
            else:
                keep = background_keep(yy, config, pp)
                if keep.sum() >= 5:
                    baseline = protected_savgol_background(tt, yy, keep, window)
                    if np.any(~np.isfinite(baseline)) or np.any(baseline <= 0):
                        raise ValueError("Invalid Savitzky-Golay background")
                    yy = yy / baseline
                    scales[indices] /= baseline
                else:
                    flags.append("insufficient_savgol_background")
            stages[name][indices] = yy
        yy, fringe_warning = fringe_stage(tt, yy, config, pp)
        if fringe_warning:
            flags.append(fringe_warning)
        keep = background_keep(yy, config, pp)
        yy -= np.median(yy[keep]) if keep.any() else np.median(yy)
        stages["final"][indices] = yy
        noise[indices] = rolling_noise(tt, yy, config.noise_hours, config.noise_clip_sigma)
        metadata.append({"start": float(tt[0]), "end": float(tt[-1]),
                         "cosine_frequencies_per_day": frequencies, "warnings": flags})
    return stages["final"], noise, scales, stages, metadata


# ======================================================================
# PIPELINE
# ======================================================================



import argparse
import hashlib
import json
import time
from dataclasses import dataclass, asdict
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack

from tqdm import tqdm





@dataclass(frozen=True)
class Config:
    """Frozen settings; thresholds can be calibrated on independent controls."""
    already_detrended: bool = False
    flux_is_residual: bool = False
    flux_column: str = ""
    quality_zero_only: bool = True  # Required; retained in manifests for provenance.
    stellar_catalog_paths: tuple = ()
    distance_column: str = "distance_pc"
    tess_band_luminosity_erg_s: float | None = None
    energy_method: str = "supplied"
    tess_effective_bandwidth_hz: float = 1.2e14
    flare_model: str = "tovar2022"
    rise_factors: tuple = (.5, 1., 2.)
    detection_statistic: str = "alternatives"
    candidate_sigma: float = 3.0
    expand_sigma: float = 1.0
    gap_factor: float = 5.0
    sigma_clip: float = 3.0
    sigma_clip_iterations: int = 10
    broad_fwhm_points: int = 41
    broad_tpeak_points: int = 9
    chi2_fit: bool = True
    chi2_max_nfev: int = 200
    detailed_fwhm_points: int = 121
    detailed_tpeak_points: int = 61
    detailed_mass: float = 0.995
    fwhm_min_minutes: float = 0.1
    fwhm_max_minutes: float = 2880.0
    amplitude_prior_scales: tuple = (3e-4, 1e-3, 3e-3, .01, .03, .1, .3, 1.)
    logbf_threshold: float = float(np.log(4.0))
    high_snr_logbf_threshold: float | None = None
    high_snr_boundary: float = 8.0
    direction_threshold: float = float(np.log(10.0))
    impulse_margin: float = 1.0
    symmetric_margin: float = 0.0
    raw_step_margin: float = 2.0
    raw_signal_threshold: float = 5.0
    local_window_hours: float = 12.0
    max_candidates: int = 0  # zero processes all; overflow is never silently lost
    candidate_workers: int = 1
    ilin: IlinConfig = IlinConfig()
    noise_window_hours: float = 2.0
    noise_min_points: int = 41
    edge_fraction: float = 0.05
    protection_iterations: int = 2
    protection_padding: float = 2.0
    edge_scope: str = "lightcurve"
    calibration_file: str = ""
    builtin_calibration: bool = True
    allow_uncalibrated: bool = False

    def __post_init__(self):
        if self.detection_statistic not in ("background","alternatives"):
            raise ValueError("Unknown detection statistic")
        if not self.rise_factors or len(set(self.rise_factors))!=len(self.rise_factors) or any(x not in (.5,1.,2.,4.) for x in self.rise_factors):
            raise ValueError("rise_factors must be unique supported model labels")
        if not np.isfinite(self.noise_window_hours) or self.noise_window_hours <= 0:
            raise ValueError("noise_window_hours must be positive")
        if self.noise_min_points < 15:
            raise ValueError("noise_min_points must be >= 15")
        if self.protection_iterations < 1 or self.protection_padding < 0:
            raise ValueError("Flare protection requires positive iterations and nonnegative padding")
        if not np.isfinite(self.edge_fraction) or not 0 <= self.edge_fraction < 0.5:
            raise ValueError("edge_fraction must be in [0, 0.5)")
        if self.edge_scope not in {"lightcurve", "segment"}:
            raise ValueError("edge_scope must be lightcurve or segment")
        if not np.isfinite(self.logbf_threshold) or self.logbf_threshold < 0:
            raise ValueError("logbf_threshold must be finite and nonnegative")
        if self.high_snr_logbf_threshold is not None and (not np.isfinite(self.high_snr_logbf_threshold) or self.high_snr_logbf_threshold < 0):
            raise ValueError("high_snr_logbf_threshold must be nonnegative finite or None")
        if not np.isfinite(self.high_snr_boundary) or self.high_snr_boundary < 3:
            raise ValueError("high_snr_boundary must be at least 3")
        if not 0 < self.fwhm_min_minutes < self.fwhm_max_minutes:
            raise ValueError("Invalid physical FWHM bounds")
        if self.quality_zero_only is not True:
            raise ValueError("QUALITY == 0 filtering is mandatory and cannot be disabled")
        if isinstance(self.chi2_max_nfev, bool) or not isinstance(self.chi2_max_nfev, (int, np.integer)) or self.chi2_max_nfev < 2:
            raise ValueError("chi2_max_nfev must be an integer >= 2")


def flare_protection(t, residual, sigma, segment_id, config):
    """Protect >=3-sigma positive excursions and symmetric surrounding wings.

    Bootstrap sigma uses adjacent differences and lower-tail scatter to limit
    positive-flare contamination while retaining red noise. Padding avoids
    privileging forward flares over reverse flares or Gaussian controls.
    This only selects background points; it does not accept an event.
    """
    mask = np.zeros(len(t), bool)
    for segment in np.unique(segment_id[segment_id >= 0]):
        ii = np.flatnonzero(segment_id == segment)
        dt = cadence_days(t[ii])
        above = residual[ii] > config.expand_sigma*sigma[ii]
        indices = np.flatnonzero(above)
        for run in np.split(indices, np.flatnonzero(np.diff(indices) > 1)+1):
            if not len(run) or not np.any(residual[ii[run]] >= config.candidate_sigma*sigma[ii[run]]):
                continue
            left, right = t[ii[run[0]]], t[ii[run[-1]]]
            padding = max(3*dt, config.protection_padding*(right-left+dt))
            mask[ii] |= (t[ii] >= left-padding) & (t[ii] <= right+padding)
    return mask


def prepare(lc, config):
    """Refit the original flux while protecting likely flares and their tails.

    Two mask updates use only the observed light curve, never injection truth.
    The final rolling scatter uses quiet points and a formal-error floor. Both
    seeding and likelihood use this same local sigma; seeding remains 3 sigma.
    """
    scale = float(np.median(lc.flux))
    if config.flux_is_residual:
        normalized = 1.0 + lc.flux
        scale = 1.0
    else:
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("Nonpositive normalization; declare --residual for zero-centered input")
        normalized = lc.flux / scale
    input_error = None if lc.flux_err is None else lc.flux_err / scale
    segment_id = np.full(len(normalized), -1, int)
    for number, indices in enumerate(split_segments(lc.time, config.gap_factor)):
        if len(indices) >= 5:
            segment_id[indices] = number
    protected = np.zeros(len(normalized), bool)
    mask_counts = []
    for iteration in range(config.protection_iterations+1):
        if config.already_detrended:
            residual = normalized - np.median(normalized)
            ilin_noise, error_scale, stages, metadata = None, np.ones(len(normalized)), {}, []
        else:
            residual, ilin_noise, error_scale, stages, metadata = detrend(
                lc.time, normalized, config.ilin, protected)
        error = None if input_error is None else input_error*error_scale
        sigma = np.full(len(residual), np.nan)
        uncertain = np.zeros(len(residual), bool)
        for segment in np.unique(segment_id[segment_id >= 0]):
            ii = np.flatnonzero(segment_id == segment)
            quiet = ii[~protected[ii]]
            if len(quiet) < max(15, int(.2*len(ii))):
                uncertain[ii] = True
            if len(quiet) < 5:
                quiet = ii
            center, scatter, _ = sigma_clip_location_scale(residual[quiet],
                clip_sigma=config.sigma_clip, max_iter=config.sigma_clip_iterations)
            residual[ii] -= center
            sigma[ii] = scatter
        prepared = PreparedLightCurve(lc, normalized, normalized-residual, residual,
            error, segment_id, sigma, np.zeros(len(residual), bool), protected, uncertain, error_scale)
        if iteration == config.protection_iterations:
            break
        bootstrap, _ = local_noise(prepared, config, difference_only=True)
        updated = protected | flare_protection(lc.time, residual, bootstrap, segment_id, config)
        mask_counts.append(int(updated.sum()))
        if np.array_equal(updated, protected):
            break
        protected = updated
    prepared.segment_sigma, noise_info = local_noise(prepared, config)
    prepared.local_baseline = quiet_local_baseline(prepared, config)
    prepared.detrend_edge_mask = edge_mask(prepared, config)
    exposure = lc.exposure_days or cadence_days(lc.time)
    usable = (prepared.segment_id >= 0) & ~prepared.detrend_edge_mask & ~prepared.background_uncertain_mask
    return prepared, {"stages": stages, "ilin_noise": ilin_noise, "portions": metadata,
        "local_noise": noise_info, "edge_points": int(prepared.detrend_edge_mask.sum()),
        "protection_mask_counts": mask_counts, "protected_points": int(prepared.protection_mask.sum()),
        "background_uncertain_points": int(prepared.background_uncertain_mask.sum()),
        "searchable_exposure_days": float(usable.sum()*exposure),
        "observed_exposure_days": float(len(lc.time)*exposure)}


def refined_grid(t, y, variance, peaks, widths, exposure, config):
    """Compatibility entry point: integrate the broad grid once, without refit."""
    broad = evaluate_grid(t, y, variance, peaks, widths, exposure, config)
    return broad, grid_log_evidence(broad)


def summarize_grid(grid):
    """Coherent joint fit plus marginalized intervals for peak, width, and A.

    Choose a peak/width MAP density in (tpeak, log width), marginalizing the
    prior-scale label. Then use the conditional amplitude-mixture median.
    """
    weights = grid_component_weights(grid)
    theta_weights = weights.sum(axis=1)
    density = special.logsumexp(grid.logbf + np.log(grid.amplitude_prior_mass), axis=1)
    best = int(np.argmax(density))
    conditional = weights[best] / weights[best].sum()
    amplitude = amplitude_mixture_quantile(grid.amplitude_mean[best],
                                            grid.amplitude_sd[best], conditional, .5)
    models=grid.model_names if grid.model_names is not None else np.full(len(grid.tpeaks),"tovar2022")
    model_name=str(models[best])
    result = {"flare_model_name":model_name,"flare_model_probability":float(theta_weights[models==model_name].sum()),
              "tpeak": float(grid.tpeaks[best]), "fwhm_days": float(grid.fwhms[best]),
              "amplitude": amplitude, "log_bayes_factor": grid_log_evidence(grid)}
    for name, values in [("tpeak", grid.tpeaks), ("fwhm_days", grid.fwhms)]:
        qs = weighted_quantile(values, theta_weights, [.16, .5, .84])
        result.update({name+"_p"+str(p): float(q) for p, q in zip((16,50,84), qs)})
    amplitude_quantiles = amplitude_mixture_quantiles(
        grid.amplitude_mean, grid.amplitude_sd, weights, [.16, .5, .84])
    for p, value in zip((16,50,84), amplitude_quantiles):
        result["amplitude_p"+str(p)] = float(value)
    # ED is A times a width-dependent constant; propagate their covariance.
    conversion = grid.fwhms[:,None] * np.array([full_equivalent_duration_days(1.,1.,str(m)) for m in models])[:,None] * 86400
    result["equivalent_duration_s"] = full_equivalent_duration_days(
        result["fwhm_days"], amplitude, model_name) * 86400
    ed_quantiles = amplitude_mixture_quantiles(
        grid.amplitude_mean*conversion, grid.amplitude_sd*conversion, weights, [.16, .5, .84])
    for p, value in zip((16,50,84), ed_quantiles):
        result["equivalent_duration_s_p"+str(p)] = float(value)
    widths = np.unique(grid.fwhms)
    boundary = (grid.fwhms <= widths[1]) | (grid.fwhms >= widths[-2])
    result["width_prior_boundary_mass"] = float(theta_weights[boundary].sum())
    return result


def template_evidence(templates, t, y, variance, config, signed=False, baseline_degree=1):
    """Integrate amplitude and equal template weights for a diagnostic family."""
    design, weight, inverse, bwy = _baseline_projection(t,y,variance,baseline_degree)
    d,m = _projected_template_stats(templates,y,design,weight,inverse,bwy)
    scales = np.asarray(config.amplitude_prior_scales)[None,:]
    if signed:
        precision = m[:,None]+1/scales**2
        logbf = -np.log(scales)-.5*np.log(precision)+.5*d[:,None]**2/precision
    else:
        logbf = positive_amplitude_logbf(d,m,scales.ravel())[0]
    return float(special.logsumexp(logbf)-np.log(logbf.size))


def gaussian_grid_evidence(t, y, variance, peaks, widths, exposure, config, baseline_degree=1):
    """Exposure-integrated symmetric Gaussian alternative, same proper priors.

    Width is Gaussian FWHM. This alternative tests symmetry directly; a
    forward-vs-reversed-flare comparison alone does not exclude a smooth bump.
    """
    pm, wm = np.meshgrid(peaks, widths, indexing="xy")
    pp, ww = pm.ravel(), wm.ravel()
    mass = np.outer(prior_cell_masses(widths,"log_uniform"),prior_cell_masses(peaks,"uniform")).ravel()
    baseline, weight, inverse, bwy = _baseline_projection(t,y,variance,baseline_degree)
    values = []
    for start in range(0,len(pp),256):
        peak = pp[start:start+256][None,:]
        sd = ww[start:start+256][None,:]/np.sqrt(8*np.log(2))
        if exposure > 0:
            upper = (t[:,None]+exposure/2-peak)/(np.sqrt(2)*sd)
            lower = (t[:,None]-exposure/2-peak)/(np.sqrt(2)*sd)
            templates = sd*np.sqrt(np.pi/2)/exposure*(special.erf(upper)-special.erf(lower))
        else:
            templates = np.exp(-.5*((t[:,None]-peak)/sd)**2)
        d,m = _projected_template_stats(templates,y,baseline,weight,inverse,bwy)
        logbf = positive_amplitude_logbf(d,m,config.amplitude_prior_scales)[0]
        values.append(special.logsumexp(logbf,axis=1)-np.log(len(config.amplitude_prior_scales)))
    return float(special.logsumexp(np.concatenate(values)+np.log(mass)))


def step_grid_evidence(t, y, variance, exposure, config, baseline_degree=2, order=32):
    """Marginalize signed exposure-integrated jumps over a uniform time prior.

    Jumps can occur anywhere within an exposure. Integrate every piecewise
    linear interval with Gauss-Legendre quadrature, including gaps, rather
    than testing only boundaries between observations. Cumulative weighted
    moments evaluate templates without an N-observations x N-jumps matrix.
    """
    t=np.asarray(t,float); x=t-t[0]
    design,weight,inverse,bwy=_baseline_projection(t,y,variance,baseline_degree)
    if len(t)<2 or x[-1]<=0:raise ValueError("Step prior needs a time span")
    exposure=max(float(exposure),np.finfo(float).eps)
    breaks=np.unique(np.clip(np.r_[0.,x-exposure/2,x+exposure/2,x[-1]],0,x[-1]))
    nodes,weights=np.polynomial.legendre.leggauss(order)
    lo,hi=breaks[:-1],breaks[1:]
    pp=((hi+lo)[:,None]/2+(hi-lo)[:,None]/2*nodes).ravel()
    mass=((hi-lo)[:,None]/(2*x[-1])*weights).ravel()
    # Cumulative sums for the ramp a*t+b and the fully raised suffix.
    columns=np.column_stack((weight,weight*x,weight*x*x,weight*y,weight*x*y,
                             weight[:,None]*design,weight[:,None]*design*x[:,None]))
    sums=np.vstack((np.zeros(columns.shape[1]),np.cumsum(columns,axis=0)))
    left=np.searchsorted(x,pp-exposure/2,side="right")
    right=np.searchsorted(x,pp+exposure/2,side="left")
    ramp=sums[right]-sums[left];suffix=sums[-1]-sums[right]
    aa=1/exposure;bb=.5-pp/exposure;k=design.shape[1]
    fy=aa*ramp[:,4]+bb*ramp[:,3]+suffix[:,3]
    ff=aa*aa*ramp[:,2]+2*aa*bb*ramp[:,1]+bb*bb*ramp[:,0]+suffix[:,0]
    fb=aa*ramp[:,5+k:]+bb[:,None]*ramp[:,5:5+k]+suffix[:,5:5+k]
    d=fy-fb@(inverse@bwy)
    m=np.maximum(ff-np.einsum("ki,ij,kj->k",fb,inverse,fb),np.finfo(float).tiny)
    scales=np.asarray(config.amplitude_prior_scales)[None,:]
    precision=m[:,None]+1/scales**2
    logbf=-np.log(scales)-.5*np.log(precision)+.5*d[:,None]**2/precision
    scores=special.logsumexp(logbf,axis=1)-np.log(scales.size)
    return float(special.logsumexp(scores+np.log(mass)))


def artifact_checks(prepared, indices, t, y, variance, result, config, peaks, widths):
    """Compare impulses, symmetric bumps and raw-flux steps.

    Raw comparisons use a shared quadratic background and unmasked local
    observations, so neighboring masks cannot hide contradictory samples. The
    time prior for a jump covers the whole raw window. They do not alter the
    reported flare-vs-affine-background ln BF. All ambiguous events survive
    in follow-up with the competing-model scores.
    """
    cadence = cadence_days(prepared.lightcurve.time)
    exposure = result["exposure_seconds"]/86400
    peak = result["seed_time"]
    nearby = np.flatnonzero(abs(t-peak) <= cadence*1.01)
    impulses = np.zeros((len(t),len(nearby)))
    impulses[nearby,np.arange(len(nearby))] = 1.
    impulse = template_evidence(impulses,t,y,variance,config)
    symmetric = gaussian_grid_evidence(t,y,variance,peaks,widths,exposure,config)
    raw_difference, raw_symmetric = np.nan, np.nan
    result["raw_logbf_flare"] = np.nan
    if not config.already_detrended:
        # Cover the observed peak and at least two fitted widths: a narrow
        # seed must not hide the discontinuity producing a broad fitted decay.
        # Restore unmasked samples and fit the same curved raw background
        # under every competing model.
        half = max(8*cadence,4*result.get("observed_width_hint_days",result["fwhm_days"]),2*result["fwhm_days"])
        lc=prepared.lightcurve
        segment=prepared.segment_id[np.argmin(abs(lc.time-peak))]
        raw_indices=np.flatnonzero((prepared.segment_id==segment)&(abs(lc.time-peak)<=half))
        if len(raw_indices) >= 10:
            tt=lc.time[raw_indices]
            yy=prepared.normalized_flux[raw_indices]-1.
            vv=np.square(prepared.segment_sigma[raw_indices]/prepared.detrend_error_scale[raw_indices])
            if prepared.relative_flux_err is not None:
                ee=prepared.relative_flux_err[raw_indices]/prepared.detrend_error_scale[raw_indices]
                vv=np.maximum(vv,np.where(np.isfinite(ee)&(ee>0),ee**2,0.))
            result["raw_background_degree"]=2
            if len(tt) > 2048:
                subset = np.unique(np.linspace(0, len(tt)-1, 2048).astype(int))
                tt, yy, vv = tt[subset], yy[subset], vv[subset]
            result["raw_diagnostic_points"] = len(tt)
            # Cover the actual cadence boundaries: a detrending residual's
            # maximum can be far from the original discontinuity.
            step_logbf=step_grid_evidence(tt,yy,vv,exposure,config,baseline_degree=2)
            raw_grid=evaluate_grid(tt,yy,vv,peaks,widths,exposure,config,baseline_degree=2)
            raw_flare=grid_log_evidence(raw_grid)
            result["raw_logbf_flare"] = float(raw_flare)
            raw_difference=step_logbf-raw_flare
            raw_symmetric=gaussian_grid_evidence(tt,yy,vv,peaks,widths,exposure,config,baseline_degree=2)-raw_flare
    return impulse,raw_difference,symmetric,raw_symmetric


def minimize_flare_chi2(t, y, variance, exposure, grid, bayesian, config, model_name=None):
    """Fit peak/log width continuously, profiling A>=0 and the linear baseline.

    For each exact exposure-averaged template f, weighted projection gives
    D=f'Py and M=f'Pf, hence A=max(0,D/M). This is a two-dimensional bounded
    maximum-likelihood search, not Bayesian posterior refinement. Start from
    the better of the Bayesian shape and the best grid profile likelihood.
    Retain the best finite solution, including initialization, on failure.
    """
    started = time.perf_counter()
    if model_name is None:
        names=np.unique(grid.model_names) if grid.model_names is not None else ["tovar2022"]
        if len(names)>1:
            choices=[]
            for name in names:
                mask=grid.model_names==name
                fields=("tpeaks","fwhms","logbf","amplitude_mean","amplitude_sd","projected_d","projected_m","model_names")
                part=GridResult(**{k:getattr(grid,k)[mask] for k in fields},
                    theta_prior_mass=grid.theta_prior_mass[mask]/grid.theta_prior_mass[mask].sum(),
                    amplitude_prior_mass=grid.amplitude_prior_mass)
                choices.append(minimize_flare_chi2(t,y,variance,exposure,part,summarize_grid(part),config,str(name)))
            valid=[x for x in choices if x[0]["chi2_fit_success"]]
            answer=min(valid or choices,key=lambda x:x[0]["chi2_min"])
            answer[0]["chi2_fit_seconds"]=time.perf_counter()-started
            answer[0]["chi2_model_evaluations"]=sum(x[0]["chi2_model_evaluations"] for x in choices)
            return answer
        model_name=str(names[0])
    design, weight, inverse, bwy = _baseline_projection(t, y, variance)
    sqrt_var = np.sqrt(variance)
    y_projected = y - design @ (inverse @ bwy)
    origin, time_range = float(np.min(grid.tpeaks)), float(np.ptp(grid.tpeaks))
    log_min = float(np.log(np.min(grid.fwhms)))
    log_range = float(np.log(np.max(grid.fwhms)) - log_min)
    if time_range <= 0 or log_range <= 0:
        raise ValueError("Continuous fit requires nonzero peak and width ranges")
    evaluations = 0

    def evaluate(position):
        nonlocal evaluations
        evaluations += 1
        peak = origin + position[0] * time_range
        width = np.exp(log_min + position[1] * log_range)
        template = flare_model(t, peak, width, 1., exposure, model_name)
        f_projected = template - design @ (inverse @ (design.T @ (weight*template)))
        m = float(np.dot(weight*f_projected, f_projected))
        d = float(np.dot(weight*f_projected, y_projected))
        amplitude = max(0., d/m) if m > np.finfo(float).tiny else 0.
        flare = amplitude * template
        beta = inverse @ (design.T @ (weight*(y-flare)))
        fitted = design @ beta + flare
        residual = (y-fitted)/sqrt_var
        if not np.isfinite(residual).all():
            raise ValueError("Nonfinite chi-square residual")
        return residual, peak, width, amplitude, beta, fitted, flare

    def position(peak, width):
        return np.clip([(peak-origin)/time_range,
                        (np.log(width)-log_min)/log_range], 0., 1.)

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        score = np.maximum(grid.projected_d, 0.)**2/grid.projected_m
    score[~np.isfinite(score)] = -np.inf
    index = int(np.argmax(score))
    starts = [position(bayesian["tpeak"], bayesian["fwhm_days"]),
              position(grid.tpeaks[index], grid.fwhms[index])]
    initial = [evaluate(x) for x in starts]
    costs = [float(np.dot(v[0], v[0])) for v in initial]
    selected = int(np.argmin(costs))
    x_best, best = starts[selected], initial[selected]
    initial_cost = costs[selected]
    success, status, nfev = False, "not_run", 0
    try:
        solution = least_squares(lambda x: evaluate(x)[0], x_best,
                                 bounds=([0., 0.], [1., 1.]), method="trf",
                                 loss="linear", ftol=1e-8, xtol=1e-8, gtol=1e-8,
                                 max_nfev=config.chi2_max_nfev)
        trial = evaluate(solution.x)
        improved = float(np.dot(trial[0], trial[0])) <= initial_cost
        if improved:
            x_best, best = solution.x, trial
        success = bool(solution.success and improved)
        status = ("converged" if success else
                  "initial_retained" if not improved else "evaluation_limit")
        nfev = int(solution.nfev)
    except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
        status = "initial_retained:" + str(error)
    residual, peak, width, amplitude, beta, fitted, flare = best
    chi2 = float(np.dot(residual, residual))
    flags = []
    if not success:
        flags.append("not_converged")
    if np.any((x_best < 1e-5) | (x_best > 1.-1e-5)):
        flags.append("parameter_bound")
    if width < 1.5*exposure:
        flags.append("cadence_limited")
    if amplitude <= 0:
        flags.append("zero_amplitude_unidentified_shape")
    ed = full_equivalent_duration_days(width, amplitude, model_name) * 86400
    luminosity = config.tess_band_luminosity_erg_s
    energy = luminosity*ed if luminosity is not None and np.isfinite(luminosity) and luminosity > 0 else np.nan
    row = dict(flare_model_name_chi2=model_name,tpeak_chi2=float(peak), fwhm_days_chi2=float(width),
               amplitude_chi2=float(amplitude), baseline_offset_chi2=float(beta[0]),
               baseline_slope_scaled_chi2=float(beta[1]), chi2_min=chi2,
               chi2_red_min=chi2/max(1, len(t)-5), chi2_initial=initial_cost,
               chi2_bayesian_shape_profile=costs[0],
               equivalent_duration_s_chi2=float(ed), flare_energy_erg_chi2=float(energy),
               sampled_model_amplitude_chi2=float(np.max(flare)),
               chi2_fit_success=success, chi2_fit_status=status, chi2_fit_flags=";".join(flags),
               chi2_nfev=nfev, chi2_model_evaluations=evaluations,
               chi2_fit_seconds=time.perf_counter()-started)
    return row, dict(fitted_chi2=fitted, flare_chi2=flare)


def measurement_summary(t, y, variance, exposure, grid, config):
    """Keep the grid posterior distinct from the optional primary ML estimate."""
    result = summarize_grid(grid)
    result["flare_model_name_bayesian"]=result["flare_model_name"]
    fields = ("tpeak", "fwhm_days", "amplitude", "equivalent_duration_s")
    for field in fields:
        result[field+"_bayesian"] = result[field]
        for p in (16, 50, 84):
            result[field+"_bayesian_p"+str(p)] = result.pop(field+"_p"+str(p))
    result.update(measurement_method="bayesian_grid", chi2_fit_status="disabled",
                  chi2_fit_success=False, chi2_fit_seconds=0., chi2_fit_flags="")
    if config.chi2_fit:
        started = time.perf_counter()
        try:
            diagnostic, _ = minimize_flare_chi2(t, y, variance, exposure, grid, result, config)
            result.update(diagnostic)
            valid = diagnostic["chi2_fit_success"] and all(
                np.isfinite(diagnostic[field+"_chi2"]) for field in fields)
            if valid and diagnostic["amplitude_chi2"] > 0:
                for field in fields:
                    result[field] = diagnostic[field+"_chi2"]
                result["flare_model_name"]=diagnostic["flare_model_name_chi2"]
                result["measurement_method"] = "chi2_profile_likelihood"
            else:
                result["measurement_method"] = "bayesian_fallback"
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
            result.update(measurement_method="bayesian_fallback", chi2_fit_status="failed:"+str(error),
                          chi2_fit_flags="fit_failed", chi2_fit_seconds=time.perf_counter()-started)
    return result


def characterize(prepared, candidate, candidates, config):
    """Fit a seed with neighboring events excluded from its background window."""
    start = time.perf_counter()
    lc, residual = prepared.lightcurve, prepared.residual
    cadence = cadence_days(lc.time)
    exposure = lc.exposure_days or cadence
    peak = lc.time[candidate.peak_index]
    event_span = max(cadence, np.ptp(lc.time[candidate.event_indices]) + cadence)
    hint = candidate.width_hint or max(cadence,event_span/4)
    half_window = max(12*cadence, 8*hint)*config.local_window_hours/12
    mask = (prepared.segment_id == candidate.segment) & (abs(lc.time-peak) <= half_window)
    noise_mask = mask.copy()
    # Padding protects the noise estimate, but must not erase measurements
    # that could contradict the target flare's rise or decay in the likelihood.
    for other in candidates:
        if other is candidate or other.segment != candidate.segment:
            continue
        lo, hi = lc.time[other.event_indices[[0,-1]]]
        padding = max(cadence, min(hi-lo, 2*(other.width_hint or cadence)))
        noise_mask &= ~((lc.time >= lo-padding) & (lc.time <= hi+2*padding))
        mask[other.event_indices] = False
    mask[candidate.event_indices] = True
    noise_mask[candidate.event_indices] = True
    indices = np.flatnonzero(mask)
    noise_indices = np.flatnonzero(noise_mask)
    quiet = ~np.isin(noise_indices, candidate.event_indices)
    if quiet.sum() < 8:
        if config.local_window_hours < 48:
            result, fit = characterize(prepared, candidate, candidates,
                replace(config, local_window_hours=min(48., 2*config.local_window_hours)))
            result["fit_context_expanded"] = True
            return result, fit
        return {"tier": "followup", "reason": "insufficient_local_background",
                "tpeak": float(peak), "seed_time": float(peak)}, None
    t, y = lc.time[indices], residual[indices]
    # Use the same time-varying TOTAL scatter in detection and likelihood.
    # Never rescale reported errors downward or impose a global noisy-segment floor.
    variance = np.square(prepared.segment_sigma[indices])
    if prepared.relative_flux_err is not None:
        err = prepared.relative_flux_err[indices]
        variance = np.maximum(variance, np.where(np.isfinite(err) & (err > 0), err**2, 0.))
    if np.any(~np.isfinite(variance)) or np.any(variance <= 0):
        raise ValueError("Invalid local noise variance")
    sigma = float(np.sqrt(variance[np.flatnonzero(indices == candidate.peak_index)[0]]))
    # Physical prior: short flares remain possible during long exposures.
    # Exposure integration, not a cadence-dependent width floor, handles dilution.
    minimum = config.fwhm_min_minutes/1440
    maximum = min(config.fwhm_max_minutes/1440, np.ptp(lc.time[noise_indices])/3)
    if maximum <= minimum:
        raise ValueError("No width prior supported by local observations")
    widths = np.geomspace(minimum, maximum, config.broad_fwhm_points)
    # Symmetric, identical peak prior for forward and time-reversed shapes.
    peaks = candidate_peak_grid(lc.time, candidate, cadence, config)
    grid, broad_logbf = refined_grid(t, y, variance, peaks, widths, exposure, config)
    result = measurement_summary(t, y, variance, exposure, grid, config)
    result.update(seed_time=float(peak), observed_width_hint_days=float(hint), sigma=float(sigma), seed_snr=float((residual[candidate.peak_index]-(prepared.local_baseline[candidate.peak_index] if prepared.local_baseline is not None else 0.))/sigma),
                  logbf_broad=float(broad_logbf), cadence_seconds=cadence*86400,
                  exposure_seconds=exposure*86400, n_local=len(t), n_quiet=int(quiet.sum()),
                  local_sigma_min=float(np.sqrt(variance.min())), local_sigma_max=float(np.sqrt(variance.max())))
    flare = flare_model(t, result["tpeak"], result["fwhm_days"], result["amplitude"], exposure,result["flare_model_name"])
    design, weight, inverse, _ = _baseline_projection(t, y, variance)
    beta = inverse @ (design.T @ (weight * (y-flare)))
    fitted = design@beta + flare
    standardized = (y-fitted)/np.sqrt(variance)
    result["chi2_red"] = float(np.sum(standardized**2)/max(1,len(y)-5))
    supported = flare >= max(.05*max(flare), sigma)
    result["chi2_event"] = float(np.mean(standardized[supported]**2)) if supported.any() else np.nan
    result["sampled_model_amplitude"] = float(np.max(flare))
    result["fitted_peak_snr"] = result["sampled_model_amplitude"] / sigma
    result["observed_peak_amplitude"] = float(np.max(y-design@beta))
    result["baseline_offset"], result["baseline_slope_scaled"] = map(float, beta)
    result["fit_center"], result["fit_span"] = float(np.median(t)), float(np.ptp(t))
    result["tstart"], result["tend"] = map(float, template_energy_bounds(
        result["tpeak"], result["fwhm_days"], result["flare_model_name"], .99))
    result["duration_days"] = result["tend"]-result["tstart"]
    total = result["equivalent_duration_s"] / 86400
    segment = np.flatnonzero(prepared.segment_id == candidate.segment)
    observed = flare_model(lc.time[segment], result["tpeak"], result["fwhm_days"],
                             result["amplitude"], exposure,result["flare_model_name"]).sum()*exposure
    result["observed_model_fraction"] = float(min(1., observed/total))
    reasons, warnings = [], []
    if result["log_bayes_factor"] < 0.:
        tier = "rejected"
        reasons.append("log_bayes_factor_below_threshold")
        result["logbf_forward_minus_reverse"] = np.nan
    else:
        reverse, _ = refined_grid(-t[::-1], y[::-1], variance[::-1], -peaks[::-1], widths, exposure, config)
        result["logbf_forward_minus_reverse"] = result["log_bayes_factor"] - grid_log_evidence(reverse)
        if config.detection_statistic == "background" and result["logbf_forward_minus_reverse"] < config.direction_threshold:
            reasons.append("flare_direction_not_established")
        impulse,raw_step,symmetric,raw_symmetric = artifact_checks(prepared,indices,t,y,variance,result,config,peaks,widths)
        result["logbf_impulse"] = impulse
        result["raw_step_minus_flare_diagnostic"] = raw_step
        result["logbf_symmetric"] = symmetric
        result["logbf_flare_minus_symmetric"] = result["log_bayes_factor"]-symmetric
        result["raw_symmetric_minus_flare_diagnostic"] = raw_symmetric
        # Proper equal-weight mixture of four alternatives on the SAME
        # detrended samples, variance, and affine background: no flare,
        # reversed flare family, symmetric bump, and single-sample impulse.
        result["logbf_flare_vs_alternatives"] = result["log_bayes_factor"] - (
            special.logsumexp([0.,result["log_bayes_factor"]-result["logbf_forward_minus_reverse"],
                              symmetric,impulse])-np.log(4.))
        if config.detection_statistic == "background" and result["logbf_flare_minus_symmetric"] < config.symmetric_margin:
            reasons.append("symmetric_shape_not_excluded")
        if np.isfinite(raw_symmetric) and raw_symmetric > np.log(10.0):
            reasons.append("raw_symmetric_shape_not_excluded")
        if config.detection_statistic == "background" and result["log_bayes_factor"]-impulse < config.impulse_margin:
            reasons.append("single_exposure_shape_ambiguous")
        if np.isfinite(raw_step) and raw_step > -config.raw_step_margin:
            reasons.append("raw_discontinuity_not_excluded")
        if not config.already_detrended and (not np.isfinite(result["raw_logbf_flare"]) or
                result["raw_logbf_flare"] < config.raw_signal_threshold):
            reasons.append("raw_flare_signal_not_established")
        # An unresolved width alone does not invalidate a flare detection.
        # Direction and impulse comparisons still guard its morphology.
        # Gaps crossing the peak/core are not a confident morphology detection.
        lower, upper = lc.time[segment[[0,-1]]]
        if min(result["tpeak"]-lower, upper-result["tpeak"]) < 2*cadence:
            reasons.append("peak_near_gap_or_edge")
        extra = np.flatnonzero((standardized > 3) & supported)
        if any(len(run) >= 2 for run in _contiguous_groups(extra, t, cadence, 1.5)):
            # Imperfect single-template measurements do not negate a detection.
            warnings.append("structured_positive_residual")
        # User-requested review rule, applied to the exposure-averaged signal
        # above its fitted baseline, not the unobserved instantaneous peak.
        if result["fitted_peak_snr"] < 3.0:
            reasons.append("fitted_peak_below_three_sigma")
        clear_artifact=(result["logbf_forward_minus_reverse"] < -np.log(10.) or
            result["logbf_flare_minus_symmetric"] < -np.log(10.) or
            (np.isfinite(raw_step) and raw_step > np.log(10.)) or
            (np.isfinite(raw_symmetric) and raw_symmetric > np.log(10.)) or
            result["log_bayes_factor"]-impulse < -np.log(10.) or
            result["fitted_peak_snr"] < 3. or
            (not config.already_detrended and np.isfinite(result["raw_logbf_flare"]) and result["raw_logbf_flare"] < config.raw_signal_threshold))
        # The alternative mixture already contains reverse/symmetric/impulse
        # models. Requiring separate weak-evidence margins double-gates coarse
        # exposures. Require the joint score to exceed ln(4): because each of
        # four alternatives has weight 1/4, this also exceeds EVERY individual
        # alternative's evidence. Keep raw-data artifact checks independent.
        weak = []
        for label, value, margin in (
            ("weak_direction_evidence", result["logbf_forward_minus_reverse"], config.direction_threshold),
            ("weak_symmetric_separation", result["logbf_flare_minus_symmetric"], config.symmetric_margin),
            ("weak_impulse_separation", result["log_bayes_factor"]-impulse, config.impulse_margin)):
            if value < margin: weak.append(label)
        result["morphology_diagnostics"] = ";".join(weak)
        if config.detection_statistic == "alternatives" and result["logbf_flare_vs_alternatives"] < np.log(4.0):
            reasons.append("insufficient_competing_model_evidence")
            clear_artifact = True
        tier = "rejected" if clear_artifact else "followup" if reasons else "accepted"
    if result["fwhm_days"] < 1.5*exposure:
        warnings.append("cadence_limited")
    if result["fwhm_days_bayesian_p84"] / max(result["fwhm_days_bayesian_p16"], 1e-15) > 3:
        warnings.append("broad_width_posterior")
    if result["width_prior_boundary_mass"] > .1:
        warnings.append("width_prior_boundary")
    if result["observed_model_fraction"] < .9:
        warnings.append("extrapolated_model_area")
    if config.chi2_fit and not result["chi2_fit_success"]:
        warnings.append("chi2_not_converged")
    if "parameter_bound" in result["chi2_fit_flags"]:
        warnings.append("chi2_parameter_bound")
    result.update(tier=tier, reason=";".join(reasons), warnings=";".join(warnings),
                  parameter_usable=not warnings and tier=="accepted", fit_seconds=time.perf_counter()-start)
    # A physical band luminosity must be supplied explicitly. Do not quietly
    # interpret a Vega-system TESS magnitude as an AB magnitude.
    luminosity = config.tess_band_luminosity_erg_s
    if luminosity is not None and np.isfinite(luminosity) and luminosity > 0:
        result["flare_energy_erg"] = luminosity*result["equivalent_duration_s"]
        result["flare_energy_erg_bayesian"] = luminosity*result["equivalent_duration_s_bayesian"]
        for p in (16,50,84):
            result["flare_energy_erg_bayesian_p"+str(p)] = luminosity*result["equivalent_duration_s_bayesian_p"+str(p)]
        result["energy_status"] = "model_energy_conditional_on_supplied_band_luminosity"
    else:
        result["flare_energy_erg"] = np.nan
        result["flare_energy_erg_bayesian"] = np.nan
        result["energy_status"] = "requires_calibrated_tess_band_luminosity"
    result["fit_seconds"] = time.perf_counter()-start
    return result, dict(indices=indices, fitted=fitted, flare=flare, variance=variance)


_CANDIDATE_WORKER_STATE = None


def _initialize_candidate_worker(prepared,candidates,config):
    """Copy read-only light-curve state once per candidate worker."""
    global _CANDIDATE_WORKER_STATE
    _CANDIDATE_WORKER_STATE = prepared,candidates,config


def _fit_one(prepared,candidates,index,config):
    """Fit one independent event; preserve explicit numerical failure reasons."""
    candidate=candidates[index]
    if prepared.detrend_edge_mask[candidate.peak_index]:
        return dict(tier="rejected", reason="edge_fraction_cut",
                    seed_time=float(prepared.lightcurve.time[candidate.peak_index]),
                    tpeak=float(prepared.lightcurve.time[candidate.peak_index]),
                    parameter_usable=False), None
    try:
        return characterize(prepared,candidate,candidates,config)
    except (ValueError,FloatingPointError,np.linalg.LinAlgError) as error:
        return dict(tier="followup",reason="fit_failed:"+str(error),
                    seed_time=float(prepared.lightcurve.time[candidate.peak_index])),None


def _candidate_worker(index):
    """Worker tasks use indices so candidate identity remains consistent."""
    prepared,candidates,config=_CANDIDATE_WORKER_STATE
    return _fit_one(prepared,candidates,index,config)


def associate_tails(results, candidates, prepared, fits, config):
    """Keep explained tail seeds from becoming independent accepted flares.

    Accepted parents retain the existing deduplication rule. A parent still
    under review can only move a child to review, never reject it. Its identity
    is recorded; its uncertain model is NOT subtracted for any parameter fit.
    """
    lc = prepared.lightcurve
    explained = np.zeros(len(lc.time))
    pending = []
    exposure = lc.exposure_days or cadence_days(lc.time)
    order = sorted(range(len(results)), key=lambda i: results[i].get("amplitude", 0), reverse=True)
    for i in order:
        row, candidate = results[i], candidates[i]
        if row["tier"] == "rejected" or fits[i] is None:
            continue
        seeds = candidate.seed_indices
        threshold = config.candidate_sigma * row["sigma"]
        new_excess = prepared.residual[seeds] - explained[seeds]
        if prepared.local_baseline is not None:new_excess-=prepared.local_baseline[seeds]
        if np.max(explained[seeds]) > .25*row["sigma"] and np.max(new_excess) < threshold:
            row.update(tier="rejected", reason="seed_explained_by_stronger_flare", parameter_usable=False)
            continue
        for parent, segment in pending:
            if segment != candidate.segment or row["tpeak"] <= parent["tpeak"]:
                continue
            tail = flare_model(lc.time[seeds], parent["tpeak"], parent["fwhm_days"],
                                 parent["amplitude"], exposure,parent.get("flare_model_name","tovar2022"))
            if np.max(tail) > .25*row["sigma"] and np.max(new_excess-tail) < threshold:
                row.update(tier="followup", parameter_usable=False,
                           parent_candidate_id=parent["candidate_id"],
                           reason=";".join(filter(None, [row["reason"], "possible_tail_of_followup_flare"])))
                break
        if row["tier"] == "accepted":
            same_segment = prepared.segment_id == candidate.segment
            explained[same_segment] += flare_model(lc.time[same_segment], row["tpeak"], row["fwhm_days"],
                                                     row["amplitude"], exposure,row.get("flare_model_name","tovar2022"))
        elif "possible_tail_of_followup_flare" not in row["reason"]:
            pending.append((row, candidate.segment))



def concise_flags(row):
    """Separate concise catalogue flags from complete scientific diagnostics."""
    reasons=str(row.get('reason','')); warnings=str(row.get('warnings',''))
    row['diagnostic_reasons']=reasons
    row['parameter_diagnostics']=warnings
    row['warnings']='cadence_limited' if 'cadence_limited' in warnings.split(';') else ''
    if row.get('tier')=='accepted':
        row['reason']=''
    elif row.get('tier')=='followup':
        if 'edge_fraction_cut' in reasons:label='edge_region'
        elif 'calibration_' in reasons:label='calibration_required'
        elif 'overflow' in reasons:label='processing_limit'
        elif 'fit_failed' in reasons:label='fit_failed'
        elif 'background' in reasons:label='background_uncertain'
        elif 'gap_or_edge' in reasons:label='incomplete_event'
        elif 'tail' in reasons:label='blended_event'
        else:label='shape_ambiguous'
        row['reason']=label
    return row


COMPACT_DIAGNOSTIC_FIELDS={'diagnostic_reasons','parameter_diagnostics','chi2_fit_flags','chi2_fit_status','morphology_diagnostics'}

def compact_table(frame):
    return frame.drop(columns=[k for k in COMPACT_DIAGNOSTIC_FIELDS if k in frame])


def route_edge_review(row):
    """Expose edge exclusions for review after all detection decisions finish.

    Edge-seeded events retain no fit; fitted events whose peak crosses the edge
    retain their existing fit. Neither may change tail association or automatic
    acceptance of interior events. Independent evidence failures stay rejected.
    """
    reasons = set(str(row.get("reason", "")).split(";"))
    if (row.get("tier") == "rejected" and "edge_fraction_cut" in reasons
            and "log_bayes_factor_below_threshold" not in reasons):
        row.update(tier="followup", parameter_usable=False)
    row["edge_review_only"] = row.get("tier") == "followup" and "edge_fraction_cut" in reasons


def analyze(lc, config=Config(), progress=None, *, associate=True, search_bounds=None):
    """Process one light curve; retain every candidate and every decision."""
    start = time.perf_counter()
    config, calibration_status = calibrated_config(lc, config)
    prepared, diagnostics = prepare(lc, config)
    detrend_seconds = time.perf_counter()-start
    all_candidates = find_candidates(prepared, config)
    indices_to_fit = [i for i,c in enumerate(all_candidates) if search_bounds is None or
                      search_bounds[0] <= lc.time[c.peak_index] <= search_bounds[1]]
    candidates = [all_candidates[i] for i in indices_to_fit]
    if progress is not None:
        progress([], len(candidates), detrend_seconds)
    results, fits = [], []
    with ExitStack() as stack:
        if config.max_candidates and len(candidates)>config.max_candidates:
            computed=((dict(tier="followup",reason="candidate_overflow_not_fitted",
                            seed_time=float(lc.time[c.peak_index])),None) for c in candidates)
        elif config.candidate_workers>1 and len(candidates)>1:
            executor=stack.enter_context(ProcessPoolExecutor(max_workers=config.candidate_workers,
                initializer=_initialize_candidate_worker,initargs=(prepared,all_candidates,config)))
            computed=executor.map(_candidate_worker,indices_to_fit)
        else:
            computed=(_fit_one(prepared,all_candidates,i,config) for i in indices_to_fit)
        for number,(result,fit) in enumerate(computed):
            result.update(candidate_id=number,gdr3_id=lc.gdr3_id,sector=lc.sector,
                          mission=lc.mission,source_path=lc.source_path)
            results.append(result);fits.append(fit)
            if progress is not None and (number+1)%25==0:
                progress(results,len(candidates),time.perf_counter()-start)
    diagnostics.update(detrend_seconds=detrend_seconds, total_seconds=time.perf_counter()-start)
    # A fitted peak can cross the edge boundary even if its seed did not.
    for row in results:
        row.update(calibration_status=calibration_status, logbf_threshold_used=config.logbf_threshold,
                   calibration_group=calibration_group(lc), edge_fraction=config.edge_fraction,
                   gaia_dr3_id=lc.gdr3_id)
        nearest = int(np.argmin(abs(lc.time-row.get("tpeak", row["seed_time"]))))
        if prepared.detrend_edge_mask[nearest] and row["tier"] != "rejected":
            row.update(tier="rejected", parameter_usable=False,
                       reason=";".join(filter(None, [row.get("reason", ""), "edge_fraction_cut"])))
        if row["tier"] == "accepted" and prepared.background_uncertain_mask[nearest]:
            row.update(tier="followup", parameter_usable=False,
                       reason=";".join(filter(None, [row.get("reason", ""), "insufficient_protected_background"])))
        if row["tier"] == "accepted" and calibration_status != "validated" and not config.allow_uncalibrated:
            row.update(tier="followup", parameter_usable=False,
                       reason=";".join(filter(None, [row.get("reason", ""), "calibration_"+calibration_status])))
    # Apply eligibility before association: an excluded parent must never
    # suppress a valid event through its extrapolated tail.
    if associate and (calibration_status == "validated" or config.allow_uncalibrated):
        associate_tails(results, candidates, prepared, fits, config)
    for row in results:
        row["detection_statistic"]=config.detection_statistic
        row["detection_log_bayes_factor"]=row.get(
            "logbf_flare_vs_alternatives" if config.detection_statistic=="alternatives" else "log_bayes_factor",np.nan)
    # Keep the morphology and tail-association decision independent of the
    # operating threshold. Otherwise removing a weak parent at a higher cut
    # can promote a previously suppressed child, invalidating score-based
    # calibration. Increasing the threshold may only remove detections.
    for row in results:
        threshold = operating_threshold(row, config)
        row["logbf_threshold_used"] = threshold
        row["operating_snr_class"] = "high" if row.get("fitted_peak_snr", -np.inf) >= config.high_snr_boundary else "ordinary"
        if (row["tier"] != "rejected" or "edge_fraction_cut" in str(row.get("reason", "")).split(";")) and row.get("detection_log_bayes_factor",np.inf) < threshold:
            row.update(tier="rejected",parameter_usable=False,
                reason=";".join(filter(None,[row.get("reason",""),"log_bayes_factor_below_threshold"])))
        # Do this only after association/thresholding: an edge event must not
        # suppress or demote a valid interior flare through an uncertain tail.
        route_edge_review(row)
        concise_flags(row)
    diagnostics["edge_review_candidates"] = sum(bool(row["edge_review_only"]) for row in results)
    diagnostics["calibration_status"] = calibration_status
    diagnostics["calibration_group"] = calibration_group(lc)
    diagnostics["logbf_threshold_used"] = config.logbf_threshold
    diagnostics["high_snr_logbf_threshold"] = config.high_snr_logbf_threshold
    frame=pd.DataFrame(results)
    if frame.empty:
        frame=pd.DataFrame(columns=['candidate_id','gdr3_id','sector','mission','source_path',
            'tier','reason','seed_time','tpeak','fwhm_days','amplitude','equivalent_duration_s',
            'flare_energy_erg','energy_status','gaia_dr3_id','calibration_status',
            'diagnostic_reasons','parameter_diagnostics','warnings',
            'logbf_threshold_used','calibration_group','edge_fraction','edge_review_only','detection_statistic','detection_log_bayes_factor'])
    return frame, prepared, fits, diagnostics


def save_plots(frame, prepared, fits, folder):
    """Save overview and all-tier contact sheets with sampled and continuous fits."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    folder.mkdir(parents=True, exist_ok=True)
    t = prepared.lightcurve.time
    colors = dict(accepted="#00835c", followup="#bc7300", rejected="#a72b48")
    fig, axes = plt.subplots(3, 1, figsize=(15,9), sharex=True, constrained_layout=True)
    axes[0].plot(t, prepared.normalized_flux, ".", ms=1.2, color="steelblue")
    axes[0].set_ylabel("Original relative flux")
    for ax in axes[1:]:
        ax.plot(t, prepared.residual, ".", ms=1.2, color="0.35")
        ax.plot(t, 3*prepared.segment_sigma, "--", lw=.8, color="royalblue")
        ax.axhline(0, color="0.6", lw=.6)
    axes[1].set_ylabel("Detrended residual")
    axes[2].set_ylabel("Candidates on residual")
    for row in frame.to_dict("records"):
        axes[2].axvline(row["seed_time"], color=colors[row["tier"]], alpha=.7, lw=.7)
    axes[2].set_xlabel("BTJD [days]")
    fig.savefig(folder/"overview.png", dpi=150); plt.close(fig)
    for tier in colors:
        rows = [(i,r) for i,r in enumerate(frame.to_dict("records")) if r["tier"]==tier]
        for offset in range(0, len(rows), 6):
            fig, axes = plt.subplots(3,2, figsize=(15,12), constrained_layout=True)
            for ax, item in zip(axes.flat, rows[offset:offset+6]):
                index, row = item
                fit = fits[index]
                center = row.get("tpeak", np.nan)
                center = center if np.isfinite(center) else row["seed_time"]
                width = row.get("fwhm_days", np.nan)
                width = width if np.isfinite(width) else 10/1440
                half = max(6*cadence_days(t), min(6*width, .2))
                local = np.flatnonzero(abs(t-center) <= half)
                if fit is not None:
                    used = np.isin(local, fit["indices"])
                    ax.plot((t[local[~used]]-center)*1440, prepared.residual[local[~used]], ".",
                            color=".75", ms=4, label="Excluded neighbor/context")
                    ax.plot((t[local[used]]-center)*1440, prepared.residual[local[used]], ".",
                            color=".25", ms=4, label="Data used in fit")
                    xx = np.linspace(t[local[0]], t[local[-1]], 1000)
                    base = row["baseline_offset"] + row["baseline_slope_scaled"]*2*(xx-row["fit_center"])/row["fit_span"]
                    method = "chi2" if row.get("measurement_method")=="chi2_profile_likelihood" else "Bayesian"
                    ax.plot((xx-center)*1440, base+flare_model(xx,center,width,row["amplitude"],
                            row["exposure_seconds"]/86400,row.get("flare_model_name","tovar2022")), color=colors[tier], lw=1.2,
                            label=f"Exposure-averaged Tovar ({method})")
                    inside = abs(t[fit["indices"]]-center) <= half
                    ax.plot((t[fit["indices"]][inside]-center)*1440,fit["fitted"][inside],"x",color=colors[tier],ms=4)
                    ax.plot((xx-center)*1440, base+3*row["sigma"],ls="--",color="royalblue",lw=.8,
                            label="Local baseline + 3 sigma")
                    if not used.all():
                        ax.legend(fontsize=6, loc="upper right")
                else:
                    ax.plot((t[local]-center)*1440, prepared.residual[local], ".", color=".25", ms=4)
                title = (f"#{row['candidate_id']} BTJD {center:.5f} | ln BF={row.get('log_bayes_factor',np.nan):.1f}"
                         f" | chi2red={row.get('chi2_red',np.nan):.2f}"
                         f" | fit peak/sigma={row.get('fitted_peak_snr',np.nan):.2f}\n"+str(row.get("reason", ""))
                         + (" [cadence limited]" if "cadence_limited" in str(row.get("warnings","")) else ""))
                ax.set_title(title, fontsize=8, wrap=True)
                ax.set_xlabel("Minutes from fitted peak"); ax.set_ylabel("Relative residual")
                ax.grid(alpha=.15)
            for ax in list(axes.flat)[len(rows[offset:offset+6]):]:
                ax.set_visible(False)
            fig.savefig(folder/f"{tier}_{offset//6+1:03d}.png",dpi=140); plt.close(fig)


ENERGY_CALIBRATION_FIELDS = ["tess_band_luminosity_erg_s", "tess_magnitude",
    "distance_pc", "stellar_catalog_source", "energy_calibration_method",
    "energy_calibration_status", "tess_effective_bandwidth_hz"]


def energy_calibration(lc, config):
    """Resolve one star's band luminosity; never substitute bolometric Lbol.

    Optional legacy effective-bandwidth approximation:
    L = 4*pi*d_cm**2 * (2416 Jy)*10**(-0.4*Tmag) * 1e-23 * bandwidth_Hz.
    Vega-like flux-density zero point: https://tess.mit.edu/public/tesstransients/pages/readme.html
    The 1.2e14 Hz bandwidth is our inherited convention, not a calibrated
    SED/response integral. No extinction, blending, or bolometric correction.
    """
    result = dict(tess_band_luminosity_erg_s=None, tess_magnitude=lc.tess_magnitude,
                  distance_pc=lc.distance_pc, stellar_catalog_source=lc.stellar_catalog_source,
                  energy_calibration_method=config.energy_method,
                  energy_calibration_status="requires_calibrated_tess_band_luminosity",
                  tess_effective_bandwidth_hz=None)
    if config.energy_method == "supplied":
        luminosity = _finite_float(config.tess_band_luminosity_erg_s)
    elif config.energy_method == "tmag-bandwidth":
        if config.tess_band_luminosity_erg_s is not None:
            raise ValueError("Choose a supplied luminosity OR TESSMAG calibration, not both")
        bandwidth = _finite_float(config.tess_effective_bandwidth_hz)
        if bandwidth is None or bandwidth <= 0:
            raise ValueError("Effective bandwidth must be positive finite Hz")
        result["tess_effective_bandwidth_hz"] = bandwidth
        result["energy_calibration_method"] = "tmag_vega_effective_bandwidth_approximation"
        distance, tmag = _finite_float(lc.distance_pc), _finite_float(lc.tess_magnitude)
        if distance is None or distance <= 0:
            result["energy_calibration_status"] = "missing_or_invalid_distance"
            return result
        if tmag is None:
            result["energy_calibration_status"] = "missing_or_invalid_tess_magnitude"
            return result
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            luminosity = float(4*np.pi*np.square(np.float64(distance)*3.0856775814913673e18)
                               * 2416.0 * np.power(10., -.4*tmag) * 1e-23 * bandwidth)
        if not np.isfinite(luminosity) or luminosity <= 0:
            result["energy_calibration_status"] = "invalid_tmag_bandwidth_luminosity"
            return result
    else:
        raise ValueError(f"Unknown energy method: {config.energy_method}")
    if luminosity is not None and np.isfinite(luminosity) and luminosity > 0:
        result["tess_band_luminosity_erg_s"] = luminosity
        result["energy_calibration_status"] = "available"
    return result


def annotate_energy(frame, calibration):
    """Keep calibration provenance and missing-input reasons in every table."""
    for name in ENERGY_CALIBRATION_FIELDS:
        frame[name] = calibration[name]
    if len(frame) and calibration["energy_calibration_method"] != "supplied":
        energy = pd.to_numeric(frame.get("flare_energy_erg", pd.Series(np.nan, index=frame.index)), errors="coerce")
        fitted = frame.get("measurement_method", pd.Series(None, index=frame.index, dtype=object)).notna()
        frame["energy_status"] = "not_fitted"
        frame.loc[fitted, "energy_status"] = calibration["energy_calibration_status"]
        frame.loc[np.isfinite(energy) & (energy > 0), "energy_status"] = "model_energy_tmag_bandwidth_approximation"
        invalid = fitted & (calibration["energy_calibration_status"] == "available") & ~(np.isfinite(energy) & (energy > 0))
        frame.loc[invalid, "energy_status"] = "invalid_model_energy"
    return frame



# ======================================================================
# LOCAL NOISE, EDGE POLICY, AND CALIBRATION CONTRACT
# ======================================================================
import csv
import fcntl
import platform
import shutil
from concurrent.futures import wait, FIRST_COMPLETED
from importlib.metadata import version, PackageNotFoundError
from scipy.stats import beta as beta_distribution

DETECTOR_VERSION = "2026.11.09.share1"

# Saved SAP calibration. The fit/morphology engine is unchanged from the source
# revision below. No file, network request, or new injection run is needed.
BUILTIN_CALIBRATION = {'calibration_id': 'SAP_20261109',
 'source_detector_version': '2026.11.03.8',
 'source_code_sha256': '67a1edfc7a21f3bfd6352327f6c51c067596e65d951781cd7018eb6886c4b9fa',
 'source_calibration_sha256': '144aec1ab768409895784507ad14cce46a13f296a2321b0a7c73123ed8a2c3f2',
 'false_alarm_target': 0.05,
 'confidence': 0.95,
 'target_unit': 'See group_provenance: 20s, 120s, 200s and 1800s use screened controls; 600s '
                'retains its prior control protocol',
 'analysis_signature': {'already_detrended': False,
                        'amplitude_prior_scales': [0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0],
                        'broad_fwhm_points': 41,
                        'broad_tpeak_points': 9,
                        'candidate_sigma': 3.0,
                        'chi2_fit': True,
                        'chi2_max_nfev': 200,
                        'detailed_fwhm_points': 121,
                        'detailed_mass': 0.995,
                        'detailed_tpeak_points': 61,
                        'detection_statistic': 'alternatives',
                        'direction_threshold': 2.302585092994046,
                        'edge_fraction': 0.05,
                        'edge_scope': 'lightcurve',
                        'expand_sigma': 1.0,
                        'flare_model': 'tovar2022',
                        'flux_column': 'SAP_FLUX',
                        'flux_is_residual': False,
                        'fwhm_max_minutes': 2880.0,
                        'fwhm_min_minutes': 0.1,
                        'gap_factor': 5.0,
                        'high_snr_boundary': 8.0,
                        'ilin': {'clip_sigma': 3.5,
                                 'cosine_iterations': 5,
                                 'gap_hours': 2.0,
                                 'noise_clip_sigma': 1.5,
                                 'noise_hours': 2.0,
                                 'savgol_hours': [6.0, 3.0],
                                 'spline_hours': 6.0,
                                 'spline_min_days': 5.0},
                        'impulse_margin': 1.0,
                        'local_window_hours': 12.0,
                        'noise_min_points': 41,
                        'noise_window_hours': 2.0,
                        'protection_iterations': 2,
                        'protection_padding': 2.0,
                        'quality_zero_only': True,
                        'raw_signal_threshold': 5.0,
                        'raw_step_margin': 2.0,
                        'rise_factors': [0.5, 1.0, 2.0],
                        'sigma_clip': 3.0,
                        'sigma_clip_iterations': 10,
                        'symmetric_margin': 0.0},
 'groups': {'QLP|SAP_FLUX|cadence=1800s|exposure=1800.00s': {'status': 'validation_failed',
                                                             'logbf_threshold': 1.3862943611198906,
                                                             'high_snr_logbf_threshold': 1.3862943611198906,
                                                             'high_snr_boundary': 8.0,
                                                             'reasons': ['holdout_false_alarm_bound_exceeds_target'],
                                                             'holdout_false_alarm_upper_95': 0.05014067956474021,
                                                             'provenance': {'date': '2026-11-09',
                                                                            'file': 'calibration_rechecked.json',
                                                                            'sha256': 'd783ebf99606bc06917dbfe54985252c6e7f7f90ffb8b3dd5f82e11be3db83f1',
                                                                            'target_unit': 'Probability '
                                                                                           'of at '
                                                                                           'least '
                                                                                           'one '
                                                                                           'accepted '
                                                                                           'event '
                                                                                           'per '
                                                                                           '24h '
                                                                                           'pre-injection '
                                                                                           'transient-screened '
                                                                                           'background '
                                                                                           'window '
                                                                                           'under '
                                                                                           'the '
                                                                                           'equal-weight '
                                                                                           'simulated '
                                                                                           'control '
                                                                                           'mixture'}},
            'QLP|SAP_FLUX|cadence=200s|exposure=200.00s': {'status': 'validated',
                                                           'logbf_threshold': 1.3862943611198906,
                                                           'high_snr_logbf_threshold': 1.3862943611198906,
                                                           'high_snr_boundary': 8.0,
                                                           'reasons': [],
                                                           'holdout_false_alarm_upper_95': 0.030605341336622634,
                                                           'provenance': {'date': '2026-11-09',
                                                                          'file': 'calibration_200s.json',
                                                                          'sha256': '4591e39fe6f6df596b0b93fe9174124e570a1fdc0a5169ccd1dc6f6f309a393d',
                                                                          'target_unit': 'Probability '
                                                                                         'of at '
                                                                                         'least '
                                                                                         'one '
                                                                                         'accepted '
                                                                                         'event '
                                                                                         'per 24h '
                                                                                         'pre-injection '
                                                                                         'transient-screened '
                                                                                         'background '
                                                                                         'window '
                                                                                         'under '
                                                                                         'the '
                                                                                         'equal-weight '
                                                                                         'simulated '
                                                                                         'control '
                                                                                         'mixture'}},
            'QLP|SAP_FLUX|cadence=600s|exposure=600.00s': {'status': 'validated',
                                                           'logbf_threshold': 1.3862943611198906,
                                                           'high_snr_logbf_threshold': 1.3862943611198906,
                                                           'high_snr_boundary': 8.0,
                                                           'reasons': [],
                                                           'holdout_false_alarm_upper_95': 0.03072367683353283,
                                                           'provenance': {'date': '2026-11-08',
                                                                          'file': 'calibration_previous_20261108.json',
                                                                          'sha256': '18950f61c999c80a7fafb6b603d4a1d4338322ed13b82a729b7b234e5c569603',
                                                                          'target_unit': 'Probability '
                                                                                         'of at '
                                                                                         'least '
                                                                                         'one '
                                                                                         'accepted '
                                                                                         'event '
                                                                                         'per 24h '
                                                                                         'quiet-background '
                                                                                         'control '
                                                                                         'window '
                                                                                         'under '
                                                                                         'the '
                                                                                         'balanced '
                                                                                         'simulated '
                                                                                         'control '
                                                                                         'mixture'}},
            'SPOC|SAP_FLUX|cadence=120s|exposure=118.80s': {'status': 'validated',
                                                            'logbf_threshold': 1.4221729029775079,
                                                            'high_snr_logbf_threshold': 2.8781335878400403,
                                                            'high_snr_boundary': 8.0,
                                                            'reasons': [],
                                                            'holdout_false_alarm_upper_95': 0.023677685118057887,
                                                            'provenance': {'date': '2026-11-09',
                                                                           'file': 'calibration_rechecked.json',
                                                                           'sha256': 'd783ebf99606bc06917dbfe54985252c6e7f7f90ffb8b3dd5f82e11be3db83f1',
                                                                           'target_unit': 'Probability '
                                                                                          'of at '
                                                                                          'least '
                                                                                          'one '
                                                                                          'accepted '
                                                                                          'event '
                                                                                          'per 24h '
                                                                                          'pre-injection '
                                                                                          'transient-screened '
                                                                                          'background '
                                                                                          'window '
                                                                                          'under '
                                                                                          'the '
                                                                                          'equal-weight '
                                                                                          'simulated '
                                                                                          'control '
                                                                                          'mixture'}},
            'SPOC|SAP_FLUX|cadence=20s|exposure=19.80s': {'status': 'validated',
                                                          'logbf_threshold': 4.1721795720870976,
                                                          'high_snr_logbf_threshold': 1.3862943611198906,
                                                          'high_snr_boundary': 8.0,
                                                          'reasons': [],
                                                          'holdout_false_alarm_upper_95': 0.012293754345837998,
                                                          'provenance': {'date': '2026-11-09',
                                                                         'file': 'calibration_rechecked.json',
                                                                         'sha256': 'd783ebf99606bc06917dbfe54985252c6e7f7f90ffb8b3dd5f82e11be3db83f1',
                                                                         'target_unit': 'Probability '
                                                                                        'of at '
                                                                                        'least one '
                                                                                        'accepted '
                                                                                        'event per '
                                                                                        '24h '
                                                                                        'pre-injection '
                                                                                        'transient-screened '
                                                                                        'background '
                                                                                        'window '
                                                                                        'under the '
                                                                                        'equal-weight '
                                                                                        'simulated '
                                                                                        'control '
                                                                                        'mixture'}}},
 'caveats': ['Small local background sample; sources were used during detector development',
             'New injected signals and star-disjoint threshold selection/checking, not external '
             'population validation',
             'Conditional binomial bounds treat randomized trials as draws from this finite '
             'background/control mixture',
             'No claim about survey catalog contamination or flare occurrence distribution',
             'Only events seeded inside the scored 24h window are fit; full original LC is '
             'detrended and all seeds enter neighbor masks',
             'Parents seeded outside the scored window are not tail-associated',
             'Placement failures stay recorded and prevent a validated label; no replacement draws',
             'The nominal binomial bounds treat windows as independent draws from the declared '
             'equal-weight control mixture. Reused stars/windows can introduce dependence; the '
             'bounds do not establish a survey-wide rate or a separate 5% bound for each artifact '
             'type.',
             'Holdout uses four source stars absent from the November 8 200-second calibration; '
             'three of the four stars appeared in its 20-second group; these sources were used in '
             'earlier detector development, so this is not external population validation',
             'Window selection removes pre-existing positive AND negative coherent excursions '
             'independent of detector fits; rates apply only to this screened background '
             'population',
             '20s retains the prior 1-training/2-holdout-star split because only three local 20s '
             'sources are available; the stars and windows are reused. 120s uses six training and '
             'five held-out stars; 1800s uses six training and four held-out stars. Those held-out '
             'stars are not in their prior cadence-specific calibration. Other-cadence and prior '
             'detector-development reuse remains; no external population validation']}


SAMPLES = ("open_clusters", "field_stars")
TABLES = {"accepted": "accepted_flares", "followup": "followup_flares", "rejected": "rejected_candidates"}
CONTROL_KINDS = ("quiet", "impulse", "step", "reverse_flare", "symmetric_bump", "correlated_noise")


def local_noise(prepared, config, difference_only=False):
    """Time-varying quiet scatter, with gap isolation and formal-error floors.

    Exclude the protected flare intervals from residual scatter AND differences.
    A sparse local window borrows nearest quiet observations within its own
    segment. Record this wider support; never difference across a masked hole.
    Bootstrap mode combines differences with lower-tail scatter, limiting
    flare contamination without discarding correlated background variability.
    Final mode also retains quiet residual scatter.
    """
    t, y = prepared.lightcurve.time, prepared.residual
    result = prepared.segment_sigma.copy()
    protected = (np.zeros(len(t), bool) if prepared.protection_mask is None
                 else prepared.protection_mask)
    diagnostics = []
    for segment in np.unique(prepared.segment_id[prepared.segment_id >= 0]):
        ii = np.flatnonzero(prepared.segment_id == segment)
        tt, yy = t[ii], y[ii]
        dt = cadence_days(tt)
        window = max(config.noise_window_hours/24, config.noise_min_points*dt)
        knots = np.unique(np.r_[np.arange(tt[0], tt[-1], window/4), tt[-1]])
        quiet = np.flatnonzero(~protected[ii])
        estimates, support, spans, fallback = [], [], [], 0
        for knot in knots:
            inside = quiet[abs(tt[quiet]-knot) <= window/2]
            if len(inside) < min(config.noise_min_points, len(quiet)):
                inside = np.sort(quiet[np.argsort(abs(tt[quiet]-knot), kind="stable")[:config.noise_min_points]])
                fallback += 1
            values = yy[inside]
            adjacent = (np.diff(inside) == 1) & (np.diff(tt[inside]) < 1.5*dt)
            diff_sigma = robust_std(np.diff(values)[adjacent])/np.sqrt(2) if adjacent.sum() >= 8 else np.nan
            estimates_here = [diff_sigma] if np.isfinite(diff_sigma) and diff_sigma > 0 else []
            if difference_only and len(values) >= 15:
                # Positive flares do not set their own masking threshold, but
                # correlated background variability must not be mistaken for
                # thousands of flares. Gaussian-normalized lower-tail spacing
                # supplies an amplitude scale as well as the difference scale.
                lower_sigma = float(np.diff(np.quantile(values, [.05, .25]))[0] /
                                    (special.ndtri(.25)-special.ndtri(.05)))
                if np.isfinite(lower_sigma) and lower_sigma > 0:
                    estimates_here.append(lower_sigma)
            if not difference_only and len(inside) >= 15:
                _, scatter, _ = sigma_clip_location_scale(values, config.sigma_clip, config.sigma_clip_iterations)
                if np.isfinite(scatter) and scatter > 0:
                    estimates_here.append(scatter)
            if not difference_only:
                # Masking positive variability can leave a deceptively flat
                # subset. Retain an independent lower-tail scale from ALL
                # nonpositive observations in this local window. Conditional
                # quantiles .1/.5 correspond to Gaussian .05/.25; positive
                # broad flare tails cannot set their own noise floor. This is a conservative noise
                # floor, including unresolved negative/background structure.
                local = yy[(abs(tt-knot) <= window/2) & (yy <= 0)]
                if len(local) >= 15:
                    floor = float(np.diff(np.quantile(local,[.1,.5]))[0] /
                                  (special.ndtri(.25)-special.ndtri(.05)))
                    if np.isfinite(floor) and floor > 0:
                        estimates_here.append(floor)
            if prepared.relative_flux_err is not None and len(inside):
                err = prepared.relative_flux_err[ii[inside]]
                err = err[np.isfinite(err) & (err > 0)]
                if len(err):
                    estimates_here.append(float(np.median(err)))
            if not estimates_here:
                estimates_here.append(float(prepared.segment_sigma[ii[0]]))
                fallback += 1
            estimates.append(max(estimates_here))
            support.append(len(inside)); spans.append(float(np.ptp(tt[inside])) if len(inside) else 0.)
        result[ii] = np.interp(tt, knots, estimates)
        if prepared.relative_flux_err is not None:
            err = prepared.relative_flux_err[ii]
            valid = np.isfinite(err) & (err > 0)
            result[ii[valid]] = np.maximum(result[ii[valid]], err[valid])
        diagnostics.append(dict(segment=int(segment), window_hours=window*24,
            knots=len(knots), fallback_knots=fallback, min_points=min(support),
            maximum_support_hours=24*max(spans), protected_points=int(protected[ii].sum()),
            sigma_min=float(np.min(result[ii])), sigma_max=float(np.max(result[ii]))))
    return result, diagnostics


def quiet_local_baseline(prepared, config):
    """Quiet-only local location for detection; never alter observed samples.

    Fits continue to use the original prepared residual with their own affine
    background. This prevents a remaining offset from hiding a local flare or
    inflating its seed significance. Protected flares do not set the location.
    """
    t,y=prepared.lightcurve.time,prepared.residual
    level=np.zeros(len(t));protected=prepared.protection_mask
    if protected is None:protected=np.zeros(len(t),bool)
    for segment in np.unique(prepared.segment_id[prepared.segment_id>=0]):
        ii=np.flatnonzero(prepared.segment_id==segment);tt=t[ii]
        quiet=np.flatnonzero(~protected[ii])
        if len(quiet)<5:continue
        window=max(config.noise_window_hours/24,config.noise_min_points*cadence_days(tt))
        knots=np.unique(np.r_[np.arange(tt[0],tt[-1],window/4),tt[-1]])
        medians=[]
        for knot in knots:
            use=quiet[abs(tt[quiet]-knot)<=window/2]
            if len(use)<min(config.noise_min_points,len(quiet)):
                use=quiet[np.argsort(abs(tt[quiet]-knot),kind="stable")[:config.noise_min_points]]
            medians.append(float(np.median(y[ii[use]])))
        level[ii]=np.interp(tt,knots,medians)
    return level


def edge_mask(prepared, config):
    """Time-based fraction at each outer edge, or at each gap-defined segment.

    The default 0.05 removes the first AND last 5% of the time span. The
    observed exposure removed can differ from 10% when gaps are present.
    Existing two-cadence gap/edge morphology checks remain independent.
    """
    t = prepared.lightcurve.time
    mask = np.zeros(len(t), bool)
    if config.edge_fraction == 0:
        return mask
    segments = [np.arange(len(t))] if config.edge_scope == "lightcurve" else [
        np.flatnonzero(prepared.segment_id == s) for s in np.unique(prepared.segment_id) if s >= 0]
    for ii in segments:
        low, high = t[ii[0]], t[ii[-1]]
        margin = config.edge_fraction*(high-low)
        mask[ii] = (t[ii] < low+margin) | (t[ii] > high-margin)
    return mask


def calibration_group(lc):
    # Do not confuse exposure with cadence (e.g. 118.8s exposure at 120s cadence).
    cadence = cadence_days(lc.time)*86400
    nominal = np.array([20, 120, 200, 600, 1800])
    closest = int(nominal[np.argmin(abs(nominal-cadence))])
    label = str(closest) if abs(closest-cadence)/closest < .02 else f"{cadence:.2f}"
    exposure = lc.exposure_days or cadence/86400
    return f"{lc.mission}|{lc.selected_flux_column}|cadence={label}s|exposure={exposure*86400:.2f}s"


def analysis_signature(config):
    settings = asdict(config)
    for key in ("calibration_file", "builtin_calibration", "allow_uncalibrated", "logbf_threshold", "high_snr_logbf_threshold", "candidate_workers",
                "max_candidates", "stellar_catalog_paths", "distance_column", "energy_method",
                "tess_band_luminosity_erg_s", "tess_effective_bandwidth_hz"):
        settings.pop(key, None)
    return settings


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=8)
def read_calibration(path, mtime_ns, size):
    data = json.loads(Path(path).read_text())
    if data.get("detector_version") != DETECTOR_VERSION or data.get("code_sha256") != file_sha256(__file__):
        raise ValueError("Calibration was made with different code; recalibrate with this file")
    return data


def operating_threshold(row, config):
    cut = config.logbf_threshold
    if config.high_snr_logbf_threshold is not None and row.get("fitted_peak_snr", -np.inf) >= config.high_snr_boundary:
        cut = config.high_snr_logbf_threshold
    return float(max(np.log(4.0), cut) if config.detection_statistic == "alternatives" else cut)


def calibrated_config(lc, config):
    if not config.calibration_file:
        if not config.builtin_calibration or config.allow_uncalibrated:
            return config, "missing"
        data = BUILTIN_CALIBRATION
        # Automatic selection and an explicit request for the same SAP column
        # use identical data. Never transfer SAP cuts to a fallback flux stream.
        selected = replace(config, flux_column=lc.selected_flux_column)
        if data["analysis_signature"] != json.loads(json.dumps(analysis_signature(selected))):
            return config, "settings_mismatch"
    else:
        path = Path(config.calibration_file)
        info = path.stat()
        data = read_calibration(str(path.resolve()), info.st_mtime_ns, info.st_size)
        if data["analysis_signature"] != json.loads(json.dumps(analysis_signature(config))):
            raise ValueError("Calibration settings differ from analysis settings; recalibrate")
    row = data["groups"].get(calibration_group(lc))
    if row is None:
        return config, "group_missing"
    if row.get("status") != "validated":
        return config, "validation_failed"
    threshold = float(row["logbf_threshold"])
    if not np.isfinite(threshold):
        raise ValueError("Nonfinite calibration threshold")
    high_threshold = row.get("high_snr_logbf_threshold", threshold)
    if high_threshold is None or not np.isfinite(high_threshold):
        raise ValueError("Nonfinite high-SNR calibration threshold")
    return replace(config, logbf_threshold=threshold, high_snr_logbf_threshold=float(high_threshold)), "validated"


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name+".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))
    temporary.replace(path)


def atomic_frame(frame, path):
    path = Path(path)
    temporary = path.with_name(path.name+".tmp")
    if path.suffix == ".parquet":
        frame.to_parquet(temporary, index=False)
    else:
        frame.to_csv(temporary, index=False)
    temporary.replace(path)


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k,v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def fingerprint(path):
    info = Path(path).stat()
    return dict(resolved=str(Path(path).resolve(strict=True)), size=info.st_size, mtime_ns=info.st_mtime_ns)


def catalog_manifest(path, distance_column):
    path = path.expanduser().resolve(strict=True)
    lookup = stellar_catalog_lookup((str(path),), distance_column)
    return dict(path=str(path), sha256=file_sha256(path), input=fingerprint(path), stars=len(lookup),
                stars_with_distance=sum(row["distance_pc"] is not None for row in lookup.values()))


def product_folder(record, output):
    path = Path(record["path"])
    digest = hashlib.sha256(record["input"]["resolved"].encode()).hexdigest()[:10]
    return Path(output)/record["sample"]/"products"/(path.stem+"_"+digest)


def completed(record, output, run_id):
    folder = product_folder(record, output)
    try:
        receipt = json.loads((folder/"complete.json").read_text())
        summary = json.loads((folder/"summary.json").read_text())
        return (receipt["record"] == record and receipt["run_id"] == run_id and summary["status"] == "complete"
                and all((folder/name).is_file() and (folder/name).stat().st_size == size
                        for name,size in receipt["outputs"].items()))
    except (OSError, ValueError, KeyError):
        return False


def run_product(record, output, config, plots, run_id):
    """Atomic products; a failure cannot leave a successful completion marker."""
    path = Path(record["path"])
    folder = product_folder(record, output)
    folder.mkdir(parents=True, exist_ok=True)
    (folder/"complete.json").unlink(missing_ok=True)
    started = time.perf_counter()
    try:
        if fingerprint(path) != record["input"]:
            raise ValueError("Input changed after inventory")
        if record.get("catalog"):
            c = record["catalog"]
            if fingerprint(c["path"]) != c["input"]:
                raise ValueError("Stellar catalog changed")
            config = replace(config, stellar_catalog_paths=(c["path"],))
        lc = load_lightcurve(path, config)
        energy = energy_calibration(lc, config)
        config = replace(config, tess_band_luminosity_erg_s=energy["tess_band_luminosity_erg_s"])
        def checkpoint(rows, total, seconds):
            atomic_json(folder/"progress.json", dict(processed=len(rows), total=total, seconds=seconds))
        frame, prepared, fits_, diagnostics = analyze(lc, config, checkpoint)
        annotate_energy(frame, energy)
        frame["sample"] = {"field_stars":"FS", "open_clusters":"OC"}.get(record["sample"], record["sample"])
        frame["gaia_dr3_id"] = frame["gdr3_id"].astype("string")
        frame["gdr3_id"] = frame["gdr3_id"].astype("string")
        for name, subset in [("all_events", frame)]+[(name, compact_table(frame.loc[frame.tier == tier])) for tier,name in TABLES.items()]:
            atomic_frame(subset, folder/(name+".csv"))
            atomic_frame(subset, folder/(name+".parquet"))
        if plots:
            np.savez_compressed(folder/"prepared.npz", time=lc.time, original=prepared.normalized_flux,
                                residual=prepared.residual, sigma=prepared.segment_sigma,
                                edge_mask=prepared.detrend_edge_mask, segment_id=prepared.segment_id)
            save_plots(frame, prepared, fits_, folder)
        summary = dict(source=str(path), sample=record["sample"], output=str(folder), status="complete",
            candidates=len(frame), counts={tier:int((frame.tier == tier).sum()) for tier in TABLES},
            flux_column=lc.selected_flux_column, n_observations=len(lc.time),
            n_input=lc.n_input, n_quality_rejected=lc.n_quality_rejected,
            quality_filter_applied=lc.quality_filter_applied, n_duplicates_merged=lc.n_duplicates_merged,
            energy_calibration=energy, **{k:v for k,v in diagnostics.items() if k not in {"stages","ilin_noise"}})
        if fingerprint(path) != record["input"]:
            raise ValueError("Input changed during processing")
        if record.get("catalog") and fingerprint(record["catalog"]["path"]) != record["catalog"]["input"]:
            raise ValueError("Stellar catalog changed during processing")
        atomic_json(folder/"summary.json", json_safe(summary))
        names = ["summary.json"]+[name+suffix for name in ("all_events", *TABLES.values()) for suffix in (".csv", ".parquet")]
        atomic_json(folder/"complete.json", dict(record=record, run_id=run_id,
                    outputs={name:(folder/name).stat().st_size for name in names}))
    except Exception as error:
        summary = dict(source=str(path), sample=record["sample"], output=str(folder), status="failed", error=repr(error))
        atomic_json(folder/"summary.json", summary)
    summary["wall_seconds"] = time.perf_counter()-started
    return summary


def bounded_jobs(function, jobs, workers, description):
    """Bound memory and queued work; worker errors stay visible."""
    with ProcessPoolExecutor(max_workers=workers) as pool:
        iterator = iter(jobs)
        active = set()
        def submit():
            job = next(iterator, None)
            if job is not None:
                active.add(pool.submit(function, *job))
        for _ in range(min(2*workers, len(jobs))):
            submit()
        with tqdm(total=len(jobs), desc=description, unit="job") as bar:
            while active:
                done, _ = wait(active, timeout=10, return_when=FIRST_COMPLETED)
                if not done:
                    bar.refresh()
                for future in done:
                    active.remove(future)
                    yield future.result()
                    bar.update(1)
                    submit()


def discover_inputs(args):
    groups = [(sample, [args.input_root/sample]) for sample in args.samples] if args.server else [("inputs", args.input)]
    records = []
    for sample, roots in groups:
        seen, count = set(), 0
        for root in roots:
            if not root.exists():
                raise FileNotFoundError(root)
            for path in sorted(root.rglob("*")) if root.is_dir() else [root]:
                if any(part.startswith(".") for part in path.relative_to(root).parts) if root.is_dir() else path.name.startswith("."):
                    continue
                if path.suffix.lower() not in {".fits", ".fit", ".fts", ".csv", ".parquet"}:
                    continue
                if not path.is_file():
                    raise FileNotFoundError(f"Invalid or broken input: {path}")
                identity = str(path.resolve())
                if identity in seen:
                    continue
                seen.add(identity)
                if args.server and not metadata_from_path(path)[0].isdigit():
                    raise ValueError(f"Server inputs require a Gaia DR3 filename: {path}")
                records.append(dict(sample=sample, path=str(path.absolute()), input=fingerprint(path)))
                count += 1
                if args.limit and count >= args.limit:
                    break
            if args.limit and count >= args.limit:
                break
        if not count:
            raise ValueError(f"No supported inputs for {sample}")
    return records


def merge_products(records, output, run_id):
    """Stream all original columns to CSV and Parquet with one stable schema.

    Two lightweight CSV header passes avoid loading the survey into memory.
    Identifiers and metadata remain strings, including 19-digit Gaia IDs.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq
    good, incomplete, columns = [], [], []
    for record in records:
        if not completed(record, output, run_id):
            incomplete.append(record["path"])
            continue
        folder = product_folder(record, output)
        good.append(folder)
        with (folder/"all_events.csv").open(newline="") as handle:
            for col in next(csv.reader(handle)):
                if col not in columns:
                    columns.append(col)
    if not columns:
        columns = ["gaia_dr3_id", "gdr3_id", "sample", "tier", "reason", "source_path"]
    strings = {"gaia_dr3_id","gdr3_id","sample","mission","source_path","tier","reason","warnings",
        "energy_status","measurement_method","chi2_fit_status","chi2_fit_flags","calibration_status",
        "diagnostic_reasons","parameter_diagnostics","flare_model_name","flare_model_name_bayesian","flare_model_name_chi2",
        "calibration_group","detection_statistic","operating_snr_class","morphology_diagnostics","energy_method","energy_distance_source","energy_luminosity_source",
        "stellar_catalog_source", "energy_calibration_method", "energy_calibration_status"}
    # Known numerical columns may be absent in all-edge/all-overflow products.
    # Infer nonnumeric fields from actual values; never infer Gaia IDs as floats.
    for folder in good:
        sample = pd.read_csv(folder/"all_events.csv", nrows=25, dtype={k:"string" for k in strings})
        for col in sample:
            if pd.api.types.is_object_dtype(sample[col].dtype) or pd.api.types.is_string_dtype(sample[col].dtype):
                strings.add(col)
    booleans = {"parameter_usable", "chi2_fit_success"} - strings
    schema = pa.schema([(col, pa.string() if col in strings else pa.bool_() if col in booleans else pa.float64()) for col in columns])
    counts = {tier:0 for tier in TABLES}
    with ExitStack() as stack:
        csvwriters, pqwriters, pending = {}, {}, []
        for tier, name in {"all":"all_events", **TABLES}.items():
            csvpath, pqpath = output/(name+".csv"), output/(name+".parquet")
            csvtmp, pqtmp = csvpath.with_suffix(".csv.tmp"), pqpath.with_suffix(".parquet.tmp")
            handle = stack.enter_context(csvtmp.open("w", newline=""))
            export_columns=columns if tier=="all" else [c for c in columns if c not in COMPACT_DIAGNOSTIC_FIELDS]
            export_schema=pa.schema([schema.field(c) for c in export_columns])
            writer = csv.DictWriter(handle, fieldnames=export_columns)
            writer.writeheader(); csvwriters[tier] = writer
            pqwriters[tier] = stack.enter_context(pq.ParquetWriter(pqtmp, export_schema, compression="zstd"))
            pending.extend([(csvtmp,csvpath),(pqtmp,pqpath)])
        for folder in tqdm(good, desc="Combining", unit="LC"):
            for frame in pd.read_csv(folder/"all_events.csv", chunksize=10000, dtype={k:"string" for k in strings}):
                frame = frame.reindex(columns=columns)
                for col in columns:
                    if col in strings:
                        frame[col] = frame[col].astype("string")
                    elif col in booleans:
                        frame[col] = frame[col].map({True:True,False:False,"True":True,"False":False}).astype("boolean")
                    else:
                        frame[col] = pd.to_numeric(frame[col], errors="raise").astype(float)
                for tier in csvwriters:
                    part = frame if tier == "all" else compact_table(frame.loc[frame.tier == tier])
                    csvwriters[tier].writerows({k:(None if pd.isna(v) else v) for k,v in row.items()} for row in part.to_dict("records"))
                    pqwriters[tier].write_table(pa.Table.from_pandas(part, schema=pa.schema([schema.field(c) for c in part.columns]), preserve_index=False))
                    if tier != "all":
                        counts[tier] += len(part)
    for temporary, destination in pending:
        temporary.replace(destination)
    return dict(complete_products=len(good), incomplete_products=len(incomplete), incomplete_paths=incomplete, counts=counts)

# ======================================================================
# INJECTION AND NEGATIVE-CONTROL CALIBRATION
# ======================================================================

def binomial_upper(successes, trials, confidence=.95):
    """One-sided Clopper-Pearson bound, conditional on the sampled control mix."""
    if trials <= 0 or successes >= trials:
        return 1.0
    return float(beta_distribution.ppf(confidence, successes+1, trials-successes))


@lru_cache(maxsize=8)
def calibration_source(path, config):
    return load_lightcurve(Path(path), config)


def slice_lightcurve(lc, indices, flux=None):
    return replace(lc, time=lc.time[indices].copy(), flux=lc.flux[indices].copy() if flux is None else flux,
        flux_err=None if lc.flux_err is None else lc.flux_err[indices].copy(),
        raw_flux=None if lc.raw_flux is None else lc.raw_flux[indices].copy())


@lru_cache(maxsize=8)
def calibration_background(path, config, source_fraction):
    """Prepare the un-injected background once; define SNR in local noise units."""
    source = calibration_source(path, config)
    low, high = source_fraction
    lc = slice_lightcurve(source, np.arange(int(len(source.time)*low), int(len(source.time)*high)))
    prepared, _ = prepare(lc, config)
    sigma_raw = prepared.segment_sigma / prepared.detrend_error_scale
    usable = ((prepared.segment_id >= 0) & ~prepared.background_uncertain_mask &
              np.isfinite(sigma_raw) & (sigma_raw > 0))
    snr = (prepared.residual-prepared.local_baseline)/prepared.segment_sigma
    seeds = lc.time[snr >= config.candidate_sigma]
    return lc, sigma_raw, usable, seeds, snr


def calibration_center(lc, config, width, rng, usable=None, excluded_times=None):
    """Draw a peak only from retained times, with wings clear of edges/gaps.

    Require +/-12 FWHM (or two cadences) inside one observed segment AND the
    fractional search region. This protects both forward and reverse controls.
    The template has infinite tails; the declared finite buffer is not a claim
    of exactly zero signal beyond it. Fail explicitly if no placement exists.
    """
    t = lc.time
    margin = max(12*width, 2*cadence_days(t))
    edge = config.edge_fraction*(t[-1]-t[0])
    allowed = np.zeros(len(t), bool)
    for ii in split_segments(t, config.gap_factor):
        if len(ii) < 5:
            continue
        local_edge = config.edge_fraction*np.ptp(t[ii]) if config.edge_scope == "segment" else 0.
        low = max(t[0]+edge if config.edge_scope == "lightcurve" else t[0], t[ii[0]]+local_edge)+margin
        high = min(t[-1]-edge if config.edge_scope == "lightcurve" else t[-1], t[ii[-1]]-local_edge)-margin
        allowed[ii] = (t[ii] >= low) & (t[ii] <= high)
    if usable is not None:
        allowed &= np.asarray(usable, bool)
    if excluded_times is not None and len(excluded_times):
        # Do not count recovery of an event that was already present before
        # injection. Other background events in the scored window still count
        # conservatively against negative-control thresholds.
        pos = np.searchsorted(excluded_times, t)
        left = excluded_times[np.maximum(pos-1, 0)]
        right = excluded_times[np.minimum(pos, len(excluded_times)-1)]
        distance = np.minimum(abs(t-left), abs(t-right))
        allowed &= distance > max(width, 2*cadence_days(t))
    choices = np.flatnonzero(allowed)
    if not len(choices):
        raise ValueError("No injection placement clear of excluded edges/gaps for requested width")
    return int(rng.choice(choices)), dict(buffer_days=margin, eligible_centers=len(choices),
                                        edge_fraction=config.edge_fraction, edge_scope=config.edge_scope)



def quiet_control_window_centers(background_snr, window_n, maximum_snr=10.):
    """Centers whose entire scored window lacks pre-existing >=10-sigma data.

    This defines a selected quiet-background control population, not a survey
    false-alarm rate. Selection is identical for flare and non-flare trials.
    """
    snr=np.asarray(background_snr,float);n=len(snr)
    starts=np.clip(np.arange(n)-window_n//2,0,n-window_n)
    loud=np.r_[0,np.cumsum(np.isfinite(snr)&(snr>=maximum_snr))]
    return (loud[starts+window_n]-loud[starts])==0


def calibration_trial(spec, config):
    """Inject BEFORE detrending and rerun the whole detection/fit decision path.

    Controls are realistic challenges, not certified non-flaring stars.
    Unmodified windows can contain real flares; their detections conservatively
    count against the threshold. Targeted artifact trials also count any other
    accepted event in the window. A flare recovery must match the injected peak.
    """
    result = {k:v for k,v in spec.items() if k != "source_fraction"}
    started = time.perf_counter()
    try:
        cfg = replace(config, calibration_file="", builtin_calibration=False, allow_uncalibrated=True, candidate_workers=1, max_candidates=0)
        trial, background_sigma_raw, background_usable, background_seeds, background_snr = calibration_background(
            spec["path"], cfg, tuple(spec["source_fraction"]))
        rng = np.random.default_rng(spec["seed"])
        dt = cadence_days(trial.time)
        n = len(trial.time)
        window_n = min(n, max(128, min(4096, int(round(1.5/dt)))))
        if window_n < 64:
            raise ValueError("Too few independent-window observations for calibration")
        # Detrend the ENTIRE original LC (or disjoint temporal half), then
        # score only the sampled window. Long-baseline stages remain active.
        tt = trial.time
        exposure = trial.exposure_days or dt
        width = float(np.exp(rng.uniform(np.log(max(.5*exposure, 1/1440)), np.log(max(60/1440, 4*exposure)))))
        snr = float(rng.choice([4., 5., 10., 20., 50.]))
        # A control window containing an obvious pre-existing event is not
        # an interpretable negative control. Select on the UNMODIFIED LC;
        # retain weak noise excursions and never inspect the injected outcome.
        clean_centers = quiet_control_window_centers(background_snr, window_n)
        center_index, placement = calibration_center(trial, cfg, width, rng,
            usable=background_usable & clean_centers, excluded_times=background_seeds)
        center = float(tt[center_index])
        start = max(0, min(n-window_n, center_index-window_n//2))
        window = np.arange(start,start+window_n)
        bounds = (float(tt[window[0]]),float(tt[window[-1]]))
        norm = 1. if cfg.flux_is_residual else float(np.median(trial.flux))
        sigma = float(background_sigma_raw[center_index])
        if not np.isfinite(sigma) or sigma <= 0:
            raise ValueError("Cannot estimate injection background noise")
        kind = spec["kind"]
        factor=float(rng.choice(config.rise_factors))
        injected_model="tovar2022" if factor==1 else "tovar2022_rise"+str(factor)
        unit = flare_model(tt, center, width, 1., exposure,injected_model)
        if kind == "reverse_flare":
            unit = flare_model(-tt, -center, width, 1., exposure,injected_model)
        amplitude = snr*sigma/max(float(np.max(unit)), 1e-12)
        signal = amplitude*unit
        if kind == "quiet":
            signal = np.zeros(n)
        elif kind == "impulse":
            signal = np.zeros(n); signal[center_index] = snr*sigma
        elif kind == "step":
            signal = snr*sigma*np.clip((tt-center)/exposure+.5, 0, 1)
        elif kind == "symmetric_bump":
            signal = snr*sigma*np.exp(-.5*((tt-center)/width)**2)
        elif kind == "correlated_noise":
            # Independent AR(1) realization with a correlation time of 3 cadences.
            signal = np.empty(n); signal[0] = rng.normal()
            rho = np.exp(-1/3)
            for j in range(1,n):
                signal[j] = rho*signal[j-1]+np.sqrt(1-rho*rho)*rng.normal()
            signal *= sigma*float(rng.choice([.5,1.,2.]))
        trial = replace(trial, flux=trial.flux+norm*signal)
        frame, prepared, _, _ = analyze(trial, cfg, associate=True, search_bounds=bounds)
        if len(frame) and frame.diagnostic_reasons.fillna("").str.contains("fit_failed|overflow").any():
            raise ValueError("A calibration window has an unmeasured candidate; not a successful null trial")
        accepted = frame.loc[frame.tier == "accepted"].copy()
        if kind == "flare" and len(accepted):
            accepted = accepted.loc[abs(accepted.tpeak-center) <= max(2*dt, width)]
        score = float(accepted.detection_log_bayes_factor.max()) if len(accepted) else None
        class_scores = {}
        for label, high in (("ordinary", False), ("high", True)):
            subset = accepted.loc[accepted.fitted_peak_snr.ge(cfg.high_snr_boundary) == high] if len(accepted) else accepted
            class_scores["score_"+label] = float(subset.detection_log_bayes_factor.max()) if len(subset) else None
        row = accepted.loc[accepted.detection_log_bayes_factor.idxmax()] if len(accepted) else None
        tolerance = max(2*dt, width)
        near = frame.loc[abs(frame.seed_time-center) <= tolerance]
        if "tpeak" in frame:
            near = frame.loc[(abs(frame.seed_time-center) <= tolerance) | (abs(frame.tpeak-center) <= tolerance)]
        result.update(status="complete", score=score, injected_peak=center,
            **class_scores,
            injected_fwhm_days=width, injected_amplitude=amplitude, injected_model=injected_model, injected_sampled_snr=snr,
            exposure_seconds=exposure*86400, cadence_seconds=dt*86400,
            recovered_ed_ratio=(float(row.equivalent_duration_s)/(full_equivalent_duration_days(width,amplitude,injected_model)*86400)
                                if kind == "flare" and row is not None else None),
            window_start=bounds[0], window_end=bounds[1], n_observations=window_n,
            context_start=float(tt[0]), context_end=float(tt[-1]), context_observations=n,
            searched_window_days=bounds[1]-bounds[0], candidates=len(frame),
            unmodified_window_max_snr=float(np.nanmax(background_snr[window])),
            injected_peak_in_search_region=not bool(prepared.detrend_edge_mask[center_index]),
            placement=placement, matched_tiers=near.tier.tolist(), matched_reasons=near.reason.fillna("").tolist(),
            tier_counts=frame.tier.value_counts().to_dict(), matched_diagnostics=near.diagnostic_reasons.fillna("").tolist(),
            injection_noise_basis="pre-injection local total sigma, converted to raw normalized flux",
            injection_noise_sigma_raw=sigma, baseline_peak_snr=float(background_snr[center_index]),
            processed_peak_snr=float((prepared.residual[center_index]-prepared.local_baseline[center_index])/prepared.segment_sigma[center_index]))
        fields = ["candidate_id", "seed_time", "tpeak", "tier", "diagnostic_reasons",
                  "fitted_peak_snr", "detection_log_bayes_factor", "log_bayes_factor",
                  "logbf_forward_minus_reverse", "logbf_flare_minus_symmetric", "logbf_impulse",
                  "raw_symmetric_minus_flare_diagnostic", "raw_step_minus_flare_diagnostic",
                  "raw_logbf_flare", "fwhm_days", "amplitude", "morphology_diagnostics"]
        result["candidate_scores"] = frame[[k for k in fields if k in frame]].to_dict("records")
    except Exception as error:
        result.update(status="failed", error=repr(error), score=None)
    result["seconds"] = time.perf_counter()-started
    return json_safe(result)


def calibration_inventory(records, config, per_group, seed):
    """Inventory actual streams; draw reproducibly from distinct stars per group."""
    inventory, failed = [], []
    for record in tqdm(records, desc="Calibration inventory", unit="LC"):
        try:
            lc = load_lightcurve(Path(record["path"]), config)
            inventory.append(dict(path=record["path"], group=calibration_group(lc),
                                  star=lc.gdr3_id if lc.gdr3_id != "UNKNOWN" else record["input"]["resolved"]))
        except Exception as error:
            failed.append(dict(path=record["path"], error=repr(error)))
    if failed:
        raise ValueError(f"Calibration inventory failed for {len(failed)} inputs; first: {failed[0]}")
    rng = np.random.default_rng(seed)
    selected = {}
    for group in sorted({r["group"] for r in inventory}):
        rows = [r for r in inventory if r["group"] == group]
        stars = sorted({r["star"] for r in rows})
        rng.shuffle(stars)
        chosen = []
        for star in stars[:per_group]:
            candidates = [r for r in rows if r["star"] == star]
            chosen.append(candidates[int(rng.integers(len(candidates)))])
        selected[group] = chosen
    return inventory, selected


def choose_threshold(rows, floor, target, confidence):
    negatives = [r for r in rows if r["kind"] != "flare" and r["status"] == "complete"]
    scores = np.array([r["score"] if r["score"] is not None else -np.inf for r in negatives])
    choices = np.unique(np.r_[floor, np.nextafter(scores[np.isfinite(scores) & (scores >= floor)], np.inf)])
    for threshold in choices:
        upper = binomial_upper(int((scores >= threshold).sum()), len(scores), confidence)
        if upper <= target:
            return float(threshold)
    return None


def choose_operating_thresholds(rows, floor, target, confidence, max_false_positives=None):
    """Fit two cuts jointly against the SAME control-window false-alarm budget.

    The strength split uses fitted sampled SNR, never injected truth. Training
    labels choose the operating point; validation labels must never enter here.
    Maximize >=10-SNR training-flare recovery, then all training-flare recovery,
    then minimize false windows and total cuts. Both classes always exceed ln(4).
    """
    negative = [r for r in rows if r["kind"] != "flare" and r["status"] == "complete"]
    positive = [r for r in rows if r["kind"] == "flare" and r["status"] == "complete"]
    if not negative or binomial_upper(0, len(negative), confidence) > target:
        return None
    budget = 0
    while budget < len(negative) and binomial_upper(budget+1, len(negative), confidence) <= target:
        budget += 1
    if max_false_positives is not None:
        budget = min(budget, int(max_false_positives))
    floor = max(float(floor), float(np.log(4.0)))
    def scores(records, label):
        return np.array([r.get("score_"+label) if r.get("score_"+label) is not None else -np.inf for r in records])
    no, nh = scores(negative,"ordinary"), scores(negative,"high")
    po, ph = scores(positive,"ordinary"), scores(positive,"high")
    strong = np.array([r["injected_sampled_snr"] >= 10 for r in positive],bool)
    choices = [np.unique(np.r_[floor,np.nextafter(x[np.isfinite(x)&(x>=floor)],np.inf)]) for x in (no,nh)]
    best, answer = None, None
    for ordinary in choices[0]:
        for high in choices[1]:
            fp = int(np.sum((no>=ordinary)|(nh>=high)))
            if fp > budget: continue
            recovered = (po>=ordinary)|(ph>=high)
            rank = (int(np.sum(recovered&strong)),int(recovered.sum()),-fp,-float(ordinary+high),-float(ordinary))
            if best is None or rank > best:
                best, answer = rank, dict(ordinary=float(ordinary),high=float(high))
    return answer


def calibration_metrics(rows, threshold, confidence):
    negatives = [r for r in rows if r["kind"] != "flare" and r["status"] == "complete"]
    positives = [r for r in rows if r["kind"] == "flare" and r["status"] == "complete"]
    def passed(row):
        if isinstance(threshold, dict):
            return any(row.get("score_"+label) is not None and row["score_"+label] >= threshold[label]
                       for label in ("ordinary", "high"))
        return threshold is not None and row["score"] is not None and row["score"] >= threshold
    fp = sum(passed(r) for r in negatives)
    strong = [r for r in positives if r["injected_sampled_snr"] >= 10 and r["injected_fwhm_days"]*86400 >= 1.5*r["exposure_seconds"]]
    return dict(negative_trials=len(negatives), false_positive_windows=fp,
        false_positive_fraction=fp/len(negatives) if negatives else None,
        false_positive_upper=binomial_upper(fp,len(negatives),confidence),
        injected_flares=len(positives), recovered_flares=sum(passed(r) for r in positives),
        recovery_fraction=sum(passed(r) for r in positives)/len(positives) if positives else None,
        strong_resolved_injections=len(strong), strong_resolved_recovered=sum(passed(r) for r in strong),
        strong_resolved_recovery=sum(passed(r) for r in strong)/len(strong) if strong else None,
        failed_trials=sum(r["status"] != "complete" for r in rows),
        controls={kind:dict(n=sum(r["kind"] == kind for r in negatives),
                           positive_windows=sum(r["kind"] == kind and passed(r) for r in negatives)) for kind in CONTROL_KINDS})


def calibrate(records, config, output, args):
    """Choose on training controls, evaluate once on untouched holdout draws.

    The bound describes the balanced simulated CONTROL-WINDOW mixture. It is
    not a sector-level false-alarm probability, a catalogue contamination rate,
    or a guarantee on new stars. Sources, windows, injection distributions and
    failed trials are exported so the operating point can be audited.
    """
    starting_code_hash = file_sha256(__file__)
    output.mkdir(parents=True, exist_ok=True)
    marker = output/"calibration.json"
    if marker.exists() or (output/"trials.jsonl").exists():
        raise ValueError("Calibration output already contains trials; use a new folder or reuse its JSON")
    split_seed = args.seed if getattr(args,"split_seed",None) is None else args.split_seed
    inventory, selected = calibration_inventory(records, config, args.calibration_files_per_group, split_seed)
    atomic_json(output/"inventory.json", dict(inventory=inventory, selected=selected))
    signature = json.loads(json.dumps(analysis_signature(config)))
    spec_list, splits = [], {}
    rng = np.random.default_rng(args.seed)
    for group, sources in selected.items():
        if len(sources) >= 2:
            split = max(1, len(sources)//2)
            pools = dict(train=sources[:split], holdout=sources[split:])
            fractions = dict(train=(0.,1.), holdout=(0.,1.))
            splits[group] = "star_disjoint"
        else:
            pools = dict(train=sources, holdout=sources)
            fractions = dict(train=(0.,.45), holdout=(.55,1.))
            splits[group] = "time_disjoint_single_star"
        for phase in ("train", "holdout"):
            for kind in ("flare", *CONTROL_KINDS):
                for j in range(args.trials_per_kind):
                    source = pools[phase][j % len(pools[phase])]
                    spec_list.append(dict(group=group, phase=phase, kind=kind, trial=j,
                        path=source["path"], star=source["star"], source_fraction=fractions[phase],
                        seed=int(rng.integers(0,2**32-1))))
    rows = []
    with (output/"trials.jsonl").open("w", buffering=1) as stream:
        for result in bounded_jobs(calibration_trial, [(s,config) for s in spec_list], args.workers, "Injection controls"):
            rows.append(result); stream.write(json.dumps(result,allow_nan=False)+"\n")
    rows.sort(key=lambda r:(r["group"],r["phase"],r["kind"],r["trial"]))
    atomic_frame(pd.DataFrame(rows), output/"trials.csv")
    if file_sha256(__file__) != starting_code_hash:
        raise RuntimeError("Detector source changed during calibration; trials retained but calibration invalid")
    report = dict(detector_version=DETECTOR_VERSION, code_sha256=starting_code_hash,
        analysis_signature=signature, seed=args.seed, source_split_seed=split_seed, trials_per_kind=args.trials_per_kind,
        false_alarm_target=args.false_alarm_target, confidence=args.confidence,
        minimum_strong_resolved_recovery=args.minimum_recovery,
        injection_sampled_snr_levels=[4,5,10,20,50],
        injection_noise_basis="Pre-injection local total noise at the injected peak; raw scaling propagated",
        injection_placement="Retained supported background; +/-12 FWHM clear of edges/gaps; no pre-existing >=3 sigma seed within matching tolerance",
        target_unit="probability of >=1 accepted event per sampled quiet-background control window; balanced control mixture",
        calibration_floor=config.logbf_threshold,detection_statistic=config.detection_statistic,
        flare_rise_factors=list(config.rise_factors),alternative_model_prior="Equal 1/4: affine background, reverse flare, symmetric Gaussian, impulse",
        background_selection="Full scored window contains no >=10-sigma excursion in the unmodified light curve; identical selection for positive and negative trials",
        caveats=["Not sector FAP or survey contamination", "Selected quiet backgrounds: unmodified windows may still contain weak real flares",
                 "Noise backgrounds and trial windows may repeat within a split",
                 "Three rise-shape injections do not measure recovery of every flare morphology",
                 "Scores include tail association within the scored window; parents outside that window are not associated",
                 "Single-star groups test disjoint time intervals, not generalization to other stars"],
        sources={r["path"]:dict(sha256=file_sha256(r["path"]),input=fingerprint(r["path"])) for v in selected.values() for r in v},
        groups={})
    for group in selected:
        train = [r for r in rows if r["group"] == group and r["phase"] == "train"]
        holdout = [r for r in rows if r["group"] == group and r["phase"] == "holdout"]
        threshold = (choose_operating_thresholds(train, config.logbf_threshold, args.false_alarm_target, args.confidence)
                     if config.detection_statistic == "alternatives" else
                     choose_threshold(train, config.logbf_threshold, args.false_alarm_target, args.confidence))
        tm, hm = calibration_metrics(train,threshold,args.confidence), calibration_metrics(holdout,threshold,args.confidence)
        reasons = []
        if threshold is None:
            reasons.append("insufficient_training_controls_for_target")
        if hm["false_positive_upper"] > args.false_alarm_target:
            reasons.append("holdout_false_alarm_bound_exceeds_target")
        if tm["failed_trials"] or hm["failed_trials"]:
            reasons.append("incomplete_trials")
        if hm["strong_resolved_injections"] < 5:
            reasons.append("insufficient_strong_resolved_injections")
        elif hm["strong_resolved_recovery"] < args.minimum_recovery:
            reasons.append("strong_resolved_recovery_below_minimum")
        report["groups"][group] = dict(status="validated" if not reasons else "validation_failed",
            logbf_threshold=threshold.get("ordinary") if isinstance(threshold,dict) else threshold,
            high_snr_logbf_threshold=threshold.get("high") if isinstance(threshold,dict) else threshold,
            high_snr_boundary=config.high_snr_boundary,
            reasons=reasons, split=splits[group], train=tm, holdout=hm)
    atomic_json(marker, json_safe(report))
    print(json.dumps(report["groups"],indent=2), flush=True)
    return marker


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--version", action="version", version=f"BANSHEE_Final {DETECTOR_VERSION}")
    parser.add_argument("--input", nargs="+", type=Path, help="Individual files or folders (FITS/CSV/Parquet)")
    parser.add_argument("--server", action="store_true", help="Use FS/OC layout and Gaia filenames")
    parser.add_argument("--input-root",type=Path,default=Path("lightcurves"),help="Root containing the selected sample folders, relative to the current directory")
    parser.add_argument("--samples",nargs="+",choices=SAMPLES,default=list(SAMPLES))
    parser.add_argument("--output-dir","--output-root",dest="output",type=Path,required=True)
    parser.add_argument("--workers",type=int,default=4)
    parser.add_argument("--limit",type=int,default=0,help="Maximum inputs per sample; zero means all")
    parser.add_argument("--plots",action="store_true",help="Overview and all-candidate contact sheets; large on full surveys")
    parser.add_argument("--flux-column",default="",help="Require this exact flux stream; default auto-selects SAP first, then the product-specific fallback")
    parser.add_argument("--already-detrended",action="store_true")
    parser.add_argument("--residual",action="store_true",help="Zero-centered relative residual input; requires --already-detrended")
    parser.add_argument("--chi2-fit",action=argparse.BooleanOptionalAction,default=True)
    parser.add_argument("--max-candidates",type=int,default=0,help="Above cap retain EVERY candidate as unfitted follow-up; zero disables")
    parser.add_argument("--noise-window-hours",type=float,default=2.)
    parser.add_argument("--noise-min-points",type=int,default=41)
    parser.add_argument("--edge-fraction",type=float,default=.05,help="Fraction at EACH edge reserved for follow-up only; zero disables")
    parser.add_argument("--edge-scope",choices=("lightcurve","segment"),default="lightcurve")
    parser.add_argument("--logbf-threshold",type=float,default=float(np.log(4.0)),help="Calibration search floor / provisional threshold")
    parser.add_argument("--detection-statistic",choices=("background","alternatives"),default="alternatives")
    parser.add_argument("--calibration",type=Path,help="Matching calibration.json from this exact code/settings")
    parser.add_argument("--builtin-calibration",action=argparse.BooleanOptionalAction,default=True,
                        help="Use saved SAP cadence thresholds embedded in this file; an explicit --calibration takes priority")
    parser.add_argument("--calibrate",action="store_true",help="Only run pre-injection controls and write calibration.json")
    parser.add_argument("--auto-calibrate",action="store_true",help="Calibrate on selected inputs, then process the survey")
    parser.add_argument("--allow-uncalibrated",action="store_true",help="Explicit research override: permit provisional acceptance")
    parser.add_argument("--calibration-files-per-group",type=int,default=4,help="Distinct stars per cadence/product; preferably >=4")
    parser.add_argument("--trials-per-kind",type=int,default=32,help="Each kind, each train/holdout split, per cadence/product")
    parser.add_argument("--false-alarm-target",type=float,default=.05,help="Control-WINDOW probability target, NOT sector FAP")
    parser.add_argument("--confidence",type=float,default=.95)
    parser.add_argument("--minimum-recovery",type=float,default=.70,help="Minimum holdout recovery of SNR>=10, resolved injections")
    parser.add_argument("--seed",type=int,default=20261102)
    parser.add_argument("--split-seed",type=int,default=None,help="Keep the original star split while changing injection draws; defaults to --seed")
    parser.add_argument("--oc-catalog",type=Path)
    parser.add_argument("--fs-catalog",type=Path)
    parser.add_argument("--stellar-catalogs",nargs="+",type=Path,default=[])
    parser.add_argument("--distance-column",default="distance_pc")
    parser.add_argument("--energy-method",choices=("supplied","tmag-bandwidth"),default="supplied")
    parser.add_argument("--tess-band-luminosity",type=float)
    parser.add_argument("--tess-effective-bandwidth-hz",type=float,default=1.2e14)
    return parser


def main():
    parser = make_parser(); args = parser.parse_args()
    if args.server == bool(args.input):
        parser.error("Choose either --server or --input")
    if args.workers < 1 or args.limit < 0 or args.max_candidates < 0:
        parser.error("workers must be positive; limit and max-candidates nonnegative")
    if args.residual and not args.already_detrended:
        parser.error("--residual requires --already-detrended")
    if sum([args.calibrate,args.auto_calibrate,bool(args.calibration)]) > 1:
        parser.error("Choose only one of --calibrate, --auto-calibrate, --calibration")
    if args.trials_per_kind < 1 or args.calibration_files_per_group < 1 or args.seed < 0:
        parser.error("Calibration counts must be positive and seed nonnegative")
    if not 0 < args.false_alarm_target < 1 or not 0 < args.confidence < 1 or not 0 <= args.minimum_recovery <= 1:
        parser.error("Invalid calibration probability/recovery setting")
    if not np.isfinite(args.tess_effective_bandwidth_hz) or args.tess_effective_bandwidth_hz <= 0:
        parser.error("Bandwidth must be positive finite Hz")
    if args.tess_band_luminosity is not None and args.energy_method != "supplied":
        parser.error("Choose supplied luminosity or magnitude-based energy")
    roots = [args.input_root] if args.server else args.input
    output = args.output.expanduser().resolve()
    for root in roots:
        root = root.expanduser().resolve()
        if output == root or output in root.parents or (root.is_dir() and root in output.parents):
            parser.error("Input and output folders must be separate and non-nested")
    config = Config(already_detrended=args.already_detrended,flux_is_residual=args.residual,
        flux_column=args.flux_column, chi2_fit=args.chi2_fit,max_candidates=args.max_candidates,
        noise_window_hours=args.noise_window_hours,noise_min_points=args.noise_min_points,
        edge_fraction=args.edge_fraction,edge_scope=args.edge_scope,logbf_threshold=args.logbf_threshold,detection_statistic=args.detection_statistic,
        calibration_file=str(args.calibration.resolve()) if args.calibration else "",builtin_calibration=args.builtin_calibration,allow_uncalibrated=args.allow_uncalibrated,
        distance_column=args.distance_column,energy_method=args.energy_method,
        tess_band_luminosity_erg_s=args.tess_band_luminosity,tess_effective_bandwidth_hz=args.tess_effective_bandwidth_hz)
    records = discover_inputs(args)
    if args.tess_band_luminosity is not None:
        ids = {metadata_from_path(Path(r["path"]))[0] for r in records}
        if len(ids) != 1 or "UNKNOWN" in ids or not np.isfinite(args.tess_band_luminosity) or args.tess_band_luminosity <= 0:
            parser.error("Supplied luminosity requires one known Gaia star and positive finite erg/s")
    catalogs = {}
    if args.server:
        for sample, path in dict(open_clusters=args.oc_catalog,field_stars=args.fs_catalog).items():
            if path and sample in args.samples:
                catalogs[sample] = catalog_manifest(path,args.distance_column)
        if args.energy_method == "tmag-bandwidth" and any(s not in catalogs for s in args.samples):
            parser.error("Magnitude-based energies require a catalog for each selected sample")
        for record in records:
            if record["sample"] in catalogs:
                record["catalog"] = catalogs[record["sample"]]
    else:
        catalogs = {str(p):catalog_manifest(p,args.distance_column) for p in args.stellar_catalogs}
        config = replace(config,stellar_catalog_paths=tuple(c["path"] for c in catalogs.values()))
    output.mkdir(parents=True,exist_ok=True)
    with (output/".runner.lock").open("a") as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("Another runner is using this output folder")
        if args.calibrate:
            path = calibrate(records,config,output,args)
            print(f"Calibration report: {path}")
            report=json.loads(path.read_text())
            if any(g["status"] != "validated" for g in report["groups"].values()):
                raise SystemExit("Calibration finished, but some groups did not meet validation criteria; see calibration.json")
            return
        if args.auto_calibrate:
            calfolder = output/"precalibration"
            calfile = calfolder/"calibration.json"
            if not calfile.exists():
                calfile = calibrate(records,config,calfolder,args)
            config = replace(config,calibration_file=str(calfile.resolve()))
        calhash = file_sha256(config.calibration_file) if config.calibration_file else None
        builtin_used = None
        if config.calibration_file:
            cpath=Path(config.calibration_file); info=cpath.stat()
            cdata=read_calibration(str(cpath.resolve()),info.st_mtime_ns,info.st_size)
            if cdata["analysis_signature"] != json.loads(json.dumps(analysis_signature(config))):
                raise ValueError("Calibration settings do not match")
        elif config.builtin_calibration and not config.allow_uncalibrated:
            builtin_used = dict(BUILTIN_CALIBRATION, detector_version=DETECTOR_VERSION,
                                code_sha256=file_sha256(__file__), calibration_source="built_in",
                                edge_review_policy="Edge-only exclusions become follow-up after detection decisions; interior cuts unchanged")
            calhash = hashlib.sha256(json.dumps(builtin_used,indent=2,sort_keys=True,allow_nan=False).encode()).hexdigest()
            print("Using built-in SAP cadence thresholds; no calibration run is needed.",flush=True)
            for group,row in BUILTIN_CALIBRATION["groups"].items():
                cuts = (f"lnBF ordinary={row['logbf_threshold']:.6g}, high-SNR={row['high_snr_logbf_threshold']:.6g}"
                        if row["status"] == "validated" else "unvalidated: otherwise passing events remain follow-up")
                print(f"  {group}: {cuts}",flush=True)
            print("Other flux streams, exposures or analysis settings remain unvalidated. Bounds apply to the tested control backgrounds.",flush=True)
        packages = {}
        for name in ("numpy","scipy","pandas","astropy","lightkurve","matplotlib","tqdm","pyarrow"):
            packages[name]=version(name)  # Missing dependencies must stop before any jobs.
        codehash = file_sha256(__file__)
        manifest = json_safe(dict(detector_version=DETECTOR_VERSION, code_sha256=codehash,
            config=asdict(config),calibration_sha256=calhash,
            calibration_source="file" if config.calibration_file else "built_in" if builtin_used else "none",
            inputs=records,catalogs=catalogs,
            plots=args.plots,packages=packages,python=platform.python_version()))
        manifest=json.loads(json.dumps(manifest))
        run_id=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
        marker=output/"run_manifest.json"
        if marker.exists():
            if json.loads(marker.read_text()) != manifest:
                raise SystemExit("Code, inputs, settings, calibration or environment changed; choose a NEW output folder")
        else:
            if any(p.name not in {".runner.lock","precalibration"} for p in output.iterdir()):
                raise SystemExit("Output is nonempty without a matching manifest; choose a NEW folder")
            atomic_json(marker,manifest)
            shutil.copy2(__file__,output/"BANSHEE_Final_snapshot.py")
            if config.calibration_file:
                shutil.copy2(config.calibration_file,output/"calibration_used.json")
            elif builtin_used is not None:
                atomic_json(output/"calibration_used.json",builtin_used)
        remaining=[r for r in records if not completed(r,output,run_id)]
        print(f"Inputs {len(records)}; complete {len(records)-len(remaining)}; remaining {len(remaining)}",flush=True)
        print(f"Local 3-sigma seeds; {100*config.edge_fraction:g}% at EACH end reserved for follow-up only. Unvalidated groups remain follow-up unless explicitly overridden.",flush=True)
        failures=0; started=time.perf_counter()
        with (output/"run_log.jsonl").open("a",buffering=1) as log:
            for result in bounded_jobs(run_product,[(r,output,config,args.plots,run_id) for r in remaining],args.workers,"Light curves"):
                log.write(json.dumps(json_safe(result))+"\n")
                failures += result["status"] != "complete"
        if file_sha256(__file__) != codehash or (config.calibration_file and file_sha256(config.calibration_file) != calhash):
            raise SystemExit("Code/calibration changed during run; do not use mixed results")
        if any(file_sha256(c["path"]) != c["sha256"] for c in catalogs.values()):
            raise SystemExit("Stellar catalog changed during run")
        counts=merge_products(records,output,run_id)
        summaries=[json.loads((product_folder(r,output)/"summary.json").read_text()) for r in records if completed(r,output,run_id)]
        summary=dict(seconds_this_run=time.perf_counter()-started,attempted_this_run=len(remaining),failed_this_run=failures,
            uncalibrated_products=sum(s["calibration_status"] != "validated" for s in summaries),
            searchable_exposure_days=sum(s["searchable_exposure_days"] for s in summaries),**counts)
        atomic_json(output/"run_summary.json",json_safe(summary))
        print(json.dumps(summary,indent=2),flush=True)
        if failures or counts["incomplete_products"]:
            raise SystemExit("Some products failed; see run_log.jsonl. Rerun the same command to retry")


if __name__ == "__main__":
    main()
