#!/usr/bin/env python3
"""BANSHEE Lite: broad Bayesian detection with continuous fit measurements.

Created 2026-09-15 from the workspace BANSHEE_RELEASE2.py, SHA256
4ef545412dc9b98300579c2cd44c4568516bc81ab2f667026c6d1bc4ae04a0ad.
Only the broad grid is evaluated, for both forward and reversed flare models.
The original 41-width, up-to-18-peak broad grid is unchanged. Neighbor padding
is used for noise estimation, but only detected neighbor intervals are removed
from the likelihood. Continuous chi-square minimization is ON by default:
peak time and log FWHM are free, while nonnegative amplitude, offset and slope
are solved analytically. Main measurements use this profile-likelihood fit;
Bayesian point estimates and credible intervals are explicitly suffixed
_bayesian and _bayesian_p16/p50/p84. They are not errors on the chi-square fit.
Bayes factors retain the original amplitude-marginalized grid calculation.
Thresholds, Tovar model, exposure integration, detrending, and energy equations
are unchanged; masks and measurement-based classification can change results.
Use --no-chi2-fit to retain Bayesian point measurements as the main output.
The detailed_* Config fields remain for compatibility but are not used.
Coarser quadrature can change evidence, classifications, fitted parameters,
and credible intervals. This is not a numerically equivalent speedup.

Individual files/folders (detrending ON unless explicitly disabled):
  python3 BANSHEE_Lite_Chi.py --input /path/to/lc.fits --output-dir /path/to/new_results --plots

Audited OC/FS server trees, progress/time-left, 20 workers and resume:
  python3 BANSHEE_Lite_Chi.py --server --workers 4 --max-candidates 300
  python3 BANSHEE_Lite_Chi.py --server --help
Default server output: ~/OCFS_Flare_Table_Lite
Server mode defaults to a 300-candidate limit per light curve. Above this
limit every candidate is retained as follow-up without fitting; no first-N
subset is fitted. Use --max-candidates 0 to disable the limit. The ordinary
CLI and Config still default to unlimited candidates for compatibility.
This is separate from the existing TOVAR run. Do not replace a running job.

Optional approximate TESS-band energies (Gaia DR3 matching only):
  --server --oc-catalog /path/OC_V1.17.parquet --fs-catalog /path/FS_V1.17.parquet
  --energy-method tmag-bandwidth --distance-column distance_pc
Reads TESSMAG/TMAG from FITS and distance in pc from the matching catalog.
Uses a Vega-like 2416 Jy zero point and an explicit 1.2e14 Hz effective
bandwidth. This is an approximation, not SED integration or bolometric energy.
Catalogs are read-only. Missing metadata leaves energy NaN, with a reason.


SOURCES AND LIMITATIONS
Ilin & Poppenhaeger (2022): https://doi.org/10.1093/mnras/stac1232
Tovar Mendoza et al. (2022): https://arxiv.org/abs/2205.05706
Pitkin et al. (2014): https://arxiv.org/abs/1406.1712
Kipping (2010): https://doi.org/10.1111/j.1365-2966.2010.17242.x
Davenport (2016): https://arxiv.org/abs/1607.03494
SciPy bounded least_squares: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html
The neighbor-mask policy is an experimentally tested implementation choice,
not a paper-prescribed selection criterion. Chi-square fitting is maximum
likelihood for fixed Gaussian errors; it does not refine the grid posterior.
"""
from __future__ import annotations

# Limit numerical-library threads before importing numpy; parallelize independent
# light curves/candidates with the explicitly requested process counts instead.
import os
import sys
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

@dataclass(frozen=True)
class Candidate:
    """A three-sigma seed and its one-sigma expanded observed island."""

    seed_indices: np.ndarray
    event_indices: np.ndarray
    peak_index: int
    segment: int

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

def clean_id(value) -> str:
    """Return an integer-like identifier without decimal/exponent formatting."""
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
        choices = ["PDCSAP_FLUX", "KSPSAP_FLUX", "SAP_FLUX", "raw_flux", "flux"]
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
            "PDCSAP_FLUX_ERR",
            "KSPSAP_FLUX_ERR",
            "SAP_FLUX_ERR",
            "FLUX_ERR",
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
        [f"{flux_name}_err", "pdcsap_flux_err", "kspsap_flux_err", "sap_flux_err", "flux_err"],
    )
    quality_name = case_column(frame.columns, ["QUALITY", "quality"])
    exposure_name = case_column(
        frame.columns, ["exposure_duration_days", "exposure_days", "timedel"]
    )
    exposure = None
    if exposure_name and len(frame[exposure_name].dropna()):
        try:
            exposure = float(frame[exposure_name].dropna().iloc[0])
        except (TypeError, ValueError):
            pass
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
    """Seed every >=3-sigma island, then expand it to positive one sigma."""
    time_values = prepared.lightcurve.time
    residual = prepared.residual
    cadence = cadence_days(time_values)
    seed_mask = np.isfinite(residual) & (
        residual >= config.candidate_sigma * prepared.segment_sigma
    )
    candidates: list[Candidate] = []
    for segment in np.unique(prepared.segment_id[prepared.segment_id >= 0]):
        segment_indices = np.flatnonzero(prepared.segment_id == segment)
        seeds = np.intersect1d(np.flatnonzero(seed_mask), segment_indices)
        for seed in _contiguous_groups(seeds, time_values, cadence, config.gap_factor):
            low, high = int(seed[0]), int(seed[-1])
            while (
                low > segment_indices[0]
                and residual[low - 1] > config.expand_sigma * prepared.segment_sigma[low - 1]
                and time_values[low] - time_values[low - 1] <= config.gap_factor * cadence
            ):
                low -= 1
            while (
                high < segment_indices[-1]
                and residual[high + 1] > config.expand_sigma * prepared.segment_sigma[high + 1]
                and time_values[high + 1] - time_values[high] <= config.gap_factor * cadence
            ):
                high += 1
            event = np.arange(low, high + 1, dtype=int)
            peak = int(event[np.nanargmax(residual[event])])
            candidates.append(Candidate(seed, event, peak, int(segment)))
    # Expansion can make adjacent seeds overlap. Keep one candidate per peak island.
    candidates.sort(key=lambda item: (item.segment, int(item.event_indices[0])))
    merged: list[Candidate] = []
    for candidate in candidates:
        if (
            merged
            and merged[-1].segment == candidate.segment
            and candidate.event_indices[0] <= merged[-1].event_indices[-1]
        ):
            event = np.arange(
                min(merged[-1].event_indices[0], candidate.event_indices[0]),
                max(merged[-1].event_indices[-1], candidate.event_indices[-1]) + 1,
            )
            seeds = np.union1d(merged[-1].seed_indices, candidate.seed_indices)
            peak = int(event[np.nanargmax(residual[event])])
            merged[-1] = Candidate(seeds, event, peak, candidate.segment)
        else:
            merged.append(candidate)
    return group_overlapping_fit_regions(merged, time_values, residual, cadence,
                                         prepared.segment_sigma, config.expand_sigma)


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
    """Uniform peak prior over the event, with broad and seed-local nodes.

    Keep the original dense seed nodes and add nodes across the event so a
    noisy observed maximum cannot exclude the physical peak from the prior.
    Prior-cell weights still normalize over the complete search interval.
    """
    peak = time_values[candidate.peak_index]
    event_times = time_values[candidate.event_indices]
    return np.unique(np.r_[np.linspace(event_times[0]-cadence, event_times[-1]+cadence,
                                      config.broad_tpeak_points),
                           np.linspace(peak-cadence, peak+cadence, config.broad_tpeak_points)])

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

def _baseline_projection(time_values, values, variance):
    """Prepare weighted projection that integrates the shared linear baseline out.

    The same intercept and local slope occur under H0 and H1. With the same
    broad (improper) prior their normalization cancels in the Bayes factor;
    projecting them out is the compact form of the analytic marginalization in
    Pitkin et al. (2014, Appendix A).
    """
    time_values = np.asarray(time_values, dtype=float)
    values = np.asarray(values, dtype=float)
    variance = np.asarray(variance, dtype=float)
    span = max(float(np.ptp(time_values)), np.finfo(float).eps)
    x = 2.0 * (time_values - np.median(time_values)) / span
    baseline = np.column_stack([np.ones_like(x), x])
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
) -> GridResult:
    """Evaluate the complete (tpeak, FWHM, amplitude-prior-scale) grid."""
    tpeak_grid = np.asarray(tpeak_grid, dtype=float)
    fwhm_grid = np.asarray(fwhm_grid, dtype=float)
    tmesh, fmesh = np.meshgrid(tpeak_grid, fwhm_grid, indexing="xy")
    tflat, fflat = tmesh.ravel(), fmesh.ravel()
    t_mass = prior_cell_masses(tpeak_grid, "uniform")
    f_mass = prior_cell_masses(fwhm_grid, "log_uniform")
    theta_mass = np.outer(f_mass, t_mass).ravel()
    amp_scales = np.asarray(config.amplitude_prior_scales, dtype=float)
    amp_mass = np.full(len(amp_scales), 1.0 / len(amp_scales))
    baseline, weight, inverse, bwy = _baseline_projection(time_values, values, variance)
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
            config.flare_model,
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

def template_functions(model_name="tovar2022"):
    """Only the continuous Tovar model is allowed for inference."""
    if model_name != "tovar2022":
        raise ValueError("This detector supports only tovar2022")
    return tovar_template, tovar_antiderivative, TOVAR_DIMENSIONLESS_FWHM


# ======================================================================
# ILIN
# ======================================================================
import warnings
from scipy.interpolate import CubicSpline
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


def spline_stage(t, flux, config):
    """Subtract an interpolating cubic spline through endpoints and 6h bins.

    Median bin flux follows the archived authors' implementation of the
    paper's 'averaged' point. A portion shorter than five days is unchanged.
    """
    if np.ptp(t) < config.spline_min_days:
        return flux.copy()
    bins = np.floor((t - t[0]) / (config.spline_hours / 24)).astype(int)
    bt, bf = [t[0]], [flux[0]]
    for label in np.unique(bins):
        inside = bins == label
        mid = float(np.mean(t[inside]))
        if t[0] < mid < t[-1]:
            bt.append(mid)
            bf.append(float(np.median(flux[inside])))
    bt.append(t[-1]); bf.append(flux[-1])
    trend = CubicSpline(bt, bf)(t)
    return flux - trend + np.median(flux)


def cosine_stage(t, flux, config):
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
        keep = padded_clip(values, config.clip_sigma)
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
            values -= correction
            frequencies.append(float(fitted[1] / (2*np.pi)))
            if snr <= 1.0:
                break
        except (RuntimeError, ValueError, OptimizeWarning) as error:
            flags.append("cosine_fit_failed:" + type(error).__name__)
            break
    return values, frequencies, flags


def fringe_stage(t, flux, config):
    """Fit decaying exponentials only at >1-sigma outlying portion endpoints.

    Clipped flares do not constrain the fit. Positive decay rates are bounded
    by construction to prevent numerical exponential growth. All observed
    points receive the same fitted background correction, including flares.
    """
    keep = padded_clip(flux, config.clip_sigma)
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
        return flux - exponential(x, *pars) + pars[-1], ""
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


def detrend(t, flux, config=IlinConfig()):
    """Return residual, noise, error scale, intermediate stages, and warnings.

    Splits at >2h; never adds observations or deletes masked flare points.
    Savitzky-Golay uses Lightkurve.flatten with order 3 and explicit 6h/3h
    windows. No detrending uncertainty is invented: returned error scaling
    propagates only multiplicative normalization factors.
    """
    t, flux = np.asarray(t, float), np.asarray(flux, float)
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
        yy = spline_stage(tt, yy, config)
        stages["spline"][indices] = yy
        yy, frequencies, flags = cosine_stage(tt, yy, config)
        stages["cosine"][indices] = yy
        for hours, name in zip(config.savgol_hours, ("savgol6", "savgol3")):
            window = int(round(hours / 24 / cadence_days(tt)))
            window += 1 - window % 2
            window = min(window, len(tt) if len(tt) % 2 else len(tt)-1)
            if window < 5:
                flags.append("savgol_window_too_short")
            else:
                keep = padded_clip(yy, config.clip_sigma)
                if keep.sum() >= 5:
                    _, trend = LightkurveLightCurve(time=tt, flux=yy).flatten(
                        window_length=window, polyorder=3, mask=~keep,
                        niters=3, sigma=config.clip_sigma, return_trend=True)
                    baseline = np.asarray(trend.flux.value)
                    if np.any(~np.isfinite(baseline)) or np.any(baseline <= 0):
                        raise ValueError("Invalid Savitzky-Golay background")
                    yy = yy / baseline
                    scales[indices] /= baseline
                else:
                    flags.append("insufficient_savgol_background")
            stages[name][indices] = yy
        yy, fringe_warning = fringe_stage(tt, yy, config)
        if fringe_warning:
            flags.append(fringe_warning)
        yy -= np.median(yy[padded_clip(yy, config.clip_sigma)])
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
    """Frozen settings recorded verbatim; thresholds are not learned per star."""
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
    fwhm_max_minutes: float = 2880.0
    amplitude_prior_scales: tuple = (3e-4, 1e-3, 3e-3, .01, .03, .1, .3, 1.)
    logbf_threshold: float = 5.0
    direction_threshold: float = float(np.log(10.0))
    impulse_margin: float = 1.0
    raw_step_margin: float = 2.0
    local_window_hours: float = 12.0
    max_candidates: int = 0  # zero processes all; overflow is never silently lost
    candidate_workers: int = 1
    ilin: IlinConfig = IlinConfig()

    def __post_init__(self):
        if self.quality_zero_only is not True:
            raise ValueError("QUALITY == 0 filtering is mandatory and cannot be disabled")
        if isinstance(self.chi2_max_nfev, bool) or not isinstance(self.chi2_max_nfev, (int, np.integer)) or self.chi2_max_nfev < 2:
            raise ValueError("chi2_max_nfev must be an integer >= 2")


def prepare(lc, config):
    """Normalize once, apply Ilin stages, and estimate segment quiet noise.

    Detection uses the release's 3-sigma-clipped MAD, not Ilin's more strongly
    clipped 1.5-sigma rolling scatter. The latter is saved as a diagnostic.
    """
    scale = float(np.median(lc.flux))
    if config.flux_is_residual:
        normalized = 1.0 + lc.flux
        scale = 1.0
    else:
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("Nonpositive normalization; declare --residual for zero-centered input")
        normalized = lc.flux / scale
    error = None if lc.flux_err is None else lc.flux_err / scale
    if config.already_detrended:
        residual = normalized - np.median(normalized)
        ilin_noise, error_scale, stages, metadata = None, np.ones(len(normalized)), {}, []
    else:
        residual, ilin_noise, error_scale, stages, metadata = detrend(lc.time, normalized, config.ilin)
    if error is not None:
        error = error * error_scale
    segment_id = np.full(len(residual), -1, int)
    sigma = np.full(len(residual), np.nan)
    for number, indices in enumerate(split_segments(lc.time, config.gap_factor)):
        if len(indices) < 5:
            continue
        center, scatter, _ = sigma_clip_location_scale(residual[indices])
        if not np.isfinite(scatter) or scatter <= 0:
            continue
        residual[indices] -= center
        sigma[indices] = scatter
        segment_id[indices] = number
    prepared = PreparedLightCurve(lc, normalized, normalized-residual, residual,
                                   error, segment_id, sigma, np.zeros(len(residual), bool))
    return prepared, {"stages": stages, "ilin_noise": ilin_noise, "portions": metadata}


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
    result = {"tpeak": float(grid.tpeaks[best]), "fwhm_days": float(grid.fwhms[best]),
              "amplitude": amplitude, "log_bayes_factor": grid_log_evidence(grid)}
    for name, values in [("tpeak", grid.tpeaks), ("fwhm_days", grid.fwhms)]:
        qs = weighted_quantile(values, theta_weights, [.16, .5, .84])
        result.update({name+"_p"+str(p): float(q) for p, q in zip((16,50,84), qs)})
    amplitude_quantiles = amplitude_mixture_quantiles(
        grid.amplitude_mean, grid.amplitude_sd, weights, [.16, .5, .84])
    for p, value in zip((16,50,84), amplitude_quantiles):
        result["amplitude_p"+str(p)] = float(value)
    # ED is A times a width-dependent constant; propagate their covariance.
    conversion = grid.fwhms[:,None] * full_equivalent_duration_days(1., 1., "tovar2022") * 86400
    result["equivalent_duration_s"] = full_equivalent_duration_days(
        result["fwhm_days"], amplitude, "tovar2022") * 86400
    ed_quantiles = amplitude_mixture_quantiles(
        grid.amplitude_mean*conversion, grid.amplitude_sd*conversion, weights, [.16, .5, .84])
    for p, value in zip((16,50,84), ed_quantiles):
        result["equivalent_duration_s_p"+str(p)] = float(value)
    widths = np.unique(grid.fwhms)
    boundary = (grid.fwhms <= widths[1]) | (grid.fwhms >= widths[-2])
    result["width_prior_boundary_mass"] = float(theta_weights[boundary].sum())
    return result


def template_evidence(templates, t, y, variance, config, signed=False):
    """Integrate amplitude and equal template weights for a diagnostic family."""
    design, weight, inverse, bwy = _baseline_projection(t,y,variance)
    d,m = _projected_template_stats(templates,y,design,weight,inverse,bwy)
    scales = np.asarray(config.amplitude_prior_scales)[None,:]
    if signed:
        precision = m[:,None]+1/scales**2
        logbf = -np.log(scales)-.5*np.log(precision)+.5*d[:,None]**2/precision
    else:
        logbf = positive_amplitude_logbf(d,m,scales.ravel())[0]
    return float(special.logsumexp(logbf)-np.log(logbf.size))


def artifact_checks(prepared, indices, t, y, variance, result, config):
    """Spike support and a raw-flux step check; neither changes H0's log BF.

    The raw comparison conditions the flare width on the detrended fit. It is
    a diagnostic, NOT another fully marginalized detection Bayes factor.
    """
    cadence = cadence_days(prepared.lightcurve.time)
    peak = result["seed_time"]
    nearby = np.flatnonzero(abs(t-peak) <= cadence*1.01)
    impulses = np.zeros((len(t),len(nearby)))
    impulses[nearby,np.arange(len(nearby))] = 1.
    impulse = template_evidence(impulses,t,y,variance,config)
    raw_difference = np.nan
    if not config.already_detrended:
        half = max(5*cadence,min(2*result["fwhm_days"],3/24))
        inside = abs(t-peak) <= half
        if inside.sum() >= 10:
            tt=t[inside]
            yy=prepared.normalized_flux[indices[inside]]-1.
            vv=variance[inside]
            # A jump's onset can precede the maximum of its detrending
            # residual. Both diagnostic families receive the SAME wider
            # time prior; restricting a step to the residual maximum is biased.
            peaks=np.linspace(peak-max(result["fwhm_days"],4*cadence),
                              peak+cadence,31)
            steps=np.clip((tt[:,None]+cadence/2-peaks[None,:])/cadence,0,1)
            flares=flare_model_batch(tt,peaks,np.full(len(peaks),result["fwhm_days"]),
                                      result["exposure_seconds"]/86400,"tovar2022")
            raw_difference=(template_evidence(steps,tt,yy,vv,config,signed=True)
                            -template_evidence(flares,tt,yy,vv,config))
    return impulse,raw_difference


def minimize_flare_chi2(t, y, variance, exposure, grid, bayesian, config):
    """Fit peak/log width continuously, profiling A>=0 and the linear baseline.

    For each exact exposure-averaged template f, weighted projection gives
    D=f'Py and M=f'Pf, hence A=max(0,D/M). This is a two-dimensional bounded
    maximum-likelihood search, not Bayesian posterior refinement. Start from
    the better of the Bayesian shape and the best grid profile likelihood.
    Retain the best finite solution, including initialization, on failure.
    """
    started = time.perf_counter()
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
        template = flare_model(t, peak, width, 1., exposure)
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
    ed = full_equivalent_duration_days(width, amplitude, "tovar2022") * 86400
    luminosity = config.tess_band_luminosity_erg_s
    energy = luminosity*ed if luminosity is not None and np.isfinite(luminosity) and luminosity > 0 else np.nan
    row = dict(tpeak_chi2=float(peak), fwhm_days_chi2=float(width),
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
            if diagnostic["amplitude_chi2"] > 0:
                for field in fields:
                    result[field] = diagnostic[field+"_chi2"]
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
    half_window = max(config.local_window_hours/48, 4*event_span)
    mask = (prepared.segment_id == candidate.segment) & (abs(lc.time-peak) <= half_window)
    noise_mask = mask.copy()
    # Padding protects the noise estimate, but must not erase measurements
    # that could contradict the target flare's rise or decay in the likelihood.
    for other in candidates:
        if other is candidate or other.segment != candidate.segment:
            continue
        lo, hi = lc.time[other.event_indices[[0,-1]]]
        padding = max(cadence, hi-lo)
        noise_mask &= ~((lc.time >= lo-padding) & (lc.time <= hi+2*padding))
        mask[other.event_indices] = False
    mask[candidate.event_indices] = True
    noise_mask[candidate.event_indices] = True
    indices = np.flatnonzero(mask)
    noise_indices = np.flatnonzero(noise_mask)
    if len(noise_indices) < 8:
        return {"tier": "followup", "reason": "insufficient_local_background",
                "tpeak": float(peak), "seed_time": float(peak)}, None
    t, y = lc.time[indices], residual[indices]
    quiet = ~np.isin(noise_indices, candidate.event_indices)
    _, scatter, _ = sigma_clip_location_scale(residual[noise_indices][quiet])
    sigma = prepared.segment_sigma[candidate.peak_index]
    if np.isfinite(scatter):
        sigma = max(sigma, scatter)
    variance = noise_variance(prepared.relative_flux_err, sigma, indices, noise_indices)
    minimum = max(.1/1440, .25*exposure)
    maximum = min(config.fwhm_max_minutes/1440, np.ptp(lc.time[noise_indices])/3)
    if maximum <= minimum:
        raise ValueError("No width prior supported by local observations")
    widths = np.geomspace(minimum, maximum, config.broad_fwhm_points)
    # Symmetric, identical peak prior for forward and time-reversed shapes.
    peaks = candidate_peak_grid(lc.time, candidate, cadence, config)
    grid, broad_logbf = refined_grid(t, y, variance, peaks, widths, exposure, config)
    result = measurement_summary(t, y, variance, exposure, grid, config)
    result.update(seed_time=float(peak), sigma=float(sigma), seed_snr=float(residual[candidate.peak_index]/sigma),
                  logbf_broad=float(broad_logbf), cadence_seconds=cadence*86400,
                  exposure_seconds=exposure*86400, n_local=len(t))
    flare = flare_model(t, result["tpeak"], result["fwhm_days"], result["amplitude"], exposure)
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
        result["tpeak"], result["fwhm_days"], "tovar2022", .99))
    result["duration_days"] = result["tend"]-result["tstart"]
    total = result["equivalent_duration_s"] / 86400
    segment = np.flatnonzero(prepared.segment_id == candidate.segment)
    observed = flare_model(lc.time[segment], result["tpeak"], result["fwhm_days"],
                             result["amplitude"], exposure).sum()*exposure
    result["observed_model_fraction"] = float(min(1., observed/total))
    reasons, warnings = [], []
    if result["log_bayes_factor"] < config.logbf_threshold:
        tier = "rejected"
        reasons.append("log_bayes_factor_below_threshold")
        result["logbf_forward_minus_reverse"] = np.nan
    else:
        reverse, _ = refined_grid(-t[::-1], y[::-1], variance[::-1], -peaks[::-1], widths, exposure, config)
        result["logbf_forward_minus_reverse"] = result["log_bayes_factor"] - grid_log_evidence(reverse)
        if result["logbf_forward_minus_reverse"] < config.direction_threshold:
            reasons.append("flare_direction_not_established")
        impulse,raw_step = artifact_checks(prepared,indices,t,y,variance,result,config)
        result["logbf_impulse"] = impulse
        result["raw_step_minus_flare_diagnostic"] = raw_step
        if result["log_bayes_factor"]-impulse < config.impulse_margin:
            reasons.append("single_exposure_shape_ambiguous")
        if np.isfinite(raw_step) and raw_step > config.raw_step_margin:
            reasons.append("raw_discontinuity_preferred")
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
        tier = "followup" if reasons else "accepted"
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
        if np.max(explained[seeds]) > .25*row["sigma"] and np.max(new_excess) < threshold:
            row.update(tier="rejected", reason="seed_explained_by_stronger_flare", parameter_usable=False)
            continue
        for parent, segment in pending:
            if segment != candidate.segment or row["tpeak"] <= parent["tpeak"]:
                continue
            tail = flare_model(lc.time[seeds], parent["tpeak"], parent["fwhm_days"],
                                 parent["amplitude"], exposure)
            if np.max(tail) > .25*row["sigma"] and np.max(new_excess-tail) < threshold:
                row.update(tier="followup", parameter_usable=False,
                           parent_candidate_id=parent["candidate_id"],
                           reason=";".join(filter(None, [row["reason"], "possible_tail_of_followup_flare"])))
                break
        if row["tier"] == "accepted":
            same_segment = prepared.segment_id == candidate.segment
            explained[same_segment] += flare_model(lc.time[same_segment], row["tpeak"], row["fwhm_days"],
                                                     row["amplitude"], exposure)
        elif "possible_tail_of_followup_flare" not in row["reason"]:
            pending.append((row, candidate.segment))


def analyze(lc, config=Config(), progress=None):
    """Process one light curve; retain every candidate and every decision."""
    start = time.perf_counter()
    prepared, diagnostics = prepare(lc, config)
    detrend_seconds = time.perf_counter()-start
    candidates = find_candidates(prepared, config)
    if progress is not None:
        progress([], len(candidates), detrend_seconds)
    results, fits = [], []
    with ExitStack() as stack:
        if config.max_candidates and len(candidates)>config.max_candidates:
            computed=((dict(tier="followup",reason="candidate_overflow_not_fitted",
                            seed_time=float(lc.time[c.peak_index])),None) for c in candidates)
        elif config.candidate_workers>1 and len(candidates)>1:
            executor=stack.enter_context(ProcessPoolExecutor(max_workers=config.candidate_workers,
                initializer=_initialize_candidate_worker,initargs=(prepared,candidates,config)))
            computed=executor.map(_candidate_worker,range(len(candidates)))
        else:
            computed=(_fit_one(prepared,candidates,i,config) for i in range(len(candidates)))
        for number,(result,fit) in enumerate(computed):
            result.update(candidate_id=number,gdr3_id=lc.gdr3_id,sector=lc.sector,
                          mission=lc.mission,source_path=lc.source_path)
            results.append(result);fits.append(fit)
            if progress is not None and (number+1)%25==0:
                progress(results,len(candidates),time.perf_counter()-start)
    diagnostics.update(detrend_seconds=detrend_seconds, total_seconds=time.perf_counter()-start)
    associate_tails(results, candidates, prepared, fits, config)
    frame=pd.DataFrame(results)
    if frame.empty:
        frame=pd.DataFrame(columns=['candidate_id','gdr3_id','sector','mission','source_path',
            'tier','reason','seed_time','tpeak','fwhm_days','amplitude','equivalent_duration_s',
            'flare_energy_erg','energy_status'])
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
                            row["exposure_seconds"]/86400), color=colors[tier], lw=1.2,
                            label=f"Exposure-averaged Tovar Mendoza ({method})")
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


def run_file(job):
    """Independent worker: load once, analyze once, write a recoverable result."""
    path, root, config, plots = job
    digest = hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:10]
    folder = root/(path.stem+"_"+digest)
    folder.mkdir(parents=True, exist_ok=True)
    try:
        lc = load_lightcurve(path, config)
        calibration = energy_calibration(lc, config)
        fit_config = replace(config, tess_band_luminosity_erg_s=calibration["tess_band_luminosity_erg_s"])
        def checkpoint(rows,total,seconds):
            (folder/"progress.json").write_text(json.dumps(dict(processed=len(rows),total=total,seconds=seconds)))
            if rows:
                annotate_energy(pd.DataFrame(rows), calibration).to_csv(folder/"partial_events.csv",index=False)
        frame, prepared, fits, diagnostics = analyze(lc, fit_config, checkpoint)
        annotate_energy(frame, calibration)
        frame.to_csv(folder/"all_events.csv", index=False)
        for tier, name in [("accepted","accepted_flares"),("followup","followup_flares"),("rejected","rejected_candidates")]:
            (frame.loc[frame.tier==tier] if len(frame) else frame).to_csv(folder/(name+".csv"),index=False)
        if plots:
            np.savez_compressed(folder/"prepared.npz", time=lc.time, original=prepared.normalized_flux,
                                residual=prepared.residual, sigma=prepared.segment_sigma,
                                **diagnostics["stages"])
            save_plots(frame, prepared, fits, folder)
        summary = dict(source=str(path), output=str(folder), candidates=len(frame),
                       accepted=int((frame.tier=="accepted").sum()) if len(frame) else 0,
                       followup=int((frame.tier=="followup").sum()) if len(frame) else 0,
                       seconds=diagnostics["total_seconds"], detrend_seconds=diagnostics["detrend_seconds"],
                       flux_column=lc.selected_flux_column, n_observations=len(lc.time),
                       portions=diagnostics["portions"], status="complete", energy_calibration=calibration)
    except Exception as error:
        summary = dict(source=str(path), status="failed", error=repr(error))
    (folder/"summary.json").write_text(json.dumps(summary,indent=2))
    return summary


BAYESIAN_EXPORT_FIELDS = [field+suffix for field in
    ("tpeak", "fwhm_days", "amplitude", "equivalent_duration_s", "flare_energy_erg")
    for suffix in ("_bayesian", "_bayesian_p16", "_bayesian_p50", "_bayesian_p84")]
MEASUREMENT_EXPORT_FIELDS = ["measurement_method", "chi2_fit_status", "chi2_fit_flags"] + BAYESIAN_EXPORT_FIELDS + ENERGY_CALIBRATION_FIELDS


def combine_catalogs(summaries, root):
    """Write three compact, combined catalogues; full per-event diagnostics remain."""
    columns = ["gdr3_id","mission","sector","tstart","duration_days","tpeak","fwhm_days",
               "amplitude","equivalent_duration_s","flare_energy_erg","chi2_red",
               "log_bayes_factor","fitted_peak_snr","parameter_usable","warnings","energy_status"] + MEASUREMENT_EXPORT_FIELDS
    for tier,name in [("accepted","accepted_flares"),("followup","followup_flares"),("rejected","rejected_candidates")]:
        selected = columns + (["reason","parent_candidate_id"] if tier!="accepted" else [])
        output = root/(name+".csv")
        pd.DataFrame(columns=selected).to_csv(output,index=False)
        for summary in summaries:
            path=Path(summary.get("output",""))/"all_events.csv"
            if not path.is_file():
                continue
            try:
                frame=pd.read_csv(path,dtype={"gdr3_id":"string"})
            except pd.errors.EmptyDataError:
                continue
            if "tier" in frame:
                frame.loc[frame.tier==tier].reindex(columns=selected).to_csv(output,index=False,header=False,mode="a")


def main():
    """CLI accepts files or folders, writes a reproducibility manifest and CSVs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--workers",type=int,default=1)
    parser.add_argument("--candidate-workers",type=int,default=1,
                        help="Independent event workers per light curve; total processes are workers times this")
    parser.add_argument("--already-detrended",action="store_true")
    parser.add_argument("--residual",action="store_true")
    parser.add_argument("--flux-column",default="")
    parser.add_argument("--plots",action="store_true")
    parser.add_argument("--chi2-fit",action=argparse.BooleanOptionalAction,default=True,
                        help="Continuous primary measurements (default ON); --no-chi2-fit uses Bayesian grid points")
    parser.add_argument("--max-candidates",type=int,default=0)
    parser.add_argument("--tess-band-luminosity",type=float,default=None,
                        help="Calibrated erg/s, only when all inputs are one Gaia source")
    parser.add_argument("--stellar-catalogs", nargs="+", type=Path, default=[])
    parser.add_argument("--distance-column", default="distance_pc", help="Catalog distance in parsecs")
    parser.add_argument("--energy-method", choices=("supplied", "tmag-bandwidth"), default="supplied")
    parser.add_argument("--tess-effective-bandwidth-hz", type=float, default=1.2e14)
    args = parser.parse_args()
    if args.workers < 1 or args.candidate_workers < 1 or args.max_candidates < 0:
        parser.error("workers must be positive; max-candidates must be nonnegative")
    catalogs = [catalog_manifest(p, args.distance_column) for p in args.stellar_catalogs]
    if args.energy_method == "tmag-bandwidth" and args.tess_band_luminosity is not None:
        parser.error("Choose --tess-band-luminosity OR --energy-method tmag-bandwidth")
    if not np.isfinite(args.tess_effective_bandwidth_hz) or args.tess_effective_bandwidth_hz <= 0:
        parser.error("Effective bandwidth must be positive finite Hz")
    paths = set()
    physical_paths = set()
    for path in args.input:
        for item in path.rglob("*") if path.is_dir() else [path]:
            if item.is_file() and item.suffix.lower() in {".fits",".fit",".fts",".parquet",".csv"}:
                if item.resolve() not in physical_paths:
                    # The symlink name carries the user's Gaia DR3 identity.
                    paths.add(item.absolute())
                    physical_paths.add(item.resolve())
    if not paths:
        parser.error('No supported light-curve files found in the supplied inputs')
    config = Config(already_detrended=args.already_detrended,flux_is_residual=args.residual,
                    flux_column=args.flux_column,max_candidates=args.max_candidates,
                    tess_band_luminosity_erg_s=args.tess_band_luminosity,candidate_workers=args.candidate_workers,
                    chi2_fit=args.chi2_fit, stellar_catalog_paths=tuple(c["path"] for c in catalogs),
                    distance_column=args.distance_column, energy_method=args.energy_method,
                    tess_effective_bandwidth_hz=args.tess_effective_bandwidth_hz)
    if args.tess_band_luminosity is not None:
        identifiers = {metadata_from_path(path)[0] for path in paths}
        if (len(identifiers)!=1 or identifiers & {"UNKNOWN",""}
                or not np.isfinite(args.tess_band_luminosity) or args.tess_band_luminosity<=0):
            parser.error("A supplied luminosity requires one known Gaia filename ID and positive finite erg/s")
    root = args.output_dir
    root.mkdir(parents=True,exist_ok=True)
    manifest = dict(config=asdict(config), stellar_catalogs=catalogs, inputs=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(paths)],
                    code={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                          for name in (Path(__file__).name,)})
    (root/"manifest.json").write_text(json.dumps(manifest,indent=2))
    jobs = [(p,root,config,args.plots) for p in sorted(paths)]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        summaries = list(tqdm(executor.map(run_file,jobs),total=len(jobs),desc="Light curves",unit="LC"))
    pd.DataFrame(summaries).to_csv(root/"runtimes.csv",index=False)
    combine_catalogs(summaries,root)
    if any(s["status"]=="failed" for s in summaries):
        raise SystemExit("Some light curves failed: see runtimes.csv; no failures were hidden.")


# ======================================================================
# SERVER
# ======================================================================





import csv
import fcntl
import platform
import shutil
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import asdict
from importlib.metadata import version



CODE = (Path(__file__).name,)
SAMPLES = ("open_clusters", "field_stars")
TABLES = {"accepted": "accepted_flares.csv", "followup": "followup_flares.csv",
          "rejected": "rejected_candidates.csv"}
OUTPUTS = ("all_events.csv", *TABLES.values(), "summary.json")


def atomic_json(path, value):
    """Publish a complete checkpoint, never a partially written JSON file."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True))
    temporary.replace(path)


def fingerprint(path):
    """Cheap resume identity; following a symlink must resolve to the same file."""
    info = path.stat()
    return dict(resolved=str(path.resolve(strict=True)), size=info.st_size,
                mtime_ns=info.st_mtime_ns)


def catalog_manifest(path, distance_column):
    """Validate and fingerprint a read-only Gaia catalog before launching jobs."""
    path = path.expanduser().resolve(strict=True)
    lookup = stellar_catalog_lookup((str(path),), distance_column)
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                input=fingerprint(path), stars=len(lookup),
                stars_with_distance=sum(row["distance_pc"] is not None for row in lookup.values()))


def discover(root, samples, limit=0):
    """Keep Gaia filenames; skip only aliases of the same file for the same ID."""
    records, seen = [], set()
    for sample in samples:
        folder = root / sample
        if not folder.is_dir():
            raise FileNotFoundError(f"Missing input sample: {folder}")
        count = 0
        for path in sorted(folder.rglob("*")):
            if path.name.startswith(".") or path.suffix.lower() not in {".fits", ".fit", ".fts"}:
                continue
            if not path.is_file():
                raise FileNotFoundError(f"Broken or invalid FITS path: {path}")
            identifier, _, _ = metadata_from_path(path)
            if not identifier.isdigit():
                raise ValueError(f"A Gaia DR3-prefixed filename is required: {path}")
            signature = fingerprint(path)
            key = (sample, identifier, signature["resolved"])
            if key in seen:
                continue
            seen.add(key)
            records.append(dict(sample=sample, path=str(path), input=signature))
            count += 1
            if limit and count >= limit:
                break
        if not count:
            raise ValueError(f"No Gaia-named FITS files in {folder}")
    return records


def product_folder(record, output):
    """Match BANSHEE's deterministic per-product output folder."""
    path = Path(record["path"])
    digest = hashlib.sha256(record["input"]["resolved"].encode()).hexdigest()[:10]
    return output / record["sample"] / "products" / (path.stem + "_" + digest)


def completed(record, output):
    """Only a successful, fully written product can be skipped on resume."""
    folder = product_folder(record, output)
    try:
        receipt = json.loads((folder / "runner_complete.json").read_text())
        summary = json.loads((folder / "summary.json").read_text())
        return (receipt == record and summary["status"] == "complete"
                and all((folder / name).is_file() for name in OUTPUTS))
    except (OSError, ValueError, KeyError):
        return False


def process(record, output, config, plots):
    """Run one unchanged analysis, then mark it resumable if input stayed fixed."""
    path = Path(record["path"])
    start = time.perf_counter()
    try:
        if fingerprint(path) != record["input"]:
            raise ValueError("Input changed after inventory")
        catalog = record.get("stellar_catalog")
        if catalog:
            if fingerprint(Path(catalog["path"])) != catalog["input"]:
                raise ValueError("Stellar catalog changed after inventory")
            config = replace(config, stellar_catalog_paths=(catalog["path"],))
        summary = run_file((path, output / record["sample"] / "products", config, plots))
        if fingerprint(path) != record["input"]:
            raise ValueError("Input changed during processing; results are not resumable")
        if catalog and fingerprint(Path(catalog["path"])) != catalog["input"]:
            raise ValueError("Stellar catalog changed during processing")
        if summary["status"] == "complete":
            atomic_json(product_folder(record, output) / "runner_complete.json", record)
    except Exception as error:
        summary = dict(source=str(path), status="failed", error=repr(error))
    return dict(sample=record["sample"], wall_seconds=time.perf_counter()-start, **summary)


def merge(records, output):
    """Read each diagnostic CSV once; retain full provenance in three main tables."""
    fields = ["sample", "gdr3_id", "mission", "sector", "source_path", "candidate_id",
              "exposure_seconds", "tstart", "duration_days", "tpeak", "fwhm_days",
              "amplitude", "equivalent_duration_s", "flare_energy_erg", "chi2_red",
              "log_bayes_factor", "fitted_peak_snr", "parameter_usable", "warnings",
              "energy_status"] + MEASUREMENT_EXPORT_FIELDS
    counts = {sample: dict(complete_products=0, incomplete_products=0,
                           accepted=0, followup=0, rejected=0) for sample in SAMPLES}
    pending = []
    with ExitStack() as stack:
        writers = {}
        for tier, name in TABLES.items():
            destination = output / name
            temporary = destination.with_suffix(".csv.tmp")
            handle = stack.enter_context(temporary.open("w", newline=""))
            columns = fields + (["reason", "parent_candidate_id"] if tier != "accepted" else [])
            writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writers[tier] = writer
            pending.append((temporary, destination))
        for record in tqdm(records, desc="Combining tables", unit="LC"):
            sample = record["sample"]
            if not completed(record, output):
                counts[sample]["incomplete_products"] += 1
                continue
            counts[sample]["complete_products"] += 1
            with (product_folder(record, output) / "all_events.csv").open(newline="") as handle:
                for row in csv.DictReader(handle):
                    row["sample"] = sample
                    writers[row["tier"]].writerow(row)
                    counts[sample][row["tier"]] += 1
    for temporary, destination in pending:
        temporary.replace(destination)
    atomic_json(output / "counts.json", counts)
    return counts


def server_main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=Path.home() / "BANSHEE_READY_LIGHTCURVES")
    parser.add_argument("--output-root", type=Path, default=Path.home() / "OCFS_Flare_Table_Lite")
    parser.add_argument("--samples", nargs="+", choices=SAMPLES, default=list(SAMPLES))
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--plots", action="store_true")
    parser.add_argument("--max-candidates", type=int, default=300,
                        help="If a curve exceeds this count, retain all candidates as unfitted follow-up; 0 disables the limit (default: 300)")
    parser.add_argument("--chi2-fit",action=argparse.BooleanOptionalAction,default=True,
                        help="Continuous primary measurements (default ON); Bayesian estimates retained separately")
    parser.add_argument("--limit", type=int, default=0, help="Smoke test only: maximum inputs per sample; use a separate output")
    parser.add_argument("--oc-catalog", type=Path)
    parser.add_argument("--fs-catalog", type=Path)
    parser.add_argument("--distance-column", default="distance_pc", help="Catalog distance in parsecs")
    parser.add_argument("--energy-method", choices=("supplied", "tmag-bandwidth"), default="supplied",
                        help="Opt in to approximate TESSMAG/distance energies; no Lbol substitution")
    parser.add_argument("--tess-effective-bandwidth-hz", type=float, default=1.2e14)
    args = parser.parse_args()
    if args.workers < 1 or args.limit < 0 or args.max_candidates < 0:
        parser.error("Workers must be positive; limit and max-candidates must be nonnegative")
    if not np.isfinite(args.tess_effective_bandwidth_hz) or args.tess_effective_bandwidth_hz <= 0:
        parser.error("Effective bandwidth must be positive finite Hz")
    catalog_paths = dict(open_clusters=args.oc_catalog, field_stars=args.fs_catalog)
    if args.energy_method == "tmag-bandwidth" and any(catalog_paths[s] is None for s in args.samples):
        parser.error("TESSMAG energies require a catalog for each selected sample (--oc-catalog/--fs-catalog)")
    catalogs = {s: catalog_manifest(catalog_paths[s], args.distance_column)
                for s in args.samples if catalog_paths[s] is not None}
    root, output = args.input_root.absolute(), args.output_root.absolute()
    if root == output or root in output.parents or output in root.parents:
        parser.error("Input and output roots must be separate, non-nested folders")
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".runner.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("Another runner is already using this output folder")
        config = Config(candidate_workers=1, max_candidates=args.max_candidates, chi2_fit=args.chi2_fit,
                        distance_column=args.distance_column, energy_method=args.energy_method,
                        tess_effective_bandwidth_hz=args.tess_effective_bandwidth_hz)
        home = Path(__file__).resolve().parent
        versions = {name: version(name) for name in
                    ("numpy", "scipy", "pandas", "astropy", "lightkurve", "matplotlib", "tqdm")}
        print("Indexing audited FITS paths; no downloads or FITS copies.", flush=True)
        records = discover(root, args.samples, args.limit)
        for record in records:
            if record["sample"] in catalogs:
                record["stellar_catalog"] = catalogs[record["sample"]]
        for sample, catalog in catalogs.items():
            print(f"{sample}: {catalog['stars_with_distance']}/{catalog['stars']} catalog stars have distance; matched by Gaia DR3 ID only.", flush=True)
        hashes = {name: hashlib.sha256((home / name).read_bytes()).hexdigest()
                  for name in (*CODE, Path(__file__).name)}
        manifest = json.loads(json.dumps(dict(config=asdict(config), code=hashes, inputs=records,
                                             stellar_catalogs=catalogs, plots=args.plots, packages=versions, python=platform.python_version())))
        marker = output / "runner_manifest.json"
        if marker.exists():
            if json.loads(marker.read_text()) != manifest:
                raise SystemExit("Code, inputs, environment or settings changed. Use a NEW output folder.")
        else:
            if any(p.name != ".runner.lock" for p in output.iterdir()):
                raise SystemExit("Output is not empty and has no runner manifest. Choose a new folder.")
            atomic_json(marker, manifest)
            snapshot = output / "code"
            snapshot.mkdir(exist_ok=True)
            for name in hashes:
                shutil.copy2(home / name, snapshot / name)
        remaining = [record for record in records if not completed(record, output)]
        print(f"Inputs: {len(records)}; already complete: {len(records)-len(remaining)}; remaining: {len(remaining)}", flush=True)
        overflow = (f"curves with >{config.max_candidates} candidates retained as unfitted follow-up"
                    if config.max_candidates else "overflow OFF")
        energy_note = ("approximate TESS-band energy from TESSMAG/distance (no bolometric correction)"
                       if args.energy_method == "tmag-bandwidth" else "energy unavailable without calibrated luminosity")
        print(f"Raw-product detrending ON; ln BF=5; fitted peak<3 sigma -> follow-up after evidence cut; {overflow}; {energy_note}.", flush=True)
        started = time.perf_counter()
        failures = 0
        with (output / "run_log.jsonl").open("a", buffering=1) as log:
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                iterator = iter(remaining)
                # Bound outstanding tasks so a 100,000-file run does not queue everything.
                active = {}
                def submit():
                    record = next(iterator, None)
                    if record is not None:
                        active[pool.submit(process, record, output, config, args.plots)] = record
                for _ in range(min(2*args.workers, len(remaining))):
                    submit()
                with tqdm(total=len(remaining), desc="Light curves", unit="LC",
                          bar_format="{desc}: {n_fmt}/{total_fmt} | {percentage:3.0f}% | "
                                     "elapsed {elapsed} | time left {remaining} | {rate_fmt}{postfix}") as bar:
                    while active:
                        done, _ = wait(active, timeout=10, return_when=FIRST_COMPLETED)
                        if not done:
                            bar.refresh()
                        for future in done:
                            record = active.pop(future)
                            try:
                                result = future.result()
                            except Exception as error:
                                result = dict(source=record["path"], sample=record["sample"],
                                              status="failed", error=repr(error))
                            log.write(json.dumps(result) + "\n")
                            failures += result["status"] != "complete"
                            bar.update(1)
                            bar.set_postfix(failed=failures)
                            submit()
        if any(hashlib.sha256((home / name).read_bytes()).hexdigest() != digest for name, digest in hashes.items()):
            raise SystemExit("Code changed while running. Do not use mixed results.")
        if any(hashlib.sha256(Path(c["path"]).read_bytes()).hexdigest() != c["sha256"] for c in catalogs.values()):
            raise SystemExit("Stellar catalog changed while running. Do not use mixed results.")
        counts = merge(records, output)
        atomic_json(output / "run_summary.json", dict(seconds_this_run=time.perf_counter()-started,
                    workers=args.workers, attempted_this_run=len(remaining), failed_this_run=failures, counts=counts))
        print(json.dumps(counts, indent=2), flush=True)
        print(f"Combined CSVs: {output}", flush=True)
        if failures or any(row["incomplete_products"] for row in counts.values()):
            raise SystemExit("Some products failed. See run_log.jsonl; rerun the same command to retry them.")


if __name__ == "__main__":
    if "--server" in sys.argv:
        sys.argv.remove("--server")
        server_main()
    else:
        main()

# BANSHEE flags and what they mean
# Thresholds below are the defaults. Follow-up does not mean "not a flare";
# it means we want to look at the event before accepting it.
#
# FOLLOW-UP REASONS
# insufficient_local_background
#   Fewer than 8 points remain in the local fitting window, so no fit is made.
# fit_failed:<message>
#   The fit failed. The message tells us what went wrong.
# candidate_overflow_not_fitted
#   The curve exceeded the configured candidate limit, so all its candidates
#   were left unfitted for review. Server mode defaults to 300 candidates;
#   ordinary CLI/Config defaults leave the limit off. Zero disables the limit.
# flare_direction_not_established
#   The flare shape is not favored strongly enough over its time reverse:
#   ln BF_forward - ln BF_reverse < ln(10).
# single_exposure_shape_ambiguous
#   A single-point spike is too competitive with the flare model:
#   ln BF_flare - ln BF_spike < 1.
# raw_discontinuity_preferred
#   A step-like change fits the original selected flux better
#   (step-minus-flare diagnostic > 2). This does not prove it is an artifact.
# peak_near_gap_or_edge
#   The fitted peak is within 2 cadences of a detection-segment boundary.
# fitted_peak_below_three_sigma
#   The exposure-averaged fitted peak, sampled at the fitted observations,
#   is below 3*sigma above the local baseline.
# possible_tail_of_followup_flare
#   This could be part of another follow-up flare's tail. Check them together.
#
# REJECTION REASONS
# log_bayes_factor_below_threshold
#   ln BF < 5. There is not enough evidence under our model comparison.
# seed_explained_by_stronger_flare
#   An accepted flare already explains the excess at this seed.
#   We do not want to count its tail as a separate flare.
#
# MEASUREMENT WARNINGS
# These do not automatically reject a flare or move it to follow-up.
# They set parameter_usable=False because the measurements need caution.
# structured_positive_residual
#   At least 2 consecutive significant positive residuals remain in the
#   model-supported region. Could be complexity, a poor fit, or background.
# cadence_limited
#   FWHM < 1.5 exposure durations. Amplitude and width may be poorly resolved.
# broad_width_posterior
#   The width's p84/p16 ratio is greater than 3.
# width_prior_boundary
#   More than 10% of the posterior sits near the width-grid boundaries.
# extrapolated_model_area
#   Less than 90% of the fitted flare area is covered by observed exposures.
#
# DETRENDING WARNINGS
# These are saved in the portion summaries, not automatically as flare cuts.
# short_portion: Fewer than 5 points; ordinary detrending is skipped.
# cosine_fit_failed:<type>: Cosine fitting stopped after an error or warning.
# savgol_window_too_short: Too few samples for that smoothing stage.
# insufficient_savgol_background: Fewer than 5 unmasked background points.
# insufficient_fringe_background: Fewer than 10 unmasked edge-fit points.
# invalid_fringe_span: The edge-fit time span is not positive.
# fringe_fit_failed:<type>: Edge correction failed; keep the pre-edge flux.
#
# ENERGY AND STATUS FIELDS
# model_energy_conditional_on_supplied_band_luminosity
#   Energy was calculated as model ED times the supplied band luminosity.
# requires_calibrated_tess_band_luminosity
#   No valid band luminosity was supplied. ED may exist, but energy is NaN.
# model_energy_tmag_bandwidth_approximation
#   Energy uses model ED times luminosity estimated from TESSMAG and distance.
#   This is approximate TESS-band energy, not bolometric energy.
# missing_or_invalid_distance / missing_or_invalid_tess_magnitude
#   Calibration input is absent or invalid; the flare fit is still retained.
# invalid_tmag_bandwidth_luminosity / invalid_model_energy
#   The numerical conversion failed; energy is not treated as usable.
# not_fitted
#   No fitted event exists (for example candidate overflow); no model energy.
# Flags alone do not stop energy calculation when a fit and luminosity exist.
# parameter_usable: Accepted with no measurement warnings; not a guarantee.
# parent_candidate_id: Identifies the possible parent flare in this file.
# quality_filter_applied: QUALITY==0 is mandatory when a QUALITY column exists.
#   False means no QUALITY column was supplied, not that the cut was disabled.
# tier: The final accepted, followup, or rejected decision.
# status: Whether the light-curve processing completed or failed.
# No 3-sigma seed means no candidate row, not an entry in rejected_candidates.
