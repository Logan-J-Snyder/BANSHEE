"""Offline regression contracts, not a survey-completeness benchmark."""

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from astropy.io import fits
from scipy.integrate import quad

import BANSHEE_Lite_Chi as B


ROOT = Path(__file__).resolve().parents[1]


def fitting_problem():
    exposure = 120 / 86400
    t = np.arange(-0.04, 0.08, exposure)
    truth = (0.00023, 0.0047, 0.023)
    y = 0.0002 + 0.0001 * t + B.flare_model(t, *truth, exposure)
    variance = np.full(len(t), 1e-8)
    grid = B.evaluate_grid(t, y, variance, np.linspace(-0.003, 0.003, 9),
                          np.geomspace(0.0005, 0.02, 41), exposure, B.Config())
    return t, y, variance, exposure, grid, truth


def test_default_scientific_configuration():
    config = B.Config()
    assert config.chi2_fit and not config.already_detrended
    assert (config.broad_fwhm_points, config.broad_tpeak_points) == (41, 9)
    assert config.candidate_sigma == 3 and config.max_candidates == 0
    assert config.quality_zero_only


def test_quality_cannot_be_disabled():
    with pytest.raises(ValueError, match="mandatory"):
        B.Config(quality_zero_only=False)


def test_exact_source_identifier():
    assert B.clean_id("5853498713190525696") == "5853498713190525696"
    assert B.metadata_from_path(Path("5853498713190525696_S01_SPOC_LC.fits")) == (
        "5853498713190525696", 1, "SPOC")


@pytest.mark.parametrize("seconds", [20, 120, 200, 600, 1800])
def test_exposure_average_matches_quadrature(seconds):
    exposure = seconds / 86400
    peak, width, amplitude = 0.0, 20 / 1440, 0.025
    times = np.array([-0.012, -0.001, 0.003, 0.02, 0.07])
    model = B.flare_model(times, peak, width, amplitude, exposure)
    reference = [quad(lambda x: float(B.flare_model(np.array([x]), peak, width, amplitude)[0]),
                      t - exposure / 2, t + exposure / 2, epsabs=1e-13)[0] / exposure
                 for t in times]
    np.testing.assert_allclose(model, reference, rtol=2e-7, atol=1e-10)


def test_ed_is_full_area_in_seconds():
    width, amplitude = 0.014, 0.03
    analytic = B.full_equivalent_duration_days(width, amplitude, "tovar2022")
    area = quad(lambda x: float(B.flare_model(np.array([x]), 0, width, amplitude)[0]),
                -0.3, 2, epsabs=1e-12, points=[0])[0]
    assert analytic * 86400 == pytest.approx(area * 86400, rel=1e-6)


def test_continuous_measurements_recover_noiseless_shape():
    t, y, variance, exposure, grid, truth = fitting_problem()
    result = B.measurement_summary(t, y, variance, exposure, grid, B.Config())
    assert result["measurement_method"] == "chi2_profile_likelihood"
    assert result["chi2_fit_success"]
    assert result["chi2_min"] <= result["chi2_initial"]
    np.testing.assert_allclose([result[k] for k in ("tpeak", "fwhm_days", "amplitude")],
                               truth, rtol=1e-4, atol=1e-8)
    assert "tpeak_bayesian_p16" in result and "tpeak_p16" not in result


def test_optimizer_exception_retains_explicit_fallback(monkeypatch):
    t, y, variance, exposure, grid, _ = fitting_problem()

    def fail(*args):
        raise ValueError("forced regression-test failure")

    monkeypatch.setattr(B, "minimize_flare_chi2", fail)
    result = B.measurement_summary(t, y, variance, exposure, grid, B.Config())
    assert result["measurement_method"] == "bayesian_fallback"
    assert result["tpeak"] == result["tpeak_bayesian"]
    assert result["chi2_fit_status"].startswith("failed:")


def test_bayesian_only_mode_is_explicit():
    t, y, variance, exposure, grid, _ = fitting_problem()
    result = B.measurement_summary(t, y, variance, exposure, grid, B.Config(chi2_fit=False))
    assert result["measurement_method"] == "bayesian_grid"
    assert result["chi2_fit_status"] == "disabled"


def test_fits_quality_filter_and_exposure(tmp_path):
    path = tmp_path / "1234567890123456789_S01_SPOC_LC.fits"
    t = np.arange(12) * 120 / 86400
    q = np.zeros(len(t), dtype=int)
    q[4] = 1
    table = fits.BinTableHDU.from_columns([
        fits.Column(name="TIME", format="D", array=t),
        fits.Column(name="PDCSAP_FLUX", format="D", array=np.ones(len(t))),
        fits.Column(name="PDCSAP_FLUX_ERR", format="D", array=np.full(len(t), 0.001)),
        fits.Column(name="QUALITY", format="J", array=q),
    ])
    table.header["TIMEDEL"] = 120 / 86400
    fits.HDUList([fits.PrimaryHDU(), table]).writeto(path)
    curve = B.load_lightcurve(path, B.Config())
    assert len(curve.time) == 11 and t[4] not in curve.time
    assert curve.quality_filter_applied and curve.n_quality_rejected == 1
    assert curve.exposure_days == pytest.approx(120 / 86400)
    assert curve.gdr3_id == "1234567890123456789"


def test_input_table_roundtrip_and_duplicate_merging(tmp_path):
    frame = pd.DataFrame({"time": [5, 4, 3, 2, 1, 0, 0],
                          "flux": [1, 1, 1, 1, 1, 0.8, 1.2], "flux_err": 0.01,
                          "gdr3_id": "5853498713190525696"})
    for suffix in (".csv", ".parquet"):
        path = tmp_path / ("lightcurve" + suffix)
        if suffix == ".csv":
            frame.to_csv(path, index=False)
        else:
            frame.to_parquet(path, index=False)
        curve = B.load_lightcurve(path, B.Config())
        assert curve.gdr3_id == "5853498713190525696"
        assert curve.n_duplicates_merged == 1
        assert np.all(np.diff(curve.time) > 0)
        assert curve.flux[0] == pytest.approx(1)


def test_candidate_overflow_defers_entire_curve():
    exposure = 120 / 86400
    t = np.arange(0, 1, exposure)
    y = np.random.default_rng(31).normal(0, 0.0005, len(t))
    y += B.flare_model(t, 0.25, 0.006, 0.03, exposure)
    y += B.flare_model(t, 0.70, 0.006, 0.03, exposure)
    curve = B.LightCurve(t, y, np.full(len(t), 0.0005), "0", 1, "SPOC",
                         "synthetic", "residual_flux", "detrended", exposure)
    config = B.Config(already_detrended=True, flux_is_residual=True, max_candidates=1)
    events, _, models, _ = B.analyze(curve, config)
    assert len(events) >= 2
    assert events["tier"].eq("followup").all()
    assert events["reason"].eq("candidate_overflow_not_fitted").all()
    assert all(model is None for model in models)


@pytest.mark.parametrize("mode", [[], ["--server"]])
def test_cli_help(mode):
    result = subprocess.run([sys.executable, str(ROOT / "BANSHEE_Lite_Chi.py"), *mode, "--help"],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert "--chi2-fit" in result.stdout


def test_documented_demo_cli_and_exports(tmp_path):
    environment = dict(os.environ, MPLBACKEND="Agg", MPLCONFIGDIR=str(tmp_path / "mpl"))
    data, output = tmp_path / "inputs", tmp_path / "outputs"
    generated = subprocess.run([
        sys.executable, str(ROOT / "examples/make_demo_lightcurve.py"), "--output-dir", str(data),
    ], env=environment, capture_output=True, text=True, timeout=60)
    assert generated.returncode == 0, generated.stderr
    truth = json.loads((data / "injection_truth.json").read_text())
    source = data / "0000000000000000000_S01_SPOC_demo.csv"
    result = subprocess.run([
        sys.executable, str(ROOT / "BANSHEE_Lite_Chi.py"), "--input", str(source),
        "--output-dir", str(output), "--already-detrended", "--flux-column", "flux",
        "--tess-band-luminosity", "1e30", "--workers", "1",
    ], env=environment, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    assert pd.read_csv(output / "runtimes.csv")["status"].eq("complete").all()
    accepted = pd.read_csv(output / "accepted_flares.csv", dtype={"gdr3_id": "string"})
    assert len(accepted) >= 1
    event = accepted.iloc[(accepted["tpeak"] - truth["tpeak_btjd"]).abs().argmin()]
    assert abs(event["tpeak"] - truth["tpeak_btjd"]) < 120 / 86400
    assert event["measurement_method"] == "chi2_profile_likelihood"
    assert event["flare_energy_erg"] == pytest.approx(event["equivalent_duration_s"] * 1e30)
    assert "fwhm_days_bayesian_p84" in accepted
    for filename in ("followup_flares.csv", "rejected_candidates.csv"):
        pd.read_csv(output / filename)
    summary = subprocess.run([sys.executable, str(ROOT / "examples/summarize_results.py"), str(output)],
                             capture_output=True, text=True, timeout=30)
    assert summary.returncode == 0, summary.stderr
    assert "accepted_flares.csv" in summary.stdout
