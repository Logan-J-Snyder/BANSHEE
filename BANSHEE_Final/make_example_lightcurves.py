#!/usr/bin/env python3
"""Create artificial examples for testing installation, never real observations."""
from pathlib import Path
import argparse
import json
import numpy as np
from astropy.io import fits
import BANSHEE_Final as b


def generate(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("Choose an empty example directory; existing files are preserved")
    truth = []
    for k, (cadence, exposure, mission, width_minutes) in enumerate([
        (20, 19.8, 'SPOC', 5), (120, 118.8, 'SPOC', 12),
        (200, 200, 'QLP', 20), (600, 600, 'QLP', 40),
        (1800, 1800, 'QLP', 90),
    ]):
        rng = np.random.default_rng(7300 + k)
        t = 2000 + np.arange(720) * cadence / 86400
        peak = t[330] + 0.17 * cadence / 86400
        profile = b.flare_model(t, peak, width_minutes / 1440, 1., exposure / 86400)
        noise = 0.001
        amplitude = 12 * noise / profile.max()
        flux = 1 + rng.normal(0, noise, len(t)) + amplitude * profile
        columns = [fits.Column(name='TIME', format='D', array=t),
                   fits.Column(name='SAP_FLUX', format='D', array=flux),
                   fits.Column(name='SAP_FLUX_ERR', format='D', array=np.full(len(t), noise)),
                   fits.Column(name='QUALITY', format='J', array=np.zeros(len(t), dtype=int))]
        hdu = fits.BinTableHDU.from_columns(columns)
        hdu.header['XPOSURE'] = exposure
        hdu.header['SECTOR'] = 1
        hdu.header['SYNTHET'] = True
        filename = f'{1000000000000000001+k}_S01_{mission}_{cadence}s_LC.fits'
        fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(output / filename)
        truth.append(dict(file=filename, synthetic=True, cadence_seconds=cadence,
                          exposure_seconds=exposure, tpeak=peak,
                          fwhm_minutes=width_minutes, amplitude=amplitude,
                          noiseless_sampled_peak_snr=12.))
    (output / 'injection_truth.json').write_text(json.dumps(truth, indent=2) + '\n')
    return truth


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('example_lightcurves'))
    args = parser.parse_args()
    records = generate(args.output)
    print(f'Created {len(records)} synthetic examples in {args.output}')
