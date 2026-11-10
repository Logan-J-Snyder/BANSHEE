"""Built-in threshold selection and safe fallback regression checks."""
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
import numpy as np
import BANSHEE_Final as b


class BuiltinDefaultsTests(unittest.TestCase):
    def lc(self, cadence=120, exposure=118.8, mission='SPOC', flux='SAP_FLUX'):
        t=2000+np.arange(100)*cadence/86400
        return b.LightCurve(t,np.ones(100),None,'5482551183847322752',3,mission,
                            'synthetic',flux,'raw',exposure_days=exposure/86400)

    def test_exact_cuts_and_status_for_all_five_groups(self):
        groups=[(20,19.8,'SPOC',4.1721795720870976,1.3862943611198906),
                (120,118.8,'SPOC',1.4221729029775079,2.8781335878400403),
                (200,200.,'QLP',1.3862943611198906,1.3862943611198906),
                (600,600.,'QLP',1.3862943611198906,1.3862943611198906)]
        for cadence,exposure,mission,ordinary,high in groups:
            with self.subTest(cadence=cadence):
                got,status=b.calibrated_config(self.lc(cadence,exposure,mission),b.Config())
                self.assertEqual(status,'validated')
                self.assertEqual(got.logbf_threshold,ordinary)
                self.assertEqual(got.high_snr_logbf_threshold,high)
                self.assertEqual(b.operating_threshold({'fitted_peak_snr':7.99},got),ordinary)
                self.assertEqual(b.operating_threshold({'fitted_peak_snr':8.},got),high)
        _,status=b.calibrated_config(self.lc(1800,1800,'QLP'),b.Config())
        self.assertEqual(status,'validation_failed')

    def test_automatic_and_explicit_sap_match(self):
        for config in (b.Config(),b.Config(flux_column='SAP_FLUX')):
            self.assertEqual(b.calibrated_config(self.lc(),config)[1],'validated')

    def test_no_transfers_to_unmeasured_exposures_streams_or_products(self):
        for lc in (self.lc(exposure=120),self.lc(flux='PDCSAP_FLUX'),
                   self.lc(mission='QLP'),self.lc(cadence=60,exposure=60)):
            self.assertNotEqual(b.calibrated_config(lc,b.Config())[1],'validated')

    def test_changed_scientific_settings_remain_unvalidated(self):
        for change in ({'already_detrended':True},{'edge_fraction':0.1},
                       {'noise_window_hours':4},{'detection_statistic':'background'},
                       {'high_snr_boundary':10}):
            with self.subTest(change=change):
                self.assertEqual(b.calibrated_config(self.lc(),b.Config(**change))[1],
                                 'settings_mismatch')

    def test_production_bookkeeping_does_not_change_the_calibration(self):
        config=b.Config(candidate_workers=2,stellar_catalog_paths=('metadata.csv',))
        self.assertEqual(b.calibrated_config(self.lc(),config)[1],'validated')

    def test_sap_priority_and_product_specific_fallback(self):
        columns=['PDCSAP_FLUX','KSPSAP_FLUX','SAP_FLUX','flux']
        for mission,fallback in [('SPOC','PDCSAP_FLUX'),('QLP','KSPSAP_FLUX')]:
            self.assertEqual(b._select_flux(columns,b.Config(),mission)[0],'SAP_FLUX')
            self.assertEqual(b._select_flux(columns[:2],b.Config(),mission)[0],fallback)
        self.assertEqual(b._select_flux(columns,b.Config(flux_column='PDCSAP_FLUX'),'SPOC')[0],
                         'PDCSAP_FLUX')

    def test_external_file_has_priority_and_hash_check_stays_strict(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'custom.json'
            data=dict(detector_version=b.DETECTOR_VERSION,code_sha256=b.file_sha256(b.__file__),
                      analysis_signature=b.analysis_signature(b.Config()),
                      groups={b.calibration_group(self.lc()):dict(status='validated',
                              logbf_threshold=9,high_snr_logbf_threshold=4)})
            path.write_text(json.dumps(data))
            config=replace(b.Config(),calibration_file=str(path))
            got,status=b.calibrated_config(self.lc(),config)
            self.assertEqual((status,got.logbf_threshold,got.high_snr_logbf_threshold),('validated',9,4))
            data['code_sha256']='wrong';path.write_text(json.dumps(data));b.read_calibration.cache_clear()
            with self.assertRaisesRegex(ValueError,'different code'):
                b.calibrated_config(self.lc(),config)

    def test_explicit_research_override_does_not_inherit_saved_cuts(self):
        for cfg in (b.Config(builtin_calibration=False),b.Config(allow_uncalibrated=True)):
            self.assertEqual(b.calibrated_config(self.lc(),cfg)[1],'missing')

    def test_1800s_otherwise_accepted_event_routes_to_followup(self):
        lc=self.lc(1800,1800,'QLP')
        candidate=b.Candidate(np.array([50]),np.arange(49,54),50,0)
        prepared=type('Prepared',(),dict(lightcurve=lc,detrend_edge_mask=np.zeros(100,bool),
                                        background_uncertain_mask=np.zeros(100,bool)))()
        row=dict(tier='accepted',reason='',seed_time=lc.time[50],tpeak=lc.time[50],
                 fitted_peak_snr=20,logbf_flare_vs_alternatives=50,parameter_usable=True)
        with patch.object(b,'prepare',return_value=(prepared,{})), \
             patch.object(b,'find_candidates',return_value=[candidate]), \
             patch.object(b,'_fit_one',return_value=(row,None)):
            frame,_,_,_=b.analyze(lc,b.Config())
        self.assertEqual(frame.iloc[0].tier,'followup')
        self.assertEqual(frame.iloc[0].reason,'calibration_required')
        self.assertIn('calibration_validation_failed',frame.iloc[0].diagnostic_reasons)

    def test_command_line_defaults_need_no_calibration_argument(self):
        args=b.make_parser().parse_args(['--input','example.fits','--output-dir','results'])
        self.assertTrue(args.builtin_calibration)
        self.assertFalse(args.calibrate or args.auto_calibrate)
        self.assertIsNone(args.calibration)


if __name__=='__main__':
    unittest.main()
