import unittest,json,tempfile
from pathlib import Path
from dataclasses import replace
import numpy as np
import BANSHEE_Final as b

class Revision8Tests(unittest.TestCase):
    def test_joint_control_budget_is_not_spent_twice(self):
        rows=[dict(kind='quiet',status='complete',score=None,score_ordinary=None,score_high=None) for _ in range(384)]
        for i in range(10):rows[i].update(score=2.,score_high=2.)
        for i in range(10,20):rows[i].update(score=3.,score_ordinary=3.)
        rows += [dict(kind='flare',status='complete',injected_sampled_snr=20.,score=4.,score_ordinary=None,score_high=4.) for _ in range(20)]
        cut=b.choose_operating_thresholds(rows,np.log(4),.05,.95)
        self.assertIsNotNone(cut)
        negatives=rows[:384]
        n=sum(any(r.get('score_'+k) is not None and r['score_'+k]>=cut[k] for k in ('ordinary','high')) for r in negatives)
        self.assertLessEqual(b.binomial_upper(n,len(negatives),.95),.05)
        self.assertLessEqual(cut['high'],4.)
    def test_strength_class_uses_fitted_snr_and_enforces_mixture_floor(self):
        cfg=b.Config(logbf_threshold=7,high_snr_logbf_threshold=0,high_snr_boundary=8)
        self.assertEqual(b.operating_threshold(dict(fitted_peak_snr=7.99,injected_sampled_snr=100),cfg),7.)
        self.assertAlmostEqual(b.operating_threshold(dict(fitted_peak_snr=8),cfg),np.log(4))
        self.assertEqual(b.operating_threshold({},cfg),7.)
    def test_calibration_roundtrip_retains_both_cuts(self):
        lc=b.LightCurve(np.arange(100)*120/86400, np.ones(100),None,'123',1,'SPOC','fixture','SAP_FLUX','raw',exposure_days=120/86400)
        cfg=b.Config()
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'calibration.json'
            p.write_text(json.dumps(dict(detector_version=b.DETECTOR_VERSION,code_sha256=b.file_sha256(b.__file__),
                analysis_signature=b.analysis_signature(cfg),groups={b.calibration_group(lc):dict(status='validated',logbf_threshold=7.,high_snr_logbf_threshold=2.)})))
            got,status=b.calibrated_config(lc,replace(cfg,calibration_file=str(p)))
            self.assertEqual(status,'validated');self.assertEqual(got.logbf_threshold,7.);self.assertEqual(got.high_snr_logbf_threshold,2.)
    def test_physical_width_prior_and_strength_boundary_are_signed_settings(self):
        sig=b.analysis_signature(b.Config())
        self.assertEqual(sig['fwhm_min_minutes'],.1)
        self.assertEqual(sig['high_snr_boundary'],8.)
        self.assertNotIn('high_snr_logbf_threshold',sig)
        with self.assertRaises(ValueError):b.Config(fwhm_min_minutes=0)
        with self.assertRaises(ValueError):b.Config(high_snr_logbf_threshold=float('nan'))

if __name__=='__main__':unittest.main()
