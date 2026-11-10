"""Edge follow-up changes must not promote artifacts or affect interior fits."""
from dataclasses import replace
from unittest.mock import patch
import unittest
import numpy as np
import BANSHEE_Final as b


class EdgeReviewTests(unittest.TestCase):
    def lc(self, gap=False):
        t=2000+np.arange(1000)*20/86400
        if gap:t[500:]+=1
        return b.LightCurve(t,np.ones(1000),None,'5482551183847322752',1,'SPOC',
                            'synthetic','SAP_FLUX','raw',exposure_days=19.8/86400)

    def prepared(self,lc,config,gap=False):
        p=type('Prepared',(),{})()
        p.lightcurve=lc;p.segment_id=np.r_[np.zeros(500,int),np.full(500,int(gap))]
        p.background_uncertain_mask=np.zeros(1000,bool)
        p.detrend_edge_mask=b.edge_mask(p,config)
        return p

    def candidate(self,k,segment=0):
        return b.Candidate(np.array([k]),np.array([k]),k,segment)

    def characterize(self,prepared,candidate,candidates,config):
        t=prepared.lightcurve.time[candidate.peak_index]
        return dict(tier='accepted',reason='',seed_time=t,tpeak=t,fitted_peak_snr=5,
                    logbf_flare_vs_alternatives=20.,parameter_usable=True),None

    def test_both_outer_edges_are_unfit_followup_and_interior_is_accepted(self):
        lc=self.lc();cfg=b.Config();p=self.prepared(lc,cfg)
        candidates=[self.candidate(k) for k in (20,500,980)]
        with patch.object(b,'prepare',return_value=(p,{})), \
             patch.object(b,'find_candidates',return_value=candidates), \
             patch.object(b,'characterize',side_effect=self.characterize) as fitting:
            frame,_,fits,diagnostics=b.analyze(lc,cfg)
        self.assertEqual(frame.tier.tolist(),['followup','accepted','followup'])
        self.assertEqual(frame.reason.tolist(),['edge_region','','edge_region'])
        self.assertEqual(frame.edge_review_only.tolist(),[True,False,True])
        self.assertTrue(frame.loc[[0,2],'parameter_usable'].eq(False).all())
        self.assertTrue(frame.loc[[0,2],'fitted_peak_snr'].isna().all())
        self.assertIsNone(fits[0]);self.assertIsNone(fits[2])
        self.assertEqual(fitting.call_count,1)
        self.assertEqual(diagnostics['edge_review_candidates'],2)

    def test_segment_edge_mode_includes_both_sides_of_a_gap(self):
        lc=self.lc(gap=True);cfg=b.Config(edge_scope='segment',allow_uncalibrated=True)
        p=self.prepared(lc,cfg,gap=True)
        candidates=[self.candidate(k,int(k>=500)) for k in (10,490,510,990)]
        with patch.object(b,'prepare',return_value=(p,{})), \
             patch.object(b,'find_candidates',return_value=candidates), \
             patch.object(b,'characterize',side_effect=AssertionError('Edge seed must not be fit')):
            frame,_,_,_=b.analyze(lc,cfg)
        self.assertTrue(frame.tier.eq('followup').all())
        self.assertTrue(frame.edge_review_only.all())

    def test_disabling_edge_cut_disables_edge_review(self):
        lc=self.lc();cfg=b.Config(edge_fraction=0,allow_uncalibrated=True)
        p=self.prepared(lc,cfg);candidates=[self.candidate(20),self.candidate(980)]
        with patch.object(b,'prepare',return_value=(p,{})), \
             patch.object(b,'find_candidates',return_value=candidates), \
             patch.object(b,'characterize',side_effect=self.characterize):
            frame,_,_,_=b.analyze(lc,cfg)
        self.assertTrue(frame.tier.eq('accepted').all())
        self.assertFalse(frame.edge_review_only.any())

    def test_fitted_peak_crossing_edge_retains_fit_for_followup(self):
        lc=self.lc();cfg=b.Config();p=self.prepared(lc,cfg)
        row,_=self.characterize(p,self.candidate(51),[],cfg);row['tpeak']=lc.time[48]
        fit={'indices':np.arange(40,70),'marker':'retained'}
        with patch.object(b,'prepare',return_value=(p,{})), \
             patch.object(b,'find_candidates',return_value=[self.candidate(51)]), \
             patch.object(b,'_fit_one',return_value=(row,fit)):
            frame,_,fits,_=b.analyze(lc,cfg)
        self.assertEqual(frame.iloc[0].tier,'followup')
        self.assertFalse(frame.iloc[0].parameter_usable)
        self.assertIs(fits[0],fit)

    def test_fitted_edge_event_below_operating_cut_stays_rejected(self):
        lc=self.lc();cfg=b.Config();p=self.prepared(lc,cfg)
        row,_=self.characterize(p,self.candidate(51),[],cfg)
        row.update(tpeak=lc.time[48],logbf_flare_vs_alternatives=2.)
        with patch.object(b,'prepare',return_value=(p,{})), \
             patch.object(b,'find_candidates',return_value=[self.candidate(51)]), \
             patch.object(b,'_fit_one',return_value=(row,None)):
            frame,_,_,_=b.analyze(lc,cfg)
        self.assertEqual(frame.iloc[0].tier,'rejected')
        self.assertIn('log_bayes_factor_below_threshold',frame.iloc[0].reason)
        self.assertFalse(frame.iloc[0].edge_review_only)

    def test_independently_rejected_artifact_is_not_promoted(self):
        lc=self.lc();cfg=b.Config();p=self.prepared(lc,cfg)
        row,_=self.characterize(p,self.candidate(51),[],cfg)
        row.update(tpeak=lc.time[48],tier='rejected',reason='insufficient_competing_model_evidence')
        with patch.object(b,'prepare',return_value=(p,{})), \
             patch.object(b,'find_candidates',return_value=[self.candidate(51)]), \
             patch.object(b,'_fit_one',return_value=(row,None)):
            frame,_,_,_=b.analyze(lc,cfg)
        self.assertEqual(frame.iloc[0].tier,'rejected')
        self.assertFalse(frame.iloc[0].edge_review_only)


if __name__=='__main__':unittest.main()
