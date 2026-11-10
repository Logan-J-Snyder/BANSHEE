"""Scientific invariants and integration guards for advisory LC screening."""
import json,tempfile,unittest
from pathlib import Path
from dataclasses import replace
import numpy as np
import BANSHEE_Final as b
import BANSHEE_Precheck as q
import BANSHEE_Final_With_Precheck as runner


def fixture(cadence=120,n=6000,seed=95):
    rng=np.random.default_rng(seed);tt=2000+np.arange(n)*cadence/86400
    lc=b.LightCurve(tt,1+rng.normal(0,.001,n),None,'1006911518695850752',60,'SPOC','fixture','SAP_FLUX','raw',exposure_days=cadence/86400)
    return lc


def prepared(lc,sigma=None):
    n=len(lc.time);segment=np.empty(n,int)
    for j,ii in enumerate(b.split_segments(lc.time)):segment[ii]=j
    noise=np.full(n,.001) if sigma is None else np.asarray(sigma)
    p=b.PreparedLightCurve(lc,lc.flux.copy(),np.ones(n),lc.flux-1,None,segment,noise,
        np.zeros(n,bool),np.zeros(n,bool),np.zeros(n,bool),np.ones(n),np.zeros(n))
    p.detrend_edge_mask=b.edge_mask(p,b.Config())
    return p


class PrecheckTests(unittest.TestCase):
    def inspect(self,lc,p=None):return q.inspect(lc,prepared(lc) if p is None else p,b.Config())

    def test_quiet_all_cadences(self):
        for cadence in (20,120,200,600,1800):
            s,*_=self.inspect(fixture(cadence=cadence))
            self.assertEqual(s['status'],'ok',(cadence,s['flags']))

    def test_sustained_noise_change_all_cadences(self):
        for cadence in (20,120,200,600,1800):
            lc=fixture(cadence=cadence);sigma=np.where(np.arange(len(lc.time))<.73*len(lc.time),.001,.003)
            lc.flux=1+(lc.flux-1)*sigma/.001
            s,w,f,_=self.inspect(lc,prepared(lc,sigma))
            self.assertIn('noise_change',s['flags']);self.assertGreater(s['max_flagged_noise_ratio'],2)

    def test_ordinary_resolved_flare_is_not_automatically_unusual(self):
        for cadence in (20,120,200):
            lc=fixture(cadence=cadence)
            lc.flux+=b.flare_model(lc.time,lc.time[len(lc.time)//2],20/1440,.03,lc.exposure_days)
            s,_,f,_=self.inspect(lc)
            self.assertNotIn('transient_features',s['flags'],(cadence,f.to_dict('records')))

    def test_impulse_step_bump_and_dip_are_distinct(self):
        for kind in ('isolated_spike','flux_step','symmetric_bump','negative_excursion'):
            lc=fixture();n=len(lc.time);k=n//2
            if kind=='isolated_spike':lc.flux[k]+=.05
            if kind=='flux_step':lc.flux[k:]+=.025
            if kind=='symmetric_bump':lc.flux+=.04*np.exp(-.5*((lc.time-lc.time[k])/.03)**2)
            if kind=='negative_excursion':lc.flux[k-10:k+10]-=.025
            s,_,f,_=self.inspect(lc)
            self.assertIn(kind,set(f.kind));self.assertIn('transient_features',s['flags'])

    def test_no_jump_detection_or_difference_across_gap(self):
        lc=fixture();n=len(lc.time);lc.time[n//2:]+=10;lc.flux[n//2:]+=.1
        s,_,f,_=self.inspect(lc)
        self.assertNotIn('flux_step',set(f.kind))
        t=np.arange(10,dtype=float);y=np.zeros(10);y[5:]=100
        d=q.differences(t,y,np.array([0,1,3,5,6,7]),1.)
        np.testing.assert_array_equal(d,np.zeros(2))

    def test_edge_only_transient_does_not_request_review(self):
        lc=fixture();lc.flux[10]+=.1
        s,_,f,_=self.inspect(lc)
        self.assertGreater(len(f),0);self.assertFalse(f.in_search_region.any())
        self.assertNotIn('transient_features',s['flags'])

    def test_correlated_residuals_are_visible(self):
        lc=fixture();lc.flux+=.02*np.sin(2*np.pi*np.arange(len(lc.time))/80)
        s,_,_,_=self.inspect(lc)
        self.assertIn('structured_background',s['flags'])

    def test_short_baseline_is_not_assessed_not_clean(self):
        s,*_=self.inspect(fixture(n=25))
        self.assertEqual(s['status'],'not_assessed');self.assertFalse(s['noise_assessed'])

    def test_constant_flux_is_explicit(self):
        lc=fixture();lc.flux[:]=1
        s,*_=self.inspect(lc)
        self.assertIn('repeated_or_constant_flux',s['flags'])

    def test_input_and_prepared_arrays_are_unchanged(self):
        lc=fixture();p=prepared(lc)
        before={name:value.copy() for name,value in vars(p).items() if isinstance(value,np.ndarray)}
        flux=lc.flux.copy();self.inspect(lc,p)
        for name,value in before.items():np.testing.assert_array_equal(getattr(p,name),value)
        np.testing.assert_array_equal(lc.flux,flux)

    def test_saved_precheck_missing_output_is_not_complete(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);record=dict(path='/tmp/123_S01_SPOC.fits',sample='field_stars',input=dict(resolved='/tmp/123_S01_SPOC.fits'))
            product=b.product_folder(record,root);product.mkdir(parents=True)
            (product/'precheck.json').write_text('{}')
            b.atomic_json(product/'complete.json',dict(record=record,run_id='x',outputs={'precheck.json':b.file_sha256(product/'precheck.json')}))
            self.assertTrue(runner.cached(record,root,'x'))
            (product/'precheck.json').write_text('{"changed":1}')
            self.assertFalse(runner.cached(record,root,'x'))

    def test_threshold_validation_and_frozen_detector(self):
        for kwargs in ({'noise_ratio':1.},{'noise_window_hours':0},{'transient_sigma':float('nan')},{'min_noise_pairs':4}):
            with self.assertRaises(ValueError):q.PrecheckConfig(**kwargs)
        manifest=json.loads((Path(b.__file__).parent/'RELEASE_MANIFEST.json').read_text())
        self.assertEqual(b.file_sha256(b.__file__),manifest['files']['BANSHEE_Final.py'])


if __name__=='__main__':unittest.main()
