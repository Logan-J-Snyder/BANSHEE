"""Regression/physics tests. Run: python -m unittest -v test_BANSHEE_Final.py"""
import json,tempfile,unittest
from pathlib import Path
from dataclasses import replace
from unittest.mock import patch
import numpy as np
import pandas as pd
from astropy.io import fits
import BANSHEE_Final as b

class BansheeTests(unittest.TestCase):
    def lightcurve(self, n=2400, cadence=120, seed=7, varying=False):
        rng=np.random.default_rng(seed); t=2000+np.arange(n)*cadence/86400
        sigma=np.where(np.arange(n)<n//2,1e-3,4e-3) if varying else np.full(n,1e-3)
        return b.LightCurve(t,1+rng.normal(size=n)*sigma,None,'5482551183847322752',3,'SPOC','synthetic',
                           'SAP_FLUX','raw',exposure_days=cadence/86400)
    def test_exact_identifier(self):
        self.assertEqual(b.clean_id('5482551183847322752'),'5482551183847322752')
        with self.assertRaises(ValueError): b.clean_id(5482551183847322752.)
    def test_matching_error_column_only(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'5482551183847322752_S3_SPOC.fits'
            cols=[fits.Column(name='TIME',format='D',array=np.arange(50)*.01),
                  fits.Column(name='SAP_FLUX',format='D',array=np.ones(50)),
                  fits.Column(name='PDCSAP_FLUX_ERR',format='D',array=np.ones(50)*99)]
            fits.HDUList([fits.PrimaryHDU(),fits.BinTableHDU.from_columns(cols)]).writeto(p)
            self.assertIsNone(b.load_lightcurve(p,b.Config(flux_column='SAP_FLUX')).flux_err)
    def test_mixed_star_table_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'a.csv'
            pd.DataFrame(dict(time=np.arange(10),flux=np.ones(10),gaia_dr3_id=['11']*5+['12']*5)).to_csv(p,index=False)
            with self.assertRaises(ValueError): b.load_lightcurve(p,b.Config())
    def test_mixed_exposures_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'a.csv'
            pd.DataFrame(dict(time=np.arange(10),flux=np.ones(10),exposure_days=[.01]*5+[.02]*5)).to_csv(p,index=False)
            with self.assertRaises(ValueError): b.load_lightcurve(p,b.Config())
    def test_local_noise_tracks_change(self):
        lc=self.lightcurve(varying=True); p,_=b.prepare(lc,b.Config(already_detrended=True))
        lo,hi=np.median(p.segment_sigma[200:1000]),np.median(p.segment_sigma[1400:2200])
        self.assertTrue(.0008<lo<.0013,(lo,hi)); self.assertTrue(.0032<hi<.0052,(lo,hi))
        self.assertGreater(hi/lo,3)
    def test_broad_flare_does_not_set_its_own_noise(self):
        lc=self.lightcurve(n=7200)
        lc.flux += .005*np.sin(2*np.pi*(lc.time-lc.time[0])/2.)
        center=lc.time[3500];width=40/1440;amplitude=.05
        base,_=b.prepare(lc,b.Config())
        injected=replace(lc,flux=lc.flux+b.flare_model(lc.time,center,width,amplitude,lc.exposure_days))
        f,p,_,d=b.analyze(injected,b.Config(allow_uncalibrated=True))
        self.assertTrue(p.protection_mask[3500])
        self.assertLess(p.segment_sigma[3500]/base.segment_sigma[3500],1.8)
        a=f.loc[(f.tier=='accepted')&(abs(f.tpeak-center)<2*lc.exposure_days)]
        self.assertEqual(len(a),1,f[['tier','reason']].to_dict('records'))
        ratio=a.iloc[0].equivalent_duration_s/(b.full_equivalent_duration_days(width,amplitude,'tovar2022')*86400)
        self.assertTrue(.8<ratio<1.2,ratio)
        self.assertEqual(b.Config().candidate_sigma,3.)
    def test_short_cadence_slow_decay_survives_detrending(self):
        lc=self.lightcurve(n=18000,cadence=20)
        center=lc.time[8500];width=35/1440;amplitude=.02
        lc.flux += b.flare_model(lc.time,center,width,amplitude,lc.exposure_days)
        f,p,_,_=b.analyze(lc,b.Config(allow_uncalibrated=True))
        a=f.loc[(f.tier=='accepted')&(abs(f.tpeak-center)<width)]
        self.assertEqual(len(a),1)
        self.assertTrue(.8<a.iloc[0].fwhm_days/width<1.2)
    def test_broad_raw_gaussian_still_not_accepted(self):
        lc=self.lightcurve(n=7200)
        center=lc.time[3500]
        lc.flux += .05*np.exp(-.5*((lc.time-center)/(.025))**2)
        f,_,_,_=b.analyze(lc,b.Config(allow_uncalibrated=True))
        near=f.loc[abs(f.seed_time-center)<.1]
        self.assertGreater(len(near),0)
        self.assertFalse((near.tier=='accepted').any())
    def test_injections_avoid_edges_and_gaps_with_wings(self):
        lc=self.lightcurve(n=7200);lc.time[3600:]+=2
        width=.02;rng=np.random.default_rng(17)
        for scope in ('lightcurve','segment'):
            cfg=b.Config(edge_scope=scope)
            prepared,_=b.prepare(lc,replace(cfg,already_detrended=True))
            for _ in range(100):
                k,meta=b.calibration_center(lc,cfg,width,rng)
                self.assertFalse(prepared.detrend_edge_mask[k])
                inside=abs(lc.time-lc.time[k])<=12*width
                self.assertFalse(prepared.detrend_edge_mask[inside].any())
                segment=prepared.segment_id[k];ii=np.flatnonzero(prepared.segment_id==segment)
                self.assertGreaterEqual(lc.time[k]-lc.time[ii[0]],12*width)
                self.assertGreaterEqual(lc.time[ii[-1]]-lc.time[k],12*width)
    def test_injection_impossible_placement_is_explicit_failure(self):
        with self.assertRaisesRegex(ValueError,'No injection placement'):
            b.calibration_center(self.lightcurve(n=100),b.Config(),.1,np.random.default_rng(8))
    def test_masked_savgol_cannot_extrapolate_outside_quiet_range(self):
        t=np.arange(400)/48.;flux=1+.0001*np.sin(t)
        keep=np.ones(400,bool);keep[:40]=False;keep[100:200]=False;keep[350:]=False
        flux[~keep]+=10.  # A protected flare must never enter the background.
        trend=b.protected_savgol_background(t,flux,keep,13)
        self.assertGreaterEqual(trend.min(),flux[keep].min())
        self.assertLessEqual(trend.max(),flux[keep].max())
        self.assertTrue(np.all(np.isfinite(trend)))
    def test_coarse_cadence_broad_flare_preserved(self):
        lc=self.lightcurve(n=1000,cadence=1800)
        center=lc.time[500];width=100/1440;amplitude=.03
        lc.flux+=b.flare_model(lc.time,center,width,amplitude,lc.exposure_days)
        f,p,_,_=b.analyze(lc,b.Config(allow_uncalibrated=True))
        a=f.loc[(f.tier=='accepted')&(abs(f.tpeak-center)<width)]
        self.assertEqual(len(a),1)
        ratio=a.iloc[0].equivalent_duration_s/(b.full_equivalent_duration_days(width,amplitude,'tovar2022')*86400)
        self.assertTrue(.75<ratio<1.25,ratio)
    def test_noise_no_gap_bridge(self):
        lc=self.lightcurve(varying=True); lc.time[1200:]+=5
        p,_=b.prepare(lc,b.Config(already_detrended=True))
        self.assertNotEqual(p.segment_id[1199],p.segment_id[1200])
        self.assertLess(p.segment_sigma[1199],.002); self.assertGreater(p.segment_sigma[1200],.002)
    def test_correlated_background_does_not_mask_the_entire_curve(self):
        lc=self.lightcurve(n=6000);rng=np.random.default_rng(47)
        red=np.zeros(6000)
        for i in range(1,len(red)):red[i]=.95*red[i-1]+rng.normal()*.001*np.sqrt(1-.95**2)
        lc.flux=1+red
        p,d=b.prepare(lc,b.Config())
        self.assertLess(p.protection_mask.mean(),.75)
        self.assertFalse(p.background_uncertain_mask.any())
        self.assertGreater(np.median(p.segment_sigma),.0005)
    def test_injection_centers_respect_supported_background_and_existing_seeds(self):
        lc=self.lightcurve(n=6000);usable=np.ones(6000,bool);usable[1000:2000]=False
        seeds=lc.time[[500,3000,4000]];width=.01;rng=np.random.default_rng(9)
        for _ in range(100):
            k,_=b.calibration_center(lc,b.Config(),width,rng,usable,seeds)
            self.assertTrue(usable[k]);self.assertGreater(np.min(abs(lc.time[k]-seeds)),width)
    def test_calibration_snr_uses_preinjection_local_total_noise(self):
        with tempfile.TemporaryDirectory() as folder:
            lc=self.lightcurve(n=6000,varying=True)
            path=Path(folder)/'5482551183847322752_S3_SPOC.csv'
            pd.DataFrame(dict(time=lc.time,flux=lc.flux)).to_csv(path,index=False)
            cfg=b.Config(already_detrended=True,allow_uncalibrated=True,flux_column='flux')
            source,sigma,usable,seeds,snr=b.calibration_background(str(path),cfg,(0.,1.))
            spec=dict(path=str(path),source_fraction=(0.,1.),seed=31415,kind='flare',phase='train',trial=0,star=lc.gdr3_id,group=b.calibration_group(source))
            r=b.calibration_trial(spec,cfg);self.assertEqual(r['status'],'complete',r)
            k=np.argmin(abs(source.time-r['injected_peak']))
            self.assertAlmostEqual(r['injection_noise_sigma_raw'],sigma[k],places=12)
            unit=b.flare_model(source.time,r['injected_peak'],r['injected_fwhm_days'],r['injected_amplitude'],source.exposure_days or b.cadence_days(source.time),r['injected_model'])
            self.assertAlmostEqual(unit.max()/sigma[k],r['injected_sampled_snr'],places=8)
    def test_formal_errors_not_shrunk(self):
        lc=replace(self.lightcurve(),flux_err=np.ones(2400)*.005)
        p,_=b.prepare(lc,b.Config(already_detrended=True))
        self.assertGreaterEqual(np.nanmin(p.segment_sigma),.0049)
    def test_edges_and_exposure_accounting(self):
        p,d=b.prepare(self.lightcurve(n=1000),b.Config(already_detrended=True))
        self.assertEqual(p.detrend_edge_mask.sum(),100)
        self.assertAlmostEqual(d['searchable_exposure_days']/d['observed_exposure_days'],.9)
        z,_=b.prepare(self.lightcurve(n=1000),b.Config(already_detrended=True,edge_fraction=0))
        self.assertFalse(z.detrend_edge_mask.any())
    def test_segment_edges_option(self):
        lc=self.lightcurve(n=1000);lc.time[500:]+=3
        p,_=b.prepare(lc,b.Config(already_detrended=True,edge_scope='segment'))
        self.assertTrue(p.detrend_edge_mask[499]);self.assertTrue(p.detrend_edge_mask[500])
    def test_integrated_area_and_cadence(self):
        t=np.arange(-1,2,20/86400); width=.01; amp=.02
        ed=b.full_equivalent_duration_days(width,amp,'tovar2022')
        for exposure in (20,120,600,1800):
            sampled=np.sum(b.flare_model(t,0,width,amp,exposure/86400))*20/86400
            self.assertAlmostEqual(sampled/ed,1,places=5)
    def test_calibration_group_distinguishes_flux_and_exposure(self):
        lc=self.lightcurve();other=replace(lc,selected_flux_column='PDCSAP_FLUX')
        self.assertNotEqual(b.calibration_group(lc),b.calibration_group(other))
        self.assertNotEqual(b.calibration_group(lc),b.calibration_group(replace(lc,exposure_days=118.8/86400)))
    def test_calibration_cannot_pass_tiny_null_sample(self):
        rows=[dict(kind='quiet',status='complete',score=None) for _ in range(10)]
        self.assertIsNone(b.choose_threshold(rows,5,.05,.95))
        self.assertGreater(b.binomial_upper(0,10),.25)
    def test_threshold_uses_training_only(self):
        rows=[dict(kind='quiet',status='complete',score=None) for _ in range(100)]
        rows[0]['score']=80
        th=b.choose_threshold(rows,5,.05,.95)
        self.assertEqual(th,5) # 1/100 upper bound is <5%.
    def test_missing_calibration_abstains(self):
        lc=self.lightcurve(n=1400);lc.flux+=b.flare_model(lc.time,lc.time[650],.012,.04,lc.exposure_days)
        f,_,_,_=b.analyze(lc,b.Config(already_detrended=True,max_candidates=0,builtin_calibration=False))
        self.assertFalse((f.tier=='accepted').any())
        self.assertTrue(f.diagnostic_reasons.str.contains('calibration_missing').any())
    def test_large_flare_recovered_with_explicit_override(self):
        lc=self.lightcurve(n=1400);lc.flux+=b.flare_model(lc.time,lc.time[650],.012,.04,lc.exposure_days)
        f,_,_,_=b.analyze(lc,b.Config(already_detrended=True,allow_uncalibrated=True,max_candidates=0))
        a=f.loc[f.tier=='accepted']; self.assertTrue(len(a)>0)
        matched=a.iloc[np.argmin(abs(a.tpeak-lc.time[650]))]
        self.assertLess(abs(matched.tpeak-lc.time[650]),2*lc.exposure_days)
        self.assertTrue(.7<matched.equivalent_duration_s/(b.full_equivalent_duration_days(.012,.04,'tovar2022')*86400)<1.3)
    def test_edge_impulse_is_fit_and_rejected(self):
        lc=self.lightcurve(n=1000);lc.flux[8]+=.1
        f,_,_,_=b.analyze(lc,b.Config(already_detrended=True,allow_uncalibrated=True))
        edge=f.loc[f.seed_time.eq(lc.time[8])].iloc[0]
        self.assertEqual(edge.tier,'rejected')
        self.assertTrue(edge.edge_case)
        self.assertTrue(np.isfinite(edge.fwhm_days))
    def test_overflow_retains_all_candidates(self):
        lc=self.lightcurve(n=1400)
        lc.flux[np.arange(100,1300,100)]+=.03
        f,_,fits_,_=b.analyze(lc,b.Config(already_detrended=True,max_candidates=1))
        self.assertGreater(len(f),1);self.assertTrue(f.diagnostic_reasons.eq('candidate_overflow_not_fitted').any())
        self.assertTrue(all(x is None for x in fits_))
    def test_symmetric_bump_not_accepted_as_flare(self):
        lc=self.lightcurve(n=1400)
        center=lc.time[650]
        lc.flux+=.04*np.exp(-.5*((lc.time-center)/.012)**2)
        f,_,_,_=b.analyze(lc,b.Config(already_detrended=True,allow_uncalibrated=True))
        near=f.loc[abs(f.seed_time-center)<.05]
        self.assertTrue(len(near)>0)
        self.assertFalse((near.tier=='accepted').any())
    def test_raw_step_not_accepted_as_flare(self):
        lc=self.lightcurve(n=2000)
        center=lc.time[900];lc.flux[900:]+=.04
        f,_,_,_=b.analyze(lc,b.Config(allow_uncalibrated=True))
        self.assertFalse((f.tier=='accepted').any())
    def test_detrending_bump_without_raw_flare_is_not_accepted(self):
        lc=self.lightcurve(n=1400)
        p,d=b.prepare(lc,b.Config(already_detrended=True))
        p.residual+=b.flare_model(lc.time,lc.time[650],.012,.04,lc.exposure_days)
        p.trend=p.normalized_flux-p.residual
        with patch.object(b,'prepare',return_value=(p,d)):
            f,_,_,_=b.analyze(lc,b.Config(allow_uncalibrated=True))
        near=f.loc[abs(f.seed_time-lc.time[650])<.05]
        self.assertGreater(len(near),0)
        self.assertFalse((near.tier=='accepted').any())
        self.assertTrue(near.reason.str.contains('raw_flare_signal_not_established').any())
    def test_windowed_search_keeps_full_detrending_context(self):
        lc=self.lightcurve(n=1800)
        c=b.Config(already_detrended=True,allow_uncalibrated=True)
        lc.flux+=b.flare_model(lc.time,lc.time[850],.012,.04,lc.exposure_days)
        full,p,_,_=b.analyze(lc,c,associate=False)
        window,q,_,_=b.analyze(lc,c,associate=False,search_bounds=(lc.time[700],lc.time[1000]))
        self.assertEqual(len(q.residual),len(p.residual))
        np.testing.assert_array_equal(p.residual,q.residual)
        f=full.loc[full.tier=='accepted'].iloc[0];w=window.loc[window.tier=='accepted'].iloc[0]
        self.assertAlmostEqual(f.log_bayes_factor,w.log_bayes_factor,places=8)
    def test_unbounded_fringe_extrapolation_retains_previous_flux(self):
        rng=np.random.default_rng(17);t=np.arange(300)/720
        flux=1+rng.normal(0,.001,len(t));flux[0]+=.1
        def broken_fit(function,x,y,p0,**kwargs):
            pars=np.array(p0);pars[0]=1e12
            return pars,np.eye(len(pars))
        with patch.object(b,'curve_fit',side_effect=broken_fit):
            corrected,flag=b.fringe_stage(t,flux,b.IlinConfig())
        self.assertEqual(flag,'fringe_correction_unbounded')
        np.testing.assert_array_equal(corrected,flux)
    def test_edge_parent_cannot_suppress_valid_interior_child(self):
        lc=self.lightcurve(n=1000)
        candidates=[b.Candidate(np.array([i]),np.array([i]),i,0) for i in (51,60)]
        def fit_one(prepared,all_candidates,index,config,**kwargs):
            i=(51,60)[index]
            row=dict(seed_time=float(lc.time[i]),tpeak=float(lc.time[48 if index==0 else i]),
                     amplitude=.05 if index==0 else .01,fwhm_days=.03,sigma=.001,
                     tier='accepted',reason='',parameter_usable=True)
            return row,{'exists':True}
        with patch.object(b,'find_candidates',return_value=candidates),patch.object(b,'_fit_one',side_effect=fit_one):
            f,_,_,_=b.analyze(lc,b.Config(already_detrended=True,allow_uncalibrated=True))
        self.assertEqual(f.iloc[0].tier,'accepted')
        self.assertFalse(f.iloc[0].edge_review_only)
        self.assertEqual(f.iloc[1].tier,'accepted')
    def test_calibration_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'cal.json';path.write_text(json.dumps(dict(detector_version='wrong',code_sha256='wrong')))
            with self.assertRaises(ValueError): b.calibrated_config(self.lightcurve(),b.Config(calibration_file=str(path)))
    def test_separate_prominent_flares_on_shared_positive_background(self):
        lc=self.lightcurve(n=2400)
        cfg=b.Config(already_detrended=True,allow_uncalibrated=True)
        p,_=b.prepare(lc,cfg)
        # A smooth elevated residual must not merge two sharp peaks hours apart.
        p.residual[:]=.004+.001*np.sin(np.arange(len(lc.time))/400)
        for i in (900,1080):p.residual+=b.flare_model(lc.time,lc.time[i],.004,.025,lc.exposure_days)
        p.segment_sigma[:]=.001
        cs=b.find_candidates(p,cfg)
        self.assertTrue(any(abs(lc.time[c.peak_index]-lc.time[900])<2*lc.exposure_days for c in cs))
        self.assertTrue(any(abs(lc.time[c.peak_index]-lc.time[1080])<2*lc.exposure_days for c in cs))
        for c in cs:
            self.assertLess(np.ptp(lc.time[c.event_indices]),.15)

    def test_peak_prior_cannot_follow_a_long_tail(self):
        t=np.arange(1000)/720
        c=b.Candidate(np.array([500]),np.arange(450,800),500,0,.004)
        grid=b.candidate_peak_grid(t,c,1/720,b.Config())
        self.assertLessEqual(max(abs(grid-t[500])),.00200001)

    def test_compact_flags_preserve_details_and_quality(self):
        r=b.concise_flags(dict(tier='accepted',reason='',warnings='width_prior_boundary;cadence_limited;chi2_parameter_bound',parameter_usable=False))
        self.assertEqual(r['warnings'],'cadence_limited');self.assertEqual(r['reason'],'')
        self.assertFalse(r['parameter_usable']);self.assertIn('width_prior_boundary',r['parameter_diagnostics'])
        f=pd.DataFrame([r]);compact=b.compact_table(f)
        self.assertNotIn('parameter_diagnostics',compact);self.assertIn('parameter_diagnostics',f)
        r=b.concise_flags(dict(tier='followup',reason='symmetric_shape_not_excluded;flare_direction_not_established',warnings='width_prior_boundary'))
        self.assertEqual(r['reason'],'shape_ambiguous');self.assertEqual(r['warnings'],'')
        self.assertIn('symmetric_shape_not_excluded',r['diagnostic_reasons'])

    def test_empty_lightcurve_analysis_has_diagnostic_schema(self):
        lc=self.lightcurve(n=1000);lc.flux[:]=1.
        f,_,_,_=b.analyze(lc,b.Config(already_detrended=True))
        self.assertEqual(len(f),0);self.assertIn('diagnostic_reasons',f)

    def test_calibration_can_search_below_old_five_floor(self):
        rows=[dict(kind='quiet',status='complete',score=None) for _ in range(100)]
        self.assertEqual(b.choose_threshold(rows,0,.05,.95),0)
        rows[:5]=[dict(kind='quiet',status='complete',score=1.2) for _ in range(5)]
        threshold=b.choose_threshold(rows,0,.05,.95)
        self.assertGreater(threshold,1.2);self.assertLess(threshold,5)

    def test_quiet_controls_exclude_entire_preexisting_event_window(self):
        snr=np.zeros(200);snr[50]=10;snr[150]=9
        valid=b.quiet_control_window_centers(snr,40)
        for center in range(200):
            start=max(0,min(160,center-20))
            self.assertEqual(bool(valid[center]),not (start<=50<start+40))
        self.assertTrue(valid[150])
        snr[0]=20
        self.assertFalse(b.quiet_control_window_centers(snr,40)[0])

    def test_masking_cannot_erase_local_background_scatter(self):
        lc=self.lightcurve(n=1000)
        cfg=b.Config(already_detrended=True)
        p,_=b.prepare(lc,cfg)
        p.residual[:]=.01*np.sin(np.arange(1000)*2*np.pi/40)
        p.protection_mask[:]=np.abs(p.residual)>.001
        p.segment_sigma[:]=.0001
        sigma,_=b.local_noise(p,cfg)
        self.assertGreater(np.median(sigma[100:-100]),.002)

    def test_quadratic_diagnostic_projects_shared_curvature(self):
        t=np.linspace(-.1,.1,151);variance=np.full(len(t),1e-6)
        flare=b.flare_model(t,0,.006,.02,0)
        cfg=b.Config();peaks=np.linspace(-.003,.003,7);widths=np.geomspace(.002,.02,13)
        g=b.evaluate_grid(t,flare,variance,peaks,widths,0,cfg,baseline_degree=2)
        h=b.evaluate_grid(t,flare+.01+2*t*t-.03*t,variance,peaks,widths,0,cfg,baseline_degree=2)
        np.testing.assert_allclose(g.logbf,h.logbf,rtol=1e-9,atol=1e-8)

    def test_raw_diagnostic_restores_masked_contradictory_samples(self):
        lc=self.lightcurve(n=1000);peak=lc.time[500];width=.006
        lc.flux+=.03*np.exp(-.5*((lc.time-peak)/width)**2)
        p,_=b.prepare(lc,b.Config(already_detrended=True))
        ii=np.flatnonzero((lc.time>=peak)&(lc.time<peak+.025))
        row=dict(seed_time=peak,fwhm_days=width,observed_width_hint_days=width,exposure_seconds=120.)
        b.artifact_checks(p,ii,lc.time[ii],p.residual[ii],p.segment_sigma[ii]**2,row,
            b.Config(),np.linspace(peak-.003,peak+.003,9),np.geomspace(.001,.03,15))
        self.assertGreater(row['raw_diagnostic_points'],len(ii))
        self.assertEqual(row['raw_background_degree'],2)

    def test_operating_cut_applied_after_association(self):
        lc=self.lightcurve(n=1000)
        candidates=[b.Candidate(np.array([i]),np.array([i]),i,0) for i in (500,510)]
        def fit_one(prepared,all_candidates,index,config,**kwargs):
            i=(500,510)[index]
            return dict(seed_time=float(lc.time[i]),tpeak=float(lc.time[i]),
                amplitude=.05 if index==0 else .01,fwhm_days=.03,sigma=.001,
                log_bayes_factor=10. if index==0 else 100.,tier='accepted',reason='',
                parameter_usable=True),{'exists':True}
        with patch.object(b,'find_candidates',return_value=candidates),patch.object(b,'_fit_one',side_effect=fit_one):
            f,_,_,_=b.analyze(lc,b.Config(already_detrended=True,allow_uncalibrated=True,logbf_threshold=20,detection_statistic="background"))
        self.assertEqual(f.iloc[0].tier,'rejected')
        self.assertIn('log_bayes_factor_below_threshold',f.iloc[0].diagnostic_reasons)
        self.assertEqual(f.iloc[1].tier,'rejected')
        self.assertIn('seed_explained_by_stronger_flare',f.iloc[1].diagnostic_reasons)

    def test_negative_operating_threshold_not_supported(self):
        with self.assertRaises(ValueError):b.Config(logbf_threshold=-1.)

    def test_merged_exports_keep_ids_and_compact_diagnostics(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);src=root/'5482551183847322752_S3_SPOC.csv'
            lc=self.lightcurve(n=500)
            pd.DataFrame(dict(time=lc.time,flux=lc.flux)).to_csv(src,index=False)
            record=dict(sample='field_stars',path=str(src),input=b.fingerprint(src))
            rows=[]
            for i,tier in enumerate(('accepted','followup','rejected')):
                rows.append(b.concise_flags(dict(gdr3_id='5482551183847322752',tier=tier,
                    candidate_id=i,reason='' if tier=='accepted' else 'symmetric_shape_not_excluded',
                    warnings='cadence_limited;width_prior_boundary',parameter_usable=False)))
            with patch.object(b,'analyze',return_value=(pd.DataFrame(rows),None,[],{})):
                s=b.run_product(record,root/'out',b.Config(already_detrended=True,flux_column='flux'),False,'run')
            self.assertEqual(s['status'],'complete',s)
            merged=b.merge_products([record],root/'out','run')
            self.assertEqual(merged['counts'],dict(accepted=1,followup=1,rejected=1))
            for suffix in ('csv','parquet'):
                path=root/'out'/('accepted_flares.'+suffix)
                f=pd.read_csv(path,dtype={'gaia_dr3_id':'string'}) if suffix=='csv' else pd.read_parquet(path)
                self.assertEqual(f.gaia_dr3_id.iloc[0],'5482551183847322752')
                self.assertEqual(f.warnings.iloc[0],'cadence_limited')
                self.assertNotIn('parameter_diagnostics',f)
                self.assertFalse(f.parameter_usable.iloc[0])
            full=pd.read_parquet(root/'out/all_events.parquet')
            self.assertIn('width_prior_boundary',full.parameter_diagnostics.iloc[0])

    def test_csv_parquet_ids_and_no_stale_completion(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); src=root/'5482551183847322752_S3_SPOC.csv'
            lc=self.lightcurve(n=500)
            pd.DataFrame(dict(time=lc.time,flux=lc.flux)).to_csv(src,index=False)
            record=dict(sample='field_stars',path=str(src),input=b.fingerprint(src))
            config=b.Config(already_detrended=True,flux_column='flux')
            summary=b.run_product(record,root/'out',config,False,'run1')
            self.assertEqual(summary['status'],'complete',summary)
            self.assertTrue(b.completed(record,root/'out','run1'))
            self.assertFalse(b.completed(record,root/'out','run2'))
            summary=b.run_product(record,root/'out',replace(config,flux_column='MISSING'),False,'run1')
            self.assertEqual(summary['status'],'failed')
            self.assertFalse(b.completed(record,root/'out','run1'))

    def test_flare_family_mixture_integrates_model_prior(self):
        from scipy.special import logsumexp
        t=np.linspace(-.05,.1,101);y=b.flare_model(t,0,.01,.01,0);v=np.full(len(t),1e-6)
        peaks=np.linspace(-.003,.003,5);widths=np.geomspace(.003,.03,11)
        cfg=b.Config(rise_factors=(.5,1.,2.))
        mixture=b.evaluate_grid(t,y,v,peaks,widths,0,cfg)
        values=[b.grid_log_evidence(b.evaluate_grid(t,y,v,peaks,widths,0,replace(cfg,rise_factors=(factor,)))) for factor in cfg.rise_factors]
        self.assertAlmostEqual(b.grid_log_evidence(mixture),logsumexp(values)-np.log(3),places=9)
        self.assertAlmostEqual(mixture.theta_prior_mass.sum(),1.,places=12)

    def test_stretched_templates_preserve_physical_units(self):
        from scipy.integrate import quad
        for factor in (.5,2.,4.):
            name='tovar2022_rise'+str(factor);fn,integral,width=b.template_functions(name)
            self.assertAlmostEqual(float(fn(0)),1.,places=7)
            self.assertAlmostEqual(float(fn(-.28851193331160346*factor)),.5,places=7)
            self.assertAlmostEqual(float(fn(.7981444440820005)),.5,places=7)
            self.assertAlmostEqual(float(integral(1e-10)-integral(-1e-10)),2e-10,places=8)
            exp=.002;center=-.001;fwhm=.01
            observed=float(b.flare_model(np.array([center]),0,fwhm,.03,exp,name)[0])
            numerical=quad(lambda t:float(b.flare_model(np.array([t]),0,fwhm,.03,0,name)[0]),center-exp/2,center+exp/2,epsabs=1e-12)[0]/exp
            self.assertAlmostEqual(observed,numerical,places=9)

    def test_model_specific_fit_and_equivalent_duration(self):
        t=np.linspace(-.05,.15,151);name='tovar2022_rise2.0'
        y=b.flare_model(t,0,.01,.02,1/720,name);v=np.full(len(t),1e-8)
        cfg=b.Config();grid=b.evaluate_grid(t,y,v,np.linspace(-.003,.003,9),np.geomspace(.004,.025,21),1/720,cfg)
        fit=b.measurement_summary(t,y,v,1/720,grid,cfg)
        self.assertEqual(fit['flare_model_name'],name)
        self.assertAlmostEqual(fit['equivalent_duration_s']/b.full_equivalent_duration_days(.01,.02,name)/86400,1.,places=4)

    def test_local_candidate_significance_is_offset_invariant(self):
        lc=self.lightcurve(n=1500);lc.flux+=b.flare_model(lc.time,lc.time[700],.006,.02,lc.exposure_days)
        cfg=b.Config(already_detrended=True);p,_=b.prepare(lc,cfg)
        before=[c.peak_index for c in b.find_candidates(p,cfg)]
        p.residual-=.2;p.local_baseline-=.2
        self.assertEqual(before,[c.peak_index for c in b.find_candidates(p,cfg)])
        self.assertTrue(any(abs(i-700)<=1 for i in before))

    def test_negative_global_residual_does_not_hide_local_flare(self):
        lc=self.lightcurve(n=1500);cfg=b.Config(already_detrended=True)
        p,_=b.prepare(lc,cfg);p.residual[:]=-.02+b.flare_model(lc.time,lc.time[700],.006,.01,lc.exposure_days)
        p.segment_sigma[:]=.001;p.protection_mask[:]=False
        p.local_baseline=b.quiet_local_baseline(p,cfg)
        cs=b.find_candidates(p,cfg)
        self.assertTrue(any(abs(c.peak_index-700)<=1 for c in cs))

    def test_cosine_fit_cannot_extrapolate_unbounded_correction(self):
        lc=self.lightcurve(n=1800)
        def broken(function,x,y,p0,**kwargs):
            pars=np.array(p0);pars[0]=1e9;return pars,np.eye(len(pars))
        with patch.object(b,'curve_fit',side_effect=broken):
            corrected,frequencies,flags=b.cosine_stage(lc.time,lc.flux,b.IlinConfig())
        np.testing.assert_array_equal(corrected,lc.flux)
        self.assertIn('cosine_correction_unbounded',flags)

    def test_alternative_score_not_background_score_sets_cut(self):
        lc=self.lightcurve(n=1000);c=b.Candidate(np.array([500]),np.array([500]),500,0)
        row=dict(seed_time=lc.time[500],tpeak=lc.time[500],log_bayes_factor=100.,logbf_flare_vs_alternatives=2.,tier='accepted',reason='',parameter_usable=True)
        with patch.object(b,'find_candidates',return_value=[c]),patch.object(b,'_fit_one',return_value=(row,None)):
            f,_,_,_=b.analyze(lc,b.Config(already_detrended=True,allow_uncalibrated=True,logbf_threshold=3),associate=False)
        self.assertEqual(f.iloc[0].tier,'rejected');self.assertEqual(f.iloc[0].detection_log_bayes_factor,2.)

    def test_step_marginalization_matches_dense_templates(self):
        from scipy.special import logsumexp
        t=2000+np.arange(35)*120/86400;t[18:]+=.01
        exposure=118.8/86400;cfg=b.Config()
        y=.02*np.clip((t-t[15]) / exposure+.71,0,1)+.003*(t-t.mean())
        y+=np.random.default_rng(14).normal(0,.001,len(t));variance=np.full(len(t),1e-6)
        edges=np.unique(np.clip(np.r_[t[0],t-exposure/2,t+exposure/2,t[-1]],t[0],t[-1]))
        z,w=np.polynomial.legendre.leggauss(32)
        pp=((edges[1:]+edges[:-1])[:,None]/2+np.diff(edges)[:,None]/2*z).ravel()
        mass=(np.diff(edges)[:,None]/(2*np.ptp(t))*w).ravel()
        scores=[b.template_evidence(np.clip((t[:,None]+exposure/2-p)/exposure,0,1),t,y,variance,cfg,signed=True,baseline_degree=2) for p in pp]
        expected=logsumexp(np.array(scores)+np.log(mass))
        actual=b.step_grid_evidence(t,y,variance,exposure,cfg)
        self.assertAlmostEqual(actual,expected,places=4)
        finer=b.step_grid_evidence(t,y,variance,exposure,cfg,order=64)
        self.assertAlmostEqual(actual,finer,places=3)
    def test_partial_exposure_steps_do_not_become_flares(self):
        for phase in (.0,.29,.73):
            with self.subTest(phase=phase):
                lc=self.lightcurve(n=3500)
                center=lc.time[1700]+phase*lc.exposure_days
                lc.flux+=.05*np.clip((lc.time-center)/lc.exposure_days+.5,0,1)
                f,_,_,_=b.analyze(lc,b.Config(allow_uncalibrated=True))
                near=f[abs(f.seed_time-center)<.15]
                self.assertGreater(len(near),0)
                self.assertFalse(near.tier.eq('accepted').any(),near[['tier','reason']].to_dict('records'))

    def test_raw_step_context_includes_broad_fitted_decay(self):
        lc=self.lightcurve(n=2400)
        center=lc.time[1100]
        lc.flux+=.02*np.clip((lc.time-center)/lc.exposure_days+.5,0,1)
        cfg=b.Config();p,_=b.prepare(lc,cfg)
        seed=center+.12
        indices=np.flatnonzero(abs(lc.time-seed)<.06)
        result=dict(seed_time=seed,fwhm_days=.3,observed_width_hint_days=.01,
                    exposure_seconds=lc.exposure_days*86400)
        _,difference,_,_=b.artifact_checks(p,indices,lc.time[indices],p.residual[indices],
          p.segment_sigma[indices]**2,result,cfg,seed+np.linspace(-.01,.01,7),np.geomspace(.001,.4,10))
        self.assertGreater(result['raw_diagnostic_points'],500)
        self.assertGreater(difference,2.3)

if __name__=='__main__':unittest.main()
