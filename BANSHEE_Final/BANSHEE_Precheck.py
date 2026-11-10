"""Advisory light-curve diagnostics. Never change data or flare decisions.

Noise thresholds are explicit screening rules, not calibrated false-alarm
probabilities. Transient features are shape warnings, not physical labels.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import json
from pathlib import Path
import BANSHEE_Final as b
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.stats import levene

VERSION = '2026.11.05.1'


@dataclass(frozen=True)
class PrecheckConfig:
    noise_ratio: float = 1.5
    noise_window_hours: float = 2.
    min_noise_pairs: int = 32
    noise_family_alpha: float = .01
    transient_sigma: float = 8.
    max_unusable_fraction: float = .5

    def __post_init__(self):
        if not np.isfinite(self.noise_ratio) or self.noise_ratio < 1.3:
            raise ValueError('noise_ratio must be finite and at least 1.3')
        if not np.isfinite(self.noise_window_hours) or self.noise_window_hours <= 0:
            raise ValueError('noise_window_hours must be positive and finite')
        if not isinstance(self.min_noise_pairs, int) or self.min_noise_pairs < 16:
            raise ValueError('min_noise_pairs must be an integer >=16')
        if not np.isfinite(self.noise_family_alpha) or not 0 < self.noise_family_alpha <= .05:
            raise ValueError('noise_family_alpha must be in (0, .05]')
        if not np.isfinite(self.transient_sigma) or self.transient_sigma < 5:
            raise ValueError('transient_sigma must be finite and >=5')
        if not 0 < self.max_unusable_fraction < 1:
            raise ValueError('max_unusable_fraction must lie between 0 and 1')


def mad(x):
    x=np.asarray(x);x=x[np.isfinite(x)]
    return float(1.482602218505602*np.median(abs(x-np.median(x)))) if len(x) else np.nan


def differences(t, y, ii, dt):
    """Use disjoint sample pairs; never bridge gaps or a removed observation."""
    a, c = ii[::2][:len(ii)//2], ii[1::2]
    good=(c-a==1)&(t[c]-t[a]<1.5*dt)
    return (y[c[good]]-y[a[good]])/np.sqrt(2)


def groups(mask, t, dt):
    ii=np.flatnonzero(mask)
    if not len(ii):return []
    return np.split(ii,np.flatnonzero((np.diff(ii)>1)|(np.diff(t[ii])>=1.5*dt))+1)


WINDOW_COLUMNS=['segment','start','end','n','pairs','scatter','estimated_noise','usable_fraction','residual_correlation','residual_to_diff_scatter']
FEATURE_COLUMNS=['kind','segment','start','end','peak_time','magnitude','detail','in_search_region','screening_pvalue']


def inspect(lc, prepared, detector_config, config=PrecheckConfig()):
    """Inspect raw flux and the ordinary prepared background without mutating either."""
    t=lc.time;raw=prepared.normalized_flux;y=prepared.residual-prepared.local_baseline
    sigma=prepared.segment_sigma;dt=b.cadence_days(t)
    usable=((prepared.segment_id>=0)&~prepared.detrend_edge_mask&~prepared.background_uncertain_mask
            &np.isfinite(sigma)&(sigma>0))
    raw_sigma=np.divide(sigma,prepared.detrend_error_scale,out=np.full(len(t),np.nan),where=prepared.detrend_error_scale>0)
    z=np.divide(y,sigma,out=np.zeros(len(t)),where=np.isfinite(sigma)&(sigma>0))
    windows=[];features=[];segments=[]
    segment_indices=b.split_segments(t,detector_config.gap_factor)
    # Conservative upper bound on adjacent-window and end/middle comparisons.
    # Brown-Forsythe assumes independent differences; correlation can invalidate
    # its nominal probability, so this remains a diagnostic screening criterion.
    noise_tests_bound=len(t)//(2*config.min_noise_pairs)+2*len(segment_indices)+1
    p_cut=config.noise_family_alpha/noise_tests_bound
    def noise_pvalue(a,c):
        if min(len(a),len(c))<config.min_noise_pairs or min(mad(a),mad(c))<=0:return 1.
        return float(levene(a,c,center='median').pvalue)
    def add(kind,segment,start,end,peak,magnitude,detail,pvalue=None):
        inside=(t>=start)&(t<=end)
        search=bool(usable[np.argmin(abs(t-peak))]) if kind!='noise_change' else bool(np.any(usable[inside]))
        features.append(dict(kind=kind,segment=int(segment),start=float(start),end=float(end),
            peak_time=float(peak),magnitude=float(magnitude),detail=detail,in_search_region=search,screening_pvalue=pvalue))
    for sn,ii in enumerate(segment_indices):
        if len(ii)<5:continue
        tt=t[ii];local_dt=b.cadence_days(tt)
        # Allow headroom for quality-filtered or missing exposures while still
        # requiring the declared number of genuinely adjacent sample pairs.
        width=max(config.noise_window_hours/24,3*config.min_noise_pairs*local_dt)
        edges=np.arange(tt[0],tt[-1]+width,width)
        if len(edges)<2:edges=np.array([tt[0],tt[-1]+local_dt])
        segment_windows=[];segment_diffs=[]
        for lo,hi in zip(edges[:-1],edges[1:]):
            jj=ii[(tt>=lo)&(tt<hi)]
            if not len(jj):continue
            dif=differences(t,raw,jj,local_dt)
            scatter=mad(dif) if len(dif)>=config.min_noise_pairs else np.nan
            quiet=jj[~prepared.protection_mask[jj]] if prepared.protection_mask is not None else jj
            paired=(np.diff(quiet)==1)&(np.diff(t[quiet])<1.5*local_dt)
            left,right=y[quiet[:-1]][paired],y[quiet[1:]][paired]
            corr=float(np.corrcoef(left,right)[0,1]) if len(left)>=24 and min(np.std(left),np.std(right))>0 else np.nan
            ds=mad(differences(t,y,quiet,local_dt))
            ratio=mad(y[quiet])/ds if ds>0 else np.nan
            row=dict(segment=sn,start=float(t[jj[0]]),end=float(t[jj[-1]]),n=len(jj),pairs=len(dif),
                scatter=scatter,estimated_noise=float(np.nanmedian(raw_sigma[jj])),
                usable_fraction=float(usable[jj].mean()),residual_correlation=corr,residual_to_diff_scatter=ratio)
            windows.append(row);segment_windows.append(row);segment_diffs.append(dif)
        # Two independent windows on each side must support a sustained change.
        for k in range(2,len(segment_windows)-1):
            four=segment_windows[k-2:k+2]
            a=np.array([v['scatter'] for v in four[:2]]);c=np.array([v['scatter'] for v in four[2:]])
            if not np.isfinite(np.r_[a,c]).all() or np.min(np.r_[a,c])<=0:continue
            ratio=float(np.median(c)/np.median(a));increase=ratio>=1
            high,low=(c,a) if increase else (a,c)
            if max(ratio,1/ratio)>=config.noise_ratio and min(high)>1.15*max(low):
                pvalue=noise_pvalue(np.concatenate(segment_diffs[k-2:k]),np.concatenate(segment_diffs[k:k+2]))
                if pvalue<=p_cut:
                    add('noise_change',sn,four[0]['start'],four[-1]['end'],four[2]['start'],max(ratio,1/ratio),
                        'sustained_increase' if increase else 'sustained_decrease',pvalue)
        # Explicit end-region comparison also catches gradual changes across the segment.
        phase=(tt-tt[0])/max(np.ptp(tt),local_dt)
        middle=ii[(phase>=.35)&(phase<.60)&usable[ii]];end=ii[(phase>=.75)&(phase<.94)&usable[ii]]
        md,ed=differences(t,raw,middle,local_dt),differences(t,raw,end,local_dt)
        m,en=mad(md),mad(ed)
        ratio=en/m if m>0 and min(len(md),len(ed))>=2*config.min_noise_pairs else np.nan
        if np.isfinite(ratio) and max(ratio,1/ratio)>=config.noise_ratio and noise_pvalue(md,ed)<=p_cut:
            add('noise_change',sn,t[end[0]],t[end[-1]],t[end[len(end)//2]],max(ratio,1/ratio),
                'end_region_increase' if ratio>=1 else 'end_region_decrease',noise_pvalue(md,ed))
        segments.append(dict(segment=sn,start=float(tt[0]),end=float(tt[-1]),n=len(ii),
                             noise_windows=sum(np.isfinite(v['scatter']) for v in segment_windows),end_to_middle_noise_ratio=ratio))
        for jj in groups((np.isin(np.arange(len(t)),ii))&(z<=-config.transient_sigma),t,local_dt):
            if len(jj)>=2:
                k=jj[np.argmin(z[jj])];add('negative_excursion',sn,t[jj[0]],t[jj[-1]],t[k],-z[k],'may_be_astrophysical_or_instrumental')
        # Isolated spikes may also be unresolved genuine flares; this is advisory.
        for j in np.flatnonzero(abs(z[ii])>=config.transient_sigma):
            if j<1 or j>=len(ii)-1:continue
            k=ii[j];neighbors=ii[[j-1,j+1]]
            if np.all(abs(t[neighbors]-t[k])<1.5*local_dt) and np.all(abs(z[neighbors])<max(3.,.15*abs(z[k]))):
                add('isolated_spike',sn,t[neighbors[0]],t[neighbors[1]],t[k],abs(z[k]),'single_exposure;genuine_short_flare_possible')
        # Raw-flux jump with stable plateaus on both sides; do not compare across a gap.
        jumps=np.flatnonzero(abs(np.diff(raw[ii]))>6*np.maximum(raw_sigma[ii[:-1]],raw_sigma[ii[1:]]))+1
        for j in jumps:
            if j<10 or j+10>len(ii):continue
            jj=ii[j-10:j+10]
            if np.any(np.diff(t[jj])>=1.5*local_dt):continue
            a,c=raw[ii[j-10:j]],raw[ii[j:j+10]];noise=float(np.nanmedian(raw_sigma[jj]))
            jump=float(np.median(c)-np.median(a))
            if noise>0 and abs(jump)>config.transient_sigma*noise and max(abs(np.median(a[:5])-np.median(a[5:])),abs(np.median(c[:5])-np.median(c[5:])))<2*noise:
                add('flux_step',sn,t[jj[0]],t[jj[-1]],t[ii[j]],abs(jump)/noise,'stable_raw_flux_levels_across_jump')
        # Cheap symmetry screen. Detailed competing-model evidence remains the detector's job.
        peaks,_=find_peaks(z[ii],height=config.transient_sigma,prominence=config.transient_sigma/2,distance=3)
        for j in peaks:
            half=z[ii[j]]/2;l=j;r=j
            while l>0 and z[ii[l-1]]>half and t[ii[l]]-t[ii[l-1]]<1.5*local_dt:l-=1
            while r<len(ii)-1 and z[ii[r+1]]>half and t[ii[r+1]]-t[ii[r]]<1.5*local_dt:r+=1
            rise=t[ii[j]]-t[ii[l]];fall=t[ii[r]]-t[ii[j]]
            if min(rise,fall)<2*local_dt or not .65<rise/fall<1.55:continue
            span=min(rise,fall);x=np.linspace(0,span,16)
            a=np.interp(t[ii[j]]-x,t[ii],y[ii]);c=np.interp(t[ii[j]]+x,t[ii],y[ii])
            similarity=float(np.corrcoef(a,c)[0,1]) if min(np.std(a),np.std(c))>0 else 0.
            if similarity>.8:
                add('symmetric_bump',sn,t[ii[l]],t[ii[r]],t[ii[j]],z[ii[j]],'symmetric_resolved_peak;physical_origin_unconfirmed')
        for k in range(1,len(segment_windows)):
            a,c=segment_windows[k-1:k+1]
            if all(v['residual_correlation']>.7 and v['residual_to_diff_scatter']>2.2 for v in (a,c)):
                add('structured_background',sn,a['start'],c['end'],(a['end']+c['start'])/2,
                    max(a['residual_to_diff_scatter'],c['residual_to_diff_scatter']),'persistent_correlated_residuals_after_detrending')
    # Merge overlapping warnings of the same kind within a segment.
    merged=[]
    for f in sorted(features,key=lambda v:(v['kind'],v['segment'],v['start'])):
        if merged and (merged[-1]['kind'],merged[-1]['segment'])==(f['kind'],f['segment']) and f['start']<=merged[-1]['end']:
            old=merged[-1];old['end']=max(old['end'],f['end']);old['magnitude']=max(old['magnitude'],f['magnitude'])
            old['in_search_region']|=f['in_search_region'];old['detail']=';'.join(sorted(set(old['detail'].split(';')+f['detail'].split(';'))))
            if f['screening_pvalue'] is not None:old['screening_pvalue']=min(old['screening_pvalue'],f['screening_pvalue'])
        else:merged.append(f.copy())
    active=[f for f in merged if f['in_search_region']]
    flags=[]
    if any(f['kind']=='noise_change' for f in active):flags.append('noise_change')
    if any(f['kind'] in ('isolated_spike','negative_excursion','symmetric_bump','flux_step') for f in active):flags.append('transient_features')
    if any(f['kind']=='structured_background' for f in active):flags.append('structured_background')
    retained=len(t)/lc.n_input if lc.n_input else None
    if 1-usable.mean()>config.max_unusable_fraction or (retained is not None and retained<.5):flags.append('limited_coverage')
    dif=np.diff(raw)
    constant_fraction=float(np.mean(dif==0)) if len(dif) else 1.
    if constant_fraction>.2:flags.append('repeated_or_constant_flux')
    assessed=sum(np.isfinite(v['scatter']) for v in windows)
    noise_assessed=any(v['noise_windows']>=4 for v in segments)
    if not noise_assessed:flags.append('insufficient_noise_baseline')
    ratios=[f['magnitude'] for f in active if f['kind']=='noise_change']
    summary=dict(version=VERSION,gaia_dr3_id=str(lc.gdr3_id),sector=lc.sector,source_path=lc.source_path,
        flux_column=lc.selected_flux_column,cadence_seconds=float(dt*86400),n_observations=len(t),
        status='review' if any(x!='insufficient_noise_baseline' for x in flags) else 'not_assessed' if flags else 'ok',
        flags=';'.join(flags),noise_assessed=noise_assessed,noise_windows=assessed,segments=len(segments),
        noise_comparison_bound=noise_tests_bound,noise_pvalue_threshold=p_cut,
        max_flagged_noise_ratio=max(ratios) if ratios else None,features_in_search_region=len(active),
        features_outside_search_region=len(merged)-len(active),usable_fraction=float(usable.mean()),
        retained_input_fraction=retained,constant_difference_fraction=constant_fraction,
        quality_filter_applied=lc.quality_filter_applied,n_quality_rejected=lc.n_quality_rejected,
        n_duplicates_merged=lc.n_duplicates_merged,settings=asdict(config),advisory_only=True,
        interpretation='Flags describe unusual data, not confirmed artifacts. No data, thresholds, flare tiers, or accepted warnings are changed. Noise-change ratios are screening rules, not calibrated significance probabilities. Pixel localization and common-mode checks are not performed.')
    return b.json_safe(summary),pd.DataFrame(windows,columns=WINDOW_COLUMNS),pd.DataFrame(merged,columns=FEATURE_COLUMNS),b.json_safe(segments)


def plot(prepared, summary, windows, features, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    t=prepared.lightcurve.time;origin=t[0];x=t-origin
    fig,axes=plt.subplots(3,1,figsize=(12,8),sharex=True)
    axes[0].plot(x,(prepared.normalized_flux-1)*1000,'.',ms=1.3,color='#506d82')
    axes[0].set_ylabel('Original relative flux (ppt)')
    y=prepared.residual-prepared.local_baseline
    axes[1].plot(x,y*1000,'.',ms=1.3,color='#60737e')
    axes[1].plot(x,3*prepared.segment_sigma*1000,color='#c55546',lw=.7,label='Local 3σ')
    axes[1].set_ylabel('Residual (ppt)');axes[1].legend(loc='upper right',fontsize=8)
    raw_sigma=prepared.segment_sigma/prepared.detrend_error_scale
    axes[2].plot(x,raw_sigma*1e6,color='#087f8c',lw=.9,label='BANSHEE noise')
    for _,g in windows.groupby('segment'):
        axes[2].plot((g.start+g.end)/2-origin,g.scatter*1e6,'o-',color='#9662b5',ms=3,lw=.7)
    axes[2].plot([],[],'o-',color='#9662b5',label='Adjacent-difference scatter')
    axes[2].set(ylabel='Noise (ppm)',xlabel=f'Days from BTJD {origin:.5f}');axes[2].legend(fontsize=8)
    for f in features.to_dict('records'):
        ax=axes[2] if f['kind']=='noise_change' else axes[1]
        ax.axvspan(f['start']-origin,f['end']-origin,color='#efa746',alpha=.15)
    for ax in axes:ax.grid(alpha=.15)
    fig.suptitle(f'Gaia DR3 {summary["gaia_dr3_id"]} · {summary["cadence_seconds"]:.0f} s · {summary["status"]}\n{summary["flags"] or "No precheck rules triggered"}',fontsize=13)
    fig.tight_layout(rect=(0,0,1,.94));fig.savefig(path,dpi=150,facecolor='white');plt.close(fig)
