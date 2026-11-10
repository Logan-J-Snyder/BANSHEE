#!/usr/bin/env python3
"""Run advisory LC prechecks, then the unchanged calibrated BANSHEE detector.

All original BANSHEE arguments are supported. Add --precheck-only to inspect
without fitting flares, or --precheck-settings FILE for diagnostic thresholds.
Precheck products are saved alongside the detector directory: OUTPUT_precheck.
"""
from pathlib import Path
import sys,json,time,hashlib,fcntl,shutil
from dataclasses import fields,asdict
import BANSHEE_Final as b
import BANSHEE_Precheck as q
import pandas as pd


def config_from_args(args):
    values=vars(args)
    selected={f.name:values[f.name] for f in fields(b.Config) if f.name in values}
    selected['flux_is_residual']=args.residual
    return b.Config(**selected)


def flat(summary, record, folder):
    row={k:v for k,v in summary.items() if not isinstance(v,(list,dict)) and k not in ('interpretation',)}
    row.update(sample={'field_stars':'FS','open_clusters':'OC'}.get(record['sample'],record['sample']),
               source_path=record['path'],precheck_report=str(folder/'precheck.json'))
    return row


def cached(record,root,run_id):
    folder=b.product_folder(record,root)
    try:
        receipt=json.loads((folder/'complete.json').read_text())
        return receipt['run_id']==run_id and receipt['record']==record and all(
            (folder/name).is_file() and b.file_sha256(folder/name)==sha for name,sha in receipt['outputs'].items())
    except (OSError,ValueError,KeyError):return False


def precheck_product(record,root,config,settings,plots,run_id):
    folder=b.product_folder(record,root);folder.mkdir(parents=True,exist_ok=True)
    (folder/'complete.json').unlink(missing_ok=True);started=time.perf_counter()
    try:
        if b.fingerprint(record['path'])!=record['input']:raise ValueError('Input changed after inventory')
        lc=b.load_lightcurve(Path(record['path']),config)
        prepared,_=b.prepare(lc,config)
        summary,windows,features,segments=q.inspect(lc,prepared,config,settings)
        summary.update(source_sha256=b.file_sha256(record['path']),precheck_seconds=time.perf_counter()-started)
        b.atomic_json(folder/'precheck.json',dict(summary=summary,segments=segments))
        names=['precheck.json']
        for df,name in ((windows,'noise_windows'),(features,'features')):
            for suffix in ('.csv','.parquet'):
                filename=name+suffix;b.atomic_frame(df,folder/filename);names.append(filename)
        if plots:
            q.plot(prepared,summary,windows,features,folder/'precheck.png');names.append('precheck.png')
        if b.fingerprint(record['path'])!=record['input']:raise ValueError('Input changed during precheck')
        b.atomic_json(folder/'complete.json',dict(run_id=run_id,record=record,
            outputs={name:b.file_sha256(folder/name) for name in names}))
    except Exception as exc:
        summary=dict(status='failed',flags='precheck_failed',error=repr(exc),advisory_only=True,
                     gaia_dr3_id=str(b.metadata_from_path(Path(record['path']))[0]),source_path=record['path'])
        b.atomic_json(folder/'precheck.json',dict(summary=summary))
    return flat(summary,record,folder)


def run_prechecks(args,config,settings):
    records=b.discover_inputs(args);output=args.output.expanduser().resolve()
    root=output.with_name(output.name+'_precheck')
    for path in ([args.input_root] if args.server else args.input):
        path=path.expanduser().resolve()
        for target in (root,output):
            if target==path or target in path.parents or (path.is_dir() and path in target.parents):
                raise ValueError('Inputs and both output folders must be separate and non-nested')
    root.mkdir(parents=True,exist_ok=True)
    hashes={Path(p).name:b.file_sha256(p) for p in (b.__file__,q.__file__,__file__)}
    manifest=b.json_safe(dict(precheck_version=q.VERSION,source_hashes=hashes,inputs=records,
        detector_preparation_config=asdict(config),precheck_settings=asdict(settings),plots=args.plots,
        semantics='Advisory stage only. Does not mask observations, change calibration, or route flare tiers.'))
    run_id=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
    with (root/'.precheck.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        marker=root/'manifest.json'
        if marker.exists():
            if json.loads(marker.read_text())!=manifest:raise ValueError('Precheck code, settings or inputs changed; choose a new output directory')
        else:
            if any(p.name!='.precheck.lock' for p in root.iterdir()):raise ValueError('Precheck folder is nonempty without a matching manifest')
            b.atomic_json(marker,manifest)
            for filename in hashes:shutil.copy2(Path(__file__).parent/filename,root/(Path(filename).stem+'_snapshot.py'))
        missing=[r for r in records if not cached(r,root,run_id)]
        print(f'Precheck: {len(records)} inputs; {len(records)-len(missing)} cached; {len(missing)} to inspect.',flush=True)
        with (root/'run_log.jsonl').open('a',buffering=1) as log:
            for row in b.bounded_jobs(precheck_product,[(r,root,config,settings,args.plots,run_id) for r in missing],args.workers,'Precheck'):
                log.write(json.dumps(b.json_safe(row))+'\n')
        rows=[]
        for record in records:
            folder=b.product_folder(record,root)
            if cached(record,root,run_id):
                summary=json.loads((folder/'precheck.json').read_text())['summary']
            else:
                summary=dict(status='failed',flags='precheck_failed',gaia_dr3_id=str(b.metadata_from_path(Path(record['path']))[0]),advisory_only=True)
                try:
                    previous=json.loads((folder/'precheck.json').read_text())['summary']
                    summary.update(error=previous.get('error','Missing or changed precheck output'),gaia_dr3_id=previous.get('gaia_dr3_id',summary['gaia_dr3_id']))
                except (OSError,ValueError,KeyError):summary['error']='Missing or unreadable precheck output'
            rows.append(flat(summary,record,folder))
        table=pd.DataFrame(rows)
        table['gaia_dr3_id']=table.gaia_dr3_id.astype('string')
        for suffix in ('.csv','.parquet'):b.atomic_frame(table,root/('lightcurve_prechecks'+suffix))
        for filename,sha in hashes.items():
            if b.file_sha256(Path(__file__).parent/filename)!=sha:raise RuntimeError('Code changed during precheck')
        if any(b.fingerprint(r['path'])!=r['input'] for r in records):raise RuntimeError('Input changed during precheck')
        summary=dict(inputs=len(records),counts=table.status.value_counts().to_dict(),precheck_output=str(root),
                     new_prechecks=len(missing),advisory_only=True,detector_code_sha256=hashes[Path(b.__file__).name])
        b.atomic_json(root/'run_summary.json',summary)
        print(json.dumps(summary,indent=2),flush=True)
    return summary


def main():
    parser=b.make_parser();parser.description=__doc__
    parser.add_argument('--precheck-only',action='store_true',help='Write LC diagnostics without running flare fits')
    parser.add_argument('--precheck-settings',type=Path,help='JSON object overriding PrecheckConfig screening thresholds')
    args=parser.parse_args()
    if args.server==bool(args.input):parser.error('Choose either --server or --input')
    if args.workers<1 or args.limit<0:parser.error('workers must be positive; limit nonnegative')
    if args.residual and not args.already_detrended:parser.error('--residual requires --already-detrended')
    if args.calibrate or args.auto_calibrate:parser.error('Run calibration with the original BANSHEE_Final.py entrypoint')
    settings=q.PrecheckConfig(**(json.loads(args.precheck_settings.read_text()) if args.precheck_settings else {}))
    summary=run_prechecks(args,config_from_args(args),settings)
    if args.precheck_only:
        if summary['counts'].get('failed',0):raise SystemExit('Some prechecks failed; inspect the summary and rerun to retry')
        return
    if summary['counts'].get('failed',0):
        print('Some prechecks failed and are explicitly marked. The detector will still attempt every input.',flush=True)
    # Delegate the full flare analysis to the exact original entrypoint.
    forwarded=[];skip=False
    for arg in sys.argv[1:]:
        if skip:skip=False;continue
        if arg=='--precheck-only':continue
        if arg=='--precheck-settings':skip=True;continue
        if arg.startswith('--precheck-settings='):continue
        forwarded.append(arg)
    sys.argv=[b.__file__,*forwarded]
    b.main()


if __name__=='__main__':main()
