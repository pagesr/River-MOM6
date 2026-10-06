#!/usr/bin/env python3
"""Compare merged forcing with legacy, including decoded timestamps and static grid.

Default paths: outputs/LEGACY/MERGED/glofas_hill_YEAR_v4.1.nc and
outputs/MERGED/glofas_hill_YEAR_v4.1.nc. Exact data comparisons by default.
File format, compression, provenance, extra hill_region, and time epoch encoding
may differ; dates, flux units, static arrays, support, and finite wet-cell runoff
must agree. Requires NumPy and netCDF4. Exit 0 pass, 1 mismatch, 2 input error.
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from netCDF4 import Dataset, num2date


def read(var,key=slice(None)):
    a=np.ma.asarray(var[key])
    if a.dtype.kind not in 'fc': a=a.astype(np.float64)
    return np.asarray(np.ma.filled(a,np.nan))


def decoded_times(nc):
    v=nc['time']; a=read(v)
    if not np.isfinite(a).all(): raise ValueError('Invalid time values')
    return [str(d) for d in num2date(a,v.units,calendar=getattr(v,'calendar','standard'))]


def compare(old_path,new_path,wet,args):
    report={'legacy':str(old_path),'new':str(new_path),'rtol':args.rtol,'atol':args.atol,
            'structural_issues':[],'first_differences':[]}
    issues=report['structural_issues']
    with Dataset(old_path) as old, Dataset(new_path) as new:
        for name in ('runoff','area','lat','lon','time','y','x'):
            if name not in old.variables or name not in new.variables:
                issues.append(f'Missing variable {name}')
        if issues: report['passed']=False; return report
        a,b=old['runoff'],new['runoff']
        if a.dimensions != ('time','y','x') or b.dimensions != a.dimensions or a.shape != b.shape:
            issues.append('Runoff shape/dimension order differs')
            report['passed']=False; return report
        if a.shape[0] == 0 or a.shape[1:] != wet.shape: raise ValueError('Empty runoff or mask shape mismatch')
        old_time,new_time=decoded_times(old),decoded_times(new)
        if old_time != new_time: issues.append('Decoded timestamps differ')
        if len(old_time) != a.shape[0] or len(new_time) != b.shape[0]:
            issues.append('Time length differs from runoff length')
        for name in ('area','lat','lon','y','x'):
            av,bv=read(old[name]),read(new[name])
            if av.shape != bv.shape or not np.isfinite(av).all() or not np.isfinite(bv).all() or not np.array_equal(av,bv):
                issues.append(f'{name}: array differs or contains nonfinite values')
        if getattr(a,'units',None) != getattr(b,'units',None): issues.append('Runoff units differ')
        report['legacy_format'],report['new_format']=old.data_model,new.data_model
        report['time_encoding_differences_ignored']={n:[getattr(old['time'],n,None),getattr(new['time'],n,None)]
              for n in ('units','calendar') if getattr(old['time'],n,None) != getattr(new['time'],n,None)}
        area_a,area_b=read(old['area']),read(new['area'])
        if any(x.shape != wet.shape or not np.isfinite(x).all() or np.any(x<=0) for x in (area_a,area_b)):
            raise ValueError('Invalid area')
        diff_count=outside=support=invalid_a=invalid_b=land_a=land_b=count=0
        max_abs=sqsum=max_flow=0.
        nz_a_all=np.zeros(wet.shape,bool); nz_b_all=nz_a_all.copy()
        for it in range(a.shape[0]):
            av,bv=read(a,it),read(b,it)
            fa,fb=np.isfinite(av),np.isfinite(bv); valid=fa & fb
            invalid_a+=int((~fa).sum()); invalid_b+=int((~fb).sum())
            na,nb=fa & (av!=0),fb & (bv!=0)
            nz_a_all |= na; nz_b_all |= nb
            land_a+=int((na & ~wet).sum()); land_b+=int((nb & ~wet).sum())
            support+=int((na != nb).sum())
            exact=valid & (av != bv)
            diff_count+=int(exact.sum())
            outside+=int((valid & ~np.isclose(bv,av,rtol=args.rtol,atol=args.atol)).sum())
            delta=bv[valid]-av[valid]
            daily=float(np.max(np.abs(delta))) if delta.size else 0.
            max_abs=max(max_abs,daily); sqsum+=float(np.sum(delta*delta)); count+=delta.size
            if fa.all() and fb.all():
                max_flow=max(max_flow,abs(float(np.sum(bv*area_b/1000)-np.sum(av*area_a/1000))))
            for j,i in np.argwhere(exact | ~valid)[:max(0,5-len(report['first_differences']))]:
                report['first_differences'].append({'time_index':it,'j':int(j),'i':int(i),
                    'legacy':float(av[j,i]) if fa[j,i] else None,'new':float(bv[j,i]) if fb[j,i] else None})
            if it==0 or (it+1)%args.log_every==0 or it+1==a.shape[0]:
                print(f'  {it+1}/{a.shape[0]}: max abs diff={daily:.6g}; differing={int(exact.sum())}')
        report.update(number_of_records=a.shape[0],runoff_exactly_equal=diff_count==invalid_a==invalid_b==0,
            differing_values=diff_count,values_outside_tolerance=outside,nonzero_location_mismatches=support,
            legacy_invalid_values=invalid_a,new_invalid_values=invalid_b,
            legacy_nonzero_values_on_land=land_a,new_nonzero_values_on_land=land_b,
            legacy_distinct_nonzero_cells=int(nz_a_all.sum()),new_distinct_nonzero_cells=int(nz_b_all.sum()),
            max_absolute_difference_kg_m2_s=max_abs,rmse_kg_m2_s=(sqsum/count)**.5 if count else None,
            max_daily_discharge_difference_m3_s=max_flow,
            passed=not issues and outside==support==invalid_a==invalid_b==land_a==land_b==0)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('legacy',type=Path,nargs='?'); p.add_argument('new',type=Path,nargs='?')
    p.add_argument('--year',type=int,default=1993); p.add_argument('--root',type=Path,default=Path('.'))
    p.add_argument('--mask',default='RAW/GRID/ocean_mask.nc'); p.add_argument('--mask-var')
    p.add_argument('--report',type=Path); p.add_argument('--rtol',type=float,default=0.)
    p.add_argument('--atol',type=float,default=0.); p.add_argument('--log-every',type=int,default=30)
    args=p.parse_args()
    if (args.legacy is None)!=(args.new is None): p.error('Provide both legacy/new paths or neither')
    if args.log_every<1 or any(not np.isfinite(x) or x<0 for x in (args.atol,args.rtol)): p.error('Invalid tolerance/log interval')
    path=lambda x:(args.root/x).resolve()
    name=f'glofas_hill_{args.year}_v4.1.nc'
    old=path(args.legacy or Path('outputs/LEGACY/MERGED')/name)
    new=path(args.new or Path('outputs/MERGED')/name)
    if old==new: p.error('Legacy and new paths must differ')
    if args.report and path(args.report) in {old,new,path(args.mask)}: p.error('Report would overwrite an input')
    with Dataset(path(args.mask)) as nc:
        n=args.mask_var or next((n for n in ('mask','wet') if n in nc.variables),None)
        if n is None: raise ValueError('No mask/wet variable')
        wet=read(nc[n])
    if wet.ndim !=2 or not np.isin(wet,[0,1]).all(): raise ValueError('Invalid binary wet mask')
    print(f'Legacy: {old}\nNew: {new}')
    report=compare(old,new,wet.astype(bool),args)
    print('\n'+('PASS' if report['passed'] else 'FAIL'))
    for k,v in report.items(): print(f'  {k}: {v}')
    if args.report:
        f=path(args.report); f.parent.mkdir(parents=True,exist_ok=True)
        f.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return 0 if report['passed'] else 1


if __name__=='__main__':
    try: sys.exit(main())
    except (OSError,ValueError,KeyError,OverflowError) as e:
        print(f'ERROR: {e}',file=sys.stderr); sys.exit(2)
