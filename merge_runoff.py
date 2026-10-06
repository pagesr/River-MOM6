#!/usr/bin/env python3
"""Merge daily Hill and GloFAS MOM6 forcing, with Hill priority in a fixed region.

Example reproducing the spatial/time selection in merge_runoff_remiv2.py:
  python merge_runoff.py 1993 --hill-box 156 267 416 533
Another grid: --hill-region-mask RAW/GRID/hill_coverage.nc --region-var hill_region
Use an explicit geographic coverage mask, not each day's nonzero Hill values.
Hill zero discharge remains zero inside its coverage; GloFAS is used outside.
Requires NumPy and netCDF4 only. See MERGE.md for time and area conventions.
"""
import argparse
from contextlib import ExitStack
import datetime as dt
import glob
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
from netCDF4 import Dataset, num2date


def read(var, key=slice(None)):
    values = np.ma.asarray(var[key])
    if values.dtype.kind not in 'fc':
        values = values.astype(np.float64)
    return np.asarray(np.ma.filled(values, np.nan))


def dates(nc):
    v = nc.variables['time']
    calendar = getattr(v, 'calendar', 'standard')
    if calendar not in ('standard', 'gregorian', 'proleptic_gregorian'):
        raise ValueError(f'Unsupported calendar {calendar!r} in {nc.filepath()}')
    raw = read(v)
    if raw.ndim != 1 or not np.isfinite(raw).all() or not (np.diff(raw) > 0).all():
        raise ValueError(f'Times must be finite, unique, and increasing: {nc.filepath()}')
    decoded = num2date(raw, v.units, calendar=calendar)
    keys = [dt.date(d.year, d.month, d.day) for d in decoded]
    if len(keys) != len(set(keys)):
        raise ValueError(f'More than one record on a calendar date: {nc.filepath()}')
    return raw, keys


def load_grid(hgrid, mask_path, mask_var):
    with Dataset(hgrid) as nc:
        lon = read(nc['x'])[1::2, 1::2]
        lat = read(nc['y'])[1::2, 1::2]
        a, dx, dy = read(nc['area']), read(nc['dx']), read(nc['dy'])
        glofas_area = (a[::2, ::2]+a[1::2, 1::2])+(a[1::2, ::2]+a[::2, 1::2])
        hill_area = (dx[1::2, ::2]+dx[1::2, 1::2])*(dy[::2, 1::2]+dy[1::2, 1::2])
    with Dataset(mask_path) as nc:
        name = mask_var or next((n for n in ('mask','wet') if n in nc.variables), None)
        if name is None:
            raise ValueError('No mask/wet variable; set --mask-var')
        wet = read(nc[name])
    if lon.ndim != 2 or any(v.shape != lon.shape for v in (lat, glofas_area, hill_area, wet)):
        raise ValueError('MOM6 grid arrays and wet mask must have matching 2D shapes')
    if not all(np.isfinite(v).all() for v in (lon, lat, glofas_area, hill_area, wet)):
        raise ValueError('Nonfinite grid values')
    if not np.isin(wet, [0,1]).all() or np.any(glofas_area <= 0) or np.any(hill_area <= 0):
        raise ValueError('Expected binary wet mask and positive cell areas')
    return lon, lat, glofas_area, hill_area, wet.astype(bool)


def load_region(args, shape, path):
    if args.hill_box:
        j0,j1,i0,i1 = args.hill_box
        if not (0 <= j0 < j1 <= shape[0] and 0 <= i0 < i1 <= shape[1]):
            raise ValueError(f'Invalid Hill box {args.hill_box} for grid {shape}; upper bounds are exclusive')
        region = np.zeros(shape, dtype=bool)
        region[j0:j1, i0:i1] = True
        description = f'index box j={j0}:{j1}, i={i0}:{i1}'
    else:
        with Dataset(path(args.hill_region_mask)) as nc:
            v = nc[args.region_var]
            if v.dimensions not in (('y','x'), ('j','i')):
                raise ValueError('Hill region mask dimensions must be (y,x) or (j,i)')
            a = read(v)
        if a.shape != shape or not np.isin(a, [0,1]).all():
            raise ValueError('Hill region mask must match the tracer grid and contain only 0/1')
        region = a.astype(bool)
        description = f'{path(args.hill_region_mask)} variable {args.region_var}'
    if not region.any():
        raise ValueError('Hill coverage region is empty')
    return region, description


def hill_index(files, needed, shape):
    index = {}
    for file in files:
        with Dataset(file) as nc:
            _, days = dates(nc)
            relevant = [(it, day) for it,day in enumerate(days) if day in needed]
            if not relevant:
                continue
            v = nc['Runoff']
            if v.dimensions not in (('time','j','i'),('time','y','x')) or v.shape != (len(days),*shape):
                raise ValueError(f'Hill Runoff shape/dimension order does not match: {file}')
            if getattr(v, 'units', '') not in ('kg/m^2/sec','kg m-2 s-1','kg m^-2 s^-1'):
                raise ValueError(f'Unexpected Hill flux units: {file}')
            for it,day in relevant:
                if day in index:
                    raise ValueError(f'Overlapping Hill files provide duplicate date {day}')
                index[day] = (file,it)
    missing = sorted(needed-index.keys())
    if missing:
        raise ValueError(f'Missing Hill forcing for {len(missing)} dates; first={missing[0]}, last={missing[-1]}. '
                         'A calendar year normally needs Hill water-year files ending in YEAR and YEAR+1.')
    return index


def select_year(nc, year, mode):
    raw, days = dates(nc)
    start = dt.date(year,1,1)
    end = dt.date(year+1,1,1) if mode == 'legacy' else dt.date(year,12,31)
    take = [it for it,day in enumerate(days) if start <= day <= end]
    expected = [start+dt.timedelta(days=i) for i in range((end-start).days+1)]
    selected = [days[i] for i in take]
    if selected != expected:
        missing = sorted(set(expected)-set(selected))
        raise ValueError(f'GloFAS does not cover the full {mode} merge interval for {year}; '
                         f'missing dates: {missing[:5]}')
    return take, selected, raw


def check_glofas_grid(nc, grid):
    lon,lat,area,_,_ = grid
    if nc['runoff'].dimensions != ('time','y','x') or nc['runoff'].shape[1:] != lon.shape:
        raise ValueError('GloFAS runoff must be (time,y,x) on the selected MOM6 grid')
    if getattr(nc['runoff'], 'units', '') != 'kg m-2 s-1':
        raise ValueError('Unexpected GloFAS flux units')
    for name, expected in (('lon',lon), ('lat',lat), ('area',area)):
        a = read(nc[name])
        if a.shape != expected.shape or not np.isfinite(a).all():
            raise ValueError(f'Invalid GloFAS {name}')
        difference = a-expected
        if name == 'lon':
            difference = (difference+180) % 360-180
        if name == 'area':
            matches = np.array_equal(a,expected)
        else:
            matches = np.all(np.abs(difference) <= 1e-8)
        if not matches:
            raise ValueError(f'GloFAS {name} differs from --hgrid; use the grid that produced both inputs')


def copy_structure(src, dst, take, compress):
    dst.setncatts({n:src.getncattr(n) for n in src.ncattrs()})
    for name,dim in src.dimensions.items():
        dst.createDimension(name, None if name == 'time' else len(dim))
    for name,v in src.variables.items():
        if 'time' in v.dimensions and name not in ('runoff','time'):
            raise ValueError(f'Unsupported time-dependent auxiliary variable {name}')
        kw = dict(fill_value=v.getncattr('_FillValue') if '_FillValue' in v.ncattrs() else False)
        if compress and v.ndim >= 2:
            kw.update(zlib=True,complevel=1)
        out = dst.createVariable(name, v.dtype, v.dimensions, **kw)
        out.setncatts({n:v.getncattr(n) for n in v.ncattrs() if n != '_FillValue'})
        if name == 'time': out[:] = v[take]
        elif name != 'runoff': out[:] = v[:]


def merge_year(nc, output, take, days, index, region, description, grid, args):
    output.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp = tempfile.mkstemp(dir=output.parent,prefix=output.stem+'.',suffix='.tmp.nc')
    os.close(fd)
    _,_,_,hill_area,wet = grid
    union_outside = np.zeros(region.shape,dtype=bool)
    max_time_shift = 0.
    try:
        with ExitStack() as stack:
            hill = {file:stack.enter_context(Dataset(file)) for file in sorted({index[d][0] for d in days})}
            dst = stack.enter_context(Dataset(tmp,'w',format='NETCDF3_64BIT_OFFSET' if args.no_compress else 'NETCDF4'))
            copy_structure(nc,dst,take,not args.no_compress)
            dst.merge_method = 'Hill inside static hill_region, GloFAS outside; no addition or zero-value fallback'
            dst.hill_region_description = description
            dst.hill_time_alignment = 'Calendar date; original GloFAS timestamps retained'
            dst.hill_source_files = '\n'.join(str(p) for p in sorted(hill))
            dst.area_convention = 'Flux copied unchanged; embedded area retained from GloFAS'
            rv = dst.createVariable('hill_region','i1',('y','x'),fill_value=False,
                                    **(dict(zlib=True,complevel=1) if not args.no_compress else {}))
            rv[:] = region.astype('i1')
            rv.long_name,rv.units = 'Static source choice: 1 Hill, 0 GloFAS','1'
            for it,(source_it,day) in enumerate(zip(take,days)):
                g = read(nc['runoff'],source_it)
                file,hill_it = index[day]
                h = read(hill[file]['Runoff'],hill_it)
                if not np.isfinite(g).all() or not np.isfinite(h).all():
                    raise ValueError(f'Missing/nonfinite input runoff on {day}; missing data is not zero discharge')
                if np.any(g[~wet] != 0) or np.any(h[~wet] != 0):
                    raise ValueError(f'Input has runoff on land on {day}; check matching grid/mask')
                merged = g.copy()
                merged[region] = h[region]
                dst['runoff'][it] = merged
                union_outside |= (h != 0) & ~region
                # Native time epochs may differ; compare only fractional-day clocks here.
                t = nc['time']; ht=hill[file]['time']
                gd=num2date(t[source_it],t.units,calendar=getattr(t,'calendar','standard'))
                hd=num2date(ht[hill_it],ht.units,calendar=getattr(ht,'calendar','standard'))
                shift=abs((gd.hour-hd.hour)*3600+(gd.minute-hd.minute)*60+gd.second-hd.second)
                max_time_shift=max(max_time_shift,shift)
                if it == 0 or (it+1) % args.log_every == 0 or it+1 == len(days):
                    area=read(nc['area']) if it == 0 else area
                    removed=float(np.sum(g[region]*area[region]/1000))
                    added=float(np.sum(h[region]*area[region]/1000))
                    native=float(np.sum(h[region]*hill_area[region]/1000))
                    print(f'  {it+1}/{len(days)} {day}: removed GloFAS={removed:.12g}, '
                          f'added Hill on output area={added:.12g}, '
                          f'Hill on native area={native:.12g} m3/s',flush=True)
            dst.hill_nonzero_cells_outside_region = int(union_outside.sum())
            dst.max_source_clock_difference_seconds = max_time_shift
            print(f'Calendar-date alignment: max source clock difference {max_time_shift:g} s')
            if union_outside.any():
                points=np.argwhere(union_outside)
                print(f'WARNING: {len(points)} Hill cells with nonzero runoff lie outside the region; '
                      f'j range {points[:,0].min()}:{points[:,0].max()+1}, '
                      f'i range {points[:,1].min()}:{points[:,1].max()+1}; '
                      'GloFAS is retained there. Review or expand Hill coverage.')
                if args.require_all_hill:
                    raise ValueError('--require-all-hill: selected region excludes nonzero Hill cells')
        os.replace(tmp,output)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)
    print(f'Wrote {output}')


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('years',type=int,nargs='+')
    p.add_argument('--root',type=Path,default=Path('.'))
    p.add_argument('--hill-dir',default='outputs/HILL-on-MOM6')
    p.add_argument('--hill-files',nargs='+',help='Override discovery with explicit paths or quoted glob patterns')
    p.add_argument('--glofas-dir',default='outputs/GLOFAS-on-MOM6')
    p.add_argument('--output-dir',default='outputs/MERGED')
    p.add_argument('--hgrid',default='RAW/GRID/ocean_hgrid.nc')
    p.add_argument('--mask',default='RAW/GRID/ocean_mask.nc')
    p.add_argument('--mask-var')
    group=p.add_mutually_exclusive_group(required=True)
    group.add_argument('--hill-box',type=int,nargs=4,metavar=('J0','J1','I0','I1'))
    group.add_argument('--hill-region-mask',help='Binary MOM6 tracer-grid coverage mask; 1 Hill, 0 GloFAS')
    p.add_argument('--region-var',default='hill_region')
    p.add_argument('--require-all-hill',action='store_true',help='Fail instead of warning when Hill nonzero cells are excluded')
    p.add_argument('--time-window',choices=('legacy','calendar-year'),default='legacy',help='Legacy includes Jan 1 of YEAR+1')
    p.add_argument('--no-compress',action='store_true',help='NetCDF3 64-bit offset instead of compressed NetCDF4')
    p.add_argument('--overwrite',action='store_true')
    p.add_argument('--log-every',type=int,default=30)
    args=p.parse_args()
    if args.log_every < 1: p.error('--log-every must be positive')
    path=lambda s:(args.root/s).resolve()
    files = sorted({Path(f).resolve() for pattern in (args.hill_files or [str(Path(args.hill_dir)/'goa_dis*.nc')])
                    for f in glob.glob(str(path(pattern)))})
    if not files: raise ValueError('No Hill output files found; generate the required ending water years first')
    grid=load_grid(path(args.hgrid),path(args.mask),args.mask_var)
    region,description=load_region(args,grid[0].shape,path)
    print(f'Hill coverage: {description}; {region.sum()} cells; zeros remain Hill zeros')
    jobs=[]; needed=set()
    for year in dict.fromkeys(args.years):
        source=path(args.glofas_dir)/f'glofas_river_{year}_v4.nc'
        output=path(args.output_dir)/f'glofas_hill_{year}_v4.1.nc'
        if output.exists() and not args.overwrite: raise ValueError(f'Output exists: {output}; use --overwrite')
        inputs=set(files)|{source,path(args.hgrid),path(args.mask)}
        if args.hill_region_mask: inputs.add(path(args.hill_region_mask))
        if output in inputs: raise ValueError('Output path would overwrite an input')
        with Dataset(source) as nc:
            check_glofas_grid(nc,grid)
            take,days,_=select_year(nc,year,args.time_window)
        needed.update(days); jobs.append((source,output,take,days))
    index=hill_index(files,needed,grid[0].shape)
    for source,output,take,days in jobs:
        print(f'Processing {days[0].year}: {len(days)} dates, {days[0]} to {days[-1]}')
        with Dataset(source) as nc: merge_year(nc,output,take,days,index,region,description,grid,args)


if __name__ == '__main__':
    try: main()
    except (OSError,ValueError,KeyError,OverflowError) as e:
        print(f'ERROR: {e}',file=sys.stderr); sys.exit(2)
