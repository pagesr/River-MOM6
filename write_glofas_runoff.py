#!/usr/bin/env python3
"""GloFAS v4 dis24 -> MOM6, based on write_runoff_glofas_dis_batch_v4.py.

Keep the original stencil, seam correction, conservative footprint, nearest
coastal LocStream mapping, native-precision area, and source accumulation order.
Only paths, grid dimensions, input checks, and daily output handling change.
Hill and merging are independent steps. See --help and GLOFAS.md.
"""
import argparse
import os
from pathlib import Path
import sys
import tempfile
import time

import numpy as np
import xarray as xr
from netCDF4 import Dataset, date2num

TIME_UNITS = 'days since 1950-01-01'


def update_stencil_sum(ocean_mask):
    """Original 3x3 stencil; keep the outer boundary zero."""
    stencil = np.zeros(ocean_mask.shape)
    for jj in range(3):
        for ii in range(3):
            stencil[1:-1, 1:-1] += ocean_mask[jj:-(2-jj) or None,
                                                        ii:-(2-ii) or None]
    return stencil


def get_glofas_pour_points(ldd, glofas_lat, glofas_lon, seam_fix=True):
    ldd_modified = ldd.copy()
    if seam_fix:
        j = np.flatnonzero(np.round(glofas_lat, 3) == 68.875)
        i = np.flatnonzero(np.round(glofas_lon, 3) == 180.025)
        if len(j) > 1 or len(i) > 1:
            raise ValueError('Ambiguous coordinates for the legacy Chukotka seam fix')
        if len(j) and len(i):
            ldd_modified[j[0], i[0]] = 1
            print('Applied legacy Chukotka seam fix at 68.875N, 180.025E')
        else:
            print('Chukotka seam coordinate is outside the source subset; fix skipped')
    ocean = np.isnan(ldd)  # Original LDD, before editing the seam.
    while True:
        stencil = update_stencil_sum(ocean)
        updates = (ldd_modified == 5) & ~ocean & (stencil > 0)
        if not updates.any():
            break
        # The legacy loop uses one fixed stencil per pass: this is equivalent.
        ocean[updates] = True
    return (ldd_modified == 5) & (stencil > 0)


def load_mom_grid(hgrid_path, mask_path, mask_var=None):
    with xr.open_dataset(hgrid_path) as hgrid:
        lon = hgrid.x.values[1::2, 1::2]
        lat = hgrid.y.values[1::2, 1::2]
        lonb = hgrid.x.values[::2, ::2]
        latb = hgrid.y.values[::2, ::2]
        # Native dtype and parentheses must match the original script.
        area = ((hgrid.area[::2, ::2] + hgrid.area[1::2, 1::2]) +
                (hgrid.area[1::2, ::2] + hgrid.area[::2, 1::2])).values
    with xr.open_dataset(mask_path) as ds:
        name = mask_var or next((n for n in ('mask', 'wet') if n in ds), None)
        if name is None or name not in ds:
            raise ValueError('No mask/wet variable found; specify --mask-var')
        mask = ds[name].values
    if lon.ndim != 2 or min(lon.shape) < 3 or any(
            a.shape != lon.shape for a in (lat, area, mask)):
        raise ValueError('Tracer coordinates, area, and mask need matching 2D shapes, at least 3x3')
    if lonb.shape != tuple(n+1 for n in lon.shape) or latb.shape != lonb.shape:
        raise ValueError('MOM6 corners must have shape (ny+1, nx+1)')
    if not all(np.isfinite(a).all() for a in (lon, lat, lonb, latb, area, mask)):
        raise ValueError('Grid contains missing/nonfinite values')
    if not np.isin(mask, [0, 1]).all() or np.any(area <= 0):
        raise ValueError('Mask must be binary (1=ocean), and areas positive')
    ocean = mask.astype(bool)
    neighbors = np.zeros(ocean.shape, dtype=bool)
    neighbors[1:-1, 1:-1] = (~ocean[2:, 1:-1] | ~ocean[:-2, 1:-1] |
                            ~ocean[1:-1, :-2] | ~ocean[1:-1, 2:])
    coast = ocean & neighbors
    if not coast.any():
        raise ValueError('No interior coastal ocean cells found')
    print(f'MOM6 grid: {lon.shape}; coastal cells: {coast.sum()}; area dtype: {area.dtype}')
    return dict(lon=lon, lat=lat, lon_b=lonb, lat_b=latb,
                area=area, mask=ocean, coast=coast)


def source_coordinates(glofas):
    lat, lon = glofas.lat.values.copy(), glofas.lon.values.copy()
    if lat.ndim != 1 or lon.ndim != 1 or min(len(lat), len(lon)) < 3:
        raise ValueError('GloFAS needs one-dimensional latitude/longitude, at least 3 each')
    if not np.isfinite(lat).all() or not np.isfinite(lon).all() or not (np.diff(lat) < 0).all():
        raise ValueError('Expected finite coordinates and descending GloFAS latitude')
    lon[lon < 0] = lon[lon < 0] + 360  # Preserve native precision and ordering.
    deg_incr = abs(np.unique(np.diff(lat))[0])
    latb = np.concatenate([[lat[0] + deg_incr/2.], .5*(lat[1:] + lat[:-1]),
                           [lat[-1] - deg_incr/2.]])
    lonb = np.concatenate([[lon[0] - deg_incr/2.], .5*(lon[1:] + lon[:-1]),
                           [lon[-1] + deg_incr/2.]])
    return dict(lat=lat, lon=lon, lat_b=latb, lon_b=lonb)


def check_ldd_coordinates(values, discharge, name, atol=1.e-6):
    """Allow coordinate rounding, keeping LDD/discharge index order intact.

    The CDO and cfgrib conversion paths can differ in their last decimal places.
    The tolerance is in degrees, far smaller than a 0.05-degree GloFAS cell.
    Equivalent longitude conventions are allowed without moving array columns.
    """
    a, b = np.asarray(values, dtype=np.float64), np.asarray(discharge, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 1 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError(f'LDD {name} coordinates have a different shape or nonfinite values: '
                         f'LDD {a.shape}, discharge {b.shape}')
    delta = a-b
    if name == 'lon':
        delta = (delta+180.) % 360. - 180.
    largest = float(np.max(np.abs(delta)))
    if largest > atol:
        index = int(np.argmax(np.abs(delta)))
        hint = ''
        if name == 'lat' and np.allclose(a[::-1], b, rtol=0, atol=atol):
            hint = ' Latitude order is reversed; reverse the LDD latitude dimension and data together.'
        raise ValueError(f'LDD {name} coordinates do not align with discharge: '
                         f'max difference={largest:.12g} degrees at index {index}; '
                         f'LDD={a[index]:.15g}, discharge={b[index]:.15g}; '
                         f'tolerance={atol:g} degrees.{hint} '
                         'Use identically ordered source subsets; coordinates are not automatically reordered.')
    if not np.array_equal(a, b):
        print(f'LDD {name}: aligned within {atol:g} degrees '
              f'(max difference={largest:.12g}; equivalent longitude conventions allowed)')


def get_mom_mask_for_glofas(grid, source, xe, filename):
    mom_to_glofas = xe.Regridder(
        {n: grid[n] for n in ('lat', 'lon', 'lat_b', 'lon_b')}, source,
        method='conservative', periodic=True, reuse_weights=False, filename=filename)
    coverage = np.asarray(mom_to_glofas(np.ones(grid['mask'].shape)))
    if not np.isfinite(coverage).all():
        raise ValueError('Conservative footprint contains nonfinite values')
    coverage[coverage < .5] = 0
    coverage[coverage > .5] = 1
    return coverage.astype(bool)  # Exactly 0.5 is included, as in the original.


def write_runoff(glofas, grid, ldd, out_file, seam_fix=True, log_every=30):
    try:
        import xesmf as xe
    except ImportError as exc:
        raise ImportError('GloFAS needs xESMF/ESMPy; use environment-glofas.yml or your legacy environment') from exc
    source = source_coordinates(glofas)
    if ldd.shape != (len(source['lat']), len(source['lon'])):
        raise ValueError('LDD and GloFAS latitude/longitude shapes differ')
    pour = get_glofas_pour_points(ldd, source['lat'], source['lon'], seam_fix)
    with tempfile.TemporaryDirectory(prefix='glofas-weights-') as weights:
        pour &= get_mom_mask_for_glofas(grid, source, xe, str(Path(weights)/'footprint.nc'))
        pour_id = np.flatnonzero(pour.ravel())
        coast_id = np.flatnonzero(grid['coast'].ravel())
        if not len(pour_id):
            raise ValueError('No GloFAS pour points inside this MOM6 footprint')
        glo_lons, glo_lats = np.meshgrid(source['lon'], source['lat'])
        coast_to_mom = xe.Regridder(
            {'lat': grid['lat'].ravel()[coast_id], 'lon': grid['lon'].ravel()[coast_id]},
            {'lat': glo_lats.ravel()[pour_id], 'lon': glo_lons.ravel()[pour_id]},
            method='nearest_s2d', locstream_in=True, locstream_out=True,
            reuse_weights=False, filename=str(Path(weights)/'nearest.nc'))
        nearest = np.asarray(coast_to_mom(coast_id)).ravel()
        if not np.isfinite(nearest).all() or not np.isin(nearest, coast_id).all():
            raise ValueError('Nearest mapping did not return valid integer coastal indices')
        nearest_coast = nearest.astype(np.int64)
    print(f'Mapping: {len(pour_id)} pour points -> {len(np.unique(nearest_coast))} coastal cells')
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=out_file.stem+'.', suffix='.tmp.nc', dir=out_file.parent)
    os.close(fd)
    try:
        ny, nx = grid['mask'].shape
        with Dataset(temporary, 'w', format='NETCDF3_64BIT_OFFSET') as nc:
            nc.createDimension('time', None)
            nc.createDimension('y', ny)
            nc.createDimension('x', nx)
            runoff = nc.createVariable('runoff', 'f8', ('time', 'y', 'x'), fill_value=False)
            runoff.units = 'kg m-2 s-1'
            for name in ('area', 'lat', 'lon'):
                v = nc.createVariable(name, grid[name].dtype, ('y', 'x'), fill_value=False)
                v[:] = grid[name]
                if name != 'area':
                    v.units = 'degrees_north' if name == 'lat' else 'degrees_east'
            tv = nc.createVariable('time', 'f8', ('time',), fill_value=False)
            tv.units, tv.calendar, tv.cartesian_axis = TIME_UNITS, 'gregorian', 'T'
            for name, count in (('y', ny), ('x', nx)):
                v = nc.createVariable(name, 'i4', (name,), fill_value=False)
                v[:] = np.arange(count)
                v.cartesian_axis = name.upper()
            dates = glofas.indexes['time']
            dates = dates.to_pydatetime() if hasattr(dates, 'to_pydatetime') else list(dates)
            tv[:] = date2num(dates, TIME_UNITS, calendar='gregorian')
            for it in range(glofas.sizes['time']):
                # dis24 is already m3/s. Load one day, accumulate in source order.
                q = glofas.isel(time=it).values.ravel()[pour_id].astype(np.float64)
                filled = np.zeros(ny*nx)
                np.add.at(filled, nearest_coast, q)
                runoff[it] = 1000*filled.reshape(ny, nx)/grid['area']
                if it == 0 or (it+1) % log_every == 0 or it+1 == len(dates):
                    print(f'  {it+1}/{len(dates)} {str(dates[it])[:10]}: '
                          f'selected={q.sum():.12g}, mapped={filled.sum():.12g} m3/s; '
                          f'nonfinite={np.count_nonzero(~np.isfinite(q))}', flush=True)
        os.replace(temporary, out_file)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    print(f'Wrote {out_file}')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('years', type=int, nargs='+', help='Calendar years; legacy mode includes one day on each side')
    parser.add_argument('--root', type=Path, default=Path('.'))
    parser.add_argument('--hgrid', default='RAW/GRID/ocean_hgrid.nc')
    parser.add_argument('--mask', default='RAW/GRID/ocean_mask.nc')
    parser.add_argument('--mask-var', help='Default: mask, falling back to wet')
    parser.add_argument('--ldd', default='RAW/GRID/ldd_glofas_v4_0_subset.nc')
    parser.add_argument('--glofas-dir', default='RAW/GLOFAS')
    parser.add_argument('--output-dir', default='outputs/GLOFAS-on-MOM6')
    parser.add_argument('--time-window', choices=('legacy', 'calendar-year'), default='legacy')
    parser.add_argument('--no-seam-fix', action='store_true', help='Disable the original Chukotka correction')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--log-every', type=int, default=30)
    args = parser.parse_args()
    if args.log_every < 1:
        parser.error('--log-every must be positive')
    path = lambda p: (args.root / p).resolve()
    hgrid, mask, ldd_path = map(path, (args.hgrid, args.mask, args.ldd))
    jobs = []
    for year in dict.fromkeys(args.years):
        years = [year-1, year, year+1] if args.time_window == 'legacy' else [year]
        files = [path(args.glofas_dir)/f'glofas_{y}.nc' for y in years]
        output = path(args.output_dir)/f'glofas_river_{year}_v4.nc'
        if output.exists() and not args.overwrite:
            raise ValueError(f'Output exists: {output}; use --overwrite to replace it')
        jobs.append((year, files, output))
    inputs = {hgrid, mask, ldd_path} | {p for _, files, _ in jobs for p in files}
    missing = sorted(str(p) for p in inputs if not p.is_file())
    if missing:
        raise ValueError('Missing input files:\n  '+'\n  '.join(missing)+
                         '\nLegacy mode requires YEAR-1, YEAR, and YEAR+1 discharge files.')
    if any(output in inputs for _, _, output in jobs):
        raise ValueError('An output path would overwrite an input file')
    grid = load_mom_grid(hgrid, mask, args.mask_var)
    with xr.open_dataarray(ldd_path) as data:
        ldd = data.values
        # Shape alone cannot establish matching LDD/discharge coordinates.
        ldd_coords = {n: data.coords[n].values.copy() for n in data.dims if n in data.coords}
    for year, files, output in jobs:
        start = time.monotonic()
        with xr.open_mfdataset(files, combine='by_coords', chunks={}, engine='netcdf4') as data:
            data = data.rename({old: new for old, new in
                                (('valid_time', 'time'), ('latitude', 'lat'), ('longitude', 'lon'))
                                if old in data.dims})
            window = (slice(f'{year-1}-12-31 12:00:00', f'{year+1}-01-01 12:00:00')
                      if args.time_window == 'legacy' else slice(f'{year}-01-01', f'{year}-12-31'))
            glofas = data.sel(time=window).dis24.transpose('time', 'lat', 'lon').chunk({'time': 1})
            if not glofas.sizes['time'] or not glofas.indexes['time'].is_unique or not glofas.indexes['time'].is_monotonic_increasing:
                raise ValueError('Selected times must be nonempty, unique, and increasing')
            for aliases, coord in ((('latitude', 'lat'), 'lat'), (('longitude', 'lon'), 'lon')):
                values = next((ldd_coords[n] for n in aliases if n in ldd_coords), None)
                if values is not None:
                    check_ldd_coordinates(values, glofas[coord].values, coord)
            print(f'Processing {year}: {glofas.sizes["time"]} records, '
                  f'{str(glofas.time.values[0])} to {str(glofas.time.values[-1])}; dis24 treated as m3/s')
            write_runoff(glofas, grid, ldd, output, not args.no_seam_fix, args.log_every)
        print(f'Elapsed: {time.monotonic()-start:.1f} s')


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, KeyError, ImportError, OverflowError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        sys.exit(2)
