#!/usr/bin/env python3
"""Write Hill discharge on a MOM6 grid without PyROMS.

Run from your River-MOM6 directory:
    python write_hill_runoff.py 1993
    python write_hill_runoff.py 1993 1994
    python write_hill_runoff.py --source RAW/HILL/goa_dischargex_09011992_08311993.nc

Dependencies: numpy, netCDF4. No SciPy, xarray, xESMF, or PyROMS required.
1993 selects the water-year file ending in 1993; dates come from year/month/day.
Defaults use RAW/HILL, RAW/GRID, and outputs/HILL-on-MOM6 relative to --root.

Compatibility target: the supplied remap_river.py (Hill q in m3/day).
Preserves coastal-cell order/duplicates, sequential source accumulation,
native-precision q/86400 and dx*dy area, first-row/last-column removal,
float64 output and legacy dimensions.
The legacy asin(sin-distance) formula is preserved, including its behavior
beyond 90 degrees; this is deliberate for regression, not a generic mapper.
Author/Created attributes identify this implementation; equality means data
and relevant NetCDF structure, not byte-identical files. Validate against your
actual legacy output before using this replacement in production.

Algorithm references (ESMG/pyroms, python3 branch):
https://github.com/ESMG/pyroms/blob/python3/pyroms_toolbox/pyroms_toolbox/get_littoral.py
https://github.com/ESMG/pyroms/blob/python3/pyroms_toolbox/pyroms_toolbox/src/remap_river.f90
"""

import argparse
import datetime as dt
import hashlib
import os
from pathlib import Path
import sys
import tempfile
import time

import numpy as np
from netCDF4 import Dataset

MAPPING_VERSION = "hill-legacy-asin-eight-neighbors-v1"


def read_array(variable, key=slice(None)):
    """Decode missing values without promoting float32 before legacy arithmetic."""
    values = np.ma.asarray(variable[key])
    if values.dtype.kind not in "fc":
        values = values.astype(np.float64)
    return np.asarray(np.ma.filled(values, np.nan))


def load_grid(hgrid_path, mask_path):
    with Dataset(hgrid_path) as nc:
        # The Fortran mapper received double-precision coordinate arguments.
        lon = read_array(nc.variables["x"], (slice(1, None, 2), slice(1, None, 2))).astype(np.float64)
        lat = read_array(nc.variables["y"], (slice(1, None, 2), slice(1, None, 2))).astype(np.float64)
        dx = read_array(nc.variables["dx"])
        dy = read_array(nc.variables["dy"])
        area = (dx[1::2, ::2] + dx[1::2, 1::2]) * (
            dy[::2, 1::2] + dy[1::2, 1::2])
        print(f"Grid precision: dx={dx.dtype}, dy={dy.dtype}, area={area.dtype}", flush=True)
        corner_shape = nc.variables["x"][::2, ::2].shape
    with Dataset(mask_path) as nc:
        wet = read_array(nc.variables["wet"])
    if lon.ndim != 2 or any(a.shape != lon.shape for a in (lat, area, wet)):
        raise ValueError("Tracer lon/lat, legacy area, and wet must have the same 2D shape")
    if corner_shape != (lon.shape[0] + 1, lon.shape[1] + 1):
        raise ValueError("Supergrid corner shape must be (ny+1, nx+1)")
    if not all(np.isfinite(a).all() for a in (lon, lat, area, wet)):
        raise ValueError("Grid contains missing or nonfinite values")
    if np.any(area <= 0) or np.any(np.abs(lat) > 90):
        raise ValueError("Grid area must be positive and latitude within [-90, 90]")
    if not np.isin(wet, [0, 1]).all():
        raise ValueError("Expected a binary wet mask (0 land, 1 ocean)")
    return lon, lat, area, wet


def get_littoral_legacy(wet):
    """Eight neighbors of land, preserving legacy land/neighbor traversal order."""
    neighbors = ((1, 0), (-1, 0), (0, 1), (0, -1),
                 (1, 1), (-1, -1), (1, -1), (-1, 1))
    ny, nx = wet.shape
    cells = []
    for j, i in zip(*np.where(wet == 0)):
        for dj, di in neighbors:
            jj, ii = j + dj, i + di
            if 0 <= jj < ny and 0 <= ii < nx and wet[jj, ii] != 0:
                cells.append((jj, ii))
    if not cells:
        raise ValueError("No coastal ocean cells found in wet mask")
    return np.asarray(cells, dtype=np.int64)


def mapping_fingerprint(*arrays):
    digest = hashlib.sha256(MAPPING_VERSION.encode())
    for array in arrays:
        normalized = np.ascontiguousarray(array, dtype="<f8")
        digest.update(str(normalized.shape).encode())
        digest.update(normalized.tobytes())
    return digest.hexdigest()


def build_mapping(src_lon, src_lat, lon, lat, wet, chunk_size):
    littoral = get_littoral_legacy(wet)
    coast_lon = lon[littoral[:, 0], littoral[:, 1]][None, :]
    beta2 = (lat[littoral[:, 0], littoral[:, 1]] * (np.pi / 180))[None, :]
    sin2, cos2 = np.sin(beta2), np.cos(beta2)
    mapping = np.empty((src_lon.size, 2), dtype=np.int64)
    print(f"Building mapping: {src_lon.size} river points, "
          f"{len(littoral)} coastal entries (duplicates preserved)", flush=True)
    for start in range(0, src_lon.size, chunk_size):
        stop = min(start + chunk_size, src_lon.size)
        delta = coast_lon - src_lon[start:stop, None]
        length = np.abs(delta)
        length = np.where(length >= 180, 360 - np.abs(delta), length)
        length = length * (np.pi / 180)
        beta1 = src_lat[start:stop, None] * (np.pi / 180)
        sin1, cos1 = np.sin(beta1), np.cos(beta1)
        st = np.sqrt((np.sin(length) * cos2) ** 2 +
                     (sin2 * cos1 - sin1 * cos2 * np.cos(length)) ** 2)
        # Radius and operation order match the active spherical Fortran branch.
        # Tiny roundoff overshoots at 1 are bounded; asin is otherwise unchanged.
        distance = np.arcsin(np.clip(st, 0, 1)) * 6356750.52
        nearest = np.argmin(distance, axis=1)  # first minimum, like MINLOC
        mapping[start:stop] = littoral[nearest]
        if start == 0 or stop == src_lon.size or (start // chunk_size) % 10 == 0:
            print(f"  mapped {stop}/{src_lon.size}", flush=True)
    return mapping


def get_mapping(src_lon, src_lat, grid, cache_path, chunk_size, rebuild):
    lon, lat, area, wet = grid
    fingerprint = mapping_fingerprint(src_lon, src_lat, lon, lat, area, wet)
    if cache_path.exists() and not rebuild:
        try:
            with np.load(cache_path, allow_pickle=False) as cache:
                matches = str(cache["fingerprint"].item()) == fingerprint
                mapping = cache["mapping"].copy()
            valid = (mapping.shape == (src_lon.size, 2) and
                     np.issubdtype(mapping.dtype, np.integer))
            if valid:
                valid = (np.all(mapping >= 0) and
                         np.all(mapping[:, 0] < lon.shape[0]) and
                         np.all(mapping[:, 1] < lon.shape[1]))
            if matches and valid:
                coastal = np.zeros(lon.shape, dtype=bool)
                lit = get_littoral_legacy(wet)
                coastal[lit[:, 0], lit[:, 1]] = True
                if coastal[mapping[:, 0], mapping[:, 1]].all():
                    print(f"Reusing verified mapping: {cache_path}", flush=True)
                    return mapping
            print("Mapping inputs changed or cache invalid; rebuilding", flush=True)
        except (OSError, ValueError, KeyError, EOFError):
            print("Mapping cache unreadable; rebuilding", flush=True)
    mapping = build_mapping(src_lon, src_lat, lon, lat, wet, chunk_size)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=cache_path.parent, suffix=".npz")
    os.close(fd)
    try:
        np.savez(tmp, fingerprint=fingerprint, mapping=mapping,
                 algorithm=MAPPING_VERSION)
        os.replace(tmp, cache_path)
    finally:
        Path(tmp).unlink(missing_ok=True)
    print(f"Saved mapping: {cache_path}", flush=True)
    return mapping


def read_source_metadata(nc):
    src_lon = read_array(nc.variables["lon"])
    src_lat = read_array(nc.variables["lat"])
    if src_lon.ndim != 1 or src_lat.shape != src_lon.shape or not src_lon.size:
        raise ValueError("Hill lon/lat must be nonempty 1D river coordinates")
    if not (np.isfinite(src_lon).all() and np.isfinite(src_lat).all()):
        raise ValueError("Missing or nonfinite river coordinates")
    if np.any(np.abs(src_lat) > 90):
        raise ValueError("Source latitude outside [-90, 90]")
    fields = [read_array(nc.variables[name]) for name in ("year", "month", "day")]
    if any(a.ndim != 1 or a.shape != fields[0].shape for a in fields):
        raise ValueError("year/month/day must be matching 1D arrays")
    if any(not np.isfinite(a).all() or np.any(a != np.floor(a)) for a in fields):
        raise ValueError("year/month/day must contain valid integers")
    dates = [dt.date(int(y), int(m), int(d)) for y, m, d in zip(*fields)]
    if not dates or any(b <= a for a, b in zip(dates, dates[1:])):
        raise ValueError("Source dates must be nonempty and strictly increasing")
    if nc.variables["q"].shape != (len(dates), src_lon.size):
        raise ValueError("Expected q shaped (number of dates, number of river points)")
    units = getattr(nc.variables["q"], "units", "unspecified")
    print(f"Hill q units: {units}; stored dtype: {nc.variables['q'].dtype}; "
          "applying native-precision legacy /86400 conversion", flush=True)
    # Legacy longitude shift occurs BEFORE f2py promotes its input to float64.
    return (np.asarray(src_lon + 360.0, dtype=np.float64),
            np.asarray(src_lat, dtype=np.float64), dates)


def write_source(source, destination, grid, cache, args):
    if source.resolve() == destination.resolve():
        raise ValueError("Input and output paths must differ")
    if destination.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {destination}. Use --overwrite to replace it")
    lon, lat, area, wet = grid
    ny, nx = lon.shape
    destination.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with Dataset(source) as src:
        src_lon, src_lat, dates = read_source_metadata(src)
        mapping = get_mapping(src_lon, src_lat, grid, cache,
                              args.chunk_size, args.rebuild_mapping)
        fd, tmp = tempfile.mkstemp(dir=destination.parent, suffix=".nc")
        os.close(fd)
        try:
            with Dataset(tmp, "w", format="NETCDF4") as dst:
                dst.Author = "write_hill_runoff.py (PyROMS-free legacy-compatible implementation)"
                dst.Created = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                dst.title = "MOM6 runoff file"
                for name, size in (("IQ", nx + 1), ("JQ", ny + 1),
                                   ("i", nx), ("j", ny), ("time", None)):
                    dst.createDimension(name, size)
                tvar = dst.createVariable("time", "f8", ("time",))
                tvar.units = "days since 1900-01-01"
                tvar.calendar = "gregorian"
                rvar = dst.createVariable("Runoff", "f8", ("time", "j", "i"),
                                          fill_value=1.e30)
                rvar.long_name = "Hill River Runoff"
                rvar.units = "kg/m^2/sec"
                epoch = dt.date(1900, 1, 1)
                max_error = 0.0
                total_removed = 0.0
                for it, date in enumerate(dates):
                    q_native = read_array(src.variables["q"], (it, slice(None)))
                    q_native = q_native / 86400.0
                    if it == 0:
                        print(f"Discharge precision: after /86400={q_native.dtype}; "
                              "accumulation=float64", flush=True)
                    # Crucial: divide in the decoded source dtype, then promote,
                    # matching xarray/NumPy arithmetic followed by f2py input casting.
                    q = np.asarray(q_native, dtype=np.float64)
                    q[np.isnan(q)] = 0.0  # fill NaNs only, never silently replace infinity
                    if not np.isfinite(q).all():
                        raise ValueError(f"Infinite q on {date}")
                    runoff = np.zeros((ny, nx), dtype=np.float64)
                    # Unbuffered source-order additions preserve repeated-cell sums.
                    np.add.at(runoff, (mapping[:, 0], mapping[:, 1]), q)
                    before = float(np.sum(q))
                    mapped = float(np.sum(runoff))
                    error = abs(mapped - before)
                    max_error = max(max_error, error)
                    rounding_limit = 1e-12 * max(1.0, float(np.sum(np.abs(q))))
                    if error > rounding_limit:
                        raise ValueError(f"Discharge not conserved before boundary removal on {date}")
                    runoff[0, :] = 0.0
                    runoff[:, -1] = 0.0
                    retained = float(np.sum(runoff))
                    total_removed += mapped - retained
                    tvar[it] = (date - epoch).days
                    rvar[it] = runoff * 1000.0 / area
                    if it == 0 or (it + 1) % args.log_every == 0 or it == len(dates) - 1:
                        print(f"  {it+1}/{len(dates)} {date}: input={before:.12g}, "
                              f"mapped={mapped:.12g}, retained={retained:.12g}, "
                              f"removed={mapped-retained:.12g} m3/s", flush=True)
            os.replace(tmp, destination)
        finally:
            Path(tmp).unlink(missing_ok=True)
    print(f"Wrote {destination}\n  Elapsed: {time.perf_counter()-started:.1f} s; "
          f"max pre-boundary conservation error: {max_error:.6g} m3/s; "
          f"mean boundary removal: {total_removed/len(dates):.6g} m3/s", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("years", nargs="*", type=int, help="Hill file ending year(s), e.g. 1993")
    parser.add_argument("--root", type=Path, default=Path("."), help="River-MOM6 project root")
    parser.add_argument("--source", type=Path, action="append", help="Explicit Hill file; repeatable")
    parser.add_argument("--hill-dir", type=Path, default=Path("RAW/HILL"))
    parser.add_argument("--hgrid", type=Path, default=Path("RAW/GRID/ocean_hgrid.nc"))
    parser.add_argument("--mask", type=Path, default=Path("RAW/GRID/sea_ice_geometry.nc"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/HILL-on-MOM6"))
    parser.add_argument("--mapping", type=Path,
                        default=Path("outputs/MAPPING/hill_to_ARC11k_mapping.npz"))
    parser.add_argument("--chunk-size", type=int, default=64, help="Rivers per distance batch")
    parser.add_argument("--log-every", type=int, default=30, help="Report every N days")
    parser.add_argument("--rebuild-mapping", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not args.years and not args.source:
        parser.error("Provide a year (e.g. 1993) or --source FILE")
    if args.source and args.years:
        parser.error("Use years or --source, not both")
    if args.chunk_size <= 0 or args.log_every <= 0:
        parser.error("--chunk-size and --log-every must be positive")
    def rooted(path):
        return path if path.is_absolute() else args.root / path
    sources = []
    if args.source:
        sources = [rooted(path) for path in args.source]
    else:
        for year in args.years:
            matches = sorted(rooted(args.hill_dir).glob(f"goa_dischargex_*{year}.nc"))
            if not matches:
                parser.error(f"No Hill files ending in {year} in {rooted(args.hill_dir)}")
            sources.extend(matches)
    sources = list(dict.fromkeys(path.resolve() for path in sources))
    output_dir = rooted(args.output_dir)
    destinations = [output_dir / (s.stem + "_q_Hill_NGOA.nc") for s in sources]
    if len(set(destinations)) != len(destinations):
        parser.error("Source basenames collide in the output directory")
    for destination in destinations:
        if destination.exists() and not args.overwrite:
            parser.error(f"Output exists: {destination}; use --overwrite to replace")
    grid = load_grid(rooted(args.hgrid), rooted(args.mask))
    print(f"MOM6 grid: {grid[0].shape}; area: legacy dx*dy", flush=True)
    for source, destination in zip(sources, destinations):
        print(f"Processing {source}", flush=True)
        write_source(source, destination, grid, rooted(args.mapping), args)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)
