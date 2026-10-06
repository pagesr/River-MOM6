#!/usr/bin/env python3
"""Independently compare GloFAS runoff with a legacy MOM6 forcing file.

Examples from River-MOM6:
  python compare_glofas_with_legacy.py --year 1993 --report outputs/glofas_1993_comparison.json
  python compare_glofas_with_legacy.py path/to/legacy.nc path/to/new.nc --mask other_grid_mask.nc

Defaults: outputs/LEGACY/GLOFAS/glofas_river_YEAR_v4.nc and
outputs/GLOFAS-on-MOM6/glofas_river_YEAR_v4.nc; --year defaults to 1993.
Requires only NumPy and netCDF4, not xESMF or the GloFAS writer.
Exact comparisons by default (rtol=atol=0), including embedded area/lat/lon,
coordinate values, time metadata, NetCDF structure and runoff values.
Author/Created provenance is not compared. Matching missing/nonfinite runoff is
reported but still fails validation. Runoff must be on four-neighbor interior
coastal cells from --mask (auto variable selection: mask, then wet).
Exit statuses: 0 pass, 1 mismatch, 2 input or usage error.
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from netCDF4 import Dataset, num2date


def read_array(var, key=slice(None)):
    values = np.ma.asarray(var[key])
    if values.dtype.kind not in "fc":
        values = values.astype(np.float64)
    return np.asarray(np.ma.filled(values, np.nan))


def dimension_info(nc):
    return {n: {"size": len(d), "unlimited": d.isunlimited()}
            for n, d in nc.dimensions.items()}


def attributes_equal(a, b):
    return set(a.ncattrs()) == set(b.ncattrs()) and all(
        np.array_equal(a.getncattr(n), b.getncattr(n)) for n in a.ncattrs())


def load_coast(path, variable):
    with Dataset(path) as nc:
        name = variable or next((n for n in ("mask", "wet") if n in nc.variables), None)
        if name is None or name not in nc.variables:
            raise ValueError("No MOM6 mask variable found; use --mask-var")
        wet = read_array(nc.variables[name])
    if wet.ndim != 2 or min(wet.shape) < 3 or not np.isin(wet, [0, 1]).all():
        raise ValueError("MOM6 mask must be 2D, at least 3x3, and binary (1 ocean)")
    wet = wet.astype(bool)
    coast = np.zeros(wet.shape, dtype=bool)
    coast[1:-1, 1:-1] = wet[1:-1, 1:-1] & (
        ~wet[2:, 1:-1] | ~wet[:-2, 1:-1] | ~wet[1:-1, :-2] | ~wet[1:-1, 2:])
    return coast


def compare(old_path, new_path, coast, args):
    report = {"legacy": str(old_path), "new": str(new_path),
              "rtol": args.rtol, "atol": args.atol,
              "structural_issues": [], "static_arrays_exactly_equal": {}, "daily": []}
    issues = report["structural_issues"]
    with Dataset(old_path) as old, Dataset(new_path) as new:
        report["legacy_dimensions"] = dimension_info(old)
        report["new_dimensions"] = dimension_info(new)
        report["legacy_format"] = old.data_model
        report["new_format"] = new.data_model
        if dimension_info(old) != dimension_info(new):
            issues.append("Dimensions or unlimited flags differ")
        if old.data_model != new.data_model:
            issues.append("NetCDF formats differ")
        required = {"runoff", "area", "lat", "lon", "time", "y", "x"}
        if not required.issubset(old.variables) or not required.issubset(new.variables):
            issues.append("Both files must contain runoff, area, lat, lon, time, y, x")
            report["passed"] = False
            return report
        if set(old.variables) != set(new.variables):
            issues.append("Variable sets differ")
        for name in sorted(required):
            a, b = old.variables[name], new.variables[name]
            if a.dimensions != b.dimensions or a.dtype != b.dtype:
                issues.append(f"{name}: datatype or dimension order differs")
            if not attributes_equal(a, b):
                issues.append(f"{name}: variable attributes differ")
        a, b = old.variables["runoff"], new.variables["runoff"]
        if a.shape != b.shape or a.ndim != 3 or a.dimensions != ("time", "y", "x") or \
           b.dimensions != ("time", "y", "x"):
            issues.append("Runoff shape/order differs; expected (time,y,x)")
            report["passed"] = False
            return report
        if a.shape[0] == 0:
            raise ValueError("No runoff time records")
        if coast.shape != a.shape[1:]:
            raise ValueError("MOM6 mask shape does not match runoff")
        for name in ("area", "lat", "lon", "time", "y", "x"):
            av, bv = read_array(old.variables[name]), read_array(new.variables[name])
            equal = av.shape == bv.shape and np.isfinite(av).all() and \
                    np.isfinite(bv).all() and np.array_equal(av, bv)
            report["static_arrays_exactly_equal"][name] = bool(equal)
            if not equal:
                issues.append(f"{name}: values differ or contain nonfinite values")
        area = read_array(old.variables["area"])
        if area.shape != coast.shape or not np.isfinite(area).all() or np.any(area <= 0):
            raise ValueError("Invalid legacy area array")
        times = read_array(old.variables["time"])
        if times.shape != (a.shape[0],) or not np.isfinite(times).all():
            raise ValueError("Legacy time does not match runoff records")
        dates = num2date(times, old.variables["time"].units,
                        calendar=getattr(old.variables["time"], "calendar", "standard"))
        report["time_range"] = [str(dates[0]), str(dates[-1])]
        report["number_of_records"] = a.shape[0]
        union_old = np.zeros(coast.shape, dtype=bool)
        union_new = np.zeros(coast.shape, dtype=bool)
        total_diff = outside = support_diff = invalid_old = invalid_new = 0
        noncoast_old = noncoast_new = count = 0
        squared_sum = max_abs = max_rel = max_flow_diff = 0.0
        first = []
        for it in range(a.shape[0]):
            av, bv = read_array(a, it), read_array(b, it)
            fa, fb = np.isfinite(av), np.isfinite(bv)
            valid = fa & fb
            invalid_old += int(np.count_nonzero(~fa))
            invalid_new += int(np.count_nonzero(~fb))
            nz_a, nz_b = fa & (av != 0), fb & (bv != 0)
            union_old |= nz_a
            union_new |= nz_b
            noncoast_old += int(np.count_nonzero(nz_a & ~coast))
            noncoast_new += int(np.count_nonzero(nz_b & ~coast))
            locations = int(np.count_nonzero(nz_a != nz_b))
            support_diff += locations
            exact_mismatch = valid & (av != bv)
            different = int(np.count_nonzero(exact_mismatch))
            out = int(np.count_nonzero(valid & ~np.isclose(bv, av, rtol=args.rtol, atol=args.atol)))
            total_diff += different
            outside += out
            diff = bv[valid]-av[valid]
            absolute = np.abs(diff)
            daily_max = float(absolute.max()) if diff.size else 0.0
            max_abs = max(max_abs, daily_max)
            squared_sum += float(np.sum(diff*diff))
            count += diff.size
            denominator = np.abs(av[valid])
            selected = denominator != 0
            if selected.any():
                max_rel = max(max_rel, float(np.max(absolute[selected]/denominator[selected])))
            old_flow = float(np.sum(av*area/1000)) if fa.all() else None
            new_flow = float(np.sum(bv*area/1000)) if fb.all() else None
            flow_diff = new_flow-old_flow if old_flow is not None and new_flow is not None else None
            if flow_diff is not None:
                max_flow_diff = max(max_flow_diff, abs(flow_diff))
            for j, i in np.argwhere(exact_mismatch | ~valid)[:max(0, 5-len(first))]:
                first.append({"time_index": it, "date": str(dates[it]),
                              "j": int(j), "i": int(i),
                              "legacy": float(av[j, i]) if fa[j, i] else None,
                              "new": float(bv[j, i]) if fb[j, i] else None})
            report["daily"].append({"time_index": it, "date": str(dates[it]),
                "legacy_nonzero_cells": int(nz_a.sum()), "new_nonzero_cells": int(nz_b.sum()),
                "differing_values": different, "values_outside_tolerance": out,
                "nonzero_location_mismatches": locations, "max_absolute_difference": daily_max,
                "legacy_discharge_m3_s": old_flow, "new_discharge_m3_s": new_flow,
                "discharge_difference_m3_s": flow_diff})
            if it == 0 or (it+1) % args.log_every == 0 or it == a.shape[0]-1:
                print(f"  {it+1}/{a.shape[0]} {dates[it]}: max abs diff={daily_max:.6g}; "
                      f"differing={different}; location mismatches={locations}", flush=True)
        report.update({"runoff_exactly_equal": total_diff == invalid_old == invalid_new == 0,
            "differing_values": total_diff, "values_outside_tolerance": outside,
            "nonzero_location_mismatches": support_diff,
            "legacy_distinct_nonzero_cells": int(union_old.sum()),
            "new_distinct_nonzero_cells": int(union_new.sum()),
            "legacy_invalid_values": invalid_old, "new_invalid_values": invalid_new,
            "legacy_nonzero_values_outside_coast": noncoast_old,
            "new_nonzero_values_outside_coast": noncoast_new,
            "max_absolute_difference_kg_m2_s": max_abs,
            "max_relative_difference_where_legacy_nonzero": max_rel,
            "rmse_kg_m2_s": (squared_sum/count)**0.5 if count else None,
            "max_daily_discharge_difference_m3_s": max_flow_diff,
            "first_differing_values": first,
            "passed": not issues and outside == support_diff == invalid_old == invalid_new ==
                      noncoast_old == noncoast_new == 0})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("legacy", nargs="?", type=Path)
    parser.add_argument("new", nargs="?", type=Path)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--year", type=int, default=1993)
    parser.add_argument("--mask", type=Path, default=Path("RAW/GRID/ocean_mask.nc"))
    parser.add_argument("--mask-var")
    parser.add_argument("--rtol", type=float, default=0.)
    parser.add_argument("--atol", type=float, default=0.)
    parser.add_argument("--log-every", type=int, default=30)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if (args.legacy is None) != (args.new is None):
        parser.error("Provide both legacy/new paths or neither")
    if args.log_every <= 0 or any(not np.isfinite(x) or x < 0 for x in (args.rtol, args.atol)):
        parser.error("Tolerances must be finite/nonnegative; --log-every positive")
    def rooted(path):
        return path if path.is_absolute() else args.root/path
    name = f"glofas_river_{args.year}_v4.nc"
    old = rooted(args.legacy or Path("outputs/LEGACY/GLOFAS")/name)
    new = rooted(args.new or Path("outputs/GLOFAS-on-MOM6")/name)
    if old.resolve() == new.resolve():
        parser.error("Legacy and new paths must differ")
    if args.report and rooted(args.report).resolve() in {old.resolve(), new.resolve(), rooted(args.mask).resolve()}:
        parser.error("Report path must not overwrite an input")
    print(f"Legacy: {old}\nNew: {new}\nTolerance: rtol={args.rtol:g}, atol={args.atol:g}", flush=True)
    report = compare(old, new, load_coast(rooted(args.mask), args.mask_var), args)
    print("\n" + ("PASS" if report["passed"] else "FAIL"))
    for name, value in report.items():
        if name not in ("daily", "legacy_dimensions", "new_dimensions", "legacy", "new", "passed"):
            print(f"  {name}: {value}")
    if args.report:
        path = rooted(args.report)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
        print(f"Saved report: {path}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError, OverflowError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)
