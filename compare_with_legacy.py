#!/usr/bin/env python3
"""Compare a new Hill-on-MOM6 file with the legacy output, one day at a time.

Run from River-MOM6 (defaults target the water year ending in 1993):
    python compare_with_legacy.py
    python compare_with_legacy.py --year 1993 --report outputs/hill_1993_comparison.json
    python compare_with_legacy.py path/to/legacy.nc path/to/new.nc

Dependencies: numpy, netCDF4. This script is independent of the writer.
Default rtol=atol=0 requires exact numerical equality. To inspect tiny
roundoff differences: --rtol 1e-12 --atol 1e-15. Nonzero locations, dimensions,
time and relevant variable metadata must still match. Nonfinite/missing runoff
is always a failure. Global Author/Created provenance differences are reported
but do not fail the comparison. NetCDF binary layout is not compared.

Integrated discharge uses the same native-precision legacy dx*dy area as the
writer, not Ah.
Full 1993 validation requires your original legacy NetCDF and grid files.
Exit status: 0 pass, 1 mismatch, 2 input/usage error.
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


def legacy_area(path):
    with Dataset(path) as nc:
        dx = read_array(nc.variables["dx"])
        dy = read_array(nc.variables["dy"])
    area = (dx[1::2, ::2] + dx[1::2, 1::2]) * (
        dy[::2, 1::2] + dy[1::2, 1::2])
    if area.ndim != 2 or not np.isfinite(area).all() or np.any(area <= 0):
        raise ValueError("Invalid legacy cell areas in ocean_hgrid.nc")
    return area


def dimension_info(nc):
    return {name: {"size": len(dim), "unlimited": dim.isunlimited()}
            for name, dim in nc.dimensions.items()}


def same_attribute(a, b, name):
    left, right = getattr(a, name, None), getattr(b, name, None)
    return bool(np.array_equal(left, right))


def compare(legacy_path, new_path, area, wet, rtol, atol, log_every):
    report = {"legacy": str(legacy_path), "new": str(new_path),
              "rtol": rtol, "atol": atol, "structural_issues": [],
              "global_attributes_differing_ignored": [], "daily": []}
    issues = report["structural_issues"]
    with Dataset(legacy_path) as old, Dataset(new_path) as new:
        report["legacy_dimensions"] = dimension_info(old)
        report["new_dimensions"] = dimension_info(new)
        if report["legacy_dimensions"] != report["new_dimensions"]:
            issues.append("Dimension names, lengths, or unlimited flags differ")
        for name in set(old.ncattrs()) | set(new.ncattrs()):
            if not same_attribute(old, new, name):
                report["global_attributes_differing_ignored"].append(name)
        report["global_attributes_differing_ignored"].sort()
        required = {"time", "Runoff"}
        if not required.issubset(old.variables) or not required.issubset(new.variables):
            issues.append("Both files must contain time and Runoff")
            report["passed"] = False
            return report
        if set(old.variables) != set(new.variables):
            issues.append("Variable names differ")
        for name in ("time", "Runoff"):
            a, b = old.variables[name], new.variables[name]
            if a.dimensions != b.dimensions or a.dtype != b.dtype:
                issues.append(f"{name}: dimension order or datatype differs")
            attrs = ("units", "calendar") if name == "time" else (
                "units", "long_name", "_FillValue")
            for attr in attrs:
                if not same_attribute(a, b, attr):
                    issues.append(f"{name}: {attr} differs")
        a, b = old.variables["Runoff"], new.variables["Runoff"]
        if a.shape != b.shape or a.ndim != 3 or a.dimensions != ("time", "j", "i"):
            issues.append("Runoff shapes differ or legacy dimensions are not (time,j,i)")
            report["passed"] = False
            return report
        if b.dimensions != ("time", "j", "i"):
            issues.append("New Runoff dimensions are not (time,j,i)")
            report["passed"] = False
            return report
        if a.shape[1:] != area.shape or wet.shape != area.shape:
            raise ValueError("Runoff, wet, and legacy cell area shapes differ")
        if a.shape[0] == 0:
            raise ValueError("Runoff has no time records")
        t_old, t_new = read_array(old.variables["time"]), read_array(new.variables["time"])
        time_equal = (t_old.shape == t_new.shape == (a.shape[0],) and
                      np.isfinite(t_old).all() and np.isfinite(t_new).all() and
                      np.array_equal(t_old, t_new))
        report["time_exactly_equal"] = bool(time_equal)
        if not time_equal:
            issues.append("Raw time arrays differ, are invalid, or do not match Runoff length")
        if t_old.shape == (a.shape[0],) and np.isfinite(t_old).all():
            dates = num2date(t_old, old.variables["time"].units,
                            calendar=getattr(old.variables["time"], "calendar", "standard"))
            labels = [date.strftime("%Y-%m-%d") for date in dates]
        else:
            labels = [str(it) for it in range(a.shape[0])]

        differing = outside = location_mismatches = invalid_old = invalid_new = 0
        legacy_nonzero = new_nonzero = 0
        land_old = land_new = boundary_old = boundary_new = 0
        max_abs = max_rel = squared_sum = 0.0
        valid_count = 0
        first = []
        union_old = np.zeros(area.shape, dtype=bool)
        union_new = np.zeros(area.shape, dtype=bool)
        boundary = np.zeros(area.shape, dtype=bool)
        boundary[0, :] = True
        boundary[:, -1] = True
        for it in range(a.shape[0]):
            av, bv = read_array(a, it), read_array(b, it)
            finite_a, finite_b = np.isfinite(av), np.isfinite(bv)
            valid = finite_a & finite_b
            invalid_old += int(np.count_nonzero(~finite_a))
            invalid_new += int(np.count_nonzero(~finite_b))
            nz_a, nz_b = finite_a & (av != 0), finite_b & (bv != 0)
            legacy_nonzero += int(np.count_nonzero(nz_a))
            new_nonzero += int(np.count_nonzero(nz_b))
            union_old |= nz_a
            union_new |= nz_b
            support_mismatch = int(np.count_nonzero(nz_a != nz_b))
            location_mismatches += support_mismatch
            land_old += int(np.count_nonzero(nz_a & (wet == 0)))
            land_new += int(np.count_nonzero(nz_b & (wet == 0)))
            boundary_old += int(np.count_nonzero(nz_a & boundary))
            boundary_new += int(np.count_nonzero(nz_b & boundary))
            diff = bv[valid] - av[valid]
            absdiff = np.abs(diff)
            daily_max = float(absdiff.max()) if absdiff.size else 0.0
            max_abs = max(max_abs, daily_max)
            squared_sum += float(np.sum(diff * diff))
            valid_count += diff.size
            denominator = np.abs(av[valid])
            if np.any(denominator != 0):
                max_rel = max(max_rel, float(np.max(
                    absdiff[denominator != 0] / denominator[denominator != 0])))
            exact_mismatch = valid & (av != bv)
            tolerance_mismatch = valid & ~np.isclose(bv, av, rtol=rtol, atol=atol)
            different_day = int(np.count_nonzero(exact_mismatch))
            outside_day = int(np.count_nonzero(tolerance_mismatch))
            differing += different_day
            outside += outside_day
            if len(first) < 5:
                for j, i in np.argwhere(exact_mismatch | ~valid)[:5-len(first)]:
                    first.append({"time_index": it, "date": labels[it],
                                  "j": int(j), "i": int(i),
                                  "legacy": float(av[j, i]) if finite_a[j, i] else None,
                                  "new": float(bv[j, i]) if finite_b[j, i] else None})
            flow_a = float(np.sum(av * area / 1000)) if finite_a.all() else None
            flow_b = float(np.sum(bv * area / 1000)) if finite_b.all() else None
            report["daily"].append({"time_index": it, "date": labels[it],
                "legacy_nonzero_cells": int(np.count_nonzero(nz_a)),
                "new_nonzero_cells": int(np.count_nonzero(nz_b)),
                "nonzero_location_mismatches": support_mismatch,
                "differing_values": different_day, "outside_tolerance": outside_day,
                "max_absolute_difference": daily_max,
                "legacy_discharge_m3_s": flow_a, "new_discharge_m3_s": flow_b,
                "discharge_difference_m3_s": flow_b-flow_a
                if flow_a is not None and flow_b is not None else None})
            if it == 0 or (it + 1) % log_every == 0 or it == a.shape[0] - 1:
                print(f"  {it+1}/{a.shape[0]} {labels[it]}: "
                      f"max abs diff={daily_max:.6g}; "
                      f"differing={different_day}; location mismatches={support_mismatch}", flush=True)

        flows = [abs(day["discharge_difference_m3_s"]) for day in report["daily"]
                 if day["discharge_difference_m3_s"] is not None]
        report.update({
            "runoff_exactly_equal": differing == 0 and invalid_old == invalid_new == 0,
            "differing_values": differing, "values_outside_tolerance": outside,
            "nonzero_location_mismatches": location_mismatches,
            "legacy_nonzero_values": legacy_nonzero, "new_nonzero_values": new_nonzero,
            "legacy_distinct_nonzero_cells": int(np.count_nonzero(union_old)),
            "new_distinct_nonzero_cells": int(np.count_nonzero(union_new)),
            "legacy_invalid_values": invalid_old, "new_invalid_values": invalid_new,
            "legacy_nonzero_values_on_land": land_old, "new_nonzero_values_on_land": land_new,
            "legacy_nonzero_values_on_removed_boundaries": boundary_old,
            "new_nonzero_values_on_removed_boundaries": boundary_new,
            "max_absolute_difference_kg_m2_s": max_abs,
            "max_relative_difference_where_legacy_nonzero": max_rel,
            "rmse_kg_m2_s": (squared_sum/valid_count)**0.5 if valid_count else None,
            "max_daily_discharge_difference_m3_s": max(flows) if flows else None,
            "first_differing_values": first,
            "passed": not issues and outside == location_mismatches == 0 and
                      invalid_old == invalid_new == 0 and land_old == land_new == 0 and
                      boundary_old == boundary_new == 0})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("legacy", nargs="?", type=Path)
    parser.add_argument("new", nargs="?", type=Path)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--year", type=int, default=1993, help="Ending water year for default paths")
    parser.add_argument("--hgrid", type=Path, default=Path("RAW/GRID/ocean_hgrid.nc"))
    parser.add_argument("--mask", type=Path, default=Path("RAW/GRID/sea_ice_geometry.nc"))
    parser.add_argument("--rtol", type=float, default=0.0)
    parser.add_argument("--atol", type=float, default=0.0)
    parser.add_argument("--log-every", type=int, default=30)
    parser.add_argument("--report", type=Path, help="Optional JSON with summary and daily discharge")
    args = parser.parse_args()
    if (args.legacy is None) != (args.new is None):
        parser.error("Provide both legacy and new paths, or neither")
    if args.log_every <= 0 or any(not np.isfinite(v) or v < 0 for v in (args.rtol, args.atol)):
        parser.error("Tolerances must be finite/nonnegative and --log-every positive")
    def rooted(path):
        return path if path.is_absolute() else args.root / path
    if args.legacy is None:
        filename = f"goa_dischargex_0901{args.year-1}_0831{args.year}_q_Hill_NGOA.nc"
        args.legacy = Path("outputs/LEGACY/HILL-on-MOM6") / filename
        args.new = Path("outputs/HILL-on-MOM6") / filename
    old, new = rooted(args.legacy), rooted(args.new)
    if old.resolve() == new.resolve():
        parser.error("Legacy and new paths must differ; comparing a file to itself cannot validate it")
    if args.report and rooted(args.report).resolve() in {
            old.resolve(), new.resolve(), rooted(args.hgrid).resolve(), rooted(args.mask).resolve()}:
        parser.error("Report path must not overwrite an input")
    area = legacy_area(rooted(args.hgrid))
    with Dataset(rooted(args.mask)) as nc:
        wet = read_array(nc.variables["wet"])
    if not np.isin(wet, [0, 1]).all():
        raise ValueError("wet mask must contain only 0 and 1")
    print(f"Legacy: {old}\nNew:    {new}\nTolerance: rtol={args.rtol:g}, atol={args.atol:g}", flush=True)
    report = compare(old, new, area, wet, args.rtol, args.atol, args.log_every)
    print("\n" + ("PASS" if report["passed"] else "FAIL"))
    for issue in report["structural_issues"]:
        print(f"  Structure/time: {issue}")
    if "runoff_exactly_equal" in report:
        print(f"  Runoff exactly equal: {report['runoff_exactly_equal']}")
        for name in ("differing_values", "values_outside_tolerance", "nonzero_location_mismatches",
                     "legacy_distinct_nonzero_cells", "new_distinct_nonzero_cells",
                     "legacy_invalid_values", "new_invalid_values",
                     "legacy_nonzero_values_on_land", "new_nonzero_values_on_land",
                     "legacy_nonzero_values_on_removed_boundaries",
                     "new_nonzero_values_on_removed_boundaries",
                     "max_absolute_difference_kg_m2_s", "rmse_kg_m2_s",
                     "max_daily_discharge_difference_m3_s"):
            print(f"  {name}: {report[name]}")
        for item in report["first_differing_values"]:
            print(f"  First differences: {item}")
    print(f"  Global attributes differing (ignored): "
          f"{report['global_attributes_differing_ignored']}")
    if args.report:
        path = rooted(args.report)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(f"  Saved report: {path}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError, OverflowError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)
