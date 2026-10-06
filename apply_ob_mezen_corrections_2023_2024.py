#!/usr/bin/env python3
"""
Apply the Ob and Mezen river corrections to the 2023 and 2024 GloFAS files.

Method reproduced from RIVERS-VERIF-AND-FIXE.ipynb:
  1. Resample each observed discharge record to daily frequency.
  2. Linearly interpolate missing daily values.
  3. Compute the mean discharge for each day of year (1-366).
  4. Convert discharge to runoff with:
         runoff = discharge * 1000 / cell_area
  5. Replace runoff at:
         Ob River    -> j=374, i=1
         Mezen River -> j=485, i=1
  6. Repeat January 1 in the extra final time record.

Standalone ARC11k post-processing step; the Hill/GloFAS/merge baseline stays
uncorrected. Defaults: inputs outputs/MERGED, outputs outputs/MERGED-CORRECTED,
observations RAW/OBS. Cell indices and numerical calculations are unchanged.

Examples from the River-MOM6 root:
  python apply_ob_mezen_corrections_2023_2024.py 2023 2024
  python apply_ob_mezen_corrections_2023_2024.py 1993 --obs-dir /path/to/observations
Use --area-file to retain an external area source; otherwise use each merged
input's embedded area. No figures are produced. See --help and README.md.
"""

import argparse
import calendar
import datetime as dt
import os
import sys
import tempfile
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from netCDF4 import Dataset


# =============================================================================
# Paths and settings
# =============================================================================
YEARS = (2023, 2024)
INPUT_TEMPLATE = "glofas_hill_{year}_v4.1.nc"
EXPECTED_GRID_SHAPE = (696, 540)  # This optional correction is ARC11k-specific.
RIVERS = {
    "Ob": {"obs_file": "Ob_Salekhard_Version_20240509.xlsx", "j": 374, "i": 1},
    "Mezen": {"obs_file": "Mezen_Malonisogorskoe_Version_20240509.xlsx", "j": 485, "i": 1},
}


# =============================================================================
# Functions
# =============================================================================
def make_climatology(obs_file):
    """Create the 366-value day-of-year discharge climatology."""
    if not obs_file.exists():
        raise FileNotFoundError(f"Observation file not found: {obs_file}")

    df = pd.read_excel(obs_file)
    if "date" not in df or "discharge" not in df:
        raise ValueError(f"{obs_file.name} must contain 'date' and 'discharge'")

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")

    # Same procedure as the notebook.
    daily = df.resample("D").asfreq()
    daily["discharge"] = daily["discharge"].interpolate(method="linear")
    daily["day_of_year"] = daily.index.dayofyear

    climatology = daily.groupby("day_of_year")["discharge"].mean().to_numpy()

    if len(climatology) != 366:
        raise ValueError(
            f"Expected 366 climatology values in {obs_file.name}; "
            f"found {len(climatology)}"
        )
    if np.isnan(climatology).any():
        raise ValueError(f"NaNs remain in the climatology from {obs_file.name}")

    return climatology


def apply_correction(runoff, area, climatology, year, river_name, j, i):
    """Replace one river cell and print verification information."""
    n_days = 366 if calendar.isleap(year) else 365
    expected_length = n_days + 1

    if runoff.shape[0] != expected_length:
        raise ValueError(
            f"Unexpected time length for {year}: {runoff.shape[0]}; "
            f"expected {expected_length}"
        )

    cell_area = float(area[j, i])
    if not np.isfinite(cell_area) or cell_area <= 0:
        raise ValueError(
            f"Invalid area at {river_name} cell (j={j}, i={i}): {cell_area}"
        )

    # Save the original values only for verification printing.
    old_runoff = np.ma.filled(runoff[:n_days, j, i], np.nan).astype(float)
    old_discharge = old_runoff * cell_area / 1000.0

    # Convert m3/s to the runoff units in the NetCDF file.
    new_runoff = climatology[:n_days] * 1000.0 / cell_area
    runoff[:n_days, j, i] = new_runoff

    # The annual files contain one extra record: repeat January 1.
    runoff[n_days, j, i] = climatology[0] * 1000.0 / cell_area

    # Read the values back and convert to discharge for verification.
    stored = np.ma.filled(runoff[:, j, i], np.nan).astype(float)
    recovered = stored[:n_days] * cell_area / 1000.0
    pad_discharge = stored[-1] * cell_area / 1000.0
    max_error = np.max(np.abs(recovered - climatology[:n_days]))

    print(f"  {river_name} River: j={j}, i={i}")
    print(f"    cell area                  : {cell_area:.3f} m2")
    print(f"    original mean discharge   : {np.nanmean(old_discharge):.3f} m3/s")
    print(f"    corrected mean discharge  : {np.mean(recovered):.3f} m3/s")
    print(f"    corrected min/max         : {np.min(recovered):.3f} / "
          f"{np.max(recovered):.3f} m3/s")
    print(f"    maximum write error       : {max_error:.6g} m3/s")
    print(f"    final padded discharge    : {pad_discharge:.3f} m3/s")
    print(f"    NaNs after correction     : {np.isnan(stored).any()}")

    # Float32 NetCDF variables can introduce small rounding differences.
    values_ok = np.allclose(
        recovered, climatology[:n_days], rtol=1e-6, atol=0.1
    )
    pad_ok = np.isclose(pad_discharge, climatology[0], rtol=1e-6, atol=0.1)

    if np.isnan(stored).any() or not values_ok or not pad_ok:
        raise RuntimeError(f"Verification failed for {river_name}, {year}")


def check_input(input_file, year):
    """Check the fixed ARC11k grid and annual padded daily record convention."""
    with Dataset(input_file) as ds:
        runoff = ds.variables["runoff"]
        if runoff.dimensions != ("time", "y", "x") or runoff.shape[1:] != EXPECTED_GRID_SHAPE:
            raise ValueError(f"Expected ARC11k runoff(time,y,x), grid {EXPECTED_GRID_SHAPE}: {input_file}")
        tv = ds.variables["time"]
        from netCDF4 import num2date
        dates = num2date(tv[:], tv.units, calendar=getattr(tv, "calendar", "standard"))
        keys = [dt.date(d.year, d.month, d.day) for d in dates]
        start, end = dt.date(year, 1, 1), dt.date(year+1, 1, 1)
        expected = [start + dt.timedelta(days=i) for i in range((end-start).days+1)]
        if keys != expected:
            raise ValueError(f"Expected consecutive Jan 1 {year} through Jan 1 {year+1}, inclusive: {input_file}")


def process_year(year, area, climatologies, input_dir, output_dir, rivers, overwrite=False):
    """Copy one annual file and apply both unchanged river corrections."""
    input_file = input_dir / INPUT_TEMPLATE.format(year=year)
    output_file = output_dir / input_file.name
    if input_file.resolve() == output_file.resolve():
        raise ValueError("Correction output must differ from its uncorrected input")
    if output_file.exists() and not overwrite:
        raise FileExistsError(f"Output exists: {output_file}; use --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)
    print("\n" + "=" * 72)
    print(f"Processing {year}\nInput : {input_file}\nOutput: {output_file}")
    # Preserve the baseline and publish only after both corrections verify.
    with tempfile.TemporaryDirectory(prefix="ob-mezen-", dir=output_dir) as temporary:
        working_file = Path(temporary) / input_file.name
        shutil.copy2(input_file, working_file)
        with Dataset(working_file, "r+") as ds:
            runoff = ds.variables["runoff"]
            print(f"Runoff shape: {runoff.shape}")
            for river_name, settings in rivers.items():
                apply_correction(runoff, area, climatologies[river_name], year,
                                 river_name, settings["j"], settings["i"])
            ds.ob_mezen_correction = "Observed day-of-year climatology: Ob (374,1), Mezen (485,1); final Jan 1 repeated"
            ds.ob_mezen_observation_files = "\n".join(str(s["obs_file"]) for s in rivers.values())
            ds.sync()
        os.replace(working_file, output_file)
    print(f"Verification passed for {year}.")


def main():
    """Run the original corrections with repository-relative paths."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("years", type=int, nargs="*", default=list(YEARS), help="Default: 2023 2024")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--input-dir", default="outputs/MERGED")
    parser.add_argument("--output-dir", default="outputs/MERGED-CORRECTED")
    parser.add_argument("--obs-dir", default="RAW/OBS")
    parser.add_argument("--area-file", help="Optional original external area NetCDF; default: each input's area")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    path = lambda value: (args.root/value).resolve()
    input_dir, output_dir, obs_dir = map(path, (args.input_dir, args.output_dir, args.obs_dir))
    years = list(dict.fromkeys(args.years))
    rivers = {name: {**settings, "obs_file": obs_dir/settings["obs_file"]} for name,settings in RIVERS.items()}
    inputs = [input_dir/INPUT_TEMPLATE.format(year=y) for y in years]
    required = inputs + [s["obs_file"] for s in rivers.values()]
    if args.area_file:
        required.append(path(args.area_file))
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing input files:\n  " + "\n  ".join(missing))
    for year, input_file in zip(years, inputs):
        check_input(input_file, year)
        output = output_dir/input_file.name
        if output in required:
            raise ValueError("Correction output would overwrite an input")
        if output.exists() and not args.overwrite:
            raise FileExistsError(f"Output exists: {output}; use --overwrite")
    print("Applying ARC11k Ob and Mezen river corrections")
    print(f"Input directory : {input_dir}\nOutput directory: {output_dir}\nYears: {years}")
    climatologies = {}
    for name,settings in rivers.items():
        climatologies[name] = make_climatology(settings["obs_file"])
        clim = climatologies[name]
        print(f"{name} climatology: {len(clim)} values, mean={np.mean(clim):.3f}, "
              f"min={np.min(clim):.3f}, max={np.max(clim):.3f} m3/s")
    for year,input_file in zip(years,inputs):
        area_file = path(args.area_file) if args.area_file else input_file
        with Dataset(area_file) as ds:
            area = np.ma.filled(ds.variables["area"][:], np.nan).astype(float)
        if area.shape != EXPECTED_GRID_SHAPE:
            raise ValueError(f"Expected ARC11k area {EXPECTED_GRID_SHAPE}, found {area.shape}")
        print(f"Area source: {area_file}; shape {area.shape}")
        process_year(year, area, climatologies, input_dir, output_dir, rivers, args.overwrite)
    print("\nAll corrected files were created successfully.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)
