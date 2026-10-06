# Step 2: GloFAS v4 to MOM6

`write_glofas_runoff.py` is a standalone adaptation of the supplied
`write_runoff_glofas_dis_batch_v4.py`, which credits Andrew C. Ross.
It keeps the same named stencil, pour-point, footprint, and runoff functions.
It does not run Hill or merge the two products.

## Inputs for the 1993 legacy comparison

```text
RAW/GRID/ocean_hgrid.nc
RAW/GRID/ocean_mask.nc
RAW/GRID/ldd_glofas_v4_0_subset.nc
RAW/GLOFAS/glofas_1992.nc
RAW/GLOFAS/glofas_1993.nc
RAW/GLOFAS/glofas_1994.nc
outputs/LEGACY/GLOFAS/glofas_river_1993_v4.nc
```

The default LDD location is **RAW/GRID**, as in your current layout. The LDD is
source data rather than a MOM6 grid. The repository now permits this northern
subset in Git; check its actual size before staging. The full optional LDD
download and annual discharge files remain excluded. The discharge variable is
`dis24`, already in m³/s;
there is no division by 86400. Input dimension names may be `valid_time`,
`latitude`, `longitude`, or `time`, `lat`, `lon`.

The LDD and discharge must have the same subset, resolution, coordinate order,
and shape. Coordinate alignment is checked when the LDD provides named
latitude/longitude coordinates, allowing rounding of at most 1e-6 degrees and
equivalent longitude conventions (−180–180 or 0–360). This tolerance is much
smaller than a 0.05-degree source cell. Coordinates and data are never
automatically reordered; reversed latitude or shifted longitude columns fail
with a diagnostic showing the largest difference. The source must use one-dimensional coordinates
and descending latitude. Bounds retain the original uniform-spacing assumption
and longitude ordering; this script does not generalize arbitrary source grids.

The default **legacy** window is inclusive from December 31 of the preceding
year at 12:00 to January 1 of the following year at 12:00. For daily noon records,
1993 therefore has 367 records. This is different from the Hill ending water year.
With only your 1992 and 1993 inputs, add `glofas_1994.nc` before reproducing the
legacy 1993 file. The script reports missing inputs before constructing weights.

## Environment

If your existing environment runs the original script, use it for the first
comparison to keep xESMF/ESMPy versions consistent. Otherwise:

```bash
conda env create -f environment-glofas.yml
conda activate river-mom6
```

The optional environment adds xarray, Dask, xESMF, and ESMPy. ESMPy installation
uses conda-forge; installing xESMF alone with pip is insufficient. See the
[xESMF installation guide](https://xesmf.readthedocs.io/en/stable/installation.html).
Hill-only users can continue using `requirements.txt`.

## Run and compare

From the River-MOM6 root:

```bash
python write_glofas_runoff.py 1993
python QC/compare_glofas_with_legacy.py --year 1993 --report outputs/glofas_1993_comparison.json
```

The generated file is:

```text
outputs/GLOFAS-on-MOM6/glofas_river_1993_v4.nc
```

Existing outputs require `--overwrite` to replace. Multiple years run sequentially:

```bash
python write_glofas_runoff.py 1993 1994
python write_glofas_runoff.py 1993 --overwrite
```

Each year builds the original xESMF weights afresh. Temporary weight filenames
are isolated so concurrent runs cannot reuse or overwrite each other's weights.
The writer reads and accumulates one day's discharge at a time, then writes
through a temporary NetCDF file. The final path is replaced only after success.

An explicit calendar-year mode needs only that year's discharge file:

```bash
python write_glofas_runoff.py 1993 --time-window calendar-year
```

It selects January 1–December 31 and writes the same output filename. Use a
separate `--output-dir` for experiments: this time window generally will **not**
match the padded legacy file. Do not use this option for the initial legacy test.

## Change the MOM6 grid independently

All destination dimensions and coordinates come from the supplied files. There
are no fixed ARC11k dimensions or geographic cutoffs. For example:

```bash
python write_glofas_runoff.py 1993 \
  --hgrid RAW/GRID/another_grid/ocean_hgrid.nc \
  --mask RAW/GRID/another_grid/ocean_mask.nc \
  --output-dir outputs/another_grid/GLOFAS-on-MOM6
```

`--mask-var` selects a different variable name; otherwise `mask` is tried before
`wet`. Values must be 1 for ocean and 0 for land. `--ldd` and `--glofas-dir` select
another GloFAS subset covering the destination grid. `--root` resolves all relative
paths against another project directory. `--help` lists the options.

For a GoA grid requiring Hill only, run `write_hill_runoff.py` with its grid and
mask options; skip GloFAS entirely. For Arctic configurations requiring both,
run both writers independently. Merging remains a separate future step.
Comparisons for another grid must also pass the matching `--mask` and explicit
legacy/new file paths to `compare_glofas_with_legacy.py`.

## Calculations preserved

1. Coastal destinations are interior wet cells next to land in the four cardinal
   directions. Outer grid cells are excluded.
2. GloFAS pour points come from LDD value 5 using the original iterative 3×3
   ocean stencil, including its zero outer boundary. Vectorizing each pass
   preserves the original fixed-stencil update behavior.
3. The original Chukotka seam correction changes LDD at 68.875°N, 180.025°E to 1.
   It is skipped when this coordinate is outside the subset. `--no-seam-fix`
   disables it explicitly and can change results.
4. A conservative xESMF mapping of **ones across the MOM6 rectangle** selects
   the GloFAS footprint. This is the original geometry, not the destination wet
   mask. Coverage of exactly 0.5 remains included.
5. xESMF `nearest_s2d` maps pour points to coastal destinations through LocStreams.
   The source traversal is row-major; accumulation is float64 in that order.
6. Cell area is the original parenthesized sum of four supergrid `area` values
   in their native dtype. This differs from Hill's legacy `dx*dy` calculation.
7. Runoff is `1000 * discharge / area`, in kg m⁻² s⁻¹.
8. Output is NetCDF3 64-bit offset, unlimited `time`, `runoff(time,y,x)` float64,
   native-dtype `area`, `lat`, `lon`, and integer `x`/`y` coordinates. There are no
   `_FillValue` attributes. Time is float64 Gregorian days since 1950-01-01.

Matching missing discharge is not silently changed to zero: it propagates as in
legacy calculations, and the comparison reports invalid values as a failure.

## Validation status

The synthetic regression tested the supplied original code against this writer:
100 pour-point cases, exact runoff and variable metadata for 367 daily records,
calendar-year and multi-file selection, and missing-year handling. The independent
comparison detected deliberate runoff, time, area, attribute, noncoastal, and
missing-value changes. Those tests simulated xESMF weights, so they establish
refactor arithmetic and file-format agreement, **not real ESMF mapping accuracy**.

The user subsequently ran the actual GloFAS 1993 comparison on the ARC11k
configuration and reported no differences from the legacy output on October 6,
2026. This adds actual-data regression evidence to the synthetic tests above.
Retain the successful local JSON report in Git. This validates reproduction of
the tested forcing; other grids or source versions need separate checks.

Comparison defaults are `rtol=0`, `atol=0`. Passing requires equal runoff,
matching time/coordinates/static arrays, matching variable attributes and file
structure, no invalid values, and no runoff outside interior coastal cells.
Global provenance attributes are ignored. Exit codes: 0 pass, 1 mismatch,
2 input/usage error. The comparison needs only NumPy and netCDF4.
