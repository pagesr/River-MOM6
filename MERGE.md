# Step 3: merge Hill and GloFAS

`merge_runoff.py` is a standalone adaptation of your `merge_runoff_remiv2.py`.
It requires only NumPy and netCDF4. It reads existing MOM6 forcing products;
it does not run either source writer or download anything.

## Source priority

Choose a **fixed spatial Hill coverage region** for the MOM6 grid:

```text
merged runoff = Hill runoff inside the region
merged runoff = GloFAS runoff outside the region
```

A Hill zero is valid discharge and replaces GloFAS inside the region. There is
no addition, blending, or daily nonzero-based source selection. Missing Hill
dates or nonfinite inputs fail; they are not silently treated as zero or filled
from GloFAS. This avoids adding both datasets for the same covered region.

The region must describe where Hill should replace GloFAS. A mask consisting
only of cells that receive Hill runoff is usually insufficient: GloFAS may
place the same river at another nearby coastal cell. Include the full coastal
region represented by Hill. The merge script cannot infer catchment coverage
from runoff alone.

## First test with your existing rectangle

Your script selects `j=156:267, i=416:533` using zero-based indices with exclusive
upper bounds. To preserve that spatial choice:

```bash
python write_hill_runoff.py 1994
python merge_runoff.py 1993 --hill-box 156 267 416 533
```

This assumes the 1993 Hill and GloFAS outputs already exist. The 1994 Hill water
year supplies September–December 1993 and January 1, 1994. The 1993 water year
supplies January–August 1993. A single Hill water-year file cannot cover a whole
calendar year. Default Hill discovery finds `goa_dis*.nc` in
`outputs/HILL-on-MOM6/`; `--hill-files` accepts explicit files or quoted globs.

The output is:

```text
outputs/MERGED/glofas_hill_1993_v4.1.nc
```

**Review the rectangle before adopting it as complete Hill coverage.** Your
earlier Hill comparison showed runoff at `j=153`, outside this legacy rectangle.
The writer reports the union of all nonzero Hill cells excluded by the region,
including their index ranges. GloFAS remains at those excluded cells. Expand the
box or use a region mask if your aim is to use all Hill coverage. Add
`--require-all-hill` to make any excluded nonzero Hill cells a failure, preserving
an existing destination if the check fails. This diagnostic tests the available
records; it does not establish that the region includes every potentially dry
Hill catchment.

## Coverage on another MOM6 grid

Neither the ARC11k dimensions nor its rectangle are defaults. Every run must
explicitly select `--hill-box` or `--hill-region-mask`.

A coverage NetCDF should contain `hill_region(y,x)` or `hill_region(j,i)`, with
1 for Hill and 0 for GloFAS. It must match the exact tracer grid and index order.
The script also embeds this mask as `hill_region` in the merged output so the
spatial decision can be inspected later.

```bash
python merge_runoff.py 1993 \
  --hill-region-mask RAW/GRID/hill_coverage.nc \
  --require-all-hill
```

Use `--region-var` if the mask variable has another name. For another grid,
configure `--hgrid`, `--mask`, `--mask-var`, `--hill-dir`, `--glofas-dir`, and
`--output-dir`. Relative paths resolve against `--root`.

The GloFAS embedded coordinates and area are checked against `--hgrid`. Hill
files have no embedded longitude/latitude, so shape checks alone cannot prove
that they came from this grid: generate both products with the same grid and
supply its matching wet mask and Hill coverage mask.

A GoA configuration using Hill alone needs no merge: use the Hill product.

## Dates and output format

Both inputs are daily fields. The merge matches **calendar dates**, preserving
the original GloFAS timestamps. This explicitly pairs Hill midnight with GloFAS
noon for the same date; the maximum clock difference is recorded and printed.
Duplicate dates, missing dates, unsupported non-Gregorian calendars, or runoff
on land fail. This is daily pairing, not interpolation or a time shift of data.

The default `--time-window legacy` reproduces your merge selection: January 1
through January 1 of the following year, inclusive. For daily records, 1993 has
366 output records; leap years have 367. The preceding December 31 GloFAS halo
is omitted. `--time-window calendar-year` omits the following January 1, producing
365/366 records. Use a separate `--output-dir` when experimenting with time windows.

Output is NetCDF4 with compression level 1, unlimited time, and the GloFAS
variable dtypes/attributes retained. `--no-compress` writes NetCDF3 64-bit offset.
The `v4.1` filename follows your merge naming; it does not change the GloFAS
system version of the input. Existing outputs require `--overwrite`. Writes
use a temporary file and replace the destination only after success.

## Area convention retained from your script

Hill uses the legacy native-precision `dx*dy` cell area. GloFAS uses the sum of
four supergrid `area` values. Your merge copies Hill **mass flux** unchanged and
retains GloFAS's embedded area; this implementation does the same.

If the two areas differ, integrating the copied Hill flux with the output area
will differ from integrating it with Hill's native area. The log prints both
values. There is no implicit area rescaling: introducing a common-area discharge
conversion would be a separate method change and would not reproduce the legacy
flux values. Source replacement also does not imply conservation of the original
GloFAS domain total, since Hill replaces a selected portion of that dataset.

## Compare with a retained legacy merge

Place the separately generated legacy file at:

```text
outputs/LEGACY/MERGED/glofas_hill_1993_v4.1.nc
```

Then run:

```bash
python QC/compare_merged_with_legacy.py --year 1993 \
  --report outputs/merged_1993_comparison.json
```

Explicit legacy/new paths and `--mask` support another grid or filename. The
comparison requires matching decoded timestamps, exact static grid arrays,
flux units, runoff and nonzero locations, finite values, and no runoff on land.
Compression, container format, provenance, time epoch encoding, and the added
coverage mask are not regression targets. Default tolerances are zero; exit
codes are 0 pass, 1 mismatch, 2 input/usage error.

Synthetic tests reproduced your supplied merge's runoff exactly with aligned
input clocks. They separately checked midnight/noon date pairing, zero Hill
priority, box/mask selection, both output formats, both time windows, atomic
failure, missing water-year records, and deliberate comparator failures. The
actual ARC11k legacy merge comparison subsequently passed on October 6, 2026:
366 records, 8,563 distinct nonzero cells in both files, zero differing values,
zero support mismatches, zero RMSE and maximum absolute difference, and zero
maximum daily discharge difference. There were no structural issues, invalid
values, or runoff on land. The correct reference was
`outputs/LEGACY/MERGED/glofas_hill_1993_v4.1.nc`; both files were NetCDF4 and the
comparison used `rtol=0`, `atol=0`.

This validates the actual legacy rectangle and 1993 inputs. A changed coverage
region or daily time policy can intentionally differ from that benchmark. Keep
the successful local JSON report; an unmerged GloFAS file is not a merged reference.

## ARC11k post-processing boundary

The retained legacy merged reference and exact 1993 pass are **before manual
Ob/Mezen corrections**. The Ob mouth is just outside this grid; the separate
`apply_ob_mezen_corrections_2023_2024.py` step places the observed climatologies
at the supplied ARC11k cells. It reads `outputs/MERGED/` and writes
`outputs/MERGED-CORRECTED/`. Keep this optional correction separate from the
validated merge, and see the README for observation files and invocation.
