import cdsapi
import zipfile
import os

client = cdsapi.Client()

for y in range(2025, 2026):
    print(f"Processing {y}")
    dataset = "cems-glofas-historical"
    request = {
        "system_version": ["version_4_0"],
        "hydrological_model": ["lisflood"],
        "product_type": ["consolidated"],
        "variable": ["river_discharge_in_the_last_24_hours"],
        "hyear": [str(y)],
        "hmonth": [f"{m:02d}" for m in range(1, 13)],
        "hday": [f"{d:02d}" for d in range(1, 32)],
        "data_format": "netcdf",
        "download_format": "zip",
        "area": [90, -180, 40.025, 180]
    }

    zip_filename = f"glofas_{y}.zip"
    nc_filename = f"glofas_{y}.nc"

    # Download ZIP
    client.retrieve(dataset, request).download(zip_filename)

    # Unzip and rename the .nc file
    with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
        for name in zip_ref.namelist():
            if name.endswith('.nc'):
                zip_ref.extract(name)
                os.rename(name, nc_filename)

    # Optionally remove the zip file
    os.remove(zip_filename)

    print(f"✅ Done for {y}: {nc_filename}")

