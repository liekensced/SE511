# Import required libraries
from terracatalogueclient import Catalogue
import datetime as dt
import rasterio
import matplotlib.pyplot as plt
import numpy as np
from skimage import exposure
from pyproj import Transformer, CRS
from shapely.geometry import Polygon

# Initialize the Terrascope catalogue
catalogue = Catalogue()

# Define the date range for 2017-2019
startDate = dt.date(2017, 1, 1)
endDate = dt.date(2019, 12, 31)

# Define the coordinates from the task (WGS84)
lat = 50.379123
lon = 4.392829

# Define a bounding box (adjust size as needed for "appropriate size")
# (Small because need to download entire file anyway and to ensure the centre coordinate is in it)
bbox = [lon - 0.002, lat - 0.002, lon + 0.002, lat + 0.002]

# Define cloud cover threshold for filtering
cloudcover_threshold = 1

print(f"Searching for products from {startDate} to {endDate}")
print(f"Area of Interest: lat={lat}, lon={lon}")
print(f"Bounding box: {bbox}")

# Get all products in the date range and location
products = list(catalogue.get_products(
    "urn:eop:VITO:TERRASCOPE_S2_TOC_V2",
    start=startDate,
    end=endDate,
    bbox=bbox
))

print(f"\nTotal products found: {len(products)}")

# Filter for images with no cloud cover (cloudCover == 0)
# Note: You may need to adjust this threshold if no images with 0% cloud cover exist
no_cloud_products = [
    product for product in products 
    if product.properties["productInformation"]["cloudCover"] <= cloudcover_threshold
]

# Sort products by cloud cover percentage (ascending)
no_cloud_products.sort(key=lambda p: p.properties["productInformation"]["cloudCover"])

print(f"Products with < {cloudcover_threshold}% cloud cover: {len(no_cloud_products)}")

# Display the cloud-free images
if len(no_cloud_products) > 0:
    print("\nCloud-free products (sorted by cloud cover %):")
    for i, product in enumerate(no_cloud_products, 1):
        print(f"{i}. {product.title}: {product.properties['productInformation']['cloudCover']}% clouds")
else:
    print("\nNo products found.")

# Obtain an access token using OIDC password grant
import os
import requests
from pathlib import Path

# Load credentials from .env file if it exists
env_path = Path(".env")
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ[key.strip()] = value.strip()

token_url = "https://sso.terrascope.be/auth/realms/terrascope/protocol/openid-connect/token"

# Get credentials from environment variables
USERNAME = os.getenv("TERRASCOPE_USERNAME")
PASSWORD = os.getenv("TERRASCOPE_PASSWORD")

if not USERNAME or not PASSWORD:
    print("Error: Set TERRASCOPE_USERNAME and TERRASCOPE_PASSWORD in .env file or environment variables.")
    ACCESS_TOKEN = None
else:
    data = {
        "grant_type": "password",
        "client_id": "public",
        "username": USERNAME,
        "password": PASSWORD,
    }

    try:
        resp = requests.post(token_url, data=data, timeout=30)
        resp.raise_for_status()
        token_payload = resp.json()
        ACCESS_TOKEN = token_payload["access_token"]
        print("Token acquired. Expires in:", token_payload.get("expires_in"), "seconds")
    except Exception as e:
        ACCESS_TOKEN = None
        print("Failed to obtain token:", e)


import requests
import rasterio
from rasterio.windows import from_bounds, Window
from pathlib import Path
import re
from pyproj import Transformer, CRS as ProjCRS

crsWGS84 = ProjCRS.from_epsg(4326)

transformer = Transformer.from_crs("epsg:32631", "epsg:4326", always_xy=True)
easting, northing = transformer.transform(lon, lat)
print(f"Exact WGS84: {easting=:.6f}, {northing=:.6f}")

padding_meters = 6000                     # change freely (6000 → 12×12 km)

west  = easting  - padding_meters
east  = easting  + padding_meters
south = northing - padding_meters
north = northing + padding_meters

print(f"Window: {west:.0f}E – {east:.0f}E, {south:.0f}N – {north:.0f}N "
    f"({2*padding_meters/1000:.0f}×{2*padding_meters/1000:.0f} km)")


# ------------------------------------------------------------------
# 3. Download only a centered window around the point (in WGS84)
# ------------------------------------------------------------------
if ACCESS_TOKEN is None or len(no_cloud_products) == 0:
    raise RuntimeError("Check ACCESS_TOKEN and no_cloud_products first!")

download_dir = Path("sentinel_data_subset")
download_dir.mkdir(exist_ok=True)

# Bands we want at 10m and 20m resolution
required_bands = ['B02', 'B03', 'B04', 'B08', 'B11']  # B11 is 20m

downloaded = 0

# Define a 6x6 km box in WGS84
# At this latitude (50°N):
# - 1° latitude ≈ 111 km
# - 1° longitude ≈ 111 × cos(50°) ≈ 71 km
import math
padding_km = 3  # ±3 km → 6×6 km square
padding_lat = padding_km / 111.0  # degrees latitude
padding_lon = padding_km / (111.0 * math.cos(math.radians(lat)))  # degrees longitude

west  = lon - padding_lon
east  = lon + padding_lon
south = lat - padding_lat
north = lat + padding_lat

print(f"\nTarget window in WGS84: {west:.6f}–{east:.6f} E, {south:.6f}–{north:.6f} N "
      f"(~6x6 km centered on point)")

for idx, product in enumerate(no_cloud_products, 1):
    print(f"\n[{idx}/{len(no_cloud_products)}] Processing: {product.title}")

    with rasterio.Env(GDAL_HTTP_HEADERS=f"Authorization: Bearer {ACCESS_TOKEN}"):
        for data_url in product.data:
            url = data_url.href
            filename = url.split('/')[-1]

            # Filter only desired bands
            if not any(band in filename for band in required_bands):
                continue

            out_path = download_dir / filename.replace(".jp2", ".tif")  # better extension

            if out_path.exists():
                print(f"   [Skip] {filename}")
                continue

            print(f"   Downloading subset: {filename} ...", end=" ", flush=True)

            try:
                with rasterio.open(f"/vsicurl/{url}") as src:
                    crs_src = src.crs
                    
                    if crs_src is None:
                        print("No CRS → skip")
                        continue

                    # Transform WGS84 box corners to the dataset's CRS
                    from pyproj import Transformer as ProjTransformer
                    transformer_to_src = ProjTransformer.from_crs("epsg:4326", crs_src, always_xy=True)
                    
                    # Four corners of the WGS84 box
                    wgs84_xs = [west, east, east, west]
                    wgs84_ys = [north, north, south, south]
                    
                    # Transform to dataset CRS
                    src_coords = [transformer_to_src.transform(x, y) for x, y in zip(wgs84_xs, wgs84_ys)]
                    xs = [c[0] for c in src_coords]
                    ys = [c[1] for c in src_coords]

                    # Transform to pixel coordinates
                    pixel_coords = [~src.transform * (x, y)
                                    for x, y in zip(xs, ys)]
                    
                    col_min = min(c[0] for c in pixel_coords)
                    col_max = max(c[0] for c in pixel_coords)
                    row_min = min(c[1] for c in pixel_coords)
                    row_max = max(c[1] for c in pixel_coords)

                    # Build window in pixel space
                    window = rasterio.windows.Window(
                        col_off=max(0, int(col_min)),
                        row_off=max(0, int(row_min)),
                        width=int(col_max - col_min) + 1,
                        height=int(row_max - row_min) + 1
                    )

                    # Clip window to actual raster bounds
                    window = window.intersection(rasterio.windows.Window(0, 0, src.width, src.height))

                    if window.width <= 0 or window.height <= 0:
                        print("No overlap with tile → skip")
                        continue

                    # Read data
                    arr = src.read(1, window=window)

                    # Create new transform for subset
                    new_transform = rasterio.windows.transform(window, src.transform)

                    # Update metadata
                    out_meta = src.meta.copy()
                    out_meta.update({
                        "driver": "GTiff",
                        "height": window.height,
                        "width": window.width,
                        "transform": new_transform,
                        "crs": crs_src,
                        "compress": "deflate"
                    })

                    # Write file
                    with rasterio.open(out_path, "w", **out_meta) as dst:
                        dst.write(arr, 1)

                print("Success")
                downloaded += 1

            except Exception as e:
                print(f"Failed → {type(e).__name__}: {e}")

print(f"\nDone! Downloaded {downloaded} subsetted bands to:")
print(f"   {download_dir.resolve()}")
