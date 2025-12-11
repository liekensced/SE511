# Import required libraries
from terracatalogueclient import Catalogue
import datetime as dt
import rasterio
import matplotlib.pyplot as plt
import numpy as np
from skimage import exposure
from pyproj import Transformer, CRS
from shapely.geometry import Polygon
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, accuracy_score
import time
import joblib

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

# Also keep a date-sorted list for model training
no_cloud_products_sorted = sorted(no_cloud_products, key=lambda p: p.properties['date'])
print(f"Date-sorted products available: {len(no_cloud_products_sorted)}")

# Load and read the downloaded band files with AOI extraction
from pathlib import Path
import rasterio
from rasterio.windows import Window
from pyproj import Transformer, CRS as ProjCRS
import numpy as np

download_dir = Path("sentinel_data_subset")

if download_dir == Path("sentinel_data_subset_small"):
    print("Using small download directory for testing!")

def load_product_files(product_name, verbose=True):
    # Find files matching this product
    band_files = {
        'B02': None,  # Blue
        'B03': None,  # Green
        'B04': None,  # Red
        'B08': None,  # NIR
        'B11': None,  # SWIR
    }
    
    # Search for band files
    # for file in download_dir.glob(f"urn_eop_VITO_TERRASCOPE_S2_TOC_V2_{product_name.split('_')[0]}_{product_name.split('_')[1]}*"):
    for file in download_dir.glob(f"{product_name.split('_')[0]}_{product_name.split('_')[1]}*"):
        for band in band_files.keys():
            if f"{band}_" in file.name:
                band_files[band] = file
                break
    
    # Check if all required bands are present
    missing = [k for k, v in band_files.items() if v is None]
    if missing:
        print(f"Missing band files: {missing}")
        print("Please download the required bands first.")
    else:
        print("All required bands found:")
        # for band, path in band_files.items():
        #     print(f"  {band}: {path.name}")
        
        # The downloader already extracted the 6x6 km box, so read the entire file
        # No need to define a window - just read all data
        with rasterio.open(band_files['B04']) as dataset:
            if verbose:
                print(f"Image bounds: {dataset.bounds}")
                print(f"Image CRS: {dataset.crs}")
                print(f"Image size: {dataset.width} x {dataset.height} pixels")
                print(f"Pixel resolution: {dataset.res}")
            
            profile = dataset.profile
        
        # Read the entire bands (no window needed - downloader already extracted the area)
        with rasterio.open(band_files['B02']) as src:
            blue = src.read(1).astype(float)
            
        with rasterio.open(band_files['B03']) as src:
            green = src.read(1).astype(float)
            
        with rasterio.open(band_files['B04']) as src:
            red = src.read(1).astype(float)
            
        with rasterio.open(band_files['B08']) as src:
            nir = src.read(1).astype(float)
            
        # For SWIR (20m resolution), read entire band and resample
        with rasterio.open(band_files['B11']) as src:
            swir_20m = src.read(1).astype(float)
            
            # Resample SWIR to 10m to match other bands
            from scipy.ndimage import zoom
            scale_factor = 2.0
            swir = zoom(swir_20m, scale_factor, order=1)
            
            # Ensure same shape as other bands
            if swir.shape != red.shape:
                swir = swir[:red.shape[0], :red.shape[1]]
        
        if verbose: 
            print(f"\nAOI extracted successfully. Shape: {red.shape}")
            # Calculate actual coverage from pixel size
            pixel_size_m = dataset.res[0]  # meters per pixel
            width_km = (red.shape[1] * pixel_size_m) / 1000
            height_km = (red.shape[0] * pixel_size_m) / 1000
            print(f"AOI covers approximately {width_km:.1f} km x {height_km:.1f} km")
        return [blue, green, red, nir, swir]

# Compute NDVI, NDBI, and NDWI indices
import numpy as np

# Avoid division by zero
def safe_divide(a, b):
    """Safely divide two arrays, returning 0 where denominator is 0."""
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.where(b != 0, a / b, 0)
    return np.nan_to_num(result, nan=0)

def compute_indices(red, nir, swir, green, verbose=True):
    # Compute NDVI: (NIR - Red) / (NIR + Red)
    ndvi = safe_divide(nir - red, nir + red)

    # Ensure all arrays have the same shape before computing NDBI
    min_rows = min(swir.shape[0], nir.shape[0])
    min_cols = min(swir.shape[1], nir.shape[1])
    swir_crop = swir[:min_rows, :min_cols]
    nir_crop = nir[:min_rows, :min_cols]

    # Compute NDBI: (SWIR - NIR) / (SWIR + NIR)
    ndbi = safe_divide(swir_crop - nir_crop, swir_crop + nir_crop)

    # Compute NDWI: (Green - NIR) / (Green + NIR)
    ndwi = safe_divide(green - nir, green + nir)

    if verbose:
        print("Indices computed successfully!")
        print(f"NDVI range: [{ndvi.min():.3f}, {ndvi.max():.3f}]")
        print(f"NDBI range: [{ndbi.min():.3f}, {ndbi.max():.3f}]")
        print(f"NDWI range: [{ndwi.min():.3f}, {ndwi.max():.3f}]")
    return [ndvi, ndbi, ndwi]



# Load the first 10 products for more robust optimization
product_indices_list = []  # List to store (ndvi, ndbi, ndwi) tuples for first 10 products
if len(no_cloud_products) > 0:
    num_products = min(10, len(no_cloud_products))
    print(f"Loading first {num_products} products for optimization...")
    for i in range(num_products):
        product = no_cloud_products[i]
        product_name = product.title
        print(f"  [{i+1}/{num_products}] {product_name}")
        [blue, green, red, nir, swir] = load_product_files(product_name, verbose=False)
        [ndvi, ndbi, ndwi] = compute_indices(red, nir, swir, green, verbose=False)
        product_indices_list.append((ndvi, ndbi, ndwi))
    print(f"Loaded {len(product_indices_list)} products successfully.")
else:
    print("No products available to process.")



[ndvi, ndbi, ndwi] = compute_indices(red, nir, swir, green)


# Create binary masks for vegetation, built-up, and water
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from skimage.morphology import remove_small_objects, disk, binary_opening, binary_closing

# Thresholds 
ndvi_thr = 0.45   # vegetation
ndbi_thr = -0.25   # built-up
ndwi_thr = 0.0  # water


def compute_masks(ndvi, ndbi, ndwi, ndvi_thr=ndvi_thr, ndbi_thr=ndbi_thr, ndwi_thr=ndwi_thr):
    # Calculate dimensions to ensure all arrays match
    min_rows = min(ndvi.shape[0], ndbi.shape[0], ndwi.shape[0])
    min_cols = min(ndvi.shape[1], ndbi.shape[1], ndwi.shape[1])
    
    # Crop all arrays to match dimensions
    ndvi_crop = ndvi[:min_rows, :min_cols]
    ndbi_crop = ndbi[:min_rows, :min_cols]
    ndwi_crop = ndwi[:min_rows, :min_cols]

    # Initial thresholded masks
    veg0 = (ndvi_crop > ndvi_thr)
    built0 = (ndbi_crop > ndbi_thr)
    water0 = (ndwi_crop > ndwi_thr)

    # Simple cleaning to remove small speckles and smooth edges
    def clean(mask: np.ndarray, min_size: int = 5, radius: int = 1) -> np.ndarray:
        mask = mask.astype(bool)
        mask = remove_small_objects(mask, min_size=min_size)
        if radius and radius > 0:
            se = disk(radius)
            mask = binary_opening(mask, se)
            mask = binary_closing(mask, se)
        return mask

    # Enforce mutual exclusivity with priority: Water > Built-up > Vegetation
    water_mask = clean(water0, radius=1)
    built_mask = clean(built0 & ~water_mask, radius=1)
    veg_mask = clean(veg0 & ~water_mask & ~built_mask, radius=1)
    return veg_mask, built_mask, water_mask, ndvi_crop, veg0, built0, water0
[veg_mask, built_mask, water_mask, ndvi_crop, veg0, built0, water0] = compute_masks(ndvi, ndbi, ndwi)

# Get mask dimensions for visualization
min_rows, min_cols = ndvi_crop.shape

# Report coverage
total_px = int(ndvi_crop.size)
get_pct = lambda m: 100.0 * float(m.sum()) / total_px if total_px > 0 else 0.0

# Load WAL_OCS reference masks
from pathlib import Path
import numpy as np

mask_dir = Path("walous/masks")

# Check if masks exist
if mask_dir.exists():
    # Load the saved masks
    wal_classification = np.load(mask_dir / "wal_classification.npy")
    wal_water_mask = np.load(mask_dir / "wal_water_mask.npy")
    wal_vegetation_mask = np.load(mask_dir / "wal_vegetation_mask.npy")
    wal_artificial_mask = np.load(mask_dir / "wal_artificial_mask.npy")
    
    print("✓ WAL_OCS reference masks loaded successfully!")
    print(f"  Shape: {wal_classification.shape}")
    print(f"  Coverage:")
    total_px = wal_classification.size
    get_pct = lambda val: 100.0 * float((wal_classification == val).sum()) / total_px
    print(f"    Vegetation: {get_pct(1):.2f}%")
    print(f"    Artificial: {get_pct(2):.2f}%")
    print(f"    Water:      {get_pct(3):.2f}%")
    print(f"    Unclassified: {get_pct(0):.2f}%")
else:
    print("⚠ Mask directory not found. Run walous/test.py first to generate masks.")
    wal_classification = None


# -----------------------------------------------------------------------------
# RandomForest classifier on 5 input bands -> 3 classes (Vegetation, Built-up, Water)
# Uses first 10 date-sorted products for training
# -----------------------------------------------------------------------------

if wal_classification is None:
    raise RuntimeError("Reference masks not found; cannot train RandomForest.")

# Limit to first 10 images sorted by date
train_products = no_cloud_products_sorted[:10]
print(f"Training on {len(train_products)} products (date-sorted, first 10)")
print(train_products)
def build_dataset(products, label_mask, max_samples_per_product=None, verbose=True):
    """Stack per-pixel 5-band features and labels across products."""
    X_parts, y_parts = [], []
    total_raw = 0
    total_valid = 0
    for idx, product in enumerate(products):
        product_name = product.title
        if verbose:
            print(f"[{idx+1}/{len(products)}] {product_name}")
        [blue, green, red, nir, swir] = load_product_files(product_name, verbose=False)

        # Align bands and labels to common shape
        min_rows = min(label_mask.shape[0], blue.shape[0], green.shape[0], red.shape[0], nir.shape[0], swir.shape[0])
        min_cols = min(label_mask.shape[1], blue.shape[1], green.shape[1], red.shape[1], nir.shape[1], swir.shape[1])

        label_crop = label_mask[:min_rows, :min_cols].astype(np.uint8)
        feature_stack = np.stack([
            blue[:min_rows, :min_cols],
            green[:min_rows, :min_cols],
            red[:min_rows, :min_cols],
            nir[:min_rows, :min_cols],
            swir[:min_rows, :min_cols],
        ], axis=-1)

        valid = label_crop > 0  # ignore unclassified
        total_raw += int(valid.size)
        total_valid += int(valid.sum())
        X = feature_stack.reshape(-1, 5)[valid.ravel()]
        y = label_crop.ravel()[valid.ravel()] - 1  # map to 0,1,2

        if max_samples_per_product is not None and X.shape[0] > max_samples_per_product:
            idx_sel = np.random.choice(X.shape[0], max_samples_per_product, replace=False)
            X = X[idx_sel]
            y = y[idx_sel]

        X_parts.append(X)
        y_parts.append(y)

        if verbose:
            cls_counts = {c: int((y == c).sum()) for c in np.unique(y)}
            print(f"  Samples: {X.shape[0]} | class counts: {cls_counts}")

    if not X_parts:
        return None, None, {"raw_pixels": 0, "valid_pixels": 0}
    return (
        np.concatenate(X_parts, axis=0),
        np.concatenate(y_parts, axis=0),
        {"raw_pixels": total_raw, "valid_pixels": total_valid},
    )


X_all, y_all, ds_stats = build_dataset(train_products, wal_classification, max_samples_per_product=None, verbose=True)

if X_all is None or y_all is None or X_all.size == 0:
    raise RuntimeError("No training data assembled; check product list and masks alignment.")

print(f"Total pixels (after cropping): {ds_stats['raw_pixels']:,}")
print(f"Total labeled (used for training): {ds_stats['valid_pixels']:,}")
print(f"Total training samples (features): {X_all.shape[0]:,}")
class_names = ['Vegetation', 'Built-up', 'Water']
for cls_idx, cls_name in enumerate(class_names):
    print(f"  {cls_name}: {(y_all == cls_idx).sum():,} pixels")

# Train/validation split
X_train, X_val, y_train, y_val = train_test_split(
    X_all,
    y_all,
    test_size=0.2,
    random_state=42,
    stratify=y_all,
)
print("start training RandomForest classifier...")
tic = time.perf_counter()
rf = RandomForestClassifier(
    n_estimators=150,
    max_depth=15,
    n_jobs=-1,
    random_state=42,
    verbose=2, 
)
rf.fit(X_train, y_train)
toc = time.perf_counter()
print(f"Elapsed: {toc - tic:.3f} s")

y_pred = rf.predict(X_val)
acc = accuracy_score(y_val, y_pred)
f1 = f1_score(y_val, y_pred, average='macro')
print(f"Validation accuracy: {acc:.3f}")
print(f"Validation macro-F1: {f1:.3f}")
print("\nDetailed report:")
print(classification_report(y_val, y_pred, target_names=class_names))


# Apply the model to the first training product and visualize coverage
from matplotlib.colors import ListedColormap

demo_product = train_products[0]
[blue, green, red, nir, swir] = load_product_files(demo_product.title, verbose=False)
min_rows = min(wal_classification.shape[0], blue.shape[0], green.shape[0], red.shape[0], nir.shape[0], swir.shape[0])
min_cols = min(wal_classification.shape[1], blue.shape[1], green.shape[1], red.shape[1], nir.shape[1], swir.shape[1])

feature_stack = np.stack([
    blue[:min_rows, :min_cols],
    green[:min_rows, :min_cols],
    red[:min_rows, :min_cols],
    nir[:min_rows, :min_cols],
    swir[:min_rows, :min_cols],
], axis=-1)

pred_map = rf.predict(feature_stack.reshape(-1, 5)).reshape(min_rows, min_cols)
pred_map_classes = pred_map + 1  # back to 1,2,3 codes

total_px = pred_map_classes.size
for cls_code, cls_name in zip([1, 2, 3], class_names):
    pct = 100.0 * float((pred_map_classes == cls_code).sum()) / total_px
    print(f"{cls_name} coverage: {pct:.2f}%")

# Quick visualization
cmap = ListedColormap(['black', 'green', 'orange', 'blue'])
plt.figure(figsize=(6, 6))
im = plt.imshow(pred_map_classes, cmap=cmap, vmin=0, vmax=3)
plt.title(f"RF Prediction - {demo_product.title}")
plt.axis('off')
cbar = plt.colorbar(im, ticks=[0, 1, 2, 3], fraction=0.046)
cbar.ax.set_yticklabels(['Unclassified', 'Vegetation', 'Built-up', 'Water'])
plt.tight_layout()
plt.show()

# Save the model
joblib.dump(rf, 'random_forest_model.pkl')

# Later, load it:
rf = joblib.load('random_forest_model.pkl')

