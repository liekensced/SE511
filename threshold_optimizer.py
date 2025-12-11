
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
print(f"Pixels: total={total_px}")
print(f"  Vegetation: {get_pct(veg0):.1f}% → {get_pct(veg_mask):.1f}% ({veg_mask.sum()} px)")
print(f"  Built-up:   {get_pct(built0):.1f}% → {get_pct(built_mask):.1f}% ({built_mask.sum()} px)")
print(f"  Water:      {get_pct(water0):.1f}% → {get_pct(water_mask):.1f}% ({water_mask.sum()} px)")


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

import itertools

def compute_f1_score(y_true, y_pred, average='macro'):
    """Compute F1 score without sklearn dependency."""
    classes = np.unique(np.concatenate([y_true, y_pred]))
    f1_scores = []
    
    for cls in classes:
        tp = np.sum((y_pred == cls) & (y_true == cls))
        fp = np.sum((y_pred == cls) & (y_true != cls))
        fn = np.sum((y_pred != cls) & (y_true == cls))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0
        f1_scores.append(f1)
    
    if average == 'macro':
        return np.mean(f1_scores) if f1_scores else 0
    else:
        return np.mean(f1_scores) if f1_scores else 0

def optimize_thresholds(indices_list, wal_classification, verbose=True, print_every=10):
    """Optimize thresholds across multiple products and return average F1 score.
    
    Args:
        indices_list: List of (ndvi, ndbi, ndwi) tuples from multiple products
        wal_classification: Reference classification mask
        verbose: Print progress
        print_every: Print every N iterations
    
    Returns:
        best_thr: Tuple of (ndvi_thr, ndbi_thr, ndwi_thr)
        best_avg_score: Best average F1 score
        history: Dict with optimization history for plotting
    """
    # Define search ranges for each threshold
    ndvi_range = np.arange(0.2, 0.5, 0.05)
    ndbi_range = np.arange(-0.1, 0.3, 0.05)
    ndwi_range = np.arange(-0.1, 0.1, 0.05)

    best_avg_score = -1
    best_thr = None
    iteration = 0
    
    # Track optimization history
    history = {
        'iterations': [],
        'f1_scores': [],
        'ndvi_thresholds': [],
        'ndbi_thresholds': [],
        'ndwi_thresholds': []
    }

    # Flatten reference mask for comparison
    ref_mask = wal_classification.flatten()
    ref_mask = ref_mask[(ref_mask == 1) | (ref_mask == 2) | (ref_mask == 3)]  # Only valid classes

    for ndvi_thr, ndbi_thr, ndwi_thr in itertools.product(ndvi_range, ndbi_range, ndwi_range):
        # Calculate F1 score for each product and average them
        f1_scores = []
        
        for ndvi, ndbi, ndwi in indices_list:
            veg_mask, built_mask, water_mask, *_ = compute_masks(ndvi, ndbi, ndwi, ndvi_thr, ndbi_thr, ndwi_thr)
            # Build predicted mask: 1=veg, 2=built, 3=water, 0=other
            pred_mask = np.zeros_like(wal_classification)
            pred_mask[veg_mask] = 1
            pred_mask[built_mask] = 2
            pred_mask[water_mask] = 3
            pred_mask = pred_mask.flatten()
            pred_mask = pred_mask[(wal_classification.flatten() == 1) | (wal_classification.flatten() == 2) | (wal_classification.flatten() == 3)]

            # Calculate F1 score (macro average)
            score = compute_f1_score(ref_mask, pred_mask, average='macro')
            f1_scores.append(score)
        
        # Average F1 across all products
        avg_score = np.mean(f1_scores)
        
        # Track history
        history['iterations'].append(iteration)
        history['f1_scores'].append(avg_score)
        history['ndvi_thresholds'].append(ndvi_thr)
        history['ndbi_thresholds'].append(ndbi_thr)
        history['ndwi_thresholds'].append(ndwi_thr)
        
        if avg_score > best_avg_score:
            best_avg_score = avg_score
            best_thr = (ndvi_thr, ndbi_thr, ndwi_thr)
        if verbose and (iteration % print_every == 0):
            print(f"[{iteration}] ndvi_thr={ndvi_thr:.2f}, ndbi_thr={ndbi_thr:.2f}, ndwi_thr={ndwi_thr:.2f} => Avg F1={avg_score:.4f}")
        iteration += 1

    print(f"\nBest thresholds: NDVI={best_thr[0]:.2f}, NDBI={best_thr[1]:.2f}, NDWI={best_thr[2]:.2f} (Avg F1={best_avg_score:.4f})")
    return best_thr, best_avg_score, history


if len(product_indices_list) > 0:
    best_thr, best_score, history = optimize_thresholds(product_indices_list, wal_classification)
else:
    print("No products loaded. Cannot optimize thresholds.")


# === Visualization of optimization process ===
if len(product_indices_list) > 0:
    import pandas as pd
    
    # Convert history to DataFrame for easier manipulation
    df_history = pd.DataFrame(history)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Threshold Optimization Process', fontsize=16, fontweight='bold')
    
    # 1. F1 Score over iterations
    ax = axes[0, 0]
    ax.plot(df_history['iterations'], df_history['f1_scores'], 'b-', linewidth=1.5, alpha=0.7)
    ax.scatter([df_history.loc[df_history['f1_scores'].idxmax(), 'iterations']], 
               [df_history['f1_scores'].max()], 
               color='red', s=100, zorder=5, label=f'Best F1={df_history["f1_scores"].max():.4f}')
    ax.set_xlabel('Iteration', fontsize=11)
    ax.set_ylabel('Average F1 Score', fontsize=11)
    ax.set_title('F1 Score Convergence')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # 2. NDVI Threshold Search
    ax = axes[0, 1]
    scatter = ax.scatter(df_history['ndvi_thresholds'], df_history['f1_scores'], 
                        c=df_history['f1_scores'], cmap='RdYlGn', s=40, alpha=0.6, vmin=df_history['f1_scores'].min(), vmax=df_history['f1_scores'].max())
    best_idx = df_history['f1_scores'].idxmax()
    ax.scatter(df_history.loc[best_idx, 'ndvi_thresholds'], df_history.loc[best_idx, 'f1_scores'], 
              color='red', s=200, marker='*', zorder=5, label=f'Best: {df_history.loc[best_idx, "ndvi_thresholds"]:.2f}')
    ax.set_xlabel('NDVI Threshold', fontsize=10)
    ax.set_ylabel('Average F1 Score', fontsize=10)
    ax.set_title('NDVI Threshold Impact', fontsize=11)
    ax.legend(fontsize=9)
    cbar = plt.colorbar(scatter, ax=ax, label='F1 Score')
    cbar.ax.tick_params(labelsize=9)
    
    # 3. NDBI Threshold Search
    ax = axes[1, 0]
    scatter = ax.scatter(df_history['ndbi_thresholds'], df_history['f1_scores'], 
                        c=df_history['f1_scores'], cmap='RdYlGn', s=40, alpha=0.6, vmin=df_history['f1_scores'].min(), vmax=df_history['f1_scores'].max())
    best_idx = df_history['f1_scores'].idxmax()
    ax.scatter(df_history.loc[best_idx, 'ndbi_thresholds'], df_history.loc[best_idx, 'f1_scores'], 
              color='red', s=200, marker='*', zorder=5, label=f'Best: {df_history.loc[best_idx, "ndbi_thresholds"]:.2f}')
    ax.set_xlabel('NDBI Threshold', fontsize=11)
    ax.set_ylabel('Average F1 Score', fontsize=11)
    ax.set_title('NDBI Threshold Impact')
    ax.legend()
    plt.colorbar(scatter, ax=ax, label='F1 Score')
    
    # 4. NDWI Threshold Search
    ax = axes[1, 1]
    scatter = ax.scatter(df_history['ndwi_thresholds'], df_history['f1_scores'], 
                        c=df_history['f1_scores'], cmap='RdYlGn', s=40, alpha=0.6, vmin=df_history['f1_scores'].min(), vmax=df_history['f1_scores'].max())
    best_idx = df_history['f1_scores'].idxmax()
    ax.scatter(df_history.loc[best_idx, 'ndwi_thresholds'], df_history.loc[best_idx, 'f1_scores'], 
              color='red', s=200, marker='*', zorder=5, label=f'Best: {df_history.loc[best_idx, "ndwi_thresholds"]:.2f}')
    ax.set_xlabel('NDWI Threshold', fontsize=11)
    ax.set_ylabel('Average F1 Score', fontsize=11)
    ax.set_title('NDWI Threshold Impact')
    ax.legend()
    plt.colorbar(scatter, ax=ax, label='F1 Score')
    
    plt.tight_layout()
    plt.savefig('optimization_overview.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # === 3D Visualization: NDVI vs NDBI vs F1 Score ===
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    scatter = ax.scatter(df_history['ndvi_thresholds'], df_history['ndbi_thresholds'], df_history['f1_scores'],
                        c=df_history['f1_scores'], cmap='RdYlGn', s=50, alpha=0.7)
    
    best_idx = df_history['f1_scores'].idxmax()
    ax.scatter(df_history.loc[best_idx, 'ndvi_thresholds'], 
              df_history.loc[best_idx, 'ndbi_thresholds'], 
              df_history.loc[best_idx, 'f1_scores'],
              color='red', s=300, marker='*', zorder=5, label='Best Solution')
    
    ax.set_xlabel('NDVI Threshold', fontsize=11, fontweight='bold')
    ax.set_ylabel('NDBI Threshold', fontsize=11, fontweight='bold')
    ax.set_zlabel('Average F1 Score', fontsize=11, fontweight='bold')
    ax.set_title('3D Optimization Landscape (NDVI vs NDBI)', fontsize=13, fontweight='bold')
    ax.legend()
    
    cbar = plt.colorbar(scatter, ax=ax, pad=0.1, shrink=0.8)
    cbar.set_label('F1 Score', fontweight='bold')
    
    plt.savefig('optimization_3d_ndvi_ndbi.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # === 3D Visualization: NDVI vs NDWI vs F1 Score ===
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    scatter = ax.scatter(df_history['ndvi_thresholds'], df_history['ndwi_thresholds'], df_history['f1_scores'],
                        c=df_history['f1_scores'], cmap='RdYlGn', s=50, alpha=0.7)
    
    best_idx = df_history['f1_scores'].idxmax()
    ax.scatter(df_history.loc[best_idx, 'ndvi_thresholds'], 
              df_history.loc[best_idx, 'ndwi_thresholds'], 
              df_history.loc[best_idx, 'f1_scores'],
              color='red', s=300, marker='*', zorder=5, label='Best Solution')
    
    ax.set_xlabel('NDVI Threshold', fontsize=11, fontweight='bold')
    ax.set_ylabel('NDWI Threshold', fontsize=11, fontweight='bold')
    ax.set_zlabel('Average F1 Score', fontsize=11, fontweight='bold')
    ax.set_title('3D Optimization Landscape (NDVI vs NDWI)', fontsize=13, fontweight='bold')
    ax.legend()
    
    cbar = plt.colorbar(scatter, ax=ax, pad=0.1, shrink=0.8)
    cbar.set_label('F1 Score', fontweight='bold')
    
    plt.savefig('optimization_3d_ndvi_ndwi.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # === Heatmap: NDVI vs NDBI (with NDWI slices) ===
    # Find the best NDWI threshold
    best_ndwi = df_history.loc[df_history['f1_scores'].idxmax(), 'ndwi_thresholds']
    
    # Create a pivot table for heatmap (NDVI x NDBI for each NDWI value)
    for ndwi_val in sorted(df_history['ndwi_thresholds'].unique()):
        df_slice = df_history[df_history['ndwi_thresholds'] == ndwi_val]
        pivot = df_slice.pivot_table(values='f1_scores', 
                                     index='ndvi_thresholds', 
                                     columns='ndbi_thresholds', 
                                     aggfunc='mean')
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(pivot, annot=True, fmt='.4f', cmap='RdYlGn', center=df_history['f1_scores'].mean(),
                   cbar_kws={'label': 'Average F1 Score'}, ax=ax, linewidths=0.5)
        ax.set_title(f'F1 Score Landscape: NDVI vs NDBI (NDWI={ndwi_val:.2f})', fontsize=12, fontweight='bold')
        ax.set_xlabel('NDBI Threshold', fontsize=11)
        ax.set_ylabel('NDVI Threshold', fontsize=11)
        
        # Mark best solution if it matches this NDWI slice
        best_idx = df_history['f1_scores'].idxmax()
        if df_history.loc[best_idx, 'ndwi_thresholds'] == ndwi_val:
            ax.set_title(ax.get_title() + ' ⭐ (Contains Best Solution)', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'optimization_heatmap_ndvi_ndbi_ndwi_{ndwi_val:.2f}.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    print("\n✓ Optimization plots saved:")
    print("  - optimization_overview.png")
    print("  - optimization_3d_ndvi_ndbi.png")
    print("  - optimization_3d_ndvi_ndwi.png")
    print("  - optimization_heatmap_ndvi_ndbi_ndwi_*.png (multiple files)")