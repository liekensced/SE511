with open('downloader.py') as f:
    exec(f.read())

# Load and read the downloaded band files with AOI extraction
from pathlib import Path
import rasterio
from rasterio.windows import Window
from pyproj import Transformer, CRS as ProjCRS
import numpy as np

download_dir = Path("sentinel_data_subset")

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
# Find the downloaded files for the first product
if len(no_cloud_products) > 0:
    product = no_cloud_products[0]
    product_name = product.title
    [blue, green, red, nir, swir] = load_product_files(product_name)
else:
    print("No products available to process.")


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

[ndvi, ndbi, ndwi] = compute_indices(red, nir, swir, green)

# Visualize the RGB image and computed indices
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Plot RGB image (True Color Composite)
# Normalize RGB values to 0-1 range for display
rgb = np.stack([red, green, blue], axis=-1)
# Apply percentile stretching for better visualization
p2, p98 = np.percentile(rgb, (2, 98))
rgb_stretched = np.clip((rgb - p2) / (p98 - p2), 0, 1)

axes[0, 0].imshow(rgb_stretched)
axes[0, 0].set_title('RGB - True Color')
axes[0, 0].axis('off')

# Plot NDVI
im1 = axes[0, 1].imshow(ndvi, cmap='RdYlGn', vmin=-1, vmax=1)
axes[0, 1].set_title('NDVI - Vegetation')
axes[0, 1].axis('off')
plt.colorbar(im1, ax=axes[0, 1], label='NDVI', fraction=0.046)

# Plot NDBI
im2 = axes[1, 0].imshow(ndbi, cmap='RdYlBu_r', vmin=-1, vmax=1)
axes[1, 0].set_title('NDBI (Built-up)')
axes[1, 0].axis('off')
plt.colorbar(im2, ax=axes[1, 0], label='NDBI', fraction=0.046)

# Plot NDWI
im3 = axes[1, 1].imshow(ndwi, cmap='Blues', vmin=-1, vmax=1)
axes[1, 1].set_title('NDWI (Water)')
axes[1, 1].axis('off')
plt.colorbar(im3, ax=axes[1, 1], label='NDWI', fraction=0.046)

plt.tight_layout()
plt.show()





# Create binary masks for vegetation, built-up, and water
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from skimage.morphology import remove_small_objects, disk, binary_opening, binary_closing

# Thresholds 
ndvi_thr = 0.4   # vegetation
ndbi_thr = 0.12   # built-up
ndwi_thr = 0.2  # water

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

    # Enforce mutual exclusivity with priority: Water > Vegetation > Built-up
    water_mask = clean(water0, radius=1)
    veg_mask = clean(veg0 & ~water_mask, radius=1)
    built_mask = clean(built0 & ~water_mask & ~veg_mask, radius=1)
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

# Visualization: RGB + combined classification mask
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# RGB (crop to match mask dimensions)
rgb_crop = np.stack([red[:min_rows, :min_cols], green[:min_rows, :min_cols], blue[:min_rows, :min_cols]], axis=-1)
p2, p98 = np.percentile(rgb_crop, (2, 98))
rgb_stretched = np.clip((rgb_crop - p2) / (p98 - p2), 0, 1)
axes[0].imshow(rgb_stretched)
axes[0].set_title('RGB - True Color')
axes[0].axis('off')

# Combined classification mask: 0=unclassified, 1=vegetation, 2=built-up, 3=water
classification = np.zeros_like(veg_mask, dtype=np.uint8)
classification[veg_mask] = 1
classification[built_mask] = 2
classification[water_mask] = 3

# Custom colormap: black, green, orange, blue
from matplotlib.colors import ListedColormap
colors = ['black', 'green', 'orange', 'blue']
cmap = ListedColormap(colors)

im = axes[1].imshow(classification, cmap=cmap, vmin=0, vmax=3)
axes[1].set_title('Classification Mask')
axes[1].axis('off')

# Add colorbar with labels
cbar = plt.colorbar(im, ax=axes[1], ticks=[0, 1, 2, 3], fraction=0.046)
cbar.ax.set_yticklabels(['Unclassified', 'Vegetation', 'Built-up', 'Water'])

plt.tight_layout()
plt.show()

# Sort products by date
no_cloud_products_sorted = sorted(no_cloud_products, key=lambda p: p.properties['date'])

print(f"Processing {len(no_cloud_products_sorted)} products sorted by date\n")

# Track previous masks for change detection
prev_classification = None
prev_rgb = None
prev_date = None
use_verbose = False

# Change detection options
ignore_unclassified = False  # Set to False to include changes to/from Unclassified

# Store change visualizations for carousel
change_plots = []

for idx, product in enumerate(no_cloud_products_sorted):
    product_name = product.title
    product_date = product.properties['date'].split('T')[0]
    
    print(f"\n{'='*60}")
    print(f"[{idx+1}/{len(no_cloud_products_sorted)}] {product_name}")
    print(f"Date: {product_date}")
    print('='*60)
    
    try:
        # Load and process product
        [blue, green, red, nir, swir] = load_product_files(product_name, verbose=use_verbose)
        [ndvi, ndbi, ndwi] = compute_indices(red, nir, swir, green, verbose=use_verbose)
        [veg_mask, built_mask, water_mask, ndvi_crop, veg0, built0, water0] = compute_masks(ndvi, ndbi, ndwi)
        
        # Create classification map
        classification = np.zeros_like(veg_mask, dtype=np.uint8)
        classification[veg_mask] = 1
        classification[built_mask] = 2
        classification[water_mask] = 3
        
        # Calculate coverage
        total_px = int(classification.size)
        get_pct = lambda val: 100.0 * float((classification == val).sum()) / total_px
        
        print(f"\nCoverage:")
        print(f"  Vegetation: {get_pct(1):.1f}%")
        print(f"  Built-up:   {get_pct(2):.1f}%")
        print(f"  Water:      {get_pct(3):.1f}%")
        print(f"  Unclassified: {get_pct(0):.1f}%")
        
        # Prepare RGB crop (stretched) matching mask dimensions
        min_rows, min_cols = classification.shape
        rgb_crop = np.stack([
            red[:min_rows, :min_cols],
            green[:min_rows, :min_cols],
            blue[:min_rows, :min_cols]
        ], axis=-1)
        p2, p98 = np.percentile(rgb_crop, (2, 98))
        rgb_stretched = np.clip((rgb_crop - p2) / (p98 - p2 + 1e-12), 0, 1)
        
        # Detect changes from previous date
        if prev_classification is not None:
            # Calculate pixel-wise changes
            if ignore_unclassified:
                # Only count changes between classified pixels (exclude unclassified transitions)
                classified_mask = (prev_classification > 0) & (classification > 0)
                changed_pixels = ((classification != prev_classification) & classified_mask).sum()
            else:
                changed_pixels = (classification != prev_classification).sum()
            
            change_pct = 100.0 * changed_pixels / total_px
            
            if change_pct > 0.5:  # Report if more than 0.5% changed
                print(f"\n⚠️ CHANGE DETECTED from {prev_date} to {product_date}:")
                print(f"   {change_pct:.1f}% of pixels changed ({changed_pixels:,} pixels)")
                if ignore_unclassified:
                    print(f"   (excluding changes to/from Unclassified)")
                
                # Detailed change breakdown
                change_summary = []
                for from_class in range(4):
                    for to_class in range(4):
                        if from_class != to_class:
                            # Skip changes involving unclassified if option is set
                            if ignore_unclassified and (from_class == 0 or to_class == 0):
                                continue
                            
                            mask = (prev_classification == from_class) & (classification == to_class)
                            count = mask.sum()
                            if count > 0:
                                class_names = ['Unclassified', 'Vegetation', 'Built-up', 'Water']
                                pct = 100.0 * count / total_px
                                print(f"   {class_names[from_class]} → {class_names[to_class]}: {pct:.2f}% ({count:,} px)")
                                change_summary.append(f"{class_names[from_class]} → {class_names[to_class]}: {pct:.2f}%")
                
                # Create change visualization
                if ignore_unclassified:
                    # Only show changes between classified pixels
                    classified_mask = (prev_classification > 0) & (classification > 0)
                    change_mask = ((classification != prev_classification) & classified_mask).astype(np.uint8)
                else:
                    change_mask = (classification != prev_classification).astype(np.uint8)
                
                # Store plot data with RGB images
                change_plots.append({
                    'prev_date': prev_date,
                    'curr_date': product_date,
                    'prev_class': prev_classification.copy(),
                    'curr_class': classification.copy(),
                    'change_mask': change_mask,
                    'change_pct': change_pct,
                    'summary': change_summary,
                    'prev_rgb': prev_rgb.copy() if prev_rgb is not None else None,
                    'curr_rgb': rgb_stretched.copy()
                })
            else:
                print(f"\nNo significant change from {prev_date} (only {change_pct:.2f}% changed)")
        
        # Store for next iteration
        prev_classification = classification.copy()
        prev_rgb = rgb_stretched.copy()
        prev_date = product_date
        
    except Exception as e:
        print(f"✗ Error processing product: {e}")
        continue

print(f"\n{'='*60}")
print("Change detection complete!")
print(f"Found {len(change_plots)} significant changes")
print('='*60)



# Visualize changes in a carousel
if len(change_plots) > 0:
    from matplotlib.colors import ListedColormap
    import ipywidgets as widgets
    from IPython.display import display
    
    # Custom colormap for classification
    colors = ['black', 'green', 'orange', 'blue']
    cmap_class = ListedColormap(colors)
    
    # Change colormap: gray for no change, red for change
    cmap_change = ListedColormap(['lightgray', 'red'])
    
    def plot_change(index, show_unclassified):
        """Plot a specific change comparison with RGB images beneath."""
        if index < 0 or index >= len(change_plots):
            return
        
        data = change_plots[index]
        
        # Recompute change mask based on toggle
        if not show_unclassified:
            # Only show changes between classified pixels
            classified_mask = (data['prev_class'] > 0) & (data['curr_class'] > 0)
            change_mask_display = ((data['curr_class'] != data['prev_class']) & classified_mask).astype(np.uint8)
            changed_pixels = change_mask_display.sum()
        else:
            # Show all changes
            change_mask_display = (data['curr_class'] != data['prev_class']).astype(np.uint8)
            changed_pixels = change_mask_display.sum()
        
        total_px = data['prev_class'].size
        change_pct_display = 100.0 * changed_pixels / total_px
        
        # Create 2-row figure: classifications on top, RGB images beneath
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # ===== TOP ROW: Classifications =====
        
        # Previous classification
        im1 = axes[0, 0].imshow(data['prev_class'], cmap=cmap_class, vmin=0, vmax=3)
        axes[0, 0].set_title(f'Classification: {data["prev_date"]}', fontsize=20)
        axes[0, 0].axis('off')
        cbar1 = plt.colorbar(im1, ax=axes[0, 0], ticks=[0, 1, 2, 3], fraction=0.046)
        cbar1.ax.set_yticklabels(['Unclassified', 'Vegetation', 'Built-up', 'Water'], fontsize=16)
        
        # Current classification
        im2 = axes[0, 1].imshow(data['curr_class'], cmap=cmap_class, vmin=0, vmax=3)
        axes[0, 1].set_title(f'Classification: {data["curr_date"]}', fontsize=20)
        axes[0, 1].axis('off')
        cbar2 = plt.colorbar(im2, ax=axes[0, 1], ticks=[0, 1, 2, 3], fraction=0.046)
        cbar2.ax.set_yticklabels(['Unclassified', 'Vegetation', 'Built-up', 'Water'], fontsize=16)
        
        # Change mask (use dynamically computed mask)
        im3 = axes[0, 2].imshow(change_mask_display, cmap=cmap_change, vmin=0, vmax=1)
        title_suffix = "" if show_unclassified else " (classified only)"
        axes[0, 2].set_title(f'Changes: {change_pct_display:.1f}% of pixels{title_suffix}', fontsize=20)
        axes[0, 2].axis('off')
        cbar3 = plt.colorbar(im3, ax=axes[0, 2], ticks=[0, 1], fraction=0.046)
        cbar3.ax.set_yticklabels(['No change', 'Changed'], fontsize=16)
        
        # ===== BOTTOM ROW: RGB images =====
        
        # Previous RGB
        if data.get('prev_rgb') is not None:
            axes[1, 0].imshow(data['prev_rgb'])
            axes[1, 0].set_title(f'RGB: {data["prev_date"]}', fontsize=20)
        else:
            axes[1, 0].text(0.5, 0.5, 'RGB not available', ha='center', va='center', 
                           transform=axes[1, 0].transAxes, fontsize=14)
            axes[1, 0].set_title(f'RGB: {data["prev_date"]}', fontsize=20)
        axes[1, 0].axis('off')
        
        # Current RGB
        axes[1, 1].imshow(data['curr_rgb'])
        axes[1, 1].set_title(f'RGB: {data["curr_date"]}', fontsize=20)
        axes[1, 1].axis('off')
        
        # Add change summary text in bottom-right panel
        axes[1, 2].axis('off')
        if data['summary']:
            # Sort summary by percentage (extract percentage from string and sort descending)
            sorted_summary = sorted(data['summary'], 
                                   key=lambda x: float(x.split(':')[1].strip().rstrip('%')), 
                                   reverse=True)
            summary_text = '\n'.join(sorted_summary[:5])  # Show top 5 changes
            axes[1, 2].text(0.5, 0.5, summary_text, ha='center', va='center',
                           transform=axes[1, 2].transAxes, fontsize=25,
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout(rect=(0, 0.04, 1, 1))
        plt.show()
    
    # Create interactive carousel
    print(f"\n🎠 Change Visualization Carousel ({len(change_plots)} changes detected)")
    print("Use the slider to navigate through detected changes:\n")
    
    slider = widgets.IntSlider(
        value=0,
        min=0,
        max=len(change_plots) - 1,
        step=1,
        description='Change:',
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='d'
    )
    
    toggle = widgets.Checkbox(
        value=False,
        description='Include Unclassified',
        disabled=False,
        indent=False
    )
    
    # Display the interactive widget with controls side by side
    controls = widgets.HBox([slider, toggle])
    ui = widgets.interactive(plot_change, index=slider, show_unclassified=toggle)
    display(widgets.VBox([controls, ui.children[-1]]))

else:
    print("No significant changes detected to visualize.")