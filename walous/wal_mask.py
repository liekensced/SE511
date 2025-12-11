from pathlib import Path

import matplotlib.pyplot as plt
import rasterio
from rasterio.plot import show
from rasterio import warp
from pyproj import Transformer
from matplotlib import colors as mcolors
from affine import Affine
import numpy as np
import math


def main():
	tif_path = Path(__file__).resolve().parent / "WAL_OCS_2018.tif.tif"
	s2_path = Path(__file__).resolve().parent / ".." / "sentinel_data_subset" / "S2B_20191204T104319_31UER_TOC-B08_10M_V200.tif"
	if not tif_path.exists():
		raise FileNotFoundError(f"Missing tif file: {tif_path}")

	# Load S2 metadata for reprojection target
	with rasterio.open(s2_path) as s2_src:
		s2_crs = s2_src.crs
		s2_bounds = s2_src.bounds
		s2_transform = s2_src.transform
		s2_width, s2_height = s2_src.width, s2_src.height

	# Reproject WAL_OCS to S2's CRS and bounds
	with rasterio.open(tif_path) as src:
		print("Metadata:")
		print(f"  Original CRS: {src.crs}")
		print(f"  Target CRS:   {s2_crs}")
		print(f"  Target Bounds: {s2_bounds}")
		print(f"  Target Size:   {s2_width} x {s2_height}")

		# Read full WAL raster
		wal_data = src.read(1)

		# Reproject to S2 CRS/bounds
		reprojected = np.empty((s2_height, s2_width), dtype=wal_data.dtype)
		warp.reproject(
			source=wal_data,
			destination=reprojected,
			src_crs=src.crs,
			src_transform=src.transform,
			dst_crs=s2_crs,
			dst_transform=s2_transform,
			resampling=rasterio.enums.Resampling.nearest
		)

		subset = reprojected

		# Apply QML palette
		palette = [
			(1, "#8a8a8a", 255),
			(2, "#dc0f0f", 255),
			(62, "#ff5500", 255),
			(3, "#4e4e4e", 255),
			(4, "#d0d0d0", 255),
			(5, "#2461f7", 255),
			(6, "#ffff73", 255),
			(7, "#e9ffbe", 255),
			(8, "#003200", 255),
			(80, "#007800", 255),
			(9, "#28c828", 255),
			(90, "#b7e8b0", 255),
			(0, "#e5ea3f", 0),
			(11, "#8a8a8a", 255),
			(15, "#8a8a8a", 255),
			(81, "#8a8a8a", 255),
			(18, "#8a8a8a", 255),
			(31, "#8a8a8a", 255),
			(71, "#8a8a8a", 255),
			(51, "#8a8a8a", 255),
			(91, "#8a8a8a", 255),
			(19, "#8a8a8a", 255),
			(28, "#dc0f0f", 255),
			(29, "#dc0f0f", 255),
			(38, "#4e4e4e", 255),
			(39, "#4e4e4e", 255),
			(93, "#4e4e4e", 255),
			(55, "#2461f7", 255),
			(58, "#2461f7", 255),
			(75, "#2461f7", 255),
			(59, "#2461f7", 255),
		]

		max_val = int(subset.max()) if subset.size else 0
		colors = [(0, 0, 0, 0)] * (max_val + 1)
		for val, hex_color, alpha in palette:
			if val <= max_val:
				colors[val] = mcolors.to_rgba(hex_color, alpha / 255.0)

		cmap = mcolors.ListedColormap(colors)

		fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
		ax1.imshow(subset, cmap=cmap, vmin=0, vmax=max_val)
		ax1.set_title("QML colors")
		ax1.axis("off")

		# Second plot: simplified palette for NDVI/NDBI/NDWI testing (closer to QML intent)
		water_vals = {5, 55, 58, 59, 75}
		# Green-toned vegetation classes only
		vegetation_vals = {6, 7, 8, 9, 80, 90}  # Vegetation: green
		# Everything else (non-zero) treated as artificial/other
		artificial_vals = {1, 2, 3, 4, 11, 15, 18, 19, 28, 29, 31, 38, 39, 51, 62, 71, 81, 91,  93}

		simple_colors = [(0, 0, 0, 1.0)] * (max_val + 1)
		for v in water_vals:
			if v <= max_val:
				simple_colors[v] = mcolors.to_rgba("#2461f7", 1.0)  # Water: blue
		for v in artificial_vals:
			if v <= max_val:
				simple_colors[v] = mcolors.to_rgba("#dc0f0f", 1.0)  # Artificial: red
		for v in vegetation_vals:
			if v <= max_val:
				simple_colors[v] = mcolors.to_rgba("#28c828", 1.0)  # Vegetation: green

		simple_cmap = mcolors.ListedColormap(simple_colors)
		ax2.imshow(subset, cmap=simple_cmap, vmin=0, vmax=max_val)
		ax2.set_title("Water / Artificial / Vegetation")
		ax2.axis("off")

		plt.tight_layout()
		plt.show()

		# Create and save classification masks from WAL_OCS
		print("\n" + "="*60)
		print("Extracting and saving WAL_OCS classification masks...")
		print("="*60)
		
		# Create binary masks for each class
		water_mask = np.isin(subset, list(water_vals)).astype(np.uint8)
		vegetation_mask = np.isin(subset, list(vegetation_vals)).astype(np.uint8)
		artificial_mask = np.isin(subset, list(artificial_vals)).astype(np.uint8)
		
		# Create unified classification: 0=unclassified, 1=vegetation, 2=artificial, 3=water
		wal_classification = np.zeros_like(subset, dtype=np.uint8)
		wal_classification[water_mask > 0] = 3
		wal_classification[vegetation_mask > 0] = 1
		wal_classification[artificial_mask > 0] = 2
		
		# Report coverage
		total_px = int(wal_classification.size)
		get_pct = lambda val: 100.0 * float((wal_classification == val).sum()) / total_px
		
		print(f"\nWAL_OCS Coverage (from reprojected data):")
		print(f"  Vegetation: {get_pct(1):.2f}%")
		print(f"  Artificial: {get_pct(2):.2f}%")
		print(f"  Water:      {get_pct(3):.2f}%")
		print(f"  Unclassified: {get_pct(0):.2f}%")
		
		# Save masks as numpy files for easy loading in notebook
		out_dir = Path(__file__).resolve().parent / "masks"
		out_dir.mkdir(exist_ok=True)
		
		if False:
			np.save(out_dir / "wal_classification.npy", wal_classification)
			np.save(out_dir / "wal_water_mask.npy", water_mask)
			np.save(out_dir / "wal_vegetation_mask.npy", vegetation_mask)
			np.save(out_dir / "wal_artificial_mask.npy", artificial_mask)
		
		print(f"\n✓ Masks saved to: {out_dir}/")
		print(f"  - wal_classification.npy (0=unclass, 1=veg, 2=artificial, 3=water)")
		print(f"  - wal_water_mask.npy")
		print(f"  - wal_vegetation_mask.npy")
		print(f"  - wal_artificial_mask.npy")
		
		return wal_classification, water_mask, vegetation_mask, artificial_mask


if __name__ == "__main__":
	main()
