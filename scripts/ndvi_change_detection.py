"""
Multi-temporal NDVI Change Detection Analysis
Author: Mike Papayai Passiany
Description: Computes NDVI from multispectral imagery, performs density slice 
classification, and generates temporal trend analysis with visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio.mask import mask
import geopandas as gpd
from pathlib import Path
import json


class NDVIChangeDetector:
    """Multi-temporal NDVI analysis for vegetation change detection."""

    DENSITY_CLASSES = {
        "Water/Shadow": (-1.0, 0.0),
        "Bare Soil/Built-up": (0.0, 0.15),
        "Sparse Vegetation": (0.15, 0.3),
        "Moderate Vegetation": (0.3, 0.5),
        "Dense Vegetation": (0.5, 0.7),
        "Very Dense Vegetation": (0.7, 1.0)
    }

    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = {}

    def compute_ndvi(self, nir_path, red_path, epoch_year):
        """Compute NDVI from NIR and Red bands."""
        with rasterio.open(nir_path) as nir_src:
            nir = nir_src.read(1).astype(np.float64)
            profile = nir_src.profile.copy()

        with rasterio.open(red_path) as red_src:
            red = red_src.read(1).astype(np.float64)

        # Avoid division by zero
        denominator = nir + red
        ndvi = np.where(denominator != 0, (nir - red) / denominator, 0)
        ndvi = np.clip(ndvi, -1, 1)

        # Save NDVI raster
        profile.update(dtype=rasterio.float64, count=1)
        output_path = self.output_dir / f"ndvi_{epoch_year}.tif"
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(ndvi, 1)

        # Compute statistics
        stats = {
            "year": epoch_year,
            "min": float(np.nanmin(ndvi)),
            "max": float(np.nanmax(ndvi)),
            "mean": float(np.nanmean(ndvi)),
            "std": float(np.nanstd(ndvi)),
            "median": float(np.nanmedian(ndvi))
        }

        self.results[epoch_year] = {"ndvi": ndvi, "stats": stats, "profile": profile}
        print(f"NDVI {epoch_year}: mean={stats['mean']:.4f}, range=[{stats['min']:.4f}, {stats['max']:.4f}]")
        return ndvi, stats

    def density_slice(self, epoch_year):
        """Classify NDVI into density classes."""
        if epoch_year not in self.results:
            raise ValueError(f"NDVI not computed for {epoch_year}")

        ndvi = self.results[epoch_year]["ndvi"]
        classified = np.zeros_like(ndvi, dtype=np.int8)

        class_areas = {}
        pixel_area = 900  # 30m x 30m in sq meters

        for i, (class_name, (low, high)) in enumerate(self.DENSITY_CLASSES.items(), 1):
            mask = (ndvi >= low) & (ndvi < high)
            classified[mask] = i
            area_sqm = int(np.sum(mask) * pixel_area)
            class_areas[class_name] = area_sqm

        self.results[epoch_year]["density_slice"] = classified
        self.results[epoch_year]["class_areas"] = class_areas
        return classified, class_areas

    def temporal_change_analysis(self, year1, year2):
        """Compute NDVI change between two epochs."""
        ndvi1 = self.results[year1]["ndvi"]
        ndvi2 = self.results[year2]["ndvi"]

        change = ndvi2 - ndvi1

        stats = {
            "period": f"{year1}-{year2}",
            "mean_change": float(np.nanmean(change)),
            "improvement_pct": float(np.sum(change > 0.1) / np.sum(~np.isnan(change)) * 100),
            "degradation_pct": float(np.sum(change < -0.1) / np.sum(~np.isnan(change)) * 100),
            "stable_pct": float(np.sum(np.abs(change) <= 0.1) / np.sum(~np.isnan(change)) * 100)
        }

        print(f"Change {year1}-{year2}: mean={stats['mean_change']:.4f}, "
              f"improved={stats['improvement_pct']:.1f}%, degraded={stats['degradation_pct']:.1f}%")
        return change, stats

    def generate_report(self, epochs):
        """Generate comprehensive analysis report with visualizations."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 1. Mean NDVI trend
        years = sorted(self.results.keys())
        means = [self.results[y]["stats"]["mean"] for y in years]
        axes[0, 0].plot(years, means, 'g-o', linewidth=2, markersize=8)
        axes[0, 0].set_title("Mean NDVI Trend (1984-2024)", fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel("Year")
        axes[0, 0].set_ylabel("Mean NDVI")
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].fill_between(years, means, alpha=0.2, color='green')

        # 2. NDVI statistics boxplot-style
        stats_data = {y: self.results[y]["stats"] for y in years}
        mins = [stats_data[y]["min"] for y in years]
        maxs = [stats_data[y]["max"] for y in years]
        stds = [stats_data[y]["std"] for y in years]

        axes[0, 1].errorbar(years, means, yerr=stds, fmt='o-', capsize=5, color='darkgreen')
        axes[0, 1].set_title("NDVI Distribution by Epoch", fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel("Year")
        axes[0, 1].set_ylabel("NDVI (mean ± std)")
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Density class areas (stacked bar)
        if all("class_areas" in self.results[y] for y in years):
            class_names = list(self.DENSITY_CLASSES.keys())
            bottom = np.zeros(len(years))
            colors = ['#2166ac', '#d2b48c', '#f4a582', '#92c5de', '#4daf4a', '#006400']

            for i, cls in enumerate(class_names):
                values = [self.results[y]["class_areas"].get(cls, 0) / 1e6 for y in years]
                axes[1, 0].bar(years, values, bottom=bottom, label=cls, color=colors[i])
                bottom += values

            axes[1, 0].set_title("Vegetation Density Distribution", fontsize=14, fontweight='bold')
            axes[1, 0].set_xlabel("Year")
            axes[1, 0].set_ylabel("Area (km²)")
            axes[1, 0].legend(fontsize=8, loc='upper right')

        # 4. Change detection summary
        change_periods = []
        change_means = []
        for i in range(len(years) - 1):
            _, stats = self.temporal_change_analysis(years[i], years[i + 1])
            change_periods.append(stats["period"])
            change_means.append(stats["mean_change"])

        colors_bar = ['green' if v > 0 else 'red' for v in change_means]
        axes[1, 1].bar(change_periods, change_means, color=colors_bar, alpha=0.8)
        axes[1, 1].set_title("NDVI Change Between Epochs", fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel("Period")
        axes[1, 1].set_ylabel("Mean NDVI Change")
        axes[1, 1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        output_path = self.output_dir / "ndvi_analysis_report.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Report saved: {output_path}")
        return str(output_path)


if __name__ == "__main__":
    detector = NDVIChangeDetector("output/ndvi_analysis")
    print("NDVI Change Detection pipeline ready.")
    print(f"Density classes: {list(NDVIChangeDetector.DENSITY_CLASSES.keys())}")
