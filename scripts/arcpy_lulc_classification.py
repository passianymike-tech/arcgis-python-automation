"""
ArcGIS Pro - Automated LULC Classification Pipeline
Author: Mike Papayai Passiany
Description: Automated supervised classification of multi-temporal satellite 
imagery using Maximum Likelihood Classification in ArcGIS Pro.
Used for: Eastern Mau Forest Reserve LULC study (1984-2024)
"""

import arcpy
import os
import json
from datetime import datetime

# Environment settings
arcpy.env.overwriteOutput = True
arcpy.env.parallelProcessingFactor = "80%"


class LULCClassificationPipeline:
    """
    Automated Land Use/Land Cover classification pipeline for ArcGIS Pro.
    Supports multi-temporal Landsat and Sentinel-2 imagery processing.
    """

    def __init__(self, workspace, study_area_shapefile):
        self.workspace = workspace
        self.study_area = study_area_shapefile
        arcpy.env.workspace = workspace

        self.class_definitions = {
            1: {"name": "Dense Forest", "color": [0, 100, 0]},
            2: {"name": "Barren Land", "color": [210, 180, 140]},
            3: {"name": "Settlement", "color": [255, 0, 0]},
            4: {"name": "Grassland", "color": [144, 238, 144]},
            5: {"name": "Planted Farmland", "color": [255, 215, 0]}
        }

        self.epochs = [1984, 1986, 1995, 2002, 2014, 2024]

    def preprocess_imagery(self, input_raster, output_raster):
        """Atmospheric correction, resampling, and clipping."""
        print(f"Preprocessing: {input_raster}")

        # Atmospheric correction (DOS1)
        corrected = arcpy.sa.Minus(input_raster, arcpy.sa.ZonalStatistics(
            self.study_area, "FID", input_raster, "MINIMUM"
        ))

        # Resample Sentinel-2 (10m) to 30m for consistency with Landsat
        desc = arcpy.Describe(input_raster)
        if desc.meanCellWidth < 20:
            print("Resampling Sentinel-2 to 30m...")
            arcpy.management.Resample(
                corrected, "temp_resampled.tif", "30 30", "BILINEAR"
            )
            corrected = "temp_resampled.tif"

        # Clip to study area
        arcpy.management.Clip(
            corrected, "#", output_raster,
            self.study_area, "#", "ClippingGeometry"
        )

        print(f"Preprocessed: {output_raster}")
        return output_raster

    def create_composite(self, band_list, output_composite):
        """Stack spectral bands into composite image."""
        print(f"Creating composite from {len(band_list)} bands...")
        arcpy.management.CompositeBands(band_list, output_composite)
        return output_composite

    def generate_training_samples(self, composite, gps_points, output_training):
        """Generate training samples from GPS field data."""
        print("Generating training samples from field GPS data...")

        # Create training sample from GPS points with class labels
        arcpy.sa.CreateTrainingSamples(
            composite, gps_points, output_training
        )
        return output_training

    def classify_image(self, composite, training_samples, output_classified):
        """Run Maximum Likelihood Classification."""
        print(f"Running Maximum Likelihood Classification...")

        # Create signature file
        sig_file = os.path.join(self.workspace, "temp_signature.gsg")
        arcpy.sa.CreateSignatures(
            composite, training_samples, sig_file, "COVARIANCE"
        )

        # Maximum Likelihood Classification
        classified = arcpy.sa.MLClassify(composite, sig_file)
        classified.save(output_classified)

        print(f"Classification complete: {output_classified}")
        return output_classified

    def compute_ndvi(self, nir_band, red_band, output_ndvi):
        """Calculate Normalized Difference Vegetation Index."""
        print("Computing NDVI...")
        nir = arcpy.sa.Float(arcpy.Raster(nir_band))
        red = arcpy.sa.Float(arcpy.Raster(red_band))
        ndvi = (nir - red) / (nir + red)
        ndvi.save(output_ndvi)
        print(f"NDVI saved: {output_ndvi}")
        return output_ndvi

    def compute_ndbi(self, swir_band, nir_band, output_ndbi):
        """Calculate Normalized Difference Built-Up Index."""
        print("Computing NDBI...")
        swir = arcpy.sa.Float(arcpy.Raster(swir_band))
        nir = arcpy.sa.Float(arcpy.Raster(nir_band))
        ndbi = (swir - nir) / (swir + nir)
        ndbi.save(output_ndbi)
        return output_ndbi

    def accuracy_assessment(self, classified_raster, reference_points):
        """Perform accuracy assessment with confusion matrix."""
        print("Running accuracy assessment...")

        # Create accuracy assessment points
        aa_points = os.path.join(self.workspace, "accuracy_points.shp")
        arcpy.sa.CreateAccuracyAssessmentPoints(
            classified_raster, aa_points, "STRATIFIED_RANDOM", 50
        )

        # Update with ground truth values from reference points
        arcpy.management.JoinField(
            aa_points, "OBJECTID", reference_points, "OBJECTID", ["GrndTruth"]
        )

        # Compute confusion matrix
        confusion_matrix = arcpy.sa.ComputeConfusionMatrix(
            aa_points, os.path.join(self.workspace, "confusion_matrix.dbf")
        )

        return confusion_matrix

    def change_detection(self, classified_t1, classified_t2, output_change):
        """Thematic change detection between two epochs."""
        print(f"Computing change detection...")

        # Combine classifications
        change = arcpy.sa.Combine([classified_t1, classified_t2])
        change.save(output_change)

        # Calculate area statistics
        arcpy.sa.TabulateArea(
            self.study_area, "FID", change, "Value",
            os.path.join(self.workspace, "change_stats.dbf")
        )

        return output_change

    def batch_process_all_epochs(self, imagery_dict, training_dict):
        """Process all epochs in batch."""
        results = {}

        for year in self.epochs:
            print(f"\n{'='*60}")
            print(f"PROCESSING EPOCH: {year}")
            print(f"{'='*60}")

            try:
                # Preprocess
                preprocessed = self.preprocess_imagery(
                    imagery_dict[year],
                    f"preprocessed_{year}.tif"
                )

                # Classify
                classified = self.classify_image(
                    preprocessed,
                    training_dict[year],
                    f"classified_{year}.tif"
                )

                # NDVI
                ndvi = self.compute_ndvi(
                    f"{preprocessed}/Band_4",
                    f"{preprocessed}/Band_3",
                    f"ndvi_{year}.tif"
                )

                results[year] = {
                    "classified": classified,
                    "ndvi": ndvi,
                    "status": "success"
                }

            except Exception as e:
                print(f"Error processing {year}: {str(e)}")
                results[year] = {"status": "error", "message": str(e)}

        # Run change detection for consecutive epochs
        for i in range(len(self.epochs) - 1):
            y1, y2 = self.epochs[i], self.epochs[i + 1]
            if results.get(y1, {}).get("status") == "success" and \
               results.get(y2, {}).get("status") == "success":
                self.change_detection(
                    results[y1]["classified"],
                    results[y2]["classified"],
                    f"change_{y1}_{y2}.tif"
                )

        # Save processing report
        report = {
            "project": "Eastern Mau Forest Reserve LULC Analysis",
            "processed_at": datetime.now().isoformat(),
            "epochs": self.epochs,
            "results": {str(k): v for k, v in results.items()}
        }

        with open(os.path.join(self.workspace, "processing_report.json"), "w") as f:
            json.dump(report, f, indent=2)

        print(f"\nBatch processing complete. Report saved.")
        return results


if __name__ == "__main__":
    # Configuration
    WORKSPACE = r"D:\GIS_Projects\Eastern_Mau"
    STUDY_AREA = r"D:\GIS_Projects\Eastern_Mau\study_area.shp"

    pipeline = LULCClassificationPipeline(WORKSPACE, STUDY_AREA)

    print("LULC Classification Pipeline initialized.")
    print(f"Workspace: {WORKSPACE}")
    print(f"Study Area: {STUDY_AREA}")
    print(f"Epochs: {pipeline.epochs}")
    print(f"Classes: {[c['name'] for c in pipeline.class_definitions.values()]}")
