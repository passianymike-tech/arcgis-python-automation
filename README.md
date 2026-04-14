# ArcGIS Python Automation Workflows

A collection of Python scripts for automating GIS workflows using **ArcPy**, **ArcGIS API for Python**, and open-source geospatial libraries. These scripts support satellite image processing, land cover classification, spatial analysis, and automated map production.

## Project Structure

```
arcgis-python-automation/
├── scripts/
│   ├── arcpy_lulc_classification.py    # Supervised classification with ArcPy
│   ├── satellite_image_acquisition.py  # Automated Sentinel/Landsat download
│   ├── ndvi_change_detection.py        # Multi-temporal NDVI analysis
│   ├── batch_map_production.py         # Automated map layout export
│   ├── geodatabase_management.py       # Enterprise geodatabase operations
│   ├── deep_learning_resampling.py     # DEM + RGB overlay with rasterio
│   ├── spatial_etl_pipeline.py         # ETL pipeline for geospatial data
│   └── folium_lims_katheri_webmap.py   # Folium cadastral web map (LIMS)
├── notebooks/
│   └── hydrological_analysis.ipynb     # Watershed delineation with WhiteboxTools
├── requirements.txt
└── README.md
```

## Scripts Overview

### 1. `arcpy_lulc_classification.py`
Automated supervised classification using ArcGIS Pro's Maximum Likelihood Classifier. Processes multi-temporal Landsat and Sentinel-2 imagery for land use/land cover mapping.

**Key features:**
- Batch processing of multi-epoch satellite imagery (1984–2024)
- Automated training sample generation from field GPS data
- Maximum Likelihood Classification with accuracy assessment
- Confusion matrix and Kappa coefficient computation
- Export to enterprise geodatabase

### 2. `satellite_image_acquisition.py`
Automates download of Sentinel-2 and Landsat imagery from USGS Earth Explorer and Copernicus Open Access Hub.

**Key features:**
- Query by AOI geometry, date range, and cloud cover threshold
- Automated atmospheric correction and band stacking
- Mosaic and clip to study area boundary
- Metadata extraction and cataloging

### 3. `ndvi_change_detection.py`
Multi-temporal NDVI computation and change detection analysis.

**Key features:**
- NDVI calculation from multispectral imagery
- Density slice classification for vegetation health mapping
- Temporal trend analysis with statistical metrics
- Automated chart generation (matplotlib)

### 4. `batch_map_production.py`
Automated production of publication-quality maps using ArcPy mapping module.

**Key features:**
- Template-based map production for multiple epochs
- Dynamic legend, scale bar, and north arrow placement
- Batch export to PDF and PNG at 300 DPI
- Consistent cartographic styling across time series

### 5. `geodatabase_management.py`
Enterprise geodatabase administration and data management.

**Key features:**
- Create and manage file/enterprise geodatabases
- Import/export feature classes and rasters
- Schema management and domain creation
- Spatial reference system management

### 6. `deep_learning_resampling.py`
DEM visualization and RGB image overlay with advanced resampling techniques.

### 7. `spatial_etl_pipeline.py`
End-to-end ETL pipeline for ingesting, transforming, and loading spatial data from multiple sources into PostGIS/ArcGIS Enterprise.

### 8. `folium_lims_katheri_webmap.py`
Interactive cadastral web map for the Katheri Land Information Management System (LIMS) using Python Folium.

**Key features:**
- 4 basemap layers (OpenStreetMap, Satellite, Terrain, CartoDB Dark)
- Boundary polygon overlay (1612 Katheri administrative boundary)
- Cadastral parcel visualization with land-use classification styling
- Interactive popups with parcel ID, area, owner info, and land-use type
- Color-coded parcels: Residential, Commercial, Agricultural, Industrial, Institutional, Mixed Use
- MiniMap, mouse position, measurement tools, and fullscreen control
- Parcel statistics computation (count, total area, land-use distribution)
- Exports to standalone HTML for web deployment

## Technologies
- **ArcPy** (ArcGIS Pro 3.2)
- **ArcGIS API for Python**
- **Python**: rasterio, numpy, scikit-learn, matplotlib, geopandas, folium
- **WhiteboxTools** (hydrological modeling)
- **GDAL/OGR**
- **PostgreSQL/PostGIS** (via psycopg2)

## Sample Output
The scripts in this repository were used for the published research:

> Passiany, M.P., Kiema, J.B. (2025). *The Impact of Deforestation and Restoration Trends on Land Use and Land Cover Changes in Eastern Mau Forest Reserve, Kenya.* African Journal on Land Policy and Geospatial Sciences, Vol. 8 No. 7. DOI: [10.48346/IMIST.PRSM/ajlp-gs.v8i7.54139](https://doi.org/10.48346/IMIST.PRSM/ajlp-gs.v8i7.54139)

## Author
**Mike Papayai Passiany**
MSc Geographic Information Systems — University of Nairobi
[LinkedIn](https://www.linkedin.com/in/59b641a2/) | [Portfolio](https://papayai.droneverse.pro/)
