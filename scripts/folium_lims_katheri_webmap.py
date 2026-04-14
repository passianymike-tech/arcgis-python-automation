# ============================================================
# LIMS Katheri — Folium Interactive Web Map
# Land Information Management System (LIMS) Visualization
# Author: Mike Papayai Passiany
# University of Nairobi — MSc Geospatial Information Science
# ============================================================
#
# This script generates an interactive web map for the Katheri
# Land Registration Section using Python Folium. It visualizes
# cadastral parcels, boundary data (1612 Boundery), and the
# georeferenced RIM (Registry Index Map) sheet overlay.
#
# Data Sources:
#   - Katheri Georeferenced RIM Sheet (TIF)
#   - 1612 Boundery shapefile (cadastral boundary)
#   - Parcels shapefile (digitized parcels)
#   - Land parcel attributes (ownership, area, land use)
#
# ============================================================

import folium
from folium import plugins
import geopandas as gpd
import json
import os
import numpy as np
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

# Katheri area center coordinates (Meru County, Kenya)
CENTER_LAT = 0.05
CENTER_LON = 37.65
ZOOM_START = 15

# Output path
OUTPUT_HTML = "lims_katheri_webmap.html"

# Data paths (relative to script location)
DATA_DIR = Path(__file__).parent / "data"
PARCELS_SHP = DATA_DIR / "Parcels_Projected_UTM_37N.shp"
BOUNDARY_SHP = DATA_DIR / "1612_Boundery.shp"
PARCELS_ATTR_SHP = DATA_DIR / "Parcels_with_mock_attributes" / "Parcel_with_Attributes.shp"


# ============================================================
# STYLE FUNCTIONS
# ============================================================

def parcel_style(feature):
    """Style function for cadastral parcels."""
    return {
        'fillColor': '#3388ff',
        'color': '#1a1aff',
        'weight': 1.5,
        'fillOpacity': 0.25,
        'dashArray': '3'
    }


def boundary_style(feature):
    """Style function for section boundary."""
    return {
        'fillColor': 'transparent',
        'color': '#e74c3c',
        'weight': 3,
        'fillOpacity': 0,
        'dashArray': '8 4'
    }


def attributed_parcel_style(feature):
    """Style parcels by land use category."""
    land_use = feature['properties'].get('land_use', 'Unknown')
    colors = {
        'Residential': '#2ecc71',
        'Agricultural': '#f39c12',
        'Commercial': '#e74c3c',
        'Institutional': '#9b59b6',
        'Industrial': '#e67e22',
        'Vacant': '#95a5a6',
    }
    return {
        'fillColor': colors.get(land_use, '#3498db'),
        'color': '#2c3e50',
        'weight': 1.5,
        'fillOpacity': 0.5,
    }


# ============================================================
# MAP CREATION
# ============================================================

def create_lims_map():
    """Create the LIMS Katheri interactive web map."""

    # Initialize map
    m = folium.Map(
        location=[CENTER_LAT, CENTER_LON],
        zoom_start=ZOOM_START,
        control_scale=True,
        prefer_canvas=True,
    )

    # ----------------------------------------------------------
    # BASE LAYERS
    # ----------------------------------------------------------
    folium.TileLayer(
        tiles='https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
        attr='© OpenStreetMap contributors',
        name='OpenStreetMap',
    ).add_to(m)

    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri World Imagery',
        name='Satellite',
    ).add_to(m)

    folium.TileLayer(
        tiles='https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png',
        attr='© CARTO',
        name='Dark',
    ).add_to(m)

    folium.TileLayer(
        tiles='https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png',
        attr='© OpenTopoMap',
        name='Topographic',
    ).add_to(m)

    # ----------------------------------------------------------
    # LOAD GEOSPATIAL DATA
    # ----------------------------------------------------------

    # Section Boundary (1612)
    if BOUNDARY_SHP.exists():
        boundary_gdf = gpd.read_file(str(BOUNDARY_SHP))
        if boundary_gdf.crs and boundary_gdf.crs.to_epsg() != 4326:
            boundary_gdf = boundary_gdf.to_crs(epsg=4326)

        boundary_layer = folium.FeatureGroup(name='Section Boundary (1612)')
        folium.GeoJson(
            boundary_gdf.to_json(),
            style_function=boundary_style,
            tooltip='Section 1612 Boundary — Katheri',
        ).add_to(boundary_layer)
        boundary_layer.add_to(m)

        # Auto-fit map to boundary extent
        bounds = boundary_gdf.total_bounds  # [minx, miny, maxx, maxy]
        m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])
    else:
        print(f"Warning: Boundary shapefile not found at {BOUNDARY_SHP}")

    # Cadastral Parcels (basic)
    if PARCELS_SHP.exists():
        parcels_gdf = gpd.read_file(str(PARCELS_SHP))
        if parcels_gdf.crs and parcels_gdf.crs.to_epsg() != 4326:
            parcels_gdf = parcels_gdf.to_crs(epsg=4326)

        parcels_layer = folium.FeatureGroup(name='Cadastral Parcels')
        folium.GeoJson(
            parcels_gdf.to_json(),
            style_function=parcel_style,
            tooltip=folium.GeoJsonTooltip(
                fields=parcels_gdf.columns[:3].tolist(),
                aliases=[f'Field {i+1}:' for i in range(min(3, len(parcels_gdf.columns)))],
                sticky=True,
            ),
        ).add_to(parcels_layer)
        parcels_layer.add_to(m)
    else:
        print(f"Warning: Parcels shapefile not found at {PARCELS_SHP}")

    # Attributed Parcels (with land use, ownership)
    if PARCELS_ATTR_SHP.exists():
        attr_gdf = gpd.read_file(str(PARCELS_ATTR_SHP))
        if attr_gdf.crs and attr_gdf.crs.to_epsg() != 4326:
            attr_gdf = attr_gdf.to_crs(epsg=4326)

        # Compute area in square meters
        if 'area_sqm' not in attr_gdf.columns:
            attr_gdf_projected = attr_gdf.to_crs(epsg=32637)  # UTM 37N
            attr_gdf['area_sqm'] = attr_gdf_projected.geometry.area.round(1)
            attr_gdf['area_ha'] = (attr_gdf['area_sqm'] / 10000).round(3)

        attr_layer = folium.FeatureGroup(name='Parcels (Land Use)', show=False)

        # Build tooltip fields dynamically
        tooltip_fields = [c for c in attr_gdf.columns if c != 'geometry'][:6]
        folium.GeoJson(
            attr_gdf.to_json(),
            style_function=attributed_parcel_style,
            tooltip=folium.GeoJsonTooltip(
                fields=tooltip_fields,
                aliases=[f.replace('_', ' ').title() + ':' for f in tooltip_fields],
                sticky=True,
                style='font-size: 12px;',
            ),
        ).add_to(attr_layer)
        attr_layer.add_to(m)
    else:
        print(f"Warning: Attributed parcels not found at {PARCELS_ATTR_SHP}")

    # ----------------------------------------------------------
    # MAP CONTROLS & PLUGINS
    # ----------------------------------------------------------

    # Layer control
    folium.LayerControl(collapsed=False).add_to(m)

    # Minimap
    plugins.MiniMap(
        tile_layer=folium.TileLayer(
            tiles='https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
            attr='OSM',
        ),
        toggle_display=True,
        width=150,
        height=150,
    ).add_to(m)

    # Fullscreen button
    plugins.Fullscreen().add_to(m)

    # Mouse position coordinates display
    plugins.MousePosition(
        position='bottomleft',
        prefix='Lat/Lon:',
        num_digits=6,
    ).add_to(m)

    # Measure tool
    plugins.MeasureControl(
        position='topleft',
        primary_length_unit='meters',
        secondary_length_unit='kilometers',
        primary_area_unit='sqmeters',
        secondary_area_unit='hectares',
    ).add_to(m)

    # ----------------------------------------------------------
    # TITLE & LEGEND
    # ----------------------------------------------------------

    title_html = '''
    <div style="position: fixed; top: 10px; left: 60px; z-index: 1000;
                background: rgba(0,0,0,0.8); color: white; padding: 12px 20px;
                border-radius: 8px; font-family: Arial, sans-serif;
                box-shadow: 0 2px 8px rgba(0,0,0,0.3);">
        <h4 style="margin: 0 0 4px 0; font-size: 14px;">
            LIMS Katheri — Land Information Management System
        </h4>
        <p style="margin: 0; font-size: 11px; color: #aaa;">
            Section 1612 · Meru County · Registry Index Map
        </p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(title_html))

    legend_html = '''
    <div style="position: fixed; bottom: 30px; right: 10px; z-index: 1000;
                background: rgba(0,0,0,0.85); color: white; padding: 12px 16px;
                border-radius: 8px; font-family: Arial, sans-serif; font-size: 12px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.3);">
        <b style="font-size: 13px;">Legend</b><br>
        <span style="color: #e74c3c;">━━</span> Section Boundary (1612)<br>
        <span style="color: #3388ff;">━━</span> Cadastral Parcels<br>
        <b style="margin-top: 6px; display: block;">Land Use:</b>
        <span style="color: #2ecc71;">■</span> Residential &nbsp;
        <span style="color: #f39c12;">■</span> Agricultural<br>
        <span style="color: #e74c3c;">■</span> Commercial &nbsp;
        <span style="color: #9b59b6;">■</span> Institutional<br>
        <span style="color: #95a5a6;">■</span> Vacant
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))

    # Attribution
    attribution_html = '''
    <div style="position: fixed; bottom: 5px; left: 60px; z-index: 1000;
                font-size: 10px; color: #888; font-family: Arial, sans-serif;">
        Mike Passiany · University of Nairobi · FGS 6221 Web Mapping
    </div>
    '''
    m.get_root().html.add_child(folium.Element(attribution_html))

    return m


# ============================================================
# PARCEL STATISTICS
# ============================================================

def print_parcel_stats(shapefile_path):
    """Print summary statistics for parcel data."""
    if not Path(shapefile_path).exists():
        print("Shapefile not found — skipping statistics.")
        return

    gdf = gpd.read_file(str(shapefile_path))
    print(f"\n{'='*50}")
    print(f"LIMS Katheri — Parcel Statistics")
    print(f"{'='*50}")
    print(f"Total Parcels: {len(gdf)}")
    print(f"CRS: {gdf.crs}")

    if gdf.crs and gdf.crs.to_epsg() != 32637:
        gdf = gdf.to_crs(epsg=32637)

    areas = gdf.geometry.area
    print(f"Total Area: {areas.sum()/10000:.2f} hectares")
    print(f"Mean Parcel Size: {areas.mean():.1f} m²")
    print(f"Smallest Parcel: {areas.min():.1f} m²")
    print(f"Largest Parcel: {areas.max():.1f} m²")
    print(f"{'='*50}\n")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("Generating LIMS Katheri Web Map...")

    # Print parcel statistics
    print_parcel_stats(PARCELS_ATTR_SHP)

    # Create and save map
    web_map = create_lims_map()
    web_map.save(OUTPUT_HTML)
    print(f"Web map saved to: {OUTPUT_HTML}")
    print(f"Open in browser to view the interactive map.")
