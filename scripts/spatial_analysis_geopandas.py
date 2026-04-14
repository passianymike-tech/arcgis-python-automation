"""
Spatial Data Analysis with GeoPandas and Folium
Author: Mike Papayai Passiany
Description: Loads shapefiles, computes spatial relationships (point-in-polygon,
buffer analysis, centroid calculation), and generates interactive web maps.
Adapted from Rotterdam ITC coursework and professional drone survey work.
"""

import geopandas as gpd
import folium
from folium import GeoJson, CircleMarker, Popup
from shapely.geometry import Point, mapping
from shapely.ops import unary_union
import json
import numpy as np
from pathlib import Path


def load_and_reproject(shapefile_path: str, target_crs: str = "EPSG:4326") -> gpd.GeoDataFrame:
    """Load a shapefile and reproject to target CRS."""
    gdf = gpd.read_file(shapefile_path)
    if gdf.crs and gdf.crs.to_epsg() != int(target_crs.split(":")[1]):
        gdf = gdf.to_crs(target_crs)
    return gdf


def compute_buffer_zones(gdf: gpd.GeoDataFrame, distances_km: list) -> dict:
    """
    Compute multiple buffer zones around features.
    Used for drone no-fly zone tiered restriction analysis.
    """
    # Project to metric CRS for accurate buffering
    gdf_projected = gdf.to_crs("EPSG:32637")  # UTM Zone 37N (Kenya)
    
    buffer_zones = {}
    for dist_km in distances_km:
        dist_m = dist_km * 1000
        buffered = gdf_projected.copy()
        buffered['geometry'] = gdf_projected.geometry.buffer(dist_m)
        buffer_zones[f"{dist_km}km"] = buffered.to_crs("EPSG:4326")
    
    return buffer_zones


def point_in_zone_analysis(point_lat: float, point_lon: float,
                           zones: gpd.GeoDataFrame) -> list:
    """
    Determine which restriction zones a given point falls within.
    Returns sorted list of intersecting zones by distance.
    """
    point = Point(point_lon, point_lat)
    results = []
    
    for idx, row in zones.iterrows():
        if row.geometry.contains(point):
            centroid = row.geometry.centroid
            distance_deg = point.distance(centroid)
            # Approximate degree to km at Kenya latitude
            distance_km = distance_deg * 111.32
            results.append({
                'name': row.get('Name', row.get('NAME', f'Zone {idx}')),
                'type': row.get('Type', 'Unknown'),
                'distance_km': round(distance_km, 2)
            })
    
    return sorted(results, key=lambda x: x['distance_km'])


def create_restriction_map(study_area: gpd.GeoDataFrame,
                           airports: gpd.GeoDataFrame,
                           military: gpd.GeoDataFrame,
                           parks: gpd.GeoDataFrame,
                           output_html: str = "restriction_map.html"):
    """
    Generate an interactive Leaflet map showing drone restriction zones
    with tiered buffer visualization.
    """
    # Compute map center
    bounds = study_area.total_bounds
    center = [(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2]
    
    m = folium.Map(
        location=center,
        zoom_start=7,
        tiles='cartodbdark_matter',
        attr='CARTO'
    )
    
    # Tiered airport buffers
    tier_colors = {
        '2km': '#ef4444',   # Red - Restricted
        '4km': '#f97316',   # Orange - Controlled
        '6km': '#3b82f6',   # Blue - Advisory
        '8km': '#6b7280'    # Gray - Awareness
    }
    
    buffer_zones = compute_buffer_zones(airports, [2, 4, 6, 8])
    
    for zone_name, zone_gdf in sorted(buffer_zones.items(), 
                                        key=lambda x: x[0], reverse=True):
        color = tier_colors.get(zone_name, '#888')
        GeoJson(
            zone_gdf.__geo_interface__,
            style_function=lambda x, c=color: {
                'fillColor': c,
                'color': c,
                'weight': 1,
                'fillOpacity': 0.2
            },
            name=f"Airport Buffer {zone_name}"
        ).add_to(m)
    
    # Airport core zones (prohibited)
    GeoJson(
        airports.__geo_interface__,
        style_function=lambda x: {
            'fillColor': '#dc2626',
            'color': '#dc2626',
            'weight': 2,
            'fillOpacity': 0.5
        },
        name="Airports - Prohibited"
    ).add_to(m)
    
    # Military zones
    GeoJson(
        military.__geo_interface__,
        style_function=lambda x: {
            'fillColor': '#f97316',
            'color': '#f97316',
            'weight': 2,
            'fillOpacity': 0.3
        },
        name="Military Restricted"
    ).add_to(m)
    
    # National parks
    GeoJson(
        parks.__geo_interface__,
        style_function=lambda x: {
            'fillColor': '#22c55e',
            'color': '#22c55e',
            'weight': 1,
            'fillOpacity': 0.2
        },
        name="National Parks"
    ).add_to(m)
    
    # County boundaries
    GeoJson(
        study_area.__geo_interface__,
        style_function=lambda x: {
            'fillColor': 'transparent',
            'color': '#818cf8',
            'weight': 1,
            'dashArray': '4'
        },
        name="County Boundaries"
    ).add_to(m)
    
    folium.LayerControl().add_to(m)
    m.save(output_html)
    print(f"Map saved: {output_html}")
    return m


def compute_zone_statistics(zones: gpd.GeoDataFrame) -> dict:
    """Compute area statistics for restriction zones."""
    zones_projected = zones.to_crs("EPSG:32637")
    zones_projected['area_km2'] = zones_projected.geometry.area / 1e6
    
    return {
        'total_features': len(zones),
        'total_area_km2': round(zones_projected['area_km2'].sum(), 2),
        'mean_area_km2': round(zones_projected['area_km2'].mean(), 2),
        'max_area_km2': round(zones_projected['area_km2'].max(), 2),
        'min_area_km2': round(zones_projected['area_km2'].min(), 2)
    }


if __name__ == '__main__':
    # Example: Kenya drone restriction analysis
    counties = load_and_reproject('./shapefiles/County_Boundery.shp')
    airports = load_and_reproject('./shapefiles/Airport.shp')
    military = load_and_reproject('./shapefiles/Military.shp')
    parks = load_and_reproject('./shapefiles/National_Parks_and_Reserves.shp')
    
    print("Airport zone statistics:")
    print(json.dumps(compute_zone_statistics(airports), indent=2))
    
    # Check if a location is in a restriction zone
    nairobi = point_in_zone_analysis(-1.286389, 36.817223, airports)
    print(f"\nNairobi restrictions: {json.dumps(nairobi, indent=2)}")
    
    # Generate interactive map
    create_restriction_map(counties, airports, military, parks)
