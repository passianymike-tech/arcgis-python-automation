"""
GeoJSON Optimization Pipeline for Web GIS Applications
Author: Mike Papayai Passiany
Description: Reduces GeoJSON file sizes for performant web map rendering
by simplifying geometries, stripping unnecessary properties, and applying
coordinate precision reduction. Used for Kenya Drone No-Fly Zone platform.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional
import math


class GeoJSONOptimizer:
    """
    Optimizes GeoJSON files for web delivery.
    Applies geometry simplification, property filtering, and coordinate rounding.
    """

    def __init__(self, precision: int = 5, keep_properties: Optional[List[str]] = None):
        self.precision = precision
        self.keep_properties = keep_properties

    def load(self, filepath: str) -> dict:
        """Load a GeoJSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)

    def save(self, geojson: dict, filepath: str) -> int:
        """Save GeoJSON and return file size in bytes."""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(geojson, f, separators=(',', ':'))
        return Path(filepath).stat().st_size

    def round_coordinates(self, coords):
        """Recursively round coordinates to specified precision."""
        if isinstance(coords[0], (int, float)):
            return [round(c, self.precision) for c in coords]
        return [self.round_coordinates(c) for c in coords]

    def filter_properties(self, properties: dict) -> dict:
        """Keep only specified properties."""
        if self.keep_properties is None:
            return properties
        return {k: v for k, v in properties.items() if k in self.keep_properties}

    def douglas_peucker(self, points, epsilon):
        """
        Douglas-Peucker line simplification algorithm.
        Reduces the number of points preserving overall shape.
        """
        if len(points) < 3:
            return points

        # Find the point with the maximum distance from the line
        start, end = points[0], points[-1]
        max_dist = 0
        max_idx = 0

        for i in range(1, len(points) - 1):
            dist = self._perpendicular_distance(points[i], start, end)
            if dist > max_dist:
                max_dist = dist
                max_idx = i

        if max_dist > epsilon:
            left = self.douglas_peucker(points[:max_idx + 1], epsilon)
            right = self.douglas_peucker(points[max_idx:], epsilon)
            return left[:-1] + right
        else:
            return [start, end]

    @staticmethod
    def _perpendicular_distance(point, line_start, line_end):
        """Calculate perpendicular distance from point to line segment."""
        dx = line_end[0] - line_start[0]
        dy = line_end[1] - line_start[1]

        if dx == 0 and dy == 0:
            dx = point[0] - line_start[0]
            dy = point[1] - line_start[1]
            return math.sqrt(dx * dx + dy * dy)

        t = ((point[0] - line_start[0]) * dx + (point[1] - line_start[1]) * dy) / (dx * dx + dy * dy)
        t = max(0, min(1, t))

        closest_x = line_start[0] + t * dx
        closest_y = line_start[1] + t * dy

        dx = point[0] - closest_x
        dy = point[1] - closest_y
        return math.sqrt(dx * dx + dy * dy)

    def simplify_geometry(self, geometry: dict, epsilon: float = 0.0001) -> dict:
        """Simplify a GeoJSON geometry."""
        geom_type = geometry['type']

        if geom_type == 'Point':
            geometry['coordinates'] = self.round_coordinates(geometry['coordinates'])

        elif geom_type in ('LineString',):
            coords = geometry['coordinates']
            geometry['coordinates'] = self.douglas_peucker(coords, epsilon)
            geometry['coordinates'] = self.round_coordinates(geometry['coordinates'])

        elif geom_type == 'Polygon':
            new_rings = []
            for ring in geometry['coordinates']:
                simplified = self.douglas_peucker(ring, epsilon)
                if len(simplified) >= 4:  # Valid polygon ring
                    new_rings.append(self.round_coordinates(simplified))
                else:
                    new_rings.append(self.round_coordinates(ring))
            geometry['coordinates'] = new_rings

        elif geom_type == 'MultiPolygon':
            new_polys = []
            for polygon in geometry['coordinates']:
                new_rings = []
                for ring in polygon:
                    simplified = self.douglas_peucker(ring, epsilon)
                    if len(simplified) >= 4:
                        new_rings.append(self.round_coordinates(simplified))
                    else:
                        new_rings.append(self.round_coordinates(ring))
                new_polys.append(new_rings)
            geometry['coordinates'] = new_polys

        return geometry

    def optimize(self, input_path: str, output_path: str,
                 epsilon: float = 0.0001) -> Dict:
        """
        Full optimization pipeline.
        Returns statistics about the optimization.
        """
        geojson = self.load(input_path)
        original_size = Path(input_path).stat().st_size

        original_features = len(geojson.get('features', []))

        for feature in geojson.get('features', []):
            feature['properties'] = self.filter_properties(feature['properties'])
            if feature.get('geometry'):
                feature['geometry'] = self.simplify_geometry(
                    feature['geometry'], epsilon
                )

        output_size = self.save(geojson, output_path)
        reduction = (1 - output_size / original_size) * 100

        return {
            'input': input_path,
            'output': output_path,
            'original_size_mb': round(original_size / 1024 / 1024, 2),
            'optimized_size_mb': round(output_size / 1024 / 1024, 2),
            'reduction_pct': round(reduction, 1),
            'features': original_features
        }


def batch_optimize(input_dir: str, output_dir: str,
                   precision: int = 5, epsilon: float = 0.0001,
                   keep_properties: Optional[List[str]] = None):
    """Optimize all GeoJSON files in a directory."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    optimizer = GeoJSONOptimizer(precision=precision, keep_properties=keep_properties)
    results = []

    for geojson_file in sorted(input_path.glob('*.geojson')):
        out_file = output_path / geojson_file.name.replace('.geojson', '_slim.geojson')
        stats = optimizer.optimize(str(geojson_file), str(out_file), epsilon)
        results.append(stats)
        print(f"  {geojson_file.name}: {stats['original_size_mb']}MB → "
              f"{stats['optimized_size_mb']}MB ({stats['reduction_pct']}% reduction)")

    total_original = sum(r['original_size_mb'] for r in results)
    total_optimized = sum(r['optimized_size_mb'] for r in results)
    print(f"\nTotal: {total_original}MB → {total_optimized}MB "
          f"({round((1 - total_optimized/total_original)*100, 1)}% reduction)")

    return results


if __name__ == '__main__':
    # Example: Optimize Kenya shapefiles for drone no-fly zone platform
    batch_optimize(
        input_dir='./shapefiles/raw',
        output_dir='./shapefiles/optimized',
        precision=5,
        epsilon=0.0002,
        keep_properties=['Name', 'Type', 'COUNTY', 'NAME']
    )
