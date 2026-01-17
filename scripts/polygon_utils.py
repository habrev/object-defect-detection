from shapely.geometry import Polygon
from shapely.validation import make_valid
import pandas as pd

def parse_polygon(xy_str):
    """
    Safely parse polygon from 'x1,y1,x2,y2,...'
    Returns None if polygon is invalid
    """
    if pd.isna(xy_str):
        return None

    coords = list(map(float, xy_str.split(',')))

    # Must have at least 3 points (6 values)
    if len(coords) < 6:
        return None

    points = [(coords[i], coords[i+1]) for i in range(0, len(coords), 2)]

    # Close polygon if not closed
    if points[0] != points[-1]:
        points.append(points[0])

    try:
        poly = Polygon(points)

        if not poly.is_valid or poly.area == 0:
            poly = make_valid(poly)

        if not poly.is_valid or poly.area == 0:
            return None

        return poly

    except Exception:
        return None

def polygon_iou(poly1, poly2):
    """
    Calculate Intersection over Union (IoU) between two polygons
    """
    if not poly1.is_valid or not poly2.is_valid:
        return 0.0

    inter = poly1.intersection(poly2).area
    union = poly1.union(poly2).area

    if union == 0:
        return 0.0

    return inter / union