import math
import cv2
import numpy as np
from typing import Optional, Tuple, Dict, Any

EARTH_RADIUS_METERS = 6371000.0

def latlon_to_pixel(
    lat: float,
    lon: float,
    map_metadata: Dict[str, Any],
    map_shape: Tuple[int, int]
) -> Optional[np.ndarray]:
    """Converts Lat/Lon to pixel coords [x, y] within a map tile. (Internal use)"""
    h, w = map_shape[:2]
    required_keys = ['Top_left_lat', 'Bottom_right_lat', 'Bottom_right_long', 'Top_left_lon']
    if not all(key in map_metadata for key in required_keys): return None
    lat_range = map_metadata['Top_left_lat'] - map_metadata['Bottom_right_lat']
    lon_range = map_metadata['Bottom_right_long'] - map_metadata['Top_left_lon']
    if abs(lat_range) < 1e-9 or abs(lon_range) < 1e-9 or w <= 0 or h <= 0: return None
    lat_frac = (map_metadata['Top_left_lat'] - lat) / lat_range
    lon_frac = (lon - map_metadata['Top_left_lon']) / lon_range
    px = lon_frac * w; py = lat_frac * h
    buffer = 0.05
    if (-buffer * w) <= px <= (1 + buffer) * w and (-buffer * h) <= py <= (1 + buffer) * h:
         px_clipped = max(0.0, min(float(w - 1), px))
         py_clipped = max(0.0, min(float(h - 1), py))
         return np.array([px_clipped, py_clipped])
    else: return None

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculates haversine distance in meters."""
    try:
        lat1_rad, lon1_rad, lat2_rad, lon2_rad = map(math.radians, [lat1, lon1, lat2, lon2])
        dlon = lon2_rad - lon1_rad; dlat = lat2_rad - lat1_rad
        a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
        a = max(0.0, min(1.0, a))
        c = 2 * math.asin(math.sqrt(a))
        distance = c * EARTH_RADIUS_METERS
        return distance
    except Exception: return float('inf')

def calculate_predicted_gps(
    map_metadata: Dict[str, Any],
    normalized_center_xy: Optional[Tuple[float, float]]
) -> Tuple[Optional[float], Optional[float]]:
    """Calculates predicted GPS from normalized center."""
    if normalized_center_xy is None: return None, None
    center_x_norm, center_y_norm = normalized_center_xy
    required_keys = ['Top_left_lat', 'Bottom_right_lat', 'Bottom_right_long', 'Top_left_lon']
    if not all(key in map_metadata for key in required_keys): return None, None
    try:
        lat_diff = map_metadata['Top_left_lat'] - map_metadata['Bottom_right_lat']
        pred_lat = map_metadata['Top_left_lat'] - center_y_norm * lat_diff
        lon_diff = map_metadata['Bottom_right_long'] - map_metadata['Top_left_lon']
        pred_lon = map_metadata['Top_left_lon'] + center_x_norm * lon_diff
        return pred_lat, pred_lon
    except Exception: return None, None

def calculate_location_and_error(
    query_metadata: Dict[str, Any], 
    map_metadata: Dict[str, Any],
    query_shape: Tuple[int, int, int],
    map_shape: Tuple[int, int, int],
    homography: Optional[np.ndarray]
) -> Optional[Tuple[float, float]]: 
    """
    Calculates the normalized center of the predicted query location within the map tile.
    Pixel-level calculations are removed.

    Args:
        query_metadata: Metadata dictionary for the query image.
        map_metadata: Metadata dictionary for the map image.
        query_shape: Shape of the query image (H, W, C) used for matching.
        map_shape: Shape of the map image (H, W, C).
        homography: The 3x3 homography matrix (Query -> Map), or None.

    Returns:
        normalized_center (Tuple[float, float] or None): Predicted center
                                                         normalized (0-1) within map tile.
    """
    normalized_center = None

    if homography is not None:
        query_h, query_w = query_shape[:2]
        if query_w <= 0 or query_h <= 0:
            print(f"Warning: Invalid query shape {query_shape} for center calculation.")
            return None 

        query_center_pixels = np.array([[[query_w / 2.0, query_h / 2.0]]], dtype=np.float32)

        try:
            pred_location_pixels_arr = cv2.perspectiveTransform(query_center_pixels, homography)
            if pred_location_pixels_arr is None: raise ValueError("perspectiveTransform returned None")
            pred_location_pixels = pred_location_pixels_arr[0, 0] 

            map_h, map_w = map_shape[:2]
            if map_w > 0 and map_h > 0:
                 norm_x = pred_location_pixels[0] / map_w
                 norm_y = pred_location_pixels[1] / map_h
                 normalized_center = (norm_x, norm_y)
                 if not (-0.5 <= norm_x <= 1.5 and -0.5 <= norm_y <= 1.5):
                      map_filename = map_metadata.get('Filename', 'N/A')
                      print(f"Warning: Normalized center ({norm_x:.3f}, {norm_y:.3f}) outside [0,1] bounds for map {map_filename}.")
            else:
                 print(f"Warning: Invalid map dimensions ({map_w}x{map_h}) for normalization.")

        except (cv2.error, ValueError, TypeError) as e:
            print(f"Warning: cv2.perspectiveTransform failed: {e}")
            normalized_center = None 

    return normalized_center