"""
Image preprocessing utilities, including resizing, camera modeling,
and perspective warping for query images.
"""

import math
from typing import List, Tuple, Dict, Optional, Union, Any
from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

def process_resize(w: int, h: int, resize: Union[int, List[int]]) -> Tuple[int, int]:
    """
    Calculates new dimensions based on resize parameter.

    Args:
        w: Original width.
        h: Original height.
        resize: Target size parameter:
                - int > 0: Target maximum dimension.
                - List/Tuple [W, H]: Target exact dimensions.
                - List/Tuple [S]: Target maximum dimension.
                - -1 or invalid: Keep original size.

    Returns:
        Tuple (new_width, new_height).
    """
    if h <= 0 or w <= 0: return w, h 
    max_dim = max(h, w)

    if isinstance(resize, int) and resize > 0:
        scale = resize / max_dim
        w_new, h_new = int(round(w * scale)), int(round(h * scale))
    elif isinstance(resize, (list, tuple)) and len(resize) == 2:
        w_new, h_new = int(resize[0]), int(resize[1]) # Ensure int
    elif isinstance(resize, (list, tuple)) and len(resize) == 1 and resize[0] > 0:
        scale = resize[0] / max_dim
        w_new, h_new = int(round(w*scale)), int(round(h*scale))
    else: #
        w_new, h_new = w, h


    w_new = max(1, w_new)
    h_new = max(1, h_new)
    return w_new, h_new

@dataclass
class CameraModel:
    """
    Represents the intrinsic parameters of a camera. Calculates focal length
    in pixels based on horizontal field of view (HFOV). Assumes principal
    point is the image center.

    Attributes:
        focal_length (float): Focal length in millimeters (mm).
        resolution_width (int): Sensor/image width in pixels.
        resolution_height (int): Sensor/image height in pixels.
        hfov_deg (float): Horizontal Field of View in degrees.
        hfov_rad (float): HFOV in radians (calculated).
        aspect_ratio (float): Width / Height (calculated).
        focal_length_px (float): Approximate focal length in pixels (calculated).
        principal_point_x (float): X-coordinate of the principal point (calculated center).
        principal_point_y (float): Y-coordinate of the principal point (calculated center).
    """
    focal_length: float
    resolution_width: int
    resolution_height: int
    hfov_deg: float

    hfov_rad: float = field(init=False)
    aspect_ratio: float = field(init=False)
    focal_length_px: float = field(init=False)
    principal_point_x: float = field(init=False)
    principal_point_y: float = field(init=False)

    def __post_init__(self) -> None:
        """Calculate derived camera parameters."""
        if self.resolution_width <= 0 or self.resolution_height <= 0 or self.hfov_deg <= 0:
             raise ValueError("Camera dimensions and HFOV must be positive.")
        self.hfov_rad = math.radians(self.hfov_deg)
        self.aspect_ratio = self.resolution_width / self.resolution_height
        tan_arg = self.hfov_rad / 2.0
        if abs(math.tan(tan_arg)) < 1e-9:
             raise ValueError("HFOV results in near-zero tan value, cannot calculate focal length.")
        self.focal_length_px = (self.resolution_width / 2.0) / math.tan(tan_arg)
        self.principal_point_x = self.resolution_width / 2.0
        self.principal_point_y = self.resolution_height / 2.0

def get_intrinsics(camera_model: CameraModel, scale: float = 1.0) -> np.ndarray:
    """
    Gets the 3x3 camera intrinsics matrix K, optionally scaled.

    Args:
        camera_model: The CameraModel object.
        scale: Optional scaling factor applied to principal point and focal length.

    Returns:
        The 3x3 intrinsics matrix K.
    """
    if scale <= 0: scale = 1.0
    fx = camera_model.focal_length_px / scale
    fy = camera_model.focal_length_px / scale 
    cx = camera_model.principal_point_x / scale
    cy = camera_model.principal_point_y / scale
    intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    return intrinsics

def rotation_matrix_from_angles(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    Computes the 3x3 rotation matrix from roll, pitch, yaw angles (degrees).
    Uses the 'xyz' extrinsic Euler angle convention.

    Args:
        roll: Rotation about the x-axis (degrees).
        pitch: Rotation about the y-axis (degrees).
        yaw: Rotation about the z-axis (degrees).

    Returns:
        The 3x3 rotation matrix.
    """
    try:
        roll, pitch, yaw = float(roll), float(pitch), float(yaw)
        r = Rotation.from_euler("xyz", [roll, pitch, yaw], degrees=True).as_matrix()
        return r.astype(np.float32)
    except (TypeError, ValueError) as e:
         print(f"Error calculating rotation matrix (invalid angles?): {e}")
         return np.identity(3, dtype=np.float32)


class QueryProcessor:
    """
    Applies a sequence of preprocessing steps (e.g., resizing, warping)
    to query images based on configuration and metadata.
    """
    def __init__(
        self,
        processings: Optional[List[str]] = None,
        resize_target: Optional[Union[int, List[int]]] = None,
        camera_model: Optional[CameraModel] = None,
        target_gimbal_yaw: float = 0.0,
        target_gimbal_pitch: float = -90.0,
        target_gimbal_roll: float = 0.0,
    ) -> None:
        """
        Initializes the QueryProcessor.

        Args:
            processings: List of processing step names ('resize', 'warp'). Order matters.
            resize_target: Parameter for the 'resize' step (see process_resize).
            camera_model: CameraModel object required for the 'warp' step.
            target_gimbal_yaw: Target yaw angle (degrees) for warping.
            target_gimbal_pitch: Target pitch angle (degrees) for warping (e.g., -90 for nadir).
            target_gimbal_roll: Target roll angle (degrees) for warping.
        """
        self.resize_target = resize_target
        self.camera_model = camera_model
        self.target_gimbal_yaw = target_gimbal_yaw
        self.target_gimbal_pitch = target_gimbal_pitch
        self.target_gimbal_roll = target_gimbal_roll
        self.processings = processings if processings else []
        self._step_functions = {
            "resize": self._resize_image,
            "warp": self._warp_image,
        }

    def __call__(self, image: np.ndarray, metadata: Dict[str, Any]) -> np.ndarray:
        """
        Applies the configured preprocessing steps sequentially to the image.

        Args:
            image: The input query image as a NumPy array (BGR or Grayscale).
            metadata: Dictionary containing metadata for the query image (e.g.,
                      'Gimball_Yaw', 'Gimball_Pitch', 'Gimball_Roll', 'Flight_Yaw').

        Returns:
            The processed image as a NumPy array. Returns the original image
            if any processing step fails.
        """
        processed_image = image.copy()
        for step_name in self.processings:
            if step_name in self._step_functions:
                try:
                    processed_image = self._step_functions[step_name](processed_image, metadata)
                except Exception as e:
                    print(f"Warning: Failed processing step '{step_name}': {e}")
                    return image 
            else:
                 print(f"Warning: Unknown processing step '{step_name}' skipped.")
        return processed_image

    def _resize_image(self, image: np.ndarray, metadata: Dict[str, Any]) -> np.ndarray:
        """Internal method to apply resizing."""
        if self.resize_target is None:
            return image
        h, w = image.shape[:2]
        new_w, new_h = process_resize(w, h, self.resize_target)
        if (new_w, new_h) == (w, h):
            return image

        interp = cv2.INTER_AREA if (new_w * new_h < w * h) else cv2.INTER_LINEAR
        try:
            resized_image = cv2.resize(image, (new_w, new_h), interpolation=interp)
            return resized_image
        except cv2.error as e:
            print(f"Error during OpenCV resize: {e}")
            return image 

    def _warp_image(self, image: np.ndarray, metadata: Dict[str, Any]) -> np.ndarray:
        """Internal method to apply perspective warping for top-down view."""
        if self.camera_model is None:
            print("Warning: Camera model required for warping. Skipping warp step.")
            return image

        h_orig, w_orig = image.shape[:2]
        if h_orig <= 0 or w_orig <= 0:
             print("Warning: Invalid image dimensions for warp.")
             return image

        gimbal_yaw = float(metadata.get('Gimball_Yaw', 0.0))
        gimbal_pitch = float(metadata.get('Gimball_Pitch', -90.0))
        gimbal_roll = float(metadata.get('Gimball_Roll', 0.0))
        flight_yaw = float(metadata.get('Flight_Yaw', 0.0))

        current_yaw = gimbal_yaw + flight_yaw
        current_pitch = gimbal_pitch
        current_roll = gimbal_roll

        R_current = rotation_matrix_from_angles(current_roll, current_pitch, current_yaw)
        R_target = rotation_matrix_from_angles(
            self.target_gimbal_roll, self.target_gimbal_pitch, self.target_gimbal_yaw
        )

        K_orig = get_intrinsics(self.camera_model) 
        scale_w = w_orig / self.camera_model.resolution_width
        scale_h = h_orig / self.camera_model.resolution_height
        K_current = get_intrinsics(self.camera_model, scale=1/max(scale_w, scale_h)) 

        try:
            H = K_current @ R_target @ R_current.T @ np.linalg.inv(K_current)
            H = H.astype(np.float32)
        except np.linalg.LinAlgError:
            print("Warning: Singular matrix during homography calculation. Skipping warp.")
            return image

        corners = np.array([[0, 0], [w_orig, 0], [w_orig, h_orig], [0, h_orig]], dtype=np.float32).reshape(-1, 1, 2)
        try:
            warped_corners = cv2.perspectiveTransform(corners, H)
            if warped_corners is None: raise ValueError("perspectiveTransform returned None")
        except cv2.error as e:
            print(f"Warning: perspectiveTransform failed for corners: {e}. Skipping warp.")
            return image

        x_coords = warped_corners[:, 0, 0]
        y_coords = warped_corners[:, 0, 1]
        x_min, y_min = np.min(warped_corners, axis=0).ravel()
        x_max, y_max = np.max(warped_corners, axis=0).ravel()

        w_new = int(round(x_max - x_min))
        h_new = int(round(y_max - y_min))

        if w_new <= 0 or h_new <= 0:
            print(f"Warning: Invalid warped dimensions ({w_new}x{h_new}). Skipping warp.")
            return image
        if w_new > w_orig * 10 or h_new > h_orig * 10:
             print(f"Warning: Warped dimensions significantly larger ({w_new}x{h_new}) than original. Skipping warp.")
             return image

        T = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], dtype=np.float32)
        H_translate = T @ H

        try:
            warped_image = cv2.warpPerspective(image, H_translate, (w_new, h_new),
                                               flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0)) 
            return warped_image
        except cv2.error as e:
            print(f"Error during OpenCV warpPerspective: {e}")
            return image 