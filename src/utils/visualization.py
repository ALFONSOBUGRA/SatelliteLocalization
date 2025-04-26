"""
Image visualization function for displaying feature matches.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple, Union

def create_match_visualization(
    image0_path: Union[str, Path],
    image1_path: Union[str, Path],
    mkpts0: np.ndarray,
    mkpts1: np.ndarray,
    inliers_mask: np.ndarray,
    output_path: Union[str, Path],
    title: str = "Feature Matches", 
    line_color_inlier: Tuple[int, int, int] = (0, 255, 0),  
    point_color_inlier: Tuple[int, int, int] = (0, 255, 0), 
    line_color_outlier: Optional[Tuple[int, int, int]] = (150, 150, 150), 
    show_outliers: bool = False,
    point_size: int = 2,
    line_thickness: int = 1,
    text_info: Optional[List[str]] = None, 
    target_height: Optional[int] = 600
) -> bool:
    """
    Creates and saves a side-by-side image visualization of matches.

    Images are resized to have the same target height while maintaining aspect ratio.
    Inlier matches are drawn, and optionally outlier matches. Text display is disabled.

    Args:
        image0_path: Path to the first image.
        image1_path: Path to the second image.
        mkpts0: Matched keypoints in image 0 (N x 2). Should be in original image coordinates.
        mkpts1: Matched keypoints in image 1 (N x 2). Should be in original image coordinates.
        inliers_mask: Boolean mask indicating inliers (N,).
        output_path: Path to save the visualization.
        title: Title for the visualization (ignored).
        line_color_inlier: BGR color tuple for inlier matches.
        point_color_inlier: BGR color tuple for inlier points.
        line_color_outlier: BGR color tuple for outlier matches (if show_outliers).
        show_outliers: Whether to draw outlier matches.
        point_size: Radius of keypoint circles (if > 0).
        line_thickness: Thickness of match lines.
        text_info: List of strings for info text (ignored).
        target_height: Optional target height for the output visualization. Images
                       will be scaled proportionally. If None, uses max input height.

    Returns:
        True if visualization was saved successfully, False otherwise.
    """
    try:
        img0_vis = cv2.imread(str(image0_path), cv2.IMREAD_COLOR)
        img1_vis = cv2.imread(str(image1_path), cv2.IMREAD_COLOR)

        if img0_vis is None or img1_vis is None:
            print(f"Error reading visualization image(s): {image0_path}, {image1_path}")
            return False

        H0, W0 = img0_vis.shape[:2]
        H1, W1 = img1_vis.shape[:2]

        if H0 <=0 or W0 <=0 or H1 <=0 or W1 <=0:
             print(f"Error: Invalid image dimensions for visualization.")
             return False

        if target_height is None or target_height <= 0:
            H_target = max(H0, H1)
        else:
            H_target = int(target_height) 

        scale0 = H_target / H0
        scale1 = H_target / H1

        W0_new = int(round(W0 * scale0))
        W1_new = int(round(W1 * scale1))
        W0_new = max(1, W0_new); W1_new = max(1, W1_new) 

        interp = cv2.INTER_AREA if (scale0 < 1.0 or scale1 < 1.0) else cv2.INTER_LINEAR
        image0_resized = cv2.resize(img0_vis, (W0_new, H_target), interpolation=interp)
        image1_resized = cv2.resize(img1_vis, (W1_new, H_target), interpolation=interp)

        mkpts0_scaled = mkpts0 * scale0
        mkpts1_scaled = mkpts1 * scale1

        W_total = W0_new + W1_new
        out = 255 * np.ones((H_target, W_total, 3), np.uint8)

        out[:H_target, :W0_new] = image0_resized
        out[:H_target, W0_new:W_total] = image1_resized

        if len(inliers_mask) != len(mkpts0):
             print(f"Warning: Inlier mask length ({len(inliers_mask)}) differs from keypoint length ({len(mkpts0)}). Skipping drawing.")
             inliers_mask = np.zeros(len(mkpts0), dtype=bool) 
        else:
            inliers_mask = inliers_mask.astype(bool)

        mkpts0_draw = np.round(mkpts0_scaled).astype(int)
        mkpts1_draw = np.round(mkpts1_scaled).astype(int)

        if show_outliers and line_color_outlier is not None:
            outliers_mask = ~inliers_mask
            mkpts0_out = mkpts0_draw[outliers_mask]
            mkpts1_out = mkpts1_draw[outliers_mask]
            for i in range(len(mkpts0_out)):
                pt0 = tuple(mkpts0_out[i])
                pt1 = tuple(mkpts1_out[i] + np.array([W0_new, 0]))
                cv2.line(out, pt0, pt1, line_color_outlier, line_thickness, lineType=cv2.LINE_AA)

        mkpts0_in = mkpts0_draw[inliers_mask]
        mkpts1_in = mkpts1_draw[inliers_mask]
        for i in range(len(mkpts0_in)):
            pt0 = tuple(mkpts0_in[i])
            pt1 = tuple(mkpts1_in[i] + np.array([W0_new, 0]))
            cv2.line(out, pt0, pt1, line_color_inlier, line_thickness, lineType=cv2.LINE_AA)
            if point_size > 0:
                cv2.circle(out, pt0, point_size, point_color_inlier, -1, lineType=cv2.LINE_AA)
                cv2.circle(out, pt1, point_size, point_color_inlier, -1, lineType=cv2.LINE_AA)

        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)

        success = cv2.imwrite(str(output_path_obj), out)
        if not success:
            print(f"Error saving visualization to {output_path_obj}")
            return False
        return True

    except FileNotFoundError as e:
        print(f"Error: Image file not found during visualization - {e}")
        return False
    except cv2.error as e:
         print(f"Error: OpenCV error during visualization - {e}")
         return False
    except Exception as e:
        print(f"Error during visualization creation: {e}")
        import traceback
        traceback.print_exc()
        return False