import torch
import cv2
import numpy as np
import time
from pathlib import Path
import sys

sg_path = Path(__file__).parent.parent / 'matchers/SuperGluePretrainedNetwork'
if str(sg_path) not in sys.path:
    sys.path.append(str(sg_path))

try:
    from models.matching import Matching
    from models.utils import frame2tensor
except ImportError as e:
    print(f"ERROR: Failed importing SuperGlue components: {e}")
    sys.exit(1)

try:
    from utils.visualization import create_match_visualization
except ImportError:
    print("ERROR: Could not import create_match_visualization from utils.visualization")
    def create_match_visualization(*args, **kwargs):
        print("Visualization function unavailable.")
        return False

class SuperGluePipeline:
    """Pipeline for feature matching using SuperGlue."""

    def __init__(self, config: dict):
        """
        Initializes the SuperGlue pipeline.

        Args:
            config: Dictionary containing configuration parameters.
                    Expected keys: device, matcher_weights, matcher_params, ransac_params.
        """
        self.config = config
        self.device = torch.device(config.get('device', 'cpu'))
        sg_weights = config.get('matcher_weights', {})
        sg_params = config.get('matcher_params', {}).get('superglue', {})
        ransac_params = config.get('ransac_params', {})

        sg_config = {
            'superpoint': {
                'nms_radius': sg_params.get('superpoint_nms_radius', 3),
                'keypoint_threshold': sg_params.get('superpoint_keypoint_threshold', 0.005),
                'max_keypoints': sg_params.get('superpoint_max_keypoints', 2048)
            },
            'superglue': {
                'weights': sg_weights.get('superglue_weights', 'outdoor'),
                'sinkhorn_iterations': sg_params.get('superglue_sinkhorn_iterations', 20),
                'match_threshold': sg_params.get('superglue_match_threshold', 0.2),
            }
        }
        print("Initializing SuperGlue...")
        self.matching = Matching(sg_config).eval().to(self.device)

        self.ransac_thresh = ransac_params.get('reproj_threshold', 8.0)
        self.ransac_conf = ransac_params.get('confidence', 0.999)
        self.ransac_max_iter = ransac_params.get('max_iter', 10000)
        self.ransac_method = cv2.RANSAC

    def preprocess_image(self, image_path: Path):
        """Loads an image, converts to grayscale, and creates a tensor."""
        try:
            img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError(f"Could not read image: {image_path.name}")
            img_tensor = frame2tensor(img, self.device)
            return img_tensor, img.shape[:2] 
        except Exception as e:
            print(f"Error loading/preprocessing image {image_path.name}: {e}")
            return None, None

    def match(self, image0_path: Path, image1_path: Path) -> dict:
        """
        Matches two images using SuperGlue.

        Args:
            image0_path: Path to the first image.
            image1_path: Path to the second image.

        Returns:
            Dictionary containing match results:
            'mkpts0', 'mkpts1', 'inliers', 'homography', 'time', 'success', 'mconf'.
        """
        start_time = time.time()
        results = {
            'mkpts0': np.array([]), 'mkpts1': np.array([]), 'inliers': np.array([]),
            'homography': None, 'time': 0, 'success': False, 'mconf': np.array([])
        }
        try:
            image0_tensor, hw0 = self.preprocess_image(image0_path)
            image1_tensor, hw1 = self.preprocess_image(image1_path)
            if image0_tensor is None or image1_tensor is None:
                 results['time'] = time.time() - start_time
                 return results

            pred_input = {'image0': image0_tensor, 'image1': image1_tensor}

            with torch.no_grad():
                pred = self.matching(pred_input)

            pred = {k: v[0].detach().cpu().numpy() for k, v in pred.items()}

            kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
            matches, conf = pred['matches0'], pred['matching_scores0']

            valid = matches > -1
            mkpts0 = kpts0[valid]
            mkpts1 = kpts1[matches[valid]]
            mconf = conf[valid] 
            results['mkpts0'] = mkpts0
            results['mkpts1'] = mkpts1
            results['mconf'] = mconf

            if len(mkpts0) < 4:
                 print(f"  Warning: Found only {len(mkpts0)} initial matches. Skipping RANSAC.")
                 results['time'] = time.time() - start_time
                 return results

            H, inlier_mask = cv2.findHomography(
                mkpts0, mkpts1,
                method=self.ransac_method,
                ransacReprojThreshold=self.ransac_thresh,
                confidence=self.ransac_conf,
                maxIters=self.ransac_max_iter
            )

            if H is None or inlier_mask is None:
                 print(f"  Warning: RANSAC failed to find Homography.")
                 results['time'] = time.time() - start_time
                 return results

            results['homography'] = H
            results['inliers'] = inlier_mask.ravel().astype(bool)
            results['success'] = True

        except Exception as e:
            print(f"ERROR during SuperGlue matching: {e}")
            import traceback
            traceback.print_exc()
        finally:
            results['time'] = time.time() - start_time
            return results

    def visualize_matches(self, image0_path, image1_path, mkpts0, mkpts1, inliers, output_path):
        """Saves a visualization of the matches using the standardized function."""
        num_inliers = np.sum(inliers)
        num_total = len(mkpts0)
        weights_name = self.config.get('matcher_weights',{}).get('superglue_weights', 'N/A')
        text = [
            f'SuperGlue ({weights_name})',
            f'Matches: {num_inliers} / {num_total}',
        ]

        try:
            create_match_visualization(
                image0_path=image0_path,
                image1_path=image1_path,
                mkpts0=mkpts0,
                mkpts1=mkpts1,
                inliers_mask=inliers,
                output_path=output_path,
                title="SuperGlue Matches",
                text_info=text,
                show_outliers=False,
                target_height=600
            )
        except Exception as e:
            print(f"ERROR during SuperGlue visualization: {e}")