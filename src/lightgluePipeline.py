import torch
import cv2
import numpy as np
import time
from pathlib import Path
import sys

lightglue_path = Path(__file__).parent.parent / 'matchers/LightGlue'
if str(lightglue_path) not in sys.path:
    sys.path.append(str(lightglue_path))

try:
    from lightglue import LightGlue, SuperPoint, DISK
    from lightglue.utils import load_image, numpy_image_to_torch, rbd
except ImportError as e:
    print(f"ERROR: Failed importing LightGlue components: {e}")
    sys.exit(1)

try:
    from utils.visualization import create_match_visualization
except ImportError:
    print("ERROR: Could not import create_match_visualization from utils.visualization")
    def create_match_visualization(*args, **kwargs):
        print("Visualization function unavailable.")
        return False

class LightGluePipeline:
    """Pipeline for feature matching using LightGlue."""

    def __init__(self, config: dict):
        """
        Initializes the LightGlue pipeline.

        Args:
            config: Dictionary containing configuration parameters.
                    Expected keys: device, matcher_weights, matcher_params, ransac_params.
        """
        self.config = config
        self.device = torch.device(config.get('device', 'cpu'))
        lg_weights = config.get('matcher_weights', {})
        lg_params = config.get('matcher_params', {}).get('lightglue', {})
        ransac_params = config.get('ransac_params', {})

        feature_type = lg_weights.get('lightglue_features', 'superpoint')
        max_keypoints = lg_params.get('extractor_max_keypoints', 2048)

        print(f"Initializing LightGlue with {feature_type} features...")
        if feature_type == 'superpoint':
            self.extractor = SuperPoint(max_num_keypoints=max_keypoints).eval().to(self.device)
        elif feature_type == 'disk':
            self.extractor = DISK(max_num_keypoints=max_keypoints).eval().to(self.device)
        else:
            raise ValueError(f"Unsupported feature type for LightGlue: {feature_type}")

        self.matcher = LightGlue(features=feature_type).eval().to(self.device)

        # RANSAC params
        self.ransac_thresh = ransac_params.get('reproj_threshold', 8.0)
        self.ransac_conf = ransac_params.get('confidence', 0.999)
        self.ransac_max_iter = ransac_params.get('max_iter', 10000)
        self.ransac_method = cv2.RANSAC 

    def preprocess_image(self, image_path: Path):
        """Loads and preprocesses an image."""
        try:
            img = load_image(image_path) 
            return img.to(self.device)
        except Exception as e:
            print(f"Error loading/preprocessing image {image_path.name}: {e}")
            return None

    def match(self, image0_path: Path, image1_path: Path) -> dict:
        """
        Matches two images using LightGlue.

        Args:
            image0_path: Path to the first image.
            image1_path: Path to the second image.

        Returns:
            Dictionary containing match results:
            'mkpts0', 'mkpts1', 'inliers', 'homography', 'time', 'success'.
        """
        start_time = time.time()
        results = {
            'mkpts0': np.array([]), 'mkpts1': np.array([]), 'inliers': np.array([]),
            'homography': None, 'time': 0, 'success': False
        }
        try:
            image0_tensor = self.preprocess_image(image0_path)
            image1_tensor = self.preprocess_image(image1_path)
            if image0_tensor is None or image1_tensor is None:
                results['time'] = time.time() - start_time
                return results

            feats0 = self.extractor.extract(image0_tensor)
            feats1 = self.extractor.extract(image1_tensor)

            matches01 = self.matcher({'image0': feats0, 'image1': feats1})

            feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]
            kpts0, kpts1, matches = feats0['keypoints'], feats1['keypoints'], matches01['matches']
           

            mkpts0 = kpts0[matches[..., 0]].cpu().numpy()
            mkpts1 = kpts1[matches[..., 1]].cpu().numpy()
            results['mkpts0'] = mkpts0
            results['mkpts1'] = mkpts1

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
            print(f"ERROR during LightGlue matching: {e}")
            import traceback
            traceback.print_exc()

        finally:
            results['time'] = time.time() - start_time
            return results

    def visualize_matches(self, image0_path, image1_path, mkpts0, mkpts1, inliers, output_path):
        """Saves a visualization of the matches using the standardized function."""
        num_inliers = np.sum(inliers)
        num_total = len(mkpts0)
        feature_type = self.config.get('matcher_weights', {}).get('lightglue_features', 'N/A')
        text = [
            f'LightGlue ({feature_type})',
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
                title="LightGlue Matches",
                text_info=text,
                show_outliers=False,
                target_height=600
            )
        except Exception as e:
            print(f"ERROR during LightGlue visualization: {e}")