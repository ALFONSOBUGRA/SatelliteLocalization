import torch
import cv2
import numpy as np
import time
from pathlib import Path
import sys
import warnings
import torchvision.transforms.functional as F_vision
from typing import Dict, Optional, Tuple, Union, List

gim_path = Path(__file__).resolve().parent.parent / 'matchers/gim'
if str(gim_path) not in sys.path:
    if gim_path.exists(): sys.path.append(str(gim_path))
    else:
        print(f"WARNING: GIM directory not found at expected path: {gim_path}")
        gim_path_alt = Path(__file__).resolve().parent.parent.parent / 'gim'
        if gim_path_alt.exists() and str(gim_path_alt) not in sys.path:
             sys.path.append(str(gim_path_alt))
             print(f"Alternate GIM path added: {gim_path_alt}")

try:
    from tools import get_padding_size
    from networks.roma.roma import RoMa
    from networks.loftr.loftr import LoFTR
    from networks.loftr.misc import lower_config
    from networks.loftr.config import get_cfg_defaults
    from networks.dkm.models.model_zoo.DKMv3 import DKMv3
    from networks.lightglue.superpoint import SuperPoint
    from networks.lightglue.models.matchers.lightglue import LightGlue
except ImportError as e:
    print(f"ERROR: Failed importing GIM components: {e}")
    print(f"Check GIM path in sys.path: {sys.path}")
    sys.exit(1)

try:
    from utils.visualization import create_match_visualization
except ImportError:
    print("ERROR: Could not import create_match_visualization from utils.visualization")
    def create_match_visualization(*args, **kwargs) -> bool:
        print("Visualization function unavailable.")
        return False


def preprocess_gim(
    image: np.ndarray,
    grayscale: bool = False,
    resize_max: Optional[int] = None,
    dfactor: int = 8
) -> Optional[Tuple[torch.Tensor, np.ndarray, Tuple[int, int]]]:
    """
    Preprocesses an image array according to GIM demo requirements.

    Handles resizing, color conversion, normalization, tensor conversion,
    and making dimensions divisible by dfactor.

    Args:
        image: Input image (NumPy array HxW or HxWxC).
        grayscale: If True, convert image to grayscale.
        resize_max: If set, resize the largest dimension to this value.
        dfactor: Ensure image dimensions are divisible by this factor.

    Returns:
        A tuple (image_tensor, scale_to_original, original_size_wh) or
        (None, None, None) if preprocessing fails.
        - image_tensor: Processed torch.Tensor (CxHxW).
        - scale_to_original: NumPy array [scale_w, scale_h] to map processed coords back to original.
        - original_size_wh: Tuple (original_width, original_height).
    """
    try:
        if image is None or image.size == 0: raise ValueError("Input image is empty.")
        image = image.astype(np.float32, copy=False)
        if image.ndim < 2 or image.ndim > 3 : raise ValueError(f"Invalid image dimensions: {image.ndim}")

        original_shape = image.shape
        original_size_wh = (original_shape[1], original_shape[0])

        if resize_max and resize_max > 0:
            h, w = original_shape[:2]
            if max(h, w) > 0:
                scale = resize_max / max(h, w)
                if scale < 1.0:
                    new_w = int(round(w * scale)); new_h = int(round(h * scale))
                    if new_w > 0 and new_h > 0:
                        interp = cv2.INTER_AREA
                        image = cv2.resize(image, (new_w, new_h), interpolation=interp)
                    else: print("Warning: resize resulted in zero dimension, skipping resize.") 

        current_shape = image.shape
        if grayscale:
            if image.ndim == 3: image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            if image.ndim != 2: raise ValueError(f"Expected 2D grayscale, got {image.ndim}D")
            image_tensor = torch.from_numpy(image[None] / 255.0).float()
        else:
            if image.ndim == 2: image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            if image.ndim != 3: raise ValueError(f"Expected 3D RGB, got {image.ndim}D")
            image_tensor = torch.from_numpy(image.transpose((2, 0, 1)) / 255.0).float()

        h, w = image_tensor.shape[-2:]
        target_h = int(h // dfactor * dfactor)
        target_w = int(w // dfactor * dfactor)

        if target_h > 0 and target_w > 0 and (target_h != h or target_w != w):
            try: image_tensor = F_vision.resize(image_tensor, size=(target_h, target_w), antialias=True)
            except TypeError: image_tensor = F_vision.resize(image_tensor, size=(target_h, target_w))

        processed_shape_wh = (image_tensor.shape[-1], image_tensor.shape[-2])
        if processed_shape_wh[0] <= 0 or processed_shape_wh[1] <= 0:
            raise ValueError(f"Invalid shape after processing: {processed_shape_wh}")

        if original_size_wh[0] == 0 or original_size_wh[1] == 0: scale_to_original = np.array([1.0, 1.0])
        else: scale_to_original = np.array(original_size_wh, dtype=float) / np.array(processed_shape_wh, dtype=float)

        return image_tensor, scale_to_original, original_size_wh

    except Exception as e:
        print(f"ERROR during preprocess_gim: {e}")
        return None, None, None

class GimPipeline:
    """
    Pipeline for feature matching using various GIM models.

    This class handles loading different GIM models (DKM, RoMa, LoFTR, LightGlue-variant),
    preprocessing images according to GIM requirements, running the selected model
    for matching, applying RANSAC filtering, and saving visualizations.
    """

    def __init__(self, config: Dict):
        """
        Initializes the GIM pipeline.

        Loads the specified GIM model and weights based on configuration. Sets up
        device and RANSAC parameters.

        Args:
            config: Dictionary containing configuration parameters. Must include
                    'device', 'matcher_weights' (with 'gim_model_type' and
                    'gim_weights_path'), 'matcher_params.gim', and 'ransac_params'.
        """
        self.config = config
        self.device = torch.device(config.get('device', 'cpu'))
        gim_weights = config.get('matcher_weights', {})
        self.model_type = gim_weights.get('gim_model_type', 'dkm')
        self.weights_path = gim_weights.get('gim_weights_path')
        self.gim_params = config.get('matcher_params', {}).get('gim', {})
        ransac_params = config.get('ransac_params', {})

        if not self.weights_path or not Path(self.weights_path).is_file():
             raise FileNotFoundError(f"GIM weights file missing: {self.weights_path}")

        print(f"Initializing GIM with model type: {self.model_type}")
        self._load_model() 

        self.ransac_thresh = ransac_params.get('reproj_threshold', 8.0)
        self.ransac_conf = ransac_params.get('confidence', 0.999)
        self.ransac_max_iter = ransac_params.get('max_iter', 10000)
        self.ransac_method = cv2.RANSAC
        ransac_method_name = ransac_params.get('method', 'RANSAC')
        if ransac_method_name == 'USAC_MAGSAC' and hasattr(cv2, 'USAC_MAGSAC'):
            self.ransac_method = cv2.USAC_MAGSAC; print("Using RANSAC Method: USAC_MAGSAC")
        elif ransac_method_name != 'RANSAC': print(f"Warning: RANSAC method '{ransac_method_name}' not found, using default.")

    def _load_model(self):
        """
        Internal method to load the specified GIM model architecture and weights.
        Handles model-specific initialization and state dictionary loading.
        """
        self.model = None
        self.detector = None 

        try:
            state_dict = torch.load(self.weights_path, map_location='cpu')
            if isinstance(state_dict, dict) and 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            if not isinstance(state_dict, dict):
                 raise TypeError("Loaded state_dict is not a dictionary.")
        except Exception as e: print(f"ERROR loading checkpoint {self.weights_path}: {e}"); raise e

        try:
            if self.model_type == 'dkm':
                h = self.gim_params.get('dkm_h', 672); w = self.gim_params.get('dkm_w', 896)
                self.model = DKMv3(weights=None, h=h, w=w)
                clean_sd = {k.replace('model.', '', 1): v for k, v in state_dict.items() if 'encoder.net.fc' not in k}
                load_info = self.model.load_state_dict(clean_sd, strict=False)
                if load_info.missing_keys or load_info.unexpected_keys: print(f"DKM Load Info: Missing={load_info.missing_keys}, Unexpected={load_info.unexpected_keys}")
            elif self.model_type == 'roma':
                img_size = self.gim_params.get('roma_img_size', 672)
                if isinstance(img_size, int): img_size = [img_size, img_size]
                if not isinstance(img_size, (list, tuple)) or len(img_size) != 2: raise ValueError("RoMa img_size must be int or [H, W]")
                self.model = RoMa(img_size=img_size)
                clean_sd = {k.replace('model.', '', 1): v for k, v in state_dict.items()}
                self.model.load_state_dict(clean_sd)
            elif self.model_type == 'loftr':
                _config = lower_config(get_cfg_defaults())['loftr']
                self.model = LoFTR(_config); self.model.load_state_dict(state_dict)
            elif self.model_type == 'lightglue':
                 self.detector = SuperPoint({'max_num_keypoints': self.gim_params.get('gim_lightglue_max_keypoints', 2048),
                                             'force_num_keypoints': True, 'detection_threshold': 0.0, 'nms_radius': 3, 'trainable': False})
                 detector_sd = {k.replace('superpoint.', '', 1): v for k, v in state_dict.items() if k.startswith('superpoint.')}
                 if not detector_sd: print("Warning: No 'superpoint.*' keys found for GIM-LightGlue detector.")
                 self.detector.load_state_dict(detector_sd, strict=False); self.detector = self.detector.eval().to(self.device)
                 print("GIM-LightGlue Detector loaded.")
                 self.model = LightGlue({'filter_threshold': self.gim_params.get('gim_lightglue_filter_threshold', 0.1),
                                         'flash': False, 'checkpointed': True})
                 matcher_sd = {k.replace('model.', '', 1): v for k, v in state_dict.items() if k.startswith('model.')}
                 if not matcher_sd: print("Warning: No 'model.*' keys found for GIM-LightGlue matcher.")
                 self.model.load_state_dict(matcher_sd, strict=False)
                 print("GIM-LightGlue Matcher loaded.")
            else: raise ValueError(f"Unsupported GIM model type: '{self.model_type}'")

            if self.model: self.model = self.model.eval().to(self.device)
            print(f"GIM model '{self.model_type}' initialized successfully.")
        except Exception as e: print(f"ERROR initializing GIM model '{self.model_type}': {e}"); import traceback; traceback.print_exc(); raise e

    def _read_and_preprocess(self, image_path: Path, grayscale: bool = False) \
        -> Tuple[Optional[torch.Tensor], Optional[np.ndarray], Optional[Tuple[int, int]]]:
        """
        Reads and preprocesses a single image file for GIM models.

        Handles image reading, color conversion, and calls the preprocess_gim function.

        Args:
            image_path: Path to the image file.
            grayscale: Whether to process the image as grayscale.

        Returns:
            Tuple of (tensor, scale_factor, original_wh) or (None, None, None) on error.
        """
        try:
            mode = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
            image = cv2.imread(str(image_path), mode)
            if image is None: raise ValueError(f'Cannot read image {image_path.name}.')

            if not grayscale and image.ndim == 3: image = image[:, :, ::-1] 
            elif not grayscale and image.ndim == 2: image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif grayscale and image.ndim == 3: image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            tensor, scale, orig_size_wh = preprocess_gim(image, grayscale=grayscale,
                                                        resize_max=self.gim_params.get('resize_max'),
                                                        dfactor=self.gim_params.get('dfactor', 8))
            if tensor is None: raise ValueError("Preprocessing failed.")
            return tensor.to(self.device)[None], scale, orig_size_wh 
        except Exception as e:
            print(f"Error reading/preprocessing image {image_path.name}: {e}")
            return None, None, None

    def match(self, image0_path: Path, image1_path: Path) -> Dict:
        """
        Performs feature matching between two images using the loaded GIM model.

        This involves preprocessing, running the model-specific matching logic
        (handling padding, sampling, or direct inference), scaling results back
        to original coordinates, and applying RANSAC filtering.

        Args:
            image0_path: Path to the first image (query).
            image1_path: Path to the second image (map/reference).

        Returns:
            A dictionary containing results: 'mkpts0', 'mkpts1' (matched keypoints
            in original image coordinates), 'inliers' (boolean mask from RANSAC),
            'homography' (estimated 3x3 matrix or None), 'time' (execution time),
            'success' (boolean indicating RANSAC success), 'mconf' (confidence scores
            for the points input to RANSAC).
        """
        start_time = time.time()
        results = {'mkpts0': np.array([]), 'mkpts1': np.array([]), 'inliers': np.array([]),
                   'homography': None, 'time': 0.0, 'success': False, 'mconf': np.array([])}
        try:
            use_grayscale = self.model_type in ['loftr', 'lightglue']
            image0, scale0, orig_size0_wh = self._read_and_preprocess(image0_path, grayscale=use_grayscale)
            image1, scale1, orig_size1_wh = self._read_and_preprocess(image1_path, grayscale=use_grayscale)
            if image0 is None or image1 is None: results['time'] = time.time() - start_time; return results

            data = {'image0': image0, 'image1': image1, 'scale0': scale0, 'scale1': scale1,
                    'hw0_i': image0.shape[-2:], 'hw1_i': image1.shape[-2:],
                    'hw0_o': (orig_size0_wh[1], orig_size0_wh[0]), 'hw1_o': (orig_size1_wh[1], orig_size1_wh[0])}
            if use_grayscale: data['gray0'] = image0; data['gray1'] = image1

            kpts0_proc, kpts1_proc, mconf = None, None, torch.empty((0,), device=self.device)

            with torch.no_grad(), warnings.catch_warnings():
                 warnings.simplefilter("ignore")

                 if self.model_type in ['dkm', 'roma']:
                      target_h, target_w = (self.gim_params.get('dkm_h', 672), self.gim_params.get('dkm_w', 896)) if self.model_type == 'dkm' \
                                           else (self.gim_params.get('roma_img_size', 672), self.gim_params.get('roma_img_size', 672)) 
                      if isinstance(target_h, list): target_h, target_w = target_h[0], target_h[1] 

                      ow0, oh0, pl0, pr0, pt0, pb0 = get_padding_size(image0, target_w, target_h)
                      ow1, oh1, pl1, pr1, pt1, pb1 = get_padding_size(image1, target_w, target_h)
                      img0_pad = torch.nn.functional.pad(image0, (pl0, pr0, pt0, pb0)); img1_pad = torch.nn.functional.pad(image1, (pl1, pr1, pt1, pb1))
                      dense_matches, dense_certainty = self.model.match(img0_pad, img1_pad)
                      sparse_matches, mconf = self.model.sample(dense_matches, dense_certainty, 5000)
                      h0p, w0p = img0_pad.shape[-2:]; h1p, w1p = img1_pad.shape[-2:]
                      kpts0_pad = torch.stack((w0p*(sparse_matches[:,0]+1)/2, h0p*(sparse_matches[:,1]+1)/2), dim=-1)
                      kpts1_pad = torch.stack((w1p*(sparse_matches[:,2]+1)/2, h1p*(sparse_matches[:,3]+1)/2), dim=-1)
                      kpts0_proc = kpts0_pad - kpts0_pad.new_tensor([pl0, pt0], device=self.device)
                      kpts1_proc = kpts1_pad - kpts1_pad.new_tensor([pl1, pt1], device=self.device)
                      mask = (kpts0_proc[:, 0] >= 0) & (kpts0_proc[:, 0] < ow0) & (kpts0_proc[:, 1] >= 0) & (kpts0_proc[:, 1] < oh0) & \
                             (kpts1_proc[:, 0] >= 0) & (kpts1_proc[:, 0] < ow1) & (kpts1_proc[:, 1] >= 0) & (kpts1_proc[:, 1] < oh1)
                      kpts0_proc, kpts1_proc, mconf = kpts0_proc[mask], kpts1_proc[mask], mconf[mask]

                 elif self.model_type == 'loftr':
                      self.model(data); kpts0_proc = data.get('mkpts0_f'); kpts1_proc = data.get('mkpts1_f')
                      if kpts0_proc is None or kpts1_proc is None: raise KeyError("LoFTR missing keypoints")
                      mconf = data.get('mconf', torch.ones(len(kpts0_proc), device=self.device))

                 elif self.model_type == 'lightglue':
                     if self.detector is None or self.model is None: raise RuntimeError("GIM LightGlue modules not loaded.")
                     pred = {}
                     pred.update({k + '0': v for k, v in self.detector({"image": data["gray0"]}).items()})
                     pred.update({k + '1': v for k, v in self.detector({"image": data["gray1"]}).items()})
                     size0_wh = (data['image0'].shape[-1], data['image0'].shape[-2]); size1_wh = (data['image1'].shape[-1], data['image1'].shape[-2])
                     matcher_input = {**pred, 'image_size0': torch.tensor([size0_wh], device=self.device), 'image_size1': torch.tensor([size1_wh], device=self.device)}
                     pred.update(self.model(matcher_input))
                     kpts0_det, kpts1_det, matches_out, mconf_raw = pred.get('keypoints0'), pred.get('keypoints1'), pred.get('matches0'), pred.get('matching_scores0')
                     kpts0_proc, kpts1_proc, mconf = torch.empty((0, 2), device=self.device), torch.empty((0, 2), device=self.device), torch.empty((0,), device=self.device)
                     if all(v is not None for v in [kpts0_det, kpts1_det, matches_out, mconf_raw]) and kpts0_det.ndim > 2 and kpts1_det.ndim > 2 and matches_out.ndim > 1 and mconf_raw.ndim > 1:
                         kpts0_det, kpts1_det, matches_out, mconf_raw = kpts0_det[0], kpts1_det[0], matches_out[0], mconf_raw[0]
                         if matches_out.ndim == 1: 
                             valid_mask = matches_out > -1; idx0 = torch.where(valid_mask)[0]; idx1 = matches_out[valid_mask].long()
                             if idx0.numel() > 0:
                                 valid_idx = (idx0 < len(kpts0_det)) & (idx1 < len(kpts1_det)); idx0, idx1 = idx0[valid_idx], idx1[valid_idx]
                                 if idx0.numel() > 0: kpts0_proc, kpts1_proc = kpts0_det[idx0], kpts1_det[idx1]; mconf = mconf_raw[idx0] if len(mconf_raw) == len(matches_out) else torch.ones_like(idx0, dtype=torch.float)
                         elif matches_out.ndim == 2 and matches_out.shape[1] == 2:
                             if matches_out.numel() > 0:
                                 idx0, idx1 = matches_out[:, 0].long(), matches_out[:, 1].long(); valid_idx = (idx0 < len(kpts0_det)) & (idx1 < len(kpts1_det))
                                 idx0, idx1 = idx0[valid_idx], idx1[valid_idx]
                                 if idx0.numel() > 0: kpts0_proc, kpts1_proc = kpts0_det[idx0], kpts1_det[idx1]; mconf = mconf_raw[valid_idx] if len(mconf_raw) == len(matches_out) else torch.ones_like(idx0, dtype=torch.float)
                         else: print(f"  Warning: Unexpected matches tensor dim: {matches_out.ndim}D")

            if kpts0_proc is None: kpts0_proc = torch.empty((0, 2), device=self.device)
            if kpts1_proc is None: kpts1_proc = torch.empty((0, 2), device=self.device)
            if mconf is None: mconf = torch.empty((0,), device=self.device)

            mkpts0_orig = kpts0_proc.detach() * kpts0_proc.new_tensor(scale0, device=self.device)
            mkpts1_orig = kpts1_proc.detach() * kpts1_proc.new_tensor(scale1, device=self.device)

            mkpts0_np = mkpts0_orig.cpu().numpy(); mkpts1_np = mkpts1_orig.cpu().numpy(); mconf_np = mconf.cpu().numpy()
            results['mkpts0'] = mkpts0_np; results['mkpts1'] = mkpts1_np; results['mconf'] = mconf_np

            if len(mkpts0_np) < 4: results['time'] = time.time() - start_time; return results 

            H, inlier_mask = cv2.findHomography(mkpts0_np, mkpts1_np, self.ransac_method,
                                                self.ransac_thresh, confidence=self.ransac_conf, maxIters=self.ransac_max_iter)

            if H is not None and inlier_mask is not None:
                results['homography'] = H; results['inliers'] = inlier_mask.ravel().astype(bool); results['success'] = True

        except Exception as e:
            print(f"ERROR during GIM ({self.model_type}) match {image0_path.name}/{image1_path.name}: {e}")
            import traceback; traceback.print_exc() 
        finally:
            results['time'] = time.time() - start_time
            return results

    def visualize_matches(self, image0_path, image1_path, mkpts0, mkpts1, inliers, output_path):
        """Saves a visualization of the matches using the standardized function."""
        num_inliers = np.sum(inliers) if inliers is not None else 0
        num_total = len(mkpts0)
        text = [f'GIM ({self.model_type.upper()})', f'Matches: {num_inliers} / {num_total}']
        try:
            success = create_match_visualization(
                image0_path=image0_path, image1_path=image1_path, mkpts0=mkpts0, mkpts1=mkpts1,
                inliers_mask=inliers, output_path=output_path, title="GIM Matches", text_info=text,
                show_outliers=False, target_height=600)
            if not success: print(f"Failed to create visualization for GIM.")
        except Exception as e: print(f"ERROR during GIM visualization call: {e}")