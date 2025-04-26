import yaml
import os
import cv2
import numpy as np
import pandas as pd
import time
from pathlib import Path
import sys
import argparse
import math

src_path = Path(__file__).resolve().parent / 'src'
if str(src_path) not in sys.path: sys.path.insert(0, str(src_path))


try:
    from lightgluePipeline import LightGluePipeline
    from supergluePipeline import SuperGluePipeline
    from gimPipeline import GimPipeline
except ImportError as e: print(f"ERROR: Import pipeline failed: {e}"); sys.exit(1)


PREPROCESSING_AVAILABLE = False
try:
    from utils.preprocessing import QueryProcessor, CameraModel
    from utils.helpers import (
        haversine_distance,
        calculate_predicted_gps,
        calculate_location_and_error,
    )
    PREPROCESSING_AVAILABLE = True; print("Successfully imported utility modules.")
except ImportError as e: print(f"WARNING: Import utils failed: {e}"); QueryProcessor = None; CameraModel = None; haversine_distance = None; calculate_predicted_gps = None; calculate_location_and_error = None


def run_benchmark(config_path: str):
    print(f"Loading configuration from: {config_path}")
    try:
        with open(config_path, 'r') as f: config = yaml.safe_load(f)
    except Exception as e: print(f"ERROR loading config: {e}"); sys.exit(1)

    matcher_type = config.get('matcher_type', 'lightglue'); paths = config.get('data_paths', {})
    benchmark_params = config.get('benchmark_params', {}); ransac_params = config.get('ransac_params', {})
    preprocess_config = config.get('preprocessing', {'enabled': False}); camera_config = config.get('camera_model', None)
    matcher_weights_config = config.get('matcher_weights', {})

    required_paths = ['query_dir', 'map_dir', 'output_dir', 'query_metadata', 'map_metadata']
    if not all(p in paths and paths[p] for p in required_paths): print(f"ERROR: Missing paths in config: {required_paths}"); sys.exit(1)

    output_dir = Path(paths['output_dir']); output_dir.mkdir(parents=True, exist_ok=True)
    run_timestamp = time.strftime("%Y%m%d-%H%M%S"); preprocess_status = "preprocessed" if preprocess_config.get('enabled', False) else "original"
    gim_model_suffix = f"_{matcher_weights_config.get('gim_model_type', 'unknown')}" if matcher_type == 'gim' else ""
    run_output_dir = output_dir / f"{matcher_type}{gim_model_suffix}_{preprocess_status}_{run_timestamp}"; run_output_dir.mkdir(exist_ok=True)
    print(f"Output will be saved to: {run_output_dir}")

    query_processor = None
    if preprocess_config.get('enabled', False):
        if not PREPROCESSING_AVAILABLE: print("ERROR: Preprocessing enabled, but modules failed import."); sys.exit(1)
        print("Initializing Query Preprocessor...")
        cam_model = None
        if camera_config and CameraModel:
            try:
                cam_model_params = {k: v for k, v in camera_config.items() if k in CameraModel.__annotations__}
                cam_model = CameraModel(**cam_model_params)
            except Exception as e: print(f"ERROR: Init CameraModel failed: {e}"); sys.exit(1)
        elif 'warp' in preprocess_config.get('steps', []): print("ERROR: 'warp' enabled, but camera_model missing."); sys.exit(1)
        try:
            query_processor = QueryProcessor(processings=preprocess_config.get('steps', []), resize_target=preprocess_config.get('resize_target'),
                                             camera_model=cam_model, target_gimbal_yaw=preprocess_config.get('target_gimbal_yaw', 0.0),
                                             target_gimbal_pitch=preprocess_config.get('target_gimbal_pitch', -90.0),
                                             target_gimbal_roll=preprocess_config.get('target_gimbal_roll', 0.0))
            print(f"  Enabled Steps: {query_processor.processings or 'None'}")
        except Exception as e: print(f"ERROR: Init QueryProcessor failed: {e}"); sys.exit(1)
    else: print("Preprocessing disabled.")

    print(f"\nInitializing matcher pipeline: {matcher_type.upper()}")
    try:
        if matcher_type == 'lightglue': pipeline = LightGluePipeline(config)
        elif matcher_type == 'superglue': pipeline = SuperGluePipeline(config)
        elif matcher_type == 'gim': pipeline = GimPipeline(config)
        else: raise ValueError(f"Unsupported matcher_type: {matcher_type}")
    except Exception as e: print(f"ERROR: Failed pipeline init: {e}"); import traceback; traceback.print_exc(); sys.exit(1)

    print("\nLoading metadata...")
    try:
        query_df = pd.read_csv(paths['query_metadata'], skipinitialspace=True); map_df = pd.read_csv(paths['map_metadata'], skipinitialspace=True)
        query_df.columns = query_df.columns.str.strip(); map_df.columns = map_df.columns.str.strip()
        print(f"Loaded {len(query_df)} queries, {len(map_df)} maps.")
    except Exception as e: print(f"ERROR reading metadata: {e}"); sys.exit(1)
    required_query_cols = ['Filename', 'Latitude', 'Longitude']; required_map_cols = ['Filename', 'Top_left_lat', 'Top_left_lon', 'Bottom_right_lat', 'Bottom_right_long']
    if preprocess_config.get('enabled', False) and 'warp' in preprocess_config.get('steps', []): required_query_cols.extend(['Gimball_Yaw', 'Gimball_Pitch', 'Gimball_Roll', 'Flight_Yaw'])
    missing_q = [c for c in required_query_cols if c not in query_df.columns]; missing_m = [c for c in required_map_cols if c not in map_df.columns]
    if missing_q: print(f"ERROR: Query metadata missing cols: {missing_q}"); sys.exit(1)
    if missing_m: print(f"ERROR: Map metadata missing cols: {missing_m}"); sys.exit(1)

    results_summary = []; min_inliers_success = benchmark_params.get('min_inliers_for_success', 10)
    temp_image_dir = run_output_dir / "processed_queries"; temp_image_dir.mkdir(exist_ok=True)
    print(f"Processed queries saved to: {temp_image_dir}"); print("\nStarting benchmark loop...")
    total_start_time = time.time()


    for query_idx, query_row in query_df.iterrows():
        query_filename = query_row['Filename']
        query_path_original = Path(paths['query_dir']) / query_filename
        print(f"\n--- Processing Query: {query_filename} ({query_idx+1}/{len(query_df)}) ---")
        if not query_path_original.is_file(): print(f"  WARNING: Query image skip: {query_path_original}"); continue


        query_path_for_matcher = query_path_original; processed_query_shape = None; preprocessing_applied_steps = []
        if query_processor:
            query_img_original = cv2.imread(str(query_path_original))
            if query_img_original is None: print(f"  WARNING: Read query failed. Skipping."); continue
            metadata_dict = query_row.to_dict()
            processed_image = query_processor(query_img_original, metadata_dict)
            processed_query_shape = processed_image.shape
            preprocessing_applied_steps = query_processor.processings
            if processed_image.shape != query_img_original.shape or not np.array_equal(processed_image, query_img_original):
                processed_filename = f"{Path(query_filename).stem}_processed{Path(query_filename).suffix}"
                query_path_for_matcher = temp_image_dir / processed_filename
                try:
                    if not cv2.imwrite(str(query_path_for_matcher), processed_image): raise OSError("imwrite fail")
                except Exception as e: print(f"  WARN: Save processed fail: {e}. Using original."); query_path_for_matcher = query_path_original; processed_query_shape = query_img_original.shape
            else: query_path_for_matcher = query_path_original; processed_query_shape = query_img_original.shape
        else:
             temp_img = cv2.imread(str(query_path_original)); processed_query_shape = temp_img.shape if temp_img is not None else None; del temp_img
        if processed_query_shape is None or len(processed_query_shape) < 2: print(f"  WARNING: Invalid query shape. Skipping."); continue


        best_match_for_query = {'query_filename': query_filename, 'best_map_filename': None, 'inliers': -1, 'outliers': -1,
                                'time': 0, 'gt_latitude': query_row.get('Latitude'), 'gt_longitude': query_row.get('Longitude'),
                                'predicted_latitude': None, 'predicted_longitude': None, 'error_meters': float('inf'),
                                'success': False} 
        if best_match_for_query['gt_latitude'] is None: print(f"  WARNING: Missing GT Lat/Lon.")

        query_results_dir = run_output_dir / Path(query_filename).stem; query_results_dir.mkdir(exist_ok=True)


        for map_idx, map_row in map_df.iterrows():
            map_filename = map_row['Filename']; map_path = Path(paths['map_dir']) / map_filename
            if not map_path.is_file(): continue
            map_img_bgr = cv2.imread(str(map_path));
            if map_img_bgr is None: continue
            map_shape = map_img_bgr.shape


            try: match_results = pipeline.match(query_path_for_matcher, map_path)
            except Exception as e: print(f"  ERROR match: {e}"); continue


            match_time = match_results.get('time', 0); homography = match_results.get('homography')
            inliers_mask = match_results.get('inliers'); mkpts0 = match_results.get('mkpts0', np.array([]))
            num_inliers = np.sum(inliers_mask) if inliers_mask is not None else 0
            num_total = len(mkpts0); num_outliers = num_total - num_inliers
            ransac_successful = match_results.get('success', False) and num_inliers >= min_inliers_success


            pred_lat, pred_lon, meter_error = None, None, float('inf')
            norm_center = None 
            localization_successful_this_map = False

            if ransac_successful and homography is not None:
                norm_center = calculate_location_and_error(query_row.to_dict(), map_row.to_dict(), processed_query_shape, map_shape, homography)
                if norm_center is not None:
                     pred_lat, pred_lon = calculate_predicted_gps(map_row.to_dict(), norm_center)
                     if pred_lat is not None and best_match_for_query['gt_latitude'] is not None:
                          meter_error = haversine_distance(best_match_for_query['gt_latitude'], best_match_for_query['gt_longitude'], pred_lat, pred_lon)
                          if meter_error != float('inf'): localization_successful_this_map = True

            output_prefix = f"{Path(query_filename).stem}_vs_{Path(map_filename).stem}"
            txt_output_path = query_results_dir / f"{output_prefix}_results.txt"
            try:
                with open(txt_output_path, 'w') as f:
                    gim_type_str = f" ({matcher_weights_config.get('gim_model_type', 'unknown')})" if matcher_type == 'gim' else ""
                    f.write(f"--- Match Results: {Path(query_filename).name} vs {Path(map_filename).name} ---\n")
                    f.write(f"Matcher: {matcher_type.upper()}{gim_type_str}\n"); f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"Preprocessing: {preprocess_status.capitalize()} ({preprocessing_applied_steps or 'None'})\n"); f.write("-" * 30 + "\n")
                    f.write(f"Query Image Used: {query_path_for_matcher.name}\n"); f.write(f"Map Image: {map_filename}\n"); f.write("-" * 30 + "\n")
                    f.write("MATCHING & RANSAC:\n"); f.write(f"  Time Spent: {match_time:.4f} s\n"); f.write(f"  Putative Matches: {num_total}\n")
                    f.write(f"  Inliers: {num_inliers}\n"); f.write(f"  Outliers: {num_outliers}\n"); f.write(f"  RANSAC Method: {ransac_params.get('method', 'N/A')}\n")
                    f.write(f"  RANSAC Threshold: {ransac_params.get('reproj_threshold', 'N/A'):.1f} px\n"); f.write(f"  RANSAC/Matcher Success: {match_results.get('success', False)}\n")
                    f.write(f"  Min Inliers Required: {min_inliers_success}\n"); f.write(f"  Sufficient Inliers Found: {ransac_successful}\n"); f.write("-" * 30 + "\n")
                    f.write("LOCALIZATION:\n"); gt_lat, gt_lon = best_match_for_query['gt_latitude'], best_match_for_query['gt_longitude']
                    f.write(f"  Ground Truth (Lat, Lon):   {f'{gt_lat:.7f}, {gt_lon:.7f}' if gt_lat is not None else 'N/A'}\n")
                    f.write(f"  Predicted (Lat, Lon):  {f'{pred_lat:.7f}, {pred_lon:.7f}' if localization_successful_this_map else 'N/A'}\n")
                    f.write(f"  Difference (Error): {f'{meter_error:.3f}' if localization_successful_this_map else 'N/A'} meters\n")
                    f.write(f"  Localization Success (Pair): {localization_successful_this_map}\n"); f.write("-" * 30 + "\n")
                    f.write("DEBUG INFO (Optional):\n")
                    f.write(f"  Norm Center (X, Y): {f'({norm_center[0]:.4f}, {norm_center[1]:.4f})' if norm_center is not None else 'N/A'}\n")
                    f.write(f"  Homography (Query->Map):\n{homography}\n")
            except Exception as e: print(f"  ERROR writing results txt: {e}")


            if benchmark_params.get('save_visualization', False) and ransac_successful:
                 vis_output_path = query_results_dir / f"{output_prefix}_match.png"
                 if hasattr(pipeline, 'visualize_matches') and callable(pipeline.visualize_matches):
                      try:
                          mkpts1 = match_results.get('mkpts1', np.array([]))
                          pipeline.visualize_matches(query_path_for_matcher, map_path, mkpts0, mkpts1, inliers_mask, vis_output_path)
                      except Exception as e: print(f"  ERROR during visualization call: {e}")


            if localization_successful_this_map:
                current_is_better = False
                if not best_match_for_query['success']: current_is_better = True
                elif num_inliers > best_match_for_query['inliers']: current_is_better = True
                elif num_inliers == best_match_for_query['inliers'] and meter_error < best_match_for_query['error_meters']: current_is_better = True
                if current_is_better:
                    best_match_for_query.update({
                        'best_map_filename': map_filename, 'inliers': num_inliers, 'outliers': num_outliers,
                        'time': match_time, 'predicted_latitude': pred_lat, 'predicted_longitude': pred_lon,
                        'error_meters': meter_error,
                        'success': True 
                    })


        print(f"  Best Match: {best_match_for_query['best_map_filename'] or 'None'} ({best_match_for_query['inliers']} inliers, {best_match_for_query['error_meters']:.2f} m error)")
        results_summary.append(best_match_for_query)


    total_end_time = time.time(); total_time = total_end_time - total_start_time
    print(f"\n--- Benchmark Finished ---"); print(f"Total Time: {total_time:.2f} seconds")


    if not results_summary: print("No queries processed. Skipping summary."); return # Exit if nothing to summarize
    summary_df = pd.DataFrame(results_summary)
    
    summary_df = summary_df.drop(columns=['gt_location_pixels', 'predicted_location_pixels', 'pixel_error'], errors='ignore')
    summary_df.rename(columns={'query_filename': 'Query Image', 'best_map_filename': 'Best Map Match', 'inliers': 'Inliers', 'outliers': 'Outliers',
                               'time': 'Best Match Time (s)', 'gt_latitude':'GT Latitude', 'gt_longitude':'GT Longitude',
                               'predicted_latitude':'Pred Latitude', 'predicted_longitude':'Pred Longitude',
                               'error_meters': 'Error (m)', 'success': 'Localization Success'}, inplace=True)
    summary_output_path = run_output_dir / "benchmark_summary.csv"
    try: summary_df.to_csv(summary_output_path, index=False, float_format='%.7f'); print(f"\nBenchmark summary saved: {summary_output_path}")
    except Exception as e: print(f"ERROR saving summary CSV: {e}")

    successful_localizations_df = summary_df[summary_df['Localization Success'] == True]
    num_successful = len(successful_localizations_df); num_queries = len(query_df); num_processed = len(results_summary)
    success_rate = (num_successful / num_processed) * 100 if num_processed > 0 else 0
    if num_successful > 0:
        average_error_m = successful_localizations_df['Error (m)'].mean(); median_error_m = successful_localizations_df['Error (m)'].median()
        average_inliers = successful_localizations_df['Inliers'].mean(); median_inliers = successful_localizations_df['Inliers'].median()
        average_time = successful_localizations_df['Best Match Time (s)'].mean()
    else: average_error_m=median_error_m=average_inliers=median_inliers=average_time=float('nan')

    print("\n--- Overall Statistics ---")
    print(f"Total Queries: {num_queries} | Processed: {num_processed} | Successful: {num_successful} | Rate (vs Processed): {success_rate:.2f}%")
    if num_successful > 0:
        print(f"Avg/Median Error (m): {average_error_m:.2f} / {median_error_m:.2f}")
        print(f"Avg/Median Inliers: {average_inliers:.1f} / {median_inliers:.1f}")
        print(f"Avg Time (Success Best): {average_time:.3f} s")

    stats_output_path = run_output_dir / "benchmark_stats.txt"
    try:
        with open(stats_output_path, 'w') as f:
            gim_type_str = f" ({matcher_weights_config.get('gim_model_type', 'unknown')})" if matcher_type == 'gim' else ""
            f.write("--- Overall Benchmark Statistics ---\n"); f.write(f"Matcher: {matcher_type.upper()}{gim_type_str}\n")
            f.write(f"Timestamp: {run_timestamp}\n"); f.write(f"Preprocessing: {preprocess_status.capitalize()} ({preprocess_config.get('steps', []) or 'None'})\n"); f.write("-" * 30 + "\n")
            f.write(f"Total Queries: {num_queries}\n"); f.write(f"Queries Processed: {num_processed}\n"); f.write(f"Successful Localizations: {num_successful}\n")
            f.write(f"Success Rate (vs Processed): {success_rate:.2f}%\n"); f.write(f"Min Inliers Required: {min_inliers_success}\n")
            if num_successful > 0:
                 f.write("-" * 30 + "\n"); f.write("Stats (Successful Localizations):\n"); f.write(f"  Avg Error: {average_error_m:.2f} m\n"); f.write(f"  Median Error: {median_error_m:.2f} m\n")
                 f.write(f"  Avg Inliers: {average_inliers:.1f}\n"); f.write(f"  Median Inliers: {median_inliers:.1f}\n"); f.write(f"  Avg Time (Best Match): {average_time:.3f} s\n")
            f.write("-" * 30 + "\n"); f.write(f"Total Runtime: {total_time:.2f} seconds\n")
        print(f"Overall statistics saved: {stats_output_path}")
    except Exception as e: print(f"ERROR saving stats: {e}")

    print("Benchmark complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Visual Localization Benchmark")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config YAML file.')
    args = parser.parse_args()
    if not Path(args.config).is_file(): print(f"ERROR: Config file not found: {args.config}"); sys.exit(1)

    temp_config = {}; preproc_enabled = False
    try:
        with open(args.config, 'r') as f: temp_config = yaml.safe_load(f)
        preproc_enabled = temp_config.get('preprocessing', {}).get('enabled', False)
    except Exception as e: print(f"Error reading config pre-check: {e}"); sys.exit(1)
    if preproc_enabled and not PREPROCESSING_AVAILABLE: print("ERROR: Preprocessing enabled, but utils failed import."); sys.exit(1)
    if not all([haversine_distance, calculate_predicted_gps, calculate_location_and_error]): print("ERROR: Core benchmark helpers failed import."); sys.exit(1)

    run_benchmark(args.config)