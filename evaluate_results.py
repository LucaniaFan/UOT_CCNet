import os
import numpy as np
import torch
from tqdm import tqdm
import json
from train import compute_metrics_single_scene, compute_metrics_all_scenes
from datetime import datetime
import argparse
import re

def load_results(scene_dir):
    """
    Load prediction results and ground truth from scene directory
    """
    pred_dict = {
        'inflow': [],
        'outflow': [],
        'first_frame': 0,
        'time': 0
    }
    
    gt_dict = {
        'inflow': [],
        'outflow': [],
        'first_frame': 0,
        'time': 0
    }
    
    try:
        files = os.listdir(scene_dir)
        frame_results = {}
        
        for file in files:
            if not file.endswith('.png') or 'matches_vis' in file or file.endswith('matches.png'):
                continue
            
            # Extract frame info using regex to be more robust
            match = re.search(r'_(\d+)_(\d+)_matches_(inflow|outflow)', file)
            if not match:
                continue
                
            # Get frame numbers
            start_frame, end_frame = match.group(1), match.group(2)
            frame_info = f"{start_frame}_{end_frame}"
            flow_type = match.group(3)
            
            if frame_info not in frame_results:
                frame_results[frame_info] = {'inflow': {}, 'outflow': {}}
            
            # Extract the last number from filename as the value
            try:
                value = float(re.search(r'_([\d.]+)\.png$', file).group(1))
            except (AttributeError, ValueError):
                print(f"Warning: Could not extract value from filename: {file}")
                continue
                
            if flow_type == 'inflow':
                if 'gt' in file:
                    frame_results[frame_info]['inflow']['gt'] = value
                else:
                    frame_results[frame_info]['inflow']['pred'] = value
            elif flow_type == 'outflow':
                if 'gt' in file:
                    frame_results[frame_info]['outflow']['gt'] = value
                else:
                    frame_results[frame_info]['outflow']['pred'] = value
        
        # Sort by frame number
        sorted_frames = sorted(frame_results.keys(), 
                             key=lambda x: [int(i) for i in x.split('_')])
        
        # Organize results
        for frame in sorted_frames:
            frame_data = frame_results[frame]
            
            if 'inflow' in frame_data:
                if 'pred' in frame_data['inflow']:
                    pred_dict['inflow'].append(frame_data['inflow']['pred'])
                if 'gt' in frame_data['inflow']:
                    gt_dict['inflow'].append(frame_data['inflow']['gt'])
                    
            if 'outflow' in frame_data:
                if 'pred' in frame_data['outflow']:
                    pred_dict['outflow'].append(frame_data['outflow']['pred'])
                if 'gt' in frame_data['outflow']:
                    gt_dict['outflow'].append(frame_data['outflow']['gt'])
        
        # Set time values
        if pred_dict['inflow']:
            pred_dict['time'] = len(pred_dict['inflow'])
            gt_dict['time'] = len(gt_dict['inflow'])
            
    except Exception as e:
        print(f"Error loading results from {scene_dir}: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # Handle empty lists
    if not pred_dict['inflow']:
        pred_dict['inflow'] = [0]
    if not pred_dict['outflow']:
        pred_dict['outflow'] = [0]
    if not gt_dict['inflow']:
        gt_dict['inflow'] = [0]
    if not gt_dict['outflow']:
        gt_dict['outflow'] = [0]
    
    # Ensure consistent lengths
    max_len = max(len(pred_dict['inflow']), len(pred_dict['outflow']),
                 len(gt_dict['inflow']), len(gt_dict['outflow']))
    pred_dict['inflow'] = pred_dict['inflow'] + [0] * (max_len - len(pred_dict['inflow']))
    pred_dict['outflow'] = pred_dict['outflow'] + [0] * (max_len - len(pred_dict['outflow']))
    gt_dict['inflow'] = gt_dict['inflow'] + [0] * (max_len - len(gt_dict['inflow']))
    gt_dict['outflow'] = gt_dict['outflow'] + [0] * (max_len - len(gt_dict['outflow']))
    
    # Update time values
    pred_dict['time'] = max_len
    gt_dict['time'] = max_len
    
    return pred_dict, gt_dict

def evaluate_all_scenes(base_dir, output_file, intervals=1):
    """
    Evaluate results for all scenes and save to file
    Args:
        base_dir: Base directory path containing scene folders
        output_file: Path to save evaluation results
        intervals: Time interval parameter (default: 1)
    """
    all_metrics = []
    scenes_pred_dict = []
    scenes_gt_dict = []
    
    with open(output_file, 'w', encoding='utf-8') as f:
        # Write evaluation header
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        header = f"Evaluation Results - {current_time}\n"
        header += "=" * 50 + "\n"
        header += f"Intervals: {intervals}\n\n"
        f.write(header)
        print(header)

        # Process each scene directory
        scene_dirs = sorted([d for d in os.listdir(base_dir) 
                           if os.path.isdir(os.path.join(base_dir, d))])
        
        for scene_name in scene_dirs:
            scene_dir = os.path.join(base_dir, scene_name)
            
            # Load scene results
            pred_dict, gt_dict = load_results(scene_dir)
            scenes_pred_dict.append(pred_dict)
            scenes_gt_dict.append(gt_dict)
            
            try:
                # Calculate metrics for single scene
                MAE, MSE, WRAE, MIAE, MOAE, metrics_mae = compute_metrics_all_scenes(
                    [pred_dict], [gt_dict], intervals)
                
                # Convert tensors to scalar values
                metrics = {
                    'MAE': float(MAE.item() if torch.is_tensor(MAE) else MAE),
                    'MSE': float(MSE.item() if torch.is_tensor(MSE) else MSE),
                    'WRAE': float(WRAE.item() if torch.is_tensor(WRAE) else WRAE),
                    'MIAE': float(MIAE.item() if torch.is_tensor(MIAE) else MIAE),
                    'MOAE': float(MOAE.item() if torch.is_tensor(MOAE) else MOAE)
                }
                all_metrics.append(metrics)
                
                # Format scene results
                scene_result = f"\nScene: {scene_name}\n"
                scene_result += f"Time frames: {pred_dict['time']}\n"
                scene_result += "Final counts:\n"
                scene_result += f"  Predicted: inflow={pred_dict['inflow'][-1]:.2f}, outflow={pred_dict['outflow'][-1]:.2f}\n"
                scene_result += f"  Ground truth: inflow={gt_dict['inflow'][-1]:.2f}, outflow={gt_dict['outflow'][-1]:.2f}\n"
                scene_result += "Metrics:\n"
                scene_result += f"  MAE: {metrics['MAE']:.2f}\n"
                scene_result += f"  MSE: {metrics['MSE']:.2f}\n"
                scene_result += f"  WRAE: {metrics['WRAE']:.2f}%\n"
                scene_result += f"  MIAE: {metrics['MIAE']:.2f}\n"
                scene_result += f"  MOAE: {metrics['MOAE']:.2f}\n"
                
                f.write(scene_result)
                print(scene_result)
                
            except Exception as e:
                error_msg = f"\nError processing scene {scene_name}: {str(e)}\n"
                f.write(error_msg)
                print(error_msg)
                continue
        
        # Calculate and write overall metrics
        if all_metrics:
            # Calculate final metrics for all scenes
            MAE, MSE, WRAE, MIAE, MOAE, _ = compute_metrics_all_scenes(
                scenes_pred_dict, scenes_gt_dict, intervals)
            
            overall_metrics = {
                'MAE': float(MAE.item() if torch.is_tensor(MAE) else MAE),
                'MSE': float(MSE.item() if torch.is_tensor(MSE) else MSE),
                'WRAE': float(WRAE.item() if torch.is_tensor(WRAE) else WRAE),
                'MIAE': float(MIAE.item() if torch.is_tensor(MIAE) else MIAE),
                'MOAE': float(MOAE.item() if torch.is_tensor(MOAE) else MOAE)
            }
            
            # Write overall results
            overall_result = "\n" + "=" * 50 + "\n"
            overall_result += "Overall Metrics:\n"
            overall_result += f"  Total Scenes: {len(all_metrics)}\n"
            overall_result += f"  MAE: {overall_metrics['MAE']:.2f}\n"
            overall_result += f"  MSE: {overall_metrics['MSE']:.2f}\n"
            overall_result += f"  WRAE: {overall_metrics['WRAE']:.2f}%\n"
            overall_result += f"  MIAE: {overall_metrics['MIAE']:.2f}\n"
            overall_result += f"  MOAE: {overall_metrics['MOAE']:.2f}\n"
            
            f.write(overall_result)
            print(overall_result)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate scene results')
    parser.add_argument('--base_dir', type=str, 
                      default='../dataset/demo_den_test2_SENSE_SENSE',
                      help='Base directory containing scene folders')
    parser.add_argument('--output_file', type=str, 
                      default='evaluation_results.txt',
                      help='Path to save evaluation results')
    parser.add_argument('--intervals', type=int, default=1,
                      help='Time interval parameter (default: 1)')
    
    args = parser.parse_args()
    
    print(f"Evaluating results in: {args.base_dir}")
    print(f"Results will be saved to: {args.output_file}")
    
    evaluate_all_scenes(args.base_dir, args.output_file, args.intervals)
    print(f"\nResults have been saved to {args.output_file}") 