import os
import numpy as np
import torch
from tqdm import tqdm
import json

def compute_metrics_single_scene(pred_dict, gt_dict):
    """
    计算单个场景的评估指标
    """
    # 计算流入流出误差
    inflow_error = abs(pred_dict['pre_inflow'] - gt_dict['gt_inflow'])
    outflow_error = abs(pred_dict['pre_outflow'] - gt_dict['gt_outflow'])
    
    # 计算总人数误差
    total_error = abs((pred_dict['pre_inflow'] + pred_dict['pre_outflow']) - 
                     (gt_dict['gt_inflow'] + gt_dict['gt_outflow']))
    
    # 计算相对误差
    total_gt = gt_dict['gt_inflow'] + gt_dict['gt_outflow']
    if total_gt > 0:
        relative_error = total_error / total_gt
    else:
        relative_error = float('inf')
    
    return {
        'MAE': total_error,
        'MSE': total_error ** 2,
        'WRAE': relative_error,
        'MIAE': inflow_error,
        'MOAE': outflow_error
    }

def compute_metrics_all_scenes(output_dir):
    """
    计算所有场景的评估指标
    """
    metrics = {
        'MAE': [],
        'MSE': [],
        'WRAE': [],
        'MIAE': [],
        'MOAE': []
    }
    
    # 遍历输出目录中的所有场景
    for scene_dir in tqdm(os.listdir(output_dir)):
        scene_path = os.path.join(output_dir, scene_dir)
        if not os.path.isdir(scene_path):
            continue
            
        # 读取预测结果
        pred_file = os.path.join(scene_path, 'prediction.json')
        if not os.path.exists(pred_file):
            continue
            
        with open(pred_file, 'r') as f:
            pred_dict = json.load(f)
            
        # 读取真实结果
        gt_file = os.path.join(scene_path, 'ground_truth.json')
        if not os.path.exists(gt_file):
            continue
            
        with open(gt_file, 'r') as f:
            gt_dict = json.load(f)
            
        # 计算指标
        scene_metrics = compute_metrics_single_scene(pred_dict, gt_dict)
        
        # 添加到总指标中
        for key in metrics:
            metrics[key].append(scene_metrics[key])
    
    # 计算平均指标
    avg_metrics = {}
    for key in metrics:
        if key == 'WRAE':
            # 对于相对误差，使用几何平均
            avg_metrics[key] = np.exp(np.mean(np.log(metrics[key])))
        else:
            avg_metrics[key] = np.mean(metrics[key])
    
    return avg_metrics

if __name__ == '__main__':
    output_dir = '../dataset/demo_den_test2_SENSE'  # 修改为实际的输出目录
    metrics = compute_metrics_all_scenes(output_dir)
    
    print("\nEvaluation Metrics:")
    print("==================")
    print(f"MAE: {metrics['MAE']:.2f}")
    print(f"MSE: {metrics['MSE']:.2f}")
    print(f"WRAE: {metrics['WRAE']:.2f}")
    print(f"MIAE: {metrics['MIAE']:.2f}")
    print(f"MOAE: {metrics['MOAE']:.2f}") 