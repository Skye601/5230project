import argparse
import os
from pathlib import Path
import traceback
from typing import Optional
import numpy as np
import pandas as pd
import torch
from filelock import FileLock
from hmr2.configs import dataset_eval_config
from hmr2.datasets import create_dataset
from hmr2.utils import Evaluator, recursive_to
from tqdm import tqdm
from hmr2.datasets.vitdet_dataset import NTUDataset, DepthVideoDataset

from hmr2.configs import CACHE_DIR_4DHUMANS
from hmr2.models import HMR2, download_models, load_hmr2, DEFAULT_CHECKPOINT
from eval_utils import (
    compute_accel,
    compute_error_accel,
    compute_error_verts,
    batch_compute_similarity_transform_torch,
)

def main():
    parser = argparse.ArgumentParser(description='Evaluate trained models')
    parser.add_argument('--checkpoint', type=str, default=DEFAULT_CHECKPOINT, help='Path to pretrained model checkpoint')
    parser.add_argument('--results_file', type=str, default='results/eval_regression.csv', help='Path to results file.')
    parser.add_argument('--dataset', type=str, default='H36M-VAL-P2', help='Dataset to evaluate') # choices=['H36M-VAL-P2', '3DPW-TEST', 'MPI-INF-TEST']
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size for inference')
    parser.add_argument('--num_samples', type=int, default=1, help='Number of test samples to draw')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers used for data loading')
    parser.add_argument('--log_freq', type=int, default=10, help='How often to log results')
    parser.add_argument('--shuffle', dest='shuffle', action='store_true', default=False, help='Shuffle the dataset during evaluation')
    parser.add_argument('--exp_name', type=str, default=None, help='Experiment name')

    parser.add_argument('--rgb_path', type=str, default="/home/heyuting/dataset/NTU-RGBD/nturgb+d_rgb_s018/nturgb+d_rgb/S018C001P008R001A081_rgb.avi", help='RGB Path')
    parser.add_argument('--depth_path', type=str, default="/home/heyuting/dataset/NTU-RGBD/nturgb+d_depth_s018/nturgb+d_depth/S018C002P043R001A081/", help='Depth Path')

    args = parser.parse_args()

    # Download and load checkpoints
    model_rgb, model_cfg = load_hmr2(DEFAULT_CHECKPOINT)

    # download_models(CACHE_DIR_4DHUMANS)
    args.checkpoint = '/home/heyuting/code/4D-Humans/logs/train/runs/depth_train_t/checkpoints/epoch=8-step=239000.ckpt'
    model_depth, model_cfg = load_hmr2(args.checkpoint)

    # Setup HMR2.0 model
    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    device = 'cuda:0'
    model_depth = model_depth.to(device)
    model_depth.eval()

    model_rgb = model_rgb.to(device)
    model_rgb.eval()

    run_eval(model_depth, model_rgb, model_cfg, device, args)

def run_eval(model_depth, model_rgb, model_cfg, device, args):
    dataset = NTUDataset(model_cfg, rgb_path=args.rgb_path, depth_path=args.depth_path, force_bbox_is_whole_img=False)
    dataloader = torch.utils.data.DataLoader(dataset, args.batch_size, shuffle=False, num_workers=args.num_workers)

    evaluation_accumulators = dict.fromkeys(['pred_j3d', 'target_j3d', 'target_verts', 'pred_verts'])
    for k,v in evaluation_accumulators.items():
        evaluation_accumulators[k] = []

    # Go over the images in the dataset.
    try:
        for i, batch in enumerate(tqdm(dataloader)):
            img_rgb, img_depth = recursive_to(batch, device)
            with torch.no_grad():
                out_rgb = model_rgb(img_rgb)
                out_depth = model_depth(img_depth)
            pred_j3d = out_depth['pred_keypoints_3d'].cpu().numpy() 
            pred_verts = out_depth['pred_vertices'].cpu().numpy()
            evaluation_accumulators['pred_verts'].append(pred_verts)
            evaluation_accumulators['pred_j3d'].append(pred_j3d)

            target_j3d = out_rgb['pred_keypoints_3d'].cpu().numpy() 
            target_verts = out_rgb['pred_vertices'].cpu().numpy()
            evaluation_accumulators['target_verts'].append(target_verts)
            evaluation_accumulators['target_j3d'].append(target_j3d)
            
    except (Exception, KeyboardInterrupt) as e:
        traceback.print_exc()
        error = repr(e)
        i = 0

    for k, v in evaluation_accumulators.items():
        evaluation_accumulators[k] = np.vstack(v)

    pred_j3ds = evaluation_accumulators['pred_j3d']
    target_j3ds = evaluation_accumulators['target_j3d']

    pred_j3ds = torch.from_numpy(pred_j3ds).float()
    target_j3ds = torch.from_numpy(target_j3ds).float()

    print(f'Evaluating on {pred_j3ds.shape[0]} number of poses...')
    pred_pelvis = (pred_j3ds[:,[2],:] + pred_j3ds[:,[3],:]) / 2.0
    target_pelvis = (target_j3ds[:,[2],:] + target_j3ds[:,[3],:]) / 2.0


    pred_j3ds -= pred_pelvis
    target_j3ds -= target_pelvis

    # Absolute error (MPJPE)
    errors = torch.sqrt(((pred_j3ds - target_j3ds) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
    S1_hat = batch_compute_similarity_transform_torch(pred_j3ds, target_j3ds)
    errors_pa = torch.sqrt(((S1_hat - target_j3ds) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
    pred_verts = evaluation_accumulators['pred_verts']
    target_verts = evaluation_accumulators['target_verts']

    m2mm = 1000

    pve = np.mean(compute_error_verts(target_verts=target_verts, pred_verts=pred_verts)) * m2mm
    accel = np.mean(compute_accel(pred_j3ds)) * m2mm
    accel_err = np.mean(compute_error_accel(joints_pred=pred_j3ds, joints_gt=target_j3ds)) * m2mm
    mpjpe = np.mean(errors) * m2mm
    pa_mpjpe = np.mean(errors_pa) * m2mm

    eval_dict = {
        'mpjpe': mpjpe,
        'pa-mpjpe': pa_mpjpe,
        'pve': pve,
        'accel': accel,
        'accel_err': accel_err
    }

    log_str = ' '.join([f'{k.upper()}: {v:.4f},'for k,v in eval_dict.items()])
    print(log_str)


if __name__ == '__main__':
    main()
