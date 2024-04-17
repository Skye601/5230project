from pathlib import Path
import torch
import argparse
import os
import cv2
import numpy as np
from PIL import Image

from hmr2.configs import CACHE_DIR_4DHUMANS
from hmr2.models import HMR2, download_models, load_hmr2, DEFAULT_CHECKPOINT
from hmr2.utils import recursive_to
from hmr2.datasets.vitdet_dataset import DepthVideoDataset
from hmr2.utils.renderer import Renderer, cam_crop_to_full
from matplotlib import cm
import tqdm


LIGHT_BLUE=(0.65098039,  0.74117647,  0.85882353)


def main():
    parser = argparse.ArgumentParser(description='HMR2 demo code')
    parser.add_argument('--checkpoint', type=str, default=DEFAULT_CHECKPOINT, help='Path to pretrained model checkpoint')
    parser.add_argument('--video_path', type=str, default='./example_data/depth_video/AD_11.mp4', help='Folder with input images')
    parser.add_argument('--out_path', type=str, default='./demo_out.mp4', help='Output folder to save rendered results')
    parser.add_argument('--out_folder', type=str, default='demo_out', help='Output folder to save rendered results')
    parser.add_argument('--side_view', dest='side_view', action='store_true', default=False, help='If set, render side view also')
    parser.add_argument('--top_view', dest='top_view', action='store_true', default=False, help='If set, render top view also')
    parser.add_argument('--full_frame', dest='full_frame', action='store_true', default=False, help='If set, render all people together also')
    parser.add_argument('--save_mesh', dest='save_mesh', action='store_true', default=False, help='If set, save meshes to disk also')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for inference/fitting')

    args = parser.parse_args()

    # Download and load checkpoints
    # download_models(CACHE_DIR_4DHUMANS)
    args.checkpoint = '/home/heyuting/code/4D-Humans/logs/train/runs/depth_train_t/checkpoints/last.ckpt'
    model, model_cfg = load_hmr2(args.checkpoint)

    # Setup HMR2.0 model
    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    device = 'cuda:0'
    model = model.to(device)
    model.eval()

    renderer = Renderer(model_cfg, faces=model.smpl.faces)

    # Make output directory if it does not exist
    os.makedirs(args.out_folder, exist_ok=True)
    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)

    dataset = DepthVideoDataset(model_cfg, video_path=args.video_path, is_depth=True, force_bbox_is_whole_img=False)
    if args.video_path.endswith('.mp4'):
        vidcap = cv2.VideoCapture(args.video_path)
        fps = vidcap.get(cv2.CAP_PROP_FPS)
    else:
        fps = 20
    sv_video = None
    count = 0
    # Run HMR2.0 on all detected humans
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)
    
    pred_smpl_kp_3d =[]
    for batch in tqdm.tqdm(dataloader):
        batch = recursive_to(batch, device)
        with torch.no_grad():
            out = model(batch, is_depth=True)
        
        pred_cam = out['pred_cam']
        box_center = batch["box_center"].float()
        box_size = batch["box_size"].float()
        img_size = batch["img_size"].float()
        scaled_focal_length = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()
        pred_cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length).detach().cpu().numpy()

        # Render the result
        batch_size = batch['img'].shape[0]
        pred_smpl_kp_3d.append(out['smpl_kp_3d'].reshape(batch_size, -1, 3))

        for n in range(batch_size):
            # Get filename from path img_path
            img_fn, _ = os.path.splitext(os.path.basename(args.video_path))
            white_img = torch.ones_like(batch['img'][n]).cpu()
            input_patch = batch['disparity_color'][n].cpu() / 255.
            # input_patch = ((input_patch - input_patch.median()) / (2 * input_patch.std()) + .5).clip(0, 1)  # Better normalization to avoid outlier
            # input_patch = (input_patch - input_patch.min()) / (input_patch.max() - input_patch.min())
            input_patch = input_patch.permute(1,2,0).numpy()

            regression_img = renderer(out['pred_vertices'][n].detach().cpu().numpy(),
                                    out['pred_cam_t'][n].detach().cpu().numpy(),
                                    batch['disparity_color'][n],
                                    mesh_base_color=LIGHT_BLUE,
                                    scene_bg_color=(1, 1, 1),
                                    is_depth=True,
                                    )

            final_img = np.concatenate([input_patch, regression_img], axis=1)

            if args.side_view:
                side_img = renderer(out['pred_vertices'][n].detach().cpu().numpy(),
                                        out['pred_cam_t'][n].detach().cpu().numpy(),
                                        white_img,
                                        mesh_base_color=LIGHT_BLUE,
                                        scene_bg_color=(1, 1, 1),
                                        side_view=True)
                final_img = np.concatenate([final_img, side_img], axis=1)

            if args.top_view:
                top_img = renderer(out['pred_vertices'][n].detach().cpu().numpy(),
                                        out['pred_cam_t'][n].detach().cpu().numpy(),
                                        white_img,
                                        mesh_base_color=LIGHT_BLUE,
                                        scene_bg_color=(1, 1, 1),
                                        top_view=True)
                final_img = np.concatenate([final_img, top_img], axis=1)
            
            cv2.imwrite(os.path.join(args.out_folder, f'{n}.png'), 255*final_img[:, :, ::-1])

            if sv_video is None:
                sv_video = cv2.VideoWriter(args.out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (final_img.shape[1], final_img.shape[0]))
                sv_video_ori = cv2.VideoWriter('./ori_depth.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (input_patch.shape[1], input_patch.shape[0]))
            sv_video.write(cv2.cvtColor(255*final_img[:, :, ::-1], cv2.COLOR_RGB2BGR).astype('uint8'))
            sv_video_ori.write(cv2.cvtColor(255*input_patch[:, :, ::-1], cv2.COLOR_RGB2BGR).astype('uint8'))
            # Add all verts and cams to list
            verts = out['pred_vertices'][n].detach().cpu().numpy()
            cam_t = pred_cam_t_full[n]

            # Save all meshes to disk
            if args.save_mesh:
                camera_translation = cam_t.copy()
                tmesh = renderer.vertices_to_trimesh(verts, camera_translation, LIGHT_BLUE)
                tmesh.export(os.path.join(args.out_folder, f'{img_fn}_{count}.obj'))
            count += 1
    
    pred_kp_joints3d = torch.cat(pred_smpl_kp_3d, dim=0)
    pred_kp_joints3d = pred_kp_joints3d.cpu().numpy()
    np.save("hmr_joints3d.npy", pred_kp_joints3d)
    print(f'Video saved to {args.out_path}.')
    print(f'Keypoints saved to \"hmr_joints3d.npy\".')
    sv_video.release()


if __name__ == '__main__':
    main()
