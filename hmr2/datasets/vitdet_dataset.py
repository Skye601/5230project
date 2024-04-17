from typing import Dict
import os
import cv2
import numpy as np
from skimage.filters import gaussian
from yacs.config import CfgNode
import torch
from PIL import Image


from .utils import (convert_cvimg_to_tensor,
                    expand_to_aspect_ratio,
                    generate_image_patch_cv2)

DEFAULT_MEAN = 255. * np.array([0.485, 0.456, 0.406])
DEFAULT_STD = 255. * np.array([0.229, 0.224, 0.225])

class ViTDetDataset(torch.utils.data.Dataset):

    def __init__(self,
                 cfg: CfgNode,
                 img_cv2: np.array,
                 boxes: np.array,
                 train: bool = False,
                 **kwargs):
        super().__init__()
        self.cfg = cfg
        self.img_cv2 = img_cv2
        # self.boxes = boxes

        assert train == False, "ViTDetDataset is only for inference"
        self.train = train
        self.img_size = cfg.MODEL.IMAGE_SIZE
        self.mean = 255. * np.array(self.cfg.MODEL.IMAGE_MEAN)
        self.std = 255. * np.array(self.cfg.MODEL.IMAGE_STD)

        # Preprocess annotations
        boxes = boxes.astype(np.float32)
        self.center = (boxes[:, 2:4] + boxes[:, 0:2]) / 2.0
        self.scale = (boxes[:, 2:4] - boxes[:, 0:2]) / 200.0
        self.personid = np.arange(len(boxes), dtype=np.int32)

    def __len__(self) -> int:
        return len(self.personid)

    def __getitem__(self, idx: int) -> Dict[str, np.array]:

        center = self.center[idx].copy()
        center_x = center[0]
        center_y = center[1]

        scale = self.scale[idx]
        BBOX_SHAPE = self.cfg.MODEL.get('BBOX_SHAPE', None)
        bbox_size = expand_to_aspect_ratio(scale*200, target_aspect_ratio=BBOX_SHAPE).max()

        patch_width = patch_height = self.img_size

        # 3. generate image patch
        # if use_skimage_antialias:
        cvimg = self.img_cv2.copy()
        if True:
            # Blur image to avoid aliasing artifacts
            downsampling_factor = ((bbox_size*1.0) / patch_width)
            print(f'{downsampling_factor=}')
            downsampling_factor = downsampling_factor / 2.0
            if downsampling_factor > 1.1:
                cvimg  = gaussian(cvimg, sigma=(downsampling_factor-1)/2, channel_axis=2, preserve_range=True)


        img_patch_cv, trans = generate_image_patch_cv2(cvimg,
                                                    center_x, center_y,
                                                    bbox_size, bbox_size,
                                                    patch_width, patch_height,
                                                    False, 1.0, 0,
                                                    border_mode=cv2.BORDER_CONSTANT)
        img_patch_cv = img_patch_cv[:, :, ::-1]
        img_patch = convert_cvimg_to_tensor(img_patch_cv)

        # apply normalization
        # for n_c in range(min(self.img_cv2.shape[2], 3)):
        #     img_patch[n_c, :, :] = (img_patch[n_c, :, :] - self.mean[n_c]) / self.std[n_c]

        item = {
            'img': img_patch,
            'personid': int(self.personid[idx]),
        }
        item['box_center'] = self.center[idx].copy()
        item['box_size'] = bbox_size
        item['img_size'] = 1.0 * np.array([cvimg.shape[1], cvimg.shape[0]])
        return item


def draw_bbox_save_video(images, bboxes, sv_path):
    images = np.array(images)
    sv_video = cv2.VideoWriter(sv_path, cv2.VideoWriter_fourcc(*'mp4v'), 20, (images.shape[2], images.shape[1]))
    for img, box in zip(images, bboxes):
        box = box[0].astype('int32')
        img = cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color=(255, 0, 0), thickness=2)
        sv_video.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR).astype('uint8'))
    sv_video.release()


class DepthVideoDataset(torch.utils.data.Dataset):
    def __init__(self,
                 cfg: CfgNode,
                 video_path: str,
                 train: bool = False,
                 fov: float = np.pi/2,
                 depth_max=7,
                 force_bbox_is_whole_img=False,
                 is_depth: bool=False,
                 use_depth: bool=True, # Use depth for prediction rather than disparity=focal/depth
                 **kwargs):
        super().__init__()
        self.cfg = cfg
        self.img_size = cfg.MODEL.IMAGE_SIZE
        self.force_bbox_is_whole_img = force_bbox_is_whole_img
        if video_path.endswith('.mp4') or video_path.endswith('.avi'):
            vidcap = cv2.VideoCapture(video_path)
            success, img = vidcap.read()
            h, w = img.shape[:2]
            is_video = True
        else:
            img_paths = [os.path.join(video_path, f) for f in sorted(os.listdir(video_path))]
            img_idx = 0
            img = np.asarray(Image.open(img_paths[img_idx]))[..., None]
            h, w = img.shape[:2]
            success = True
            is_video = False            
        focal = h / 2 / np.tan(fov/2)
        disparitys = []
        disparity_colors = []
        boxes_list = []
        detection_flag_list = []
        
        # Set up detector
        from pathlib import Path
        from hmr2.utils.utils_detectron2 import DefaultPredictor_Lazy
        import hmr2
        from detectron2.config import LazyConfig
        cfg_path = Path(hmr2.__file__).parent/'configs'/'cascade_mask_rcnn_vitdet_h_75ep.py'
        detectron2_cfg = LazyConfig.load(str(cfg_path))
        detectron2_cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
        for i in range(3):
            detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
        detector = DefaultPredictor_Lazy(detectron2_cfg)

        # from ultralytics import YOLO
        # detector = YOLO('yolov8n.pt') 

        while success:
            if is_depth:
                if is_video:
                    depth = np.asarray(img, dtype=np.float32) / 256 * depth_max  # convert to m
                else:
                    depth = np.asarray(img, dtype=np.float32) / 1000
                depth = depth.mean(axis=-1)
                near, far = 0.1, depth_max
                depth = np.where(depth<=1e-3, far*np.ones_like(depth), depth)
                focal = 60
                depth_clip = np.clip(depth, near, far)
                depth_clip = np.stack([depth_clip]*3, axis=-1)
                if use_depth:
                    disparity = focal / depth_clip
                else:
                    disparity = depth_clip
                # Normalize to -1 ~ 1 (Approximate)
                disparity = disparity - np.median(disparity)
                disparity = disparity / np.absolute(disparity).mean()
                # Convert to color (0 - 255)
                disparity_color = (depth_clip - depth_clip.min()) / (depth_clip.max() - depth_clip.min())
                disparity_color = (disparity_color * 255).astype(np.uint8)
            else:
                ratio=max(img.shape)/1024
                H_new, W_new = int(img.shape[0] / ratio), int(img.shape[1] / ratio)
                img = cv2.resize(img, (W_new, H_new))
                for n_c in range(min(img.shape[2], 3)):
                    img[n_c, :, :] = (img[n_c, :, :] - DEFAULT_MEAN[n_c]) / DEFAULT_STD[n_c]
                disparity = np.asarray(img, dtype=np.float32) 
                disparity_color= np.asarray(img, dtype=np.float32) 
                
            disparitys.append(disparity)
            disparity_colors.append(disparity_color)
            if is_video:
                success, img = vidcap.read()
            else:
                img_idx += 1
                success = img_idx < len(img_paths)
                if success:
                    img = np.asarray(Image.open(img_paths[img_idx]))[..., None]
        if is_video:
            vidcap.release()
        self.disparitys = np.stack(disparitys)
        self.disparity_colors = np.stack(disparity_colors)
        print("***********disparity_colors: ", self.disparity_colors.shape)
        detect_batch = 8
        start = 0
        while start < self.disparity_colors.shape[0]:
            end = min(self.disparity_colors.shape[0], start+detect_batch)
            det_outs = detector(self.disparity_colors[start: end])
            for det_out in det_outs:
                det_instances = det_out['instances']
                valid_idx = (det_instances.pred_classes==0) & (det_instances.scores > 0.45)
                if len(valid_idx) > 0 and sum(valid_idx) > 0 and not self.force_bbox_is_whole_img:
                    print(f'Detect {sum(valid_idx).item()} objects!')
                    boxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
                    detection_flag_list.append(True)
                else:
                    print('Failed to detect objects, use the whole image as bbx!')
                    boxes = np.array([[0, 0, disparity_color.shape[1], disparity_color.shape[0]]])
                    detection_flag_list.append(False)
                boxes_list.append(boxes)

            # det_outs = detector.track(disparity_color)
            # for det_out in det_outs:
            #     valid_idx = (det_out[0].boxes.cls==0) & (det_out[0].boxes.conf > 0.4)
            #     if len(valid_idx) > 0 and sum(valid_idx) > 0:
            #         print(f'Detect {sum(valid_idx).item()} objects!')
            #         boxes = det_out[0].boxes.xyxy[valid_idx].cpu().numpy()
            #         detection_flag_list.append(True)
            #     else:
            #         print('Failed to detect objects, use the whole image as bbx!')
            #         boxes = np.array([[0, 0, disparity_color.shape[1], disparity_color.shape[0]]])
            #         detection_flag_list.append(False)
            #     boxes_list.append(boxes)
            start = end
        self.boxes_list = boxes_list

        draw_bbox_save_video(disparity_colors, self.boxes_list, './raw_detection.mp4')

        # Interpolate the missing detected bboxes
        if not self.force_bbox_is_whole_img and not np.all(detection_flag_list):
            interval_idx1, interval_idx2 = 0, 0
            total_len = len(detection_flag_list)
            while True:
                while interval_idx1 < total_len and detection_flag_list[interval_idx1]:
                    interval_idx1 += 1
                if interval_idx1 == total_len:
                    break
                interval_idx2 = interval_idx1 + 1
                while interval_idx2 < total_len and not detection_flag_list[interval_idx2]:
                    interval_idx2 += 1
                if interval_idx1 == 0 and interval_idx2 == total_len:
                    break
                boxes1 = boxes_list[interval_idx1-1] if interval_idx1 > 0 else boxes_list[interval_idx2]
                boxes2 = boxes_list[interval_idx2] if interval_idx2 < total_len else boxes_list[interval_idx1-1]
                weights = np.linspace(0, 1, interval_idx2-interval_idx1+2, endpoint=True)[1:-1]
                for idx in range(interval_idx1, interval_idx2):
                    weight = weights[idx-interval_idx1]
                    boxes_list[idx] = boxes1 * (1 - weight) + boxes2 * weight
                print(f'Interpolating interval: {interval_idx1}-{interval_idx2}.')
                interval_idx1 = interval_idx2

        draw_bbox_save_video(disparity_colors, self.boxes_list, './filtered_detection.mp4')

        assert train == False, "ViTDetDataset is only for inference"
        self.train = train

    def __len__(self) -> int:
        return len(self.disparitys)

    def __getitem__(self, idx: int) -> Dict[str, np.array]:

        # Preprocess annotations
        boxes = self.boxes_list[idx][:1]  # Only return the first person!
        boxes = boxes.astype(np.float32)
        center = (boxes[0, 2:4] + boxes[0, 0:2]) / 2.0
        scale = (boxes[0, 2:4] - boxes[0, 0:2]) / 200.0
        personid = np.arange(len(boxes), dtype=np.int32)

        center_x = center[0]
        center_y = center[1]

        BBOX_SHAPE = self.cfg.MODEL.get('BBOX_SHAPE', None)
        bbox_size = expand_to_aspect_ratio(scale*200, target_aspect_ratio=BBOX_SHAPE).max()

        patch_width = patch_height = self.img_size

        # 3. generate image patch
        # if use_skimage_antialias:
        cvimg = self.disparitys[idx]
        if True:
            # Blur image to avoid aliasing artifacts
            downsampling_factor = ((bbox_size*1.0) / patch_width)
            print(f'{downsampling_factor=}')
            downsampling_factor = downsampling_factor / 2.0
            if downsampling_factor > 1.1:
                cvimg  = gaussian(cvimg, sigma=(downsampling_factor-1)/2, channel_axis=2, preserve_range=True)

        img_patch_cv, trans = generate_image_patch_cv2(cvimg,
                                                    center_x, center_y,
                                                    bbox_size, bbox_size,
                                                    patch_width, patch_height,
                                                    False, 1.0, 0,
                                                    border_mode=cv2.BORDER_CONSTANT)
        img_patch_cv = img_patch_cv[:, :, ::-1]
        img_patch = convert_cvimg_to_tensor(img_patch_cv)

        color_patch_cv, _ = generate_image_patch_cv2(self.disparity_colors[idx],
                                                    center_x, center_y,
                                                    bbox_size, bbox_size,
                                                    patch_width, patch_height,
                                                    False, 1.0, 0,
                                                    border_mode=cv2.BORDER_CONSTANT)
        color_patch = convert_cvimg_to_tensor(color_patch_cv)

        item = {
            'img': img_patch,
            'disparity_color': color_patch,
        }
        item['box_center'] = center
        item['box_size'] = bbox_size
        item['img_size'] = 1.0 * np.array([cvimg.shape[1], cvimg.shape[0]])
        return item


class NTUDataset(torch.utils.data.Dataset):
    def __init__(self,
                 cfg: CfgNode,
                 rgb_path: str,
                 depth_path: str,
                 train: bool = False,
                 fov: float = np.pi/2,
                 depth_max=7.,
                 force_bbox_is_whole_img=False,
                 is_depth: bool=False,
                 **kwargs):
        super().__init__()
        self.rgb_dataset = DepthVideoDataset(video_path=rgb_path, cfg=cfg, train=train, fov=fov, depth_max=depth_max, force_bbox_is_whole_img=force_bbox_is_whole_img, is_depth=False)
        self.depth_dataset = DepthVideoDataset(video_path=depth_path, cfg=cfg, train=train, fov=fov, depth_max=depth_max, force_bbox_is_whole_img=force_bbox_is_whole_img, is_depth=True)

    def __len__(self) -> int:
        return len(self.rgb_dataset)

    def __getitem__(self, idx: int) -> Dict[str, np.array]:

        depth_item = self.depth_dataset.__getitem__(idx)
        rgb_item = self.rgb_dataset.__getitem__(idx)

        # rgb, dep = rgb_item['img'], depth_item['img']
        rgb = {'img': rgb_item['img']}
        dep = {'img': depth_item['img']}
        return rgb, dep

class NTUDataset_dir(torch.utils.data.Dataset):
    def __init__(self,
                 cfg: CfgNode,
                 rgb_dir: str,
                 depth_dir: str,
                 train: bool = False,
                 **kwargs):
        super().__init__()
        self.rgb_videos = [os.path.join(rgb_dir, f) for f in os.listdir(rgb_dir) if f.endswith('.avi')]
        self.depth_videos = [os.path.join(depth_dir, f) for f in os.listdir(depth_dir) if f.endswith('.mp4')]
        assert len(self.rgb_videos) == len(self.depth_videos) 

        self.rgb_datasets = [DepthVideoDataset(v, cfg, train, **kwargs, is_depth=False) for v in self.rgb_videos]
        self.depth_datasets = [DepthVideoDataset(v, cfg, train, **kwargs, is_depth=True) for v in self.depth_videos]
        self.total_frames = sum(len(ds) for ds in self.rgb_datasets)

    def __len__(self) -> int:
        return self.total_frames

    def __getitem__(self, idx: int) -> Dict[str, np.array]:

        depth_item = self.depth_dataset.__getitem__(idx)
        rgb_item = self.rgb_dataset.__getitem__(idx)

        # rgb, dep = rgb_item['img'], depth_item['img']
        rgb = {'img': rgb_item['img']}
        dep = {'img': depth_item['img']}
        return rgb, dep
