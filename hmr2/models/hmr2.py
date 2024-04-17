import torch
import pytorch_lightning as pl
from typing import Any, Dict, Mapping, Tuple
import numpy as np
from yacs.config import CfgNode

from ..utils import SkeletonRenderer, MeshRenderer
from ..utils.geometry import aa_to_rotmat, perspective_projection
from ..utils.pylogger import get_pylogger
from .backbones import create_backbone
from .heads import build_smpl_head
from .discriminator import Discriminator
from .losses import Keypoint3DLoss, Keypoint2DLoss, ParameterLoss
from . import SMPL

from ..depth_anything.dpt import DepthAnything
from ..depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
from torchvision.transforms import Compose, transforms
import torch.nn.functional as F
import cv2

log = get_pylogger(__name__)

DEFAULT_MEAN = 255. * np.array([0.485, 0.456, 0.406])
DEFAULT_STD = 255. * np.array([0.229, 0.224, 0.225])

###########################################################
SMPL_BODY_BONES = [-0.0018, -0.2233, 0.0282, 0.0695, -0.0914, -0.0068, -0.0677, -0.0905, -0.0043,
                   -0.0025, 0.1090, -0.0267, 0.0343, -0.3752, -0.0045, -0.0383, -0.3826, -0.0089,
                   0.0055, 0.1352, 0.0011, -0.0136, -0.3980, -0.0437, 0.0158, -0.3984, -0.0423,
                   0.0015, 0.0529, 0.0254, 0.0264, -0.0558, 0.1193, -0.0254, -0.0481, 0.1233,
                   -0.0028, 0.2139, -0.0429, 0.0788, 0.1217, -0.0341, -0.0818, 0.1188, -0.0386,
                   0.0052, 0.0650, 0.0513, 0.0910, 0.0305, -0.0089, -0.0960, 0.0326, -0.0091,
                   0.2596, -0.0128, -0.0275, -0.2537, -0.0133, -0.0214, 0.2492, 0.0090, -0.0012,
                   -0.2553, 0.0078, -0.0056, 0.0840, -0.0082, -0.0149, -0.0846, -0.0061, -0.0103]

class HybrIKJointsToRotmat:
    def __init__(self):
        self.naive_hybrik = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
        self.num_nodes = 22
        self.parents = [0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]
        self.child = [-1, 4, 5, 6, 7, 8, 9, 10, 11, -1, -2, -2, 15,
                      16, 17, -2, 18, 19, 20, 21, -2, -2]
        self.bones = np.reshape(np.array(SMPL_BODY_BONES), [24, 3])[:self.num_nodes]

    def multi_child_rot(self, t, p,
                        pose_global_parent):
        """
        t: B x 3 x child_num
        p: B x 3 x child_num
        pose_global_parent: B x 3 x 3
        """
        m = np.matmul(t, np.transpose(np.matmul(np.linalg.inv(pose_global_parent), p), [0, 2, 1]))
        
        u, s, vt = np.linalg.svd(m, full_matrices=False)
        r = np.matmul(np.transpose(vt, [0, 2, 1]), np.transpose(u, [0, 2, 1]))
        err_det_mask = (np.linalg.det(r) < 0.0).reshape(-1, 1, 1)
        id_fix = np.reshape(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]]),
                            [1, 3, 3])
        r_fix = np.matmul(np.transpose(vt, [0, 2, 1]),
                          np.matmul(id_fix,
                                    np.transpose(u, [0, 2, 1])))
        r = r * (1.0 - err_det_mask) + r_fix * err_det_mask
        return r, np.matmul(pose_global_parent, r)

    def single_child_rot(self, t, p, pose_global_parent, twist=None):
        """
        t: B x 3 x 1
        p: B x 3 x 1
        pose_global_parent: B x 3 x 3
        twist: B x 2 if given, default to None
        """
        p_rot = np.matmul(np.linalg.inv(pose_global_parent), p)
        cross = np.cross(t, p_rot, axisa=1, axisb=1, axisc=1)
        sina = np.linalg.norm(cross, axis=1, keepdims=True) / (np.linalg.norm(t, axis=1, keepdims=True) *
                                                               np.linalg.norm(p_rot, axis=1, keepdims=True))
        cross = cross / np.linalg.norm(cross, axis=1, keepdims=True)
        cosa = np.sum(t * p_rot, axis=1, keepdims=True) / (np.linalg.norm(t, axis=1, keepdims=True) *
                                                           np.linalg.norm(p_rot, axis=1, keepdims=True))
        sina = np.reshape(sina, [-1, 1, 1])
        cosa = np.reshape(cosa, [-1, 1, 1])
        skew_sym_t = np.stack([0.0 * cross[:, 0], -cross[:, 2], cross[:, 1],
                               cross[:, 2], 0.0 * cross[:, 0], -cross[:, 0],
                               -cross[:, 1], cross[:, 0], 0.0 * cross[:, 0]], 1)
        skew_sym_t = np.reshape(skew_sym_t, [-1, 3, 3])
        dsw_rotmat = np.reshape(np.eye(3), [1, 3, 3]
                                ) + sina * skew_sym_t + (1.0 - cosa) * np.matmul(skew_sym_t,
                                                                                 skew_sym_t)
        if twist is not None:
            skew_sym_t = np.stack([0.0 * t[:, 0], -t[:, 2], t[:, 1],
                                   t[:, 2], 0.0 * t[:, 0], -t[:, 0],
                                   -t[:, 1], t[:, 0], 0.0 * t[:, 0]], 1)
            skew_sym_t = np.reshape(skew_sym_t, [-1, 3, 3])
            sina = np.reshape(twist[:, 1], [-1, 1, 1])
            cosa = np.reshape(twist[:, 0], [-1, 1, 1])
            dtw_rotmat = np.reshape(np.eye(3), [1, 3, 3]
                                    ) + sina * skew_sym_t + (1.0 - cosa) * np.matmul(skew_sym_t,
                                                                                     skew_sym_t)
            dsw_rotmat = np.matmul(dsw_rotmat, dtw_rotmat)
        return dsw_rotmat, np.matmul(pose_global_parent, dsw_rotmat)

    def __call__(self, joints, twist=None):
        """
        joints: B x N x 3
        twist: B x N x 2 if given, default to None
        """
        expand_dim = False
        if len(joints.shape) == 2:
            expand_dim = True
            joints = np.expand_dims(joints, 0)
            if twist is not None:
                twist = np.expand_dims(twist, 0)
        assert (len(joints.shape) == 3)
        batch_size = np.shape(joints)[0]
        joints_rel = joints - joints[:, self.parents]
        joints_hybrik = 0.0 * joints_rel
        pose_global = np.zeros([batch_size, self.num_nodes, 3, 3])
        pose = np.zeros([batch_size, self.num_nodes, 3, 3])
        for i in range(self.num_nodes):
            if i == 0:
                joints_hybrik[:, 0] = joints[:, 0]
            else:
                joints_hybrik[:, i] = np.matmul(pose_global[:, self.parents[i]],
                                                np.reshape(self.bones[i], [1, 3, 1])).reshape(-1, 3) + \
                                      joints_hybrik[:, self.parents[i]]
            if self.child[i] == -2:
                pose[:, i] = pose[:, i] + np.eye(3).reshape(1, 3, 3)
                pose_global[:, i] = pose_global[:, self.parents[i]]
                continue
            if i == 0:
                r, rg = self.multi_child_rot(np.transpose(self.bones[[1, 2, 3]].reshape(1, 3, 3), [0, 2, 1]),
                                             np.transpose(joints_rel[:, [1, 2, 3]], [0, 2, 1]),
                                             np.eye(3).reshape(1, 3, 3))

            elif i == 9:
                r, rg = self.multi_child_rot(np.transpose(self.bones[[12, 13, 14]].reshape(1, 3, 3), [0, 2, 1]),
                                             np.transpose(joints_rel[:, [12, 13, 14]], [0, 2, 1]),
                                             pose_global[:, self.parents[9]])
            else:
                p = joints_rel[:, self.child[i]]
                if self.naive_hybrik[i] == 0:
                    p = joints[:, self.child[i]] - joints_hybrik[:, i]
                twi = None
                if twist is not None:
                    twi = twist[:, i]
                r, rg = self.single_child_rot(self.bones[self.child[i]].reshape(1, 3, 1),
                                              p.reshape(-1, 3, 1),
                                              pose_global[:, self.parents[i]],
                                              twi)
            pose[:, i] = r
            pose_global[:, i] = rg
        if expand_dim:
            pose = pose[0]
        return pose
###########################################################
    
class HMR2(pl.LightningModule):

    def __init__(self, cfg: CfgNode, init_renderer: bool = True):
        """
        Setup HMR2 model
        Args:
            cfg (CfgNode): Config file as a yacs CfgNode
        """
        super().__init__()

        # Save hyperparameters
        self.save_hyperparameters(logger=False, ignore=['init_renderer'])

        self.cfg = cfg
        # Create backbone feature extractor
        self.backbone = create_backbone(cfg)
        if cfg.MODEL.BACKBONE.get('PRETRAINED_WEIGHTS', None):
            log.info(f'Loading backbone weights from {cfg.MODEL.BACKBONE.PRETRAINED_WEIGHTS}')
            self.backbone.load_state_dict(torch.load(cfg.MODEL.BACKBONE.PRETRAINED_WEIGHTS, map_location='cpu')['state_dict'])

        # Create SMPL head
        self.smpl_head = build_smpl_head(cfg)

        # Create discriminator
        if self.cfg.LOSS_WEIGHTS.ADVERSARIAL > 0:
            self.discriminator = Discriminator()

        # Define loss functions
        self.keypoint_3d_loss = Keypoint3DLoss(loss_type='l1')
        self.keypoint_2d_loss = Keypoint2DLoss(loss_type='l1')
        self.smpl_parameter_loss = ParameterLoss()

        # Instantiate SMPL model
        smpl_cfg = {k.lower(): v for k,v in dict(cfg.SMPL).items()}
        self.smpl = SMPL(**smpl_cfg)

        # Buffer that shows whetheer we need to initialize ActNorm layers
        self.register_buffer('initialized', torch.tensor(False))
        # Setup renderer for visualization
        if init_renderer:
            self.renderer = SkeletonRenderer(self.cfg)
            self.mesh_renderer = MeshRenderer(self.cfg, faces=self.smpl.faces)
        else:
            self.renderer = None
            self.mesh_renderer = None

        # Disable automatic optimization since we use adversarial training
        self.automatic_optimization = False

        # Depth anthing
        self.depth_anything = DepthAnything.from_pretrained('LiheYoung/depth_anything_{}14'.format("vitl")).eval()
        self.depth_resize = transforms.Resize([266,266], interpolation=transforms.InterpolationMode.BICUBIC)

    def get_parameters(self):
        all_params = list(self.smpl_head.parameters())
        all_params += list(self.backbone.parameters())
        return all_params

    def configure_optimizers(self) -> Tuple[torch.optim.Optimizer, torch.optim.Optimizer]:
        """
        Setup model and distriminator Optimizers
        Returns:
            Tuple[torch.optim.Optimizer, torch.optim.Optimizer]: Model and discriminator optimizers
        """
        param_groups = [{'params': filter(lambda p: p.requires_grad, self.get_parameters()), 'lr': self.cfg.TRAIN.LR}]

        optimizer = torch.optim.AdamW(params=param_groups,
                                        # lr=self.cfg.TRAIN.LR,
                                        weight_decay=self.cfg.TRAIN.WEIGHT_DECAY)
        optimizer_disc = torch.optim.AdamW(params=self.discriminator.parameters(),
                                            lr=self.cfg.TRAIN.LR,
                                            weight_decay=self.cfg.TRAIN.WEIGHT_DECAY)

        return optimizer, optimizer_disc

    def forward_step(self, batch: Dict, train: bool = False, is_depth: bool=False) -> Dict:
        """
        Run a forward step of the network
        Args:
            batch (Dict): Dictionary containing batch data
            train (bool): Flag indicating whether it is training or validation mode
        Returns:
            Dict: Dictionary containing the regression output
        """

        # Use RGB image as input
        x = batch['img']  # B, 3, H, W
        batch_size = x.shape[0]

        #########################################depth-anything 20240131#################################
        if train:
            print("*************Depth anything*******************")
            with torch.no_grad():
                self.depth_anything.to(x.device)
                image_d = self.depth_resize(x.clone())
                depth = self.depth_anything(image_d)
            resize = transforms.Resize([256,256], interpolation=transforms.InterpolationMode.BICUBIC)
            depth = resize(depth)
            # depth = (depth - depth.min()) / (depth.max() - depth.min())
            depth = depth - depth.median(dim=0).values
            depth = depth / depth.abs().mean(dim=0)
            depth = depth.unsqueeze(1).repeat(1, 3, 1, 1)
            x = depth
        # depth = depth.cpu().numpy().astype(np.uint8)
        # depth_color = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)
        # x = torch.from_numpy(depth).to(x.device)
        #########################################depth-anything 20240131#################################

        # Compute conditioning features using the backbone
        # if using ViT backbone, we need to use a different aspect ratio  ?? why
        conditioning_feats = self.backbone(x[:,:,:,32:-32])

        pred_smpl_params, pred_cam, _ = self.smpl_head(conditioning_feats)

        # Store useful regression outputs to the output dict
        output = {}
        output['pred_cam'] = pred_cam
        output['pred_smpl_params'] = {k: v.clone() for k,v in pred_smpl_params.items()}

        # Compute camera translation
        device = pred_smpl_params['body_pose'].device
        dtype = pred_smpl_params['body_pose'].dtype
        focal_length = self.cfg.EXTRA.FOCAL_LENGTH * torch.ones(batch_size, 2, device=device, dtype=dtype)
        pred_cam_t = torch.stack([pred_cam[:, 1],
                                  pred_cam[:, 2],
                                  2*focal_length[:, 0]/(self.cfg.MODEL.IMAGE_SIZE * pred_cam[:, 0] +1e-9)],dim=-1)
        output['pred_cam_t'] = pred_cam_t
        output['focal_length'] = focal_length

        # Compute model vertices, joints and the projected joints
        pred_smpl_params['global_orient'] = pred_smpl_params['global_orient'].reshape(batch_size, -1, 3, 3)
        pred_smpl_params['body_pose'] = pred_smpl_params['body_pose'].reshape(batch_size, -1, 3, 3)
        pred_smpl_params['betas'] = pred_smpl_params['betas'].reshape(batch_size, -1)


        smpl_output, smpl_kpt = self.smpl(**{k: v.float() for k,v in pred_smpl_params.items()}, pose2rot=False)
        
        ############### stupid nou try###############################
        smpl_kpt = smpl_kpt.cpu().numpy()
        smpl_kpt = smpl_kpt - smpl_kpt[0, 0]
        pose_generator = HybrIKJointsToRotmat()
        pose = pose_generator(smpl_kpt) 
        pose = np.concatenate([pose, np.stack([np.stack([np.eye(3)] * pose.shape[0], 0)] * 2, 1)], 1)
        from scipy.spatial.transform import Rotation as RRR
        r = RRR.from_rotvec(np.array([np.pi, 0.0, 0.0]))
        pose[:, 0] = np.matmul(r.as_matrix().reshape(1, 3, 3), pose[:, 0])

        poses = pose
        pred_rotmats = []
        for pose in poses:
            if pose.size == 72:
                pose = pose.reshape(-1, 3)
                pose = RRR.from_rotvec(pose).as_matrix()
                pose = pose.reshape(1, 24, 3, 3)
            pred_rotmats.append(
                torch.from_numpy(pose.astype(np.float32)[None]).to(torch.device("cuda")))
        pred_rotmat = torch.cat(pred_rotmats, dim=0)

        smpl_output_new, smpl_kpt = self.smpl(
            betas=pred_smpl_params['betas'],
            body_pose=pred_rotmat[:, 1:],
            global_orient=pred_rotmat[:, 0].unsqueeze(1),
            pose2rot=False
        )
        ############### yuting try###############################

        
        pred_keypoints_3d = smpl_output.joints
        pred_vertices = smpl_output.vertices
        output['pred_keypoints_3d'] = pred_keypoints_3d.reshape(batch_size, -1, 3)
        output['pred_vertices'] = pred_vertices.reshape(batch_size, -1, 3)
        output['smpl_kp_3d'] = smpl_kpt

        pred_cam_t = pred_cam_t.reshape(-1, 3)
        focal_length = focal_length.reshape(-1, 2)
        pred_keypoints_2d = perspective_projection(pred_keypoints_3d,
                                                   translation=pred_cam_t,
                                                   focal_length=focal_length / self.cfg.MODEL.IMAGE_SIZE)

        output['pred_keypoints_2d'] = pred_keypoints_2d.reshape(batch_size, -1, 2)
        return output

    def compute_loss(self, batch: Dict, output: Dict, train: bool = True) -> torch.Tensor:
        """
        Compute losses given the input batch and the regression output
        Args:
            batch (Dict): Dictionary containing batch data
            output (Dict): Dictionary containing the regression output
            train (bool): Flag indicating whether it is training or validation mode
        Returns:
            torch.Tensor : Total loss for current batch
        """

        pred_smpl_params = output['pred_smpl_params']
        pred_keypoints_2d = output['pred_keypoints_2d']
        pred_keypoints_3d = output['pred_keypoints_3d']


        batch_size = pred_smpl_params['body_pose'].shape[0]
        device = pred_smpl_params['body_pose'].device
        dtype = pred_smpl_params['body_pose'].dtype

        # Get annotations
        gt_keypoints_2d = batch['keypoints_2d']
        gt_keypoints_3d = batch['keypoints_3d']
        gt_smpl_params = batch['smpl_params']
        has_smpl_params = batch['has_smpl_params']
        is_axis_angle = batch['smpl_params_is_axis_angle']

        # Compute 3D keypoint loss
        loss_keypoints_2d = self.keypoint_2d_loss(pred_keypoints_2d, gt_keypoints_2d)
        loss_keypoints_3d = self.keypoint_3d_loss(pred_keypoints_3d, gt_keypoints_3d, pelvis_id=25+14)

        # Compute loss on SMPL parameters
        loss_smpl_params = {}
        for k, pred in pred_smpl_params.items():
            gt = gt_smpl_params[k].view(batch_size, -1)
            if is_axis_angle[k].all():
                gt = aa_to_rotmat(gt.reshape(-1, 3)).view(batch_size, -1, 3, 3)
            has_gt = has_smpl_params[k]
            loss_smpl_params[k] = self.smpl_parameter_loss(pred.reshape(batch_size, -1), gt.reshape(batch_size, -1), has_gt)

        loss = self.cfg.LOSS_WEIGHTS['KEYPOINTS_3D'] * loss_keypoints_3d+\
               self.cfg.LOSS_WEIGHTS['KEYPOINTS_2D'] * loss_keypoints_2d+\
               sum([loss_smpl_params[k] * self.cfg.LOSS_WEIGHTS[k.upper()] for k in loss_smpl_params])

        losses = dict(loss=loss.detach(),
                      loss_keypoints_2d=loss_keypoints_2d.detach(),
                      loss_keypoints_3d=loss_keypoints_3d.detach())

        for k, v in loss_smpl_params.items():
            losses['loss_' + k] = v.detach()

        output['losses'] = losses

        return loss

    # Tensoroboard logging should run from first rank only
    @pl.utilities.rank_zero.rank_zero_only
    def tensorboard_logging(self, batch: Dict, output: Dict, step_count: int, train: bool = True, write_to_summary_writer: bool = True) -> None:
        """
        Log results to Tensorboard
        Args:
            batch (Dict): Dictionary containing batch data
            output (Dict): Dictionary containing the regression output
            step_count (int): Global training step count
            train (bool): Flag indicating whether it is training or validation mode
        """

        mode = 'train' if train else 'val'
        batch_size = batch['keypoints_2d'].shape[0]
        images = batch['img']

        #########################################depth-anything 20240131#################################
        x = batch['img']
        with torch.no_grad():
            self.depth_anything.to(x.device)
            image_d = self.depth_resize(x.clone())
            depth = self.depth_anything(image_d)
        resize = transforms.Resize([256,256], interpolation=transforms.InterpolationMode.BICUBIC)
        depth = resize(depth)
        depth = depth - depth.median(dim=0).values
        depth = depth / depth.abs().mean(dim=0)
        depth = (depth - depth.min(dim=0).values) / (depth.max(dim=0).values - depth.min(dim=0).values)
        depth = depth.unsqueeze(1).repeat(1, 3, 1, 1)
        #########################################depth-anything 20240131#################################

        images = images * torch.tensor([0.229, 0.224, 0.225], device=images.device).reshape(1,3,1,1)
        images = images + torch.tensor([0.485, 0.456, 0.406], device=images.device).reshape(1,3,1,1)
        #images = 255*images.permute(0, 2, 3, 1).cpu().numpy()

        pred_keypoints_3d = output['pred_keypoints_3d'].detach().reshape(batch_size, -1, 3)
        pred_vertices = output['pred_vertices'].detach().reshape(batch_size, -1, 3)
        focal_length = output['focal_length'].detach().reshape(batch_size, 2)
        gt_keypoints_3d = batch['keypoints_3d']
        gt_keypoints_2d = batch['keypoints_2d']
        losses = output['losses']
        pred_cam_t = output['pred_cam_t'].detach().reshape(batch_size, 3)
        pred_keypoints_2d = output['pred_keypoints_2d'].detach().reshape(batch_size, -1, 2)

        if write_to_summary_writer:
            summary_writer = self.logger.experiment
            for loss_name, val in losses.items():
                summary_writer.add_scalar(mode +'/' + loss_name, val.detach().item(), step_count)
        num_images = min(batch_size, self.cfg.EXTRA.NUM_LOG_IMAGES)

        gt_keypoints_3d = batch['keypoints_3d']
        pred_keypoints_3d = output['pred_keypoints_3d'].detach().reshape(batch_size, -1, 3)

        # We render the skeletons instead of the full mesh because rendering a lot of meshes will make the training slow.
        #predictions = self.renderer(pred_keypoints_3d[:num_images],
        #                            gt_keypoints_3d[:num_images],
        #                            2 * gt_keypoints_2d[:num_images],
        #                            images=images[:num_images],
        #                            camera_translation=pred_cam_t[:num_images])
        predictions = self.mesh_renderer.visualize_tensorboard(pred_vertices[:num_images].cpu().numpy(),
                                                               pred_cam_t[:num_images].cpu().numpy(),
                                                               images[:num_images].cpu().numpy(),
                                                               pred_keypoints_2d[:num_images].cpu().numpy(),
                                                               gt_keypoints_2d[:num_images].cpu().numpy(),
                                                               focal_length=focal_length[:num_images].cpu().numpy(),
                                                               depths=depth.cpu().numpy())
        if write_to_summary_writer:
            summary_writer.add_image('%s/predictions' % mode, predictions, step_count)

        return predictions

    def forward(self, batch: Dict, **kwargs) -> Dict:
        """
        Run a forward step of the network in val mode
        Args:
            batch (Dict): Dictionary containing batch data
        Returns:
            Dict: Dictionary containing the regression output
        """
        return self.forward_step(batch, train=False, **kwargs)

    def training_step_discriminator(self, batch: Dict,
                                    body_pose: torch.Tensor,
                                    betas: torch.Tensor,
                                    optimizer: torch.optim.Optimizer) -> torch.Tensor:
        """
        Run a discriminator training step
        Args:
            batch (Dict): Dictionary containing mocap batch data
            body_pose (torch.Tensor): Regressed body pose from current step
            betas (torch.Tensor): Regressed betas from current step
            optimizer (torch.optim.Optimizer): Discriminator optimizer
        Returns:
            torch.Tensor: Discriminator loss
        """
        batch_size = body_pose.shape[0]
        gt_body_pose = batch['body_pose']
        gt_betas = batch['betas']
        gt_rotmat = aa_to_rotmat(gt_body_pose.view(-1,3)).view(batch_size, -1, 3, 3)
        disc_fake_out = self.discriminator(body_pose.detach(), betas.detach())
        loss_fake = ((disc_fake_out - 0.0) ** 2).sum() / batch_size
        disc_real_out = self.discriminator(gt_rotmat, gt_betas)
        loss_real = ((disc_real_out - 1.0) ** 2).sum() / batch_size
        loss_disc = loss_fake + loss_real
        loss = self.cfg.LOSS_WEIGHTS.ADVERSARIAL * loss_disc
        optimizer.zero_grad()
        self.manual_backward(loss)
        optimizer.step()
        return loss_disc.detach()

    def training_step(self, joint_batch: Dict, batch_idx: int) -> Dict:
        """
        Run a full training step
        Args:
            joint_batch (Dict): Dictionary containing image and mocap batch data
            batch_idx (int): Unused.
            batch_idx (torch.Tensor): Unused.
        Returns:
            Dict: Dictionary containing regression output.
        """
        batch = joint_batch['img']
        mocap_batch = joint_batch['mocap']
        optimizer = self.optimizers(use_pl_optimizer=True)
        if self.cfg.LOSS_WEIGHTS.ADVERSARIAL > 0:
            optimizer, optimizer_disc = optimizer

        batch_size = batch['img'].shape[0]
        output = self.forward_step(batch, train=True)
        pred_smpl_params = output['pred_smpl_params']
        if self.cfg.get('UPDATE_GT_SPIN', False):
            self.update_batch_gt_spin(batch, output)
        loss = self.compute_loss(batch, output, train=True)
        if self.cfg.LOSS_WEIGHTS.ADVERSARIAL > 0:
            disc_out = self.discriminator(pred_smpl_params['body_pose'].reshape(batch_size, -1), pred_smpl_params['betas'].reshape(batch_size, -1))
            loss_adv = ((disc_out - 1.0) ** 2).sum() / batch_size
            loss = loss + self.cfg.LOSS_WEIGHTS.ADVERSARIAL * loss_adv

        # Error if Nan
        if torch.isnan(loss):
            raise ValueError('Loss is NaN')

        optimizer.zero_grad()
        self.manual_backward(loss)
        # Clip gradient
        if self.cfg.TRAIN.get('GRAD_CLIP_VAL', 0) > 0:
            gn = torch.nn.utils.clip_grad_norm_(self.get_parameters(), self.cfg.TRAIN.GRAD_CLIP_VAL, error_if_nonfinite=True)
            self.log('train/grad_norm', gn, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        optimizer.step()
        if self.cfg.LOSS_WEIGHTS.ADVERSARIAL > 0:
            loss_disc = self.training_step_discriminator(mocap_batch, pred_smpl_params['body_pose'].reshape(batch_size, -1), pred_smpl_params['betas'].reshape(batch_size, -1), optimizer_disc)
            output['losses']['loss_gen'] = loss_adv
            output['losses']['loss_disc'] = loss_disc

        if self.global_step > 0 and self.global_step % self.cfg.GENERAL.LOG_STEPS == 0:
            self.tensorboard_logging(batch, output, self.global_step, train=True)

        self.log('train/loss', output['losses']['loss'], on_step=True, on_epoch=True, prog_bar=True, logger=False)

        return output

    def validation_step(self, batch: Dict, batch_idx: int, dataloader_idx=0) -> Dict:
        """
        Run a validation step and log to Tensorboard
        Args:
            batch (Dict): Dictionary containing batch data
            batch_idx (int): Unused.
        Returns:
            Dict: Dictionary containing regression output.
        """
        # batch_size = batch['img'].shape[0]
        output = self.forward_step(batch, train=False)
        loss = self.compute_loss(batch, output, train=False)
        output['loss'] = loss
        self.tensorboard_logging(batch, output, self.global_step, train=False)

        return output
