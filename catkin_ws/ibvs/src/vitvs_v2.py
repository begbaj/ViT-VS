# Core imports
import rospy
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Pose, Twist
from gazebo_msgs.srv import GetModelState, SpawnModel, SetModelState, DeleteModel
from sensor_msgs.msg import Image as ImageMsg
from cv_bridge import CvBridge

# Scientific computing
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy import stats

# Deep learning & vision
import torch
from torch import nn
import torch.nn.modules.utils as nn_utils
import torch.nn.functional as F
import timm
import cv2
from PIL import Image
from torchvision import transforms

# Visualization
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch

# System
import os
import sys
import yaml
import time
import types
from pathlib import Path
from typing import Union, List, Tuple
import argparse

# ROS transforms
import tf2_ros
import tf_conversions



def chunk_cosine_sim(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Computes cosine similarity between all possible pairs in two sets of vectors."""
    result_list = []
    num_token_x = x.shape[2]
    for token_idx in range(num_token_x):
        token = x[:, :, token_idx, :].unsqueeze(dim=2)  # Bx1x1xd'
        result_list.append(torch.nn.CosineSimilarity(dim=3)(token, y))  # Bx1xt
    return torch.stack(result_list, dim=2)  # Bx1x(t_x)x(t_y)

def _to_cartesian(coords, shape):
    """Takes raveled coordinates and returns them in a cartesian coordinate frame"""
    if torch.is_tensor(coords):
        coords = coords.long()

    # Calculate rows and columns for all indices
    width = shape[1]
    rows = coords // width
    cols = coords % width

    # Stack coordinates
    result = torch.stack([rows, cols], dim=-1)
    return result

def find_correspondences_batch(descriptors1, descriptors2, num_pairs=18, distance_threshold=1):
    """Find correspondences between two images using their descriptors."""
    B, _, t_m_1, d_h = descriptors1.size()
    num_patches = (int(np.sqrt(t_m_1)), int(np.sqrt(t_m_1)))

    # Calculate similarities
    similarities = chunk_cosine_sim(descriptors1, descriptors2)

    sim_1, nn_1 = torch.max(similarities, dim=-1)
    sim_2, nn_2 = torch.max(similarities, dim=-2)

    # Check if we're dealing with the same image
    is_same_image = sim_1.mean().item() > 0.99

    if is_same_image:
        # For same image, take random points
        num_points = min(num_pairs, t_m_1)
        perm = torch.randperm(t_m_1, device=descriptors1.device)
        indices = perm[:num_points]
        points1 = _to_cartesian(indices, num_patches)
        points2 = points1.clone()  # Same points for same image
        sim_scores = torch.ones(num_points, device=descriptors1.device)
        return points1, points2, sim_scores

    else:
        nn_1, nn_2 = nn_1[:, 0, :], nn_2[:, 0, :]
        cyclical_idxs = torch.gather(nn_2, dim=-1, index=nn_1)

        image_idxs = torch.arange(t_m_1, device=descriptors1.device)[None, :].repeat(B, 1)
        cyclical_idxs_ij = _to_cartesian(cyclical_idxs, shape=num_patches)
        image_idxs_ij = _to_cartesian(image_idxs, shape=num_patches)

        b, hw, ij_dim = cyclical_idxs_ij.size()
        cyclical_dists = -torch.nn.PairwiseDistance(p=2)(
            cyclical_idxs_ij.view(-1, ij_dim),
            image_idxs_ij.view(-1, ij_dim)
        ).view(b, hw)

        cyclical_dists_norm = cyclical_dists - cyclical_dists.min(1, keepdim=True)[0]
        cyclical_dists_norm /= (cyclical_dists_norm.max(1, keepdim=True)[0] + 1e-8)

        sorted_vals, selected_points_image_1 = cyclical_dists_norm.sort(dim=-1, descending=True)

        mask = sorted_vals >= distance_threshold
        filtered_points = selected_points_image_1[mask]

        num_available = filtered_points.numel()
        num_to_select = min(num_pairs, num_available)

        if num_to_select > 0:
            perm = torch.randperm(num_available, device=descriptors1.device)
            selected_indices = perm[:num_to_select]
            selected_points_image_1 = filtered_points[selected_indices].unsqueeze(0)

            selected_points_image_2 = torch.gather(nn_1, dim=-1, index=selected_points_image_1)
            sim_selected_12 = torch.gather(sim_1[:, 0, :], dim=-1, index=selected_points_image_1)

            points1 = _to_cartesian(selected_points_image_1[0], num_patches)
            points2 = _to_cartesian(selected_points_image_2[0], num_patches)

            return points1, points2, sim_selected_12
        else:
            return None, None, None


class ViTExtractor:
    """ This class facilitates extraction of features, descriptors, and saliency maps from a ViT.
    We use the following notation in the documentation of the module's methods:
    B - batch size
    h - number of heads. usually takes place of the channel dimension in pytorch's convention BxCxHxW
    p - patch size of the ViT. either 8 or 16.
    t - number of tokens. equals the number of patches + 1, e.g. HW / p**2 + 1. Where H and W are the height and width
    of the input image.
    d - the embedding dimension in the ViT.
    """

    def __init__(self, model_type: str = 'dino_vits8', stride: int = 4, model: nn.Module = None, device: str = 'cuda'):
        """
        :param model_type: A string specifying the type of model to extract from.
                          [dino_vits8 | dino_vits16 | dino_vitb8 | dino_vitb16 | vit_small_patch8_224 |
                          vit_small_patch16_224 | vit_base_patch8_224 | vit_base_patch16_224]
        :param stride: stride of first convolution layer. small stride -> higher resolution.
        :param model: Optional parameter. The nn.Module to extract from instead of creating a new one in ViTExtractor.
                      should be compatible with model_type.
        """
        self.model_type = model_type
        self.device = device
        if model is not None:
            self.model = model
        else:
            self.model = ViTExtractor.create_model(model_type)

        self.model = ViTExtractor.patch_vit_resolution(self.model, stride=stride)
        self.model.eval()
        self.model.to(self.device)
        self.p = self.model.patch_embed.patch_size
        if type(self.p) == tuple:
            self.p = self.p[0]
        self.stride = self.model.patch_embed.proj.stride

        self.mean = (0.485, 0.456, 0.406) if "dino" in self.model_type else (0.5, 0.5, 0.5)
        self.std = (0.229, 0.224, 0.225) if "dino" in self.model_type else (0.5, 0.5, 0.5)

        self._feats = []
        self.hook_handlers = []
        self.load_size = None
        self.num_patches = None

    @staticmethod
    def create_model(model_type: str) -> nn.Module:
        """
        :param model_type: a string specifying which model to load. [dino_vits8 | dino_vits16 | dino_vitb8 |
                           dino_vitb16 | vit_small_patch8_224 | vit_small_patch16_224 | vit_base_patch8_224 |
                           vit_base_patch16_224]
        :return: the model
        """
        torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
        if 'v2' in model_type:
            model = torch.hub.load('facebookresearch/dinov2', model_type)
        elif 'dino' in model_type:
            model = torch.hub.load('facebookresearch/dino:main', model_type)
        else:  # model from timm -- load weights from timm to dino model (enables working on arbitrary size images).
            temp_model = timm.create_model(model_type, pretrained=True)
            model_type_dict = {
                'vit_small_patch16_224': 'dino_vits16',
                'vit_small_patch8_224': 'dino_vits8',
                'vit_base_patch16_224': 'dino_vitb16',
                'vit_base_patch8_224': 'dino_vitb8'
            }
            model = torch.hub.load('facebookresearch/dino:main', model_type_dict[model_type])
            temp_state_dict = temp_model.state_dict()
            del temp_state_dict['head.weight']
            del temp_state_dict['head.bias']
            model.load_state_dict(temp_state_dict)
        return model

    @staticmethod
    def _fix_pos_enc(patch_size: int, stride_hw: Tuple[int, int]):
        """
        Creates a method for position encoding interpolation.
        :param patch_size: patch size of the model.
        :param stride_hw: A tuple containing the new height and width stride respectively.
        :return: the interpolation method
        """

        def interpolate_pos_encoding(self, x: torch.Tensor, w: int, h: int) -> torch.Tensor:
            npatch = x.shape[1] - 1
            N = self.pos_embed.shape[1] - 1
            if npatch == N and w == h:
                return self.pos_embed
            class_pos_embed = self.pos_embed[:, 0]
            patch_pos_embed = self.pos_embed[:, 1:]
            dim = x.shape[-1]
            # compute number of tokens taking stride into account
            w0 = 1 + (w - patch_size) // stride_hw[1]
            h0 = 1 + (h - patch_size) // stride_hw[0]
            assert (w0 * h0 == npatch), f"""got wrong grid size for {h}x{w} with patch_size {patch_size} and 
                                            stride {stride_hw} got {h0}x{w0}={h0 * w0} expecting {npatch}"""
            # we add a small number to avoid floating point error in the interpolation
            # see discussion at https://github.com/facebookresearch/dino/issues/8
            w0, h0 = w0 + 0.1, h0 + 0.1
            patch_pos_embed = nn.functional.interpolate(
                patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
                scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
                mode='bicubic',
                align_corners=False, recompute_scale_factor=False
            )
            assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
            patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
            return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

        return interpolate_pos_encoding

    @staticmethod
    def patch_vit_resolution(model: nn.Module, stride: int) -> nn.Module:
        """
        change resolution of model output by changing the stride of the patch extraction.
        :param model: the model to change resolution for.
        :param stride: the new stride parameter.
        :return: the adjusted model
        """
        patch_size = model.patch_embed.patch_size
        if type(patch_size) == tuple:
            patch_size = patch_size[0]
        if stride == patch_size:  # nothing to do
            return model

        stride = nn_utils._pair(stride)
        assert all([(patch_size // s_) * s_ == patch_size for s_ in
                    stride]), f'stride {stride} should divide patch_size {patch_size}'

        # fix the stride
        model.patch_embed.proj.stride = stride
        # fix the positional encoding code
        model.interpolate_pos_encoding = types.MethodType(ViTExtractor._fix_pos_enc(patch_size, stride), model)
        return model

    def preprocess(self, image_path: Union[str, Path],
                   load_size: Union[int, Tuple[int, int]] = None, patch_size: int = 14) -> Tuple[
        torch.Tensor, Image.Image]:
        """
        Preprocesses an image before extraction.
        :param image_path: path to image to be extracted.
        :param load_size: optional. Size to resize image before the rest of preprocessing.
        :return: a tuple containing:
                    (1) the preprocessed image as a tensor to insert the model of shape BxCxHxW.
                    (2) the pil image in relevant dimensions
        """

        def divisible_by_num(num, dim):
            return num * (dim // num)

        pil_image = Image.open(image_path).convert('RGB')
        if load_size is not None:
            pil_image = transforms.Resize(load_size, interpolation=transforms.InterpolationMode.LANCZOS)(pil_image)

            width, height = pil_image.size
            new_width = divisible_by_num(patch_size, width)
            new_height = divisible_by_num(patch_size, height)
            pil_image = pil_image.resize((new_width, new_height), resample=Image.LANCZOS)

        prep = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
        prep_img = prep(pil_image)[None, ...]
        return prep_img, pil_image

    def preprocess_pil(self, pil_image):
        """
        Preprocesses an image before extraction.
        :param image_path: path to image to be extracted.
        :param load_size: optional. Size to resize image before the rest of preprocessing.
        :return: a tuple containing:
                    (1) the preprocessed image as a tensor to insert the model of shape BxCxHxW.
                    (2) the pil image in relevant dimensions
        """
        prep = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
        prep_img = prep(pil_image)[None, ...]
        return prep_img

    def _get_hook(self, facet: str):
        """
        generate a hook method for a specific block and facet.
        """
        if facet in ['attn', 'token']:
            def _hook(model, input, output):
                self._feats.append(output)

            return _hook

        if facet == 'query':
            facet_idx = 0
        elif facet == 'key':
            facet_idx = 1
        elif facet == 'value':
            facet_idx = 2
        else:
            raise TypeError(f"{facet} is not a supported facet.")

        def _inner_hook(module, input, output):
            input = input[0]
            B, N, C = input.shape
            qkv = module.qkv(input).reshape(B, N, 3, module.num_heads, C // module.num_heads).permute(2, 0, 3, 1, 4)
            self._feats.append(qkv[facet_idx])  # Bxhxtxd

        return _inner_hook

    def _register_hooks(self, layers: List[int], facet: str) -> None:
        """
        register hook to extract features.
        :param layers: layers from which to extract features.
        :param facet: facet to extract. One of the following options: ['key' | 'query' | 'value' | 'token' | 'attn']
        """
        for block_idx, block in enumerate(self.model.blocks):
            if block_idx in layers:
                if facet == 'token':
                    self.hook_handlers.append(block.register_forward_hook(self._get_hook(facet)))
                elif facet == 'attn':
                    self.hook_handlers.append(block.attn.attn_drop.register_forward_hook(self._get_hook(facet)))
                elif facet in ['key', 'query', 'value']:
                    self.hook_handlers.append(block.attn.register_forward_hook(self._get_hook(facet)))
                else:
                    raise TypeError(f"{facet} is not a supported facet.")

    def _unregister_hooks(self) -> None:
        """
        unregisters the hooks. should be called after feature extraction.
        """
        for handle in self.hook_handlers:
            handle.remove()
        self.hook_handlers = []

    def _extract_features(self, batch: torch.Tensor, layers: List[int] = 11, facet: str = 'key') -> List[torch.Tensor]:
        """
        extract features from the model
        :param batch: batch to extract features for. Has shape BxCxHxW.
        :param layers: layer to extract. A number between 0 to 11.
        :param facet: facet to extract. One of the following options: ['key' | 'query' | 'value' | 'token' | 'attn']
        :return : tensor of features.
                  if facet is 'key' | 'query' | 'value' has shape Bxhxtxd
                  if facet is 'attn' has shape Bxhxtxt
                  if facet is 'token' has shape Bxtxd
        """
        B, C, H, W = batch.shape
        self._feats = []
        self._register_hooks(layers, facet)
        _ = self.model(batch)
        self._unregister_hooks()
        self.load_size = (H, W)
        self.num_patches = (1 + (H - self.p) // self.stride[0], 1 + (W - self.p) // self.stride[1])
        return self._feats

    def _log_bin(self, x: torch.Tensor, hierarchy: int = 2) -> torch.Tensor:
        """
        create a log-binned descriptor.
        :param x: tensor of features. Has shape Bxhxtxd.
        :param hierarchy: how many bin hierarchies to use.
        """
        B = x.shape[0]
        num_bins = 1 + 8 * hierarchy

        bin_x = x.permute(0, 2, 3, 1).flatten(start_dim=-2, end_dim=-1)  # Bx(t-1)x(dxh)
        bin_x = bin_x.permute(0, 2, 1)
        bin_x = bin_x.reshape(B, bin_x.shape[1], self.num_patches[0], self.num_patches[1])
        # Bx(dxh)xnum_patches[0]xnum_patches[1]
        sub_desc_dim = bin_x.shape[1]

        avg_pools = []
        # compute bins of all sizes for all spatial locations.
        for k in range(0, hierarchy):
            # avg pooling with kernel 3**kx3**k
            win_size = 3 ** k
            avg_pool = torch.nn.AvgPool2d(win_size, stride=1, padding=win_size // 2, count_include_pad=False)
            avg_pools.append(avg_pool(bin_x))

        bin_x = torch.zeros((B, sub_desc_dim * num_bins, self.num_patches[0], self.num_patches[1])).to(self.device)
        for y in range(self.num_patches[0]):
            for x in range(self.num_patches[1]):
                part_idx = 0
                # fill all bins for a spatial location (y, x)
                for k in range(0, hierarchy):
                    kernel_size = 3 ** k
                    for i in range(y - kernel_size, y + kernel_size + 1, kernel_size):
                        for j in range(x - kernel_size, x + kernel_size + 1, kernel_size):
                            if i == y and j == x and k != 0:
                                continue
                            if 0 <= i < self.num_patches[0] and 0 <= j < self.num_patches[1]:
                                bin_x[:, part_idx * sub_desc_dim: (part_idx + 1) * sub_desc_dim, y, x] = avg_pools[k][
                                                                                                         :, :, i, j]
                            else:  # handle padding in a more delicate way than zero padding
                                temp_i = max(0, min(i, self.num_patches[0] - 1))
                                temp_j = max(0, min(j, self.num_patches[1] - 1))
                                bin_x[:, part_idx * sub_desc_dim: (part_idx + 1) * sub_desc_dim, y, x] = avg_pools[k][
                                                                                                         :, :, temp_i,
                                                                                                         temp_j]
                            part_idx += 1
        bin_x = bin_x.flatten(start_dim=-2, end_dim=-1).permute(0, 2, 1).unsqueeze(dim=1)
        # Bx1x(t-1)x(dxh)
        return bin_x

    def extract_descriptors(self, batch: torch.Tensor, layer: int = 11, facet: str = 'key',
                            bin: bool = False, include_cls: bool = False) -> torch.Tensor:
        """
        extract descriptors from the model
        :param batch: batch to extract descriptors for. Has shape BxCxHxW.
        :param layers: layer to extract. A number between 0 to 11.
        :param facet: facet to extract. One of the following options: ['key' | 'query' | 'value' | 'token']
        :param bin: apply log binning to the descriptor. default is False.
        :return: tensor of descriptors. Bx1xtxd' where d' is the dimension of the descriptors.
        """
        assert facet in ['key', 'query', 'value', 'token'], f"""{facet} is not a supported facet for descriptors. 
                                                             choose from ['key' | 'query' | 'value' | 'token'] """
        self._extract_features(batch, [layer], facet)
        x = self._feats[0]
        if facet == 'token':
            x.unsqueeze_(dim=1)  # Bx1xtxd
        if not include_cls:
            x = x[:, :, 1:, :]  # remove cls token
        else:
            assert not bin, "bin = True and include_cls = True are not supported together, set one of them False."
        if not bin:
            desc = x.permute(0, 2, 3, 1).flatten(start_dim=-2, end_dim=-1).unsqueeze(dim=1)  # Bx1xtx(dxh)
        else:
            desc = self._log_bin(x)
        return desc

    def extract_saliency_maps(self, batch: torch.Tensor) -> torch.Tensor:
        """
        extract saliency maps. The saliency maps are extracted by averaging several attention heads from the last layer
        in of the CLS token. All values are then normalized to range between 0 and 1.
        :param batch: batch to extract saliency maps for. Has shape BxCxHxW.
        :return: a tensor of saliency maps. has shape Bxt-1
        """
        assert self.model_type == "dino_vits8", f"saliency maps are supported only for dino_vits model_type."
        self._extract_features(batch, [11], 'attn')
        head_idxs = [0, 2, 4, 5]
        curr_feats = self._feats[0]  # Bxhxtxt
        cls_attn_map = curr_feats[:, head_idxs, 0, 1:].mean(dim=1)  # Bx(t-1)
        temp_mins, temp_maxs = cls_attn_map.min(dim=1)[0], cls_attn_map.max(dim=1)[0]
        cls_attn_maps = (cls_attn_map - temp_mins) / (temp_maxs - temp_mins)  # normalize to range [0,1]
        return cls_attn_maps


def scale_points_from_patch(points, vit_image_size=518, num_patches=37):
    """Scale points from patch coordinates to pixel coordinates"""
    points = (points + 0.5) / num_patches * vit_image_size
    return points


def visualize_correspondences(image1, image2, points1, points2, save_path=None):
    """Visualize correspondences between two images."""
    if isinstance(image1, Image.Image):
        image1 = np.array(image1)
    if isinstance(image2, Image.Image):
        image2 = np.array(image2)

    if torch.is_tensor(points1):
        points1 = points1.cpu().detach().numpy()
    if torch.is_tensor(points2):
        points2 = points2.cpu().detach().numpy()

    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    ax1.imshow(image1)
    ax2.imshow(image2)

    ax1.axis('off')
    ax2.axis('off')

    colors = plt.cm.rainbow(np.linspace(0, 1, len(points1)))

    for i, ((y1, x1), (y2, x2), color) in enumerate(zip(points1, points2, colors)):
        ax1.plot(x1, y1, 'o', color=color, markersize=8)
        ax1.text(x1 + 5, y1 + 5, str(i), color=color, fontsize=8)

        ax2.plot(x2, y2, 'o', color=color, markersize=8)
        ax2.text(x2 + 5, y2 + 5, str(i), color=color, fontsize=8)

        con = ConnectionPatch(
            xyA=(x1, y1), xyB=(x2, y2),
            coordsA="data", coordsB="data",
            axesA=ax1, axesB=ax2, color=color, alpha=0.5
        )
        fig.add_artist(con)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    return fig

class Controller:
    def __init__(self, desired_position, desired_orientation, config_path):
        self.config_path = config_path
        # Load parameters from YAML
        self.load_parameters()

        # Processing variables
        self.latest_image = None
        self.latest_image_depth = None
        self.latest_pil_image = None
        self.camera_position = None
        self.desired_position = desired_position
        self.desired_orientation = desired_orientation
        self.orientation_quaternion = None
        self.s_uv = None
        self.s_uv_star = None
        self.v_c = None
        self.iteration_count = 0

        # Histories and tracking variables
        self.velocity_history = []
        self.position_history = []
        self.orientation_history = []
        self.iteration_history = 0
        self.initial_position_error = None
        self.velocity_mean_100 = []
        self.velocity_mean_10 = []
        self.average_velocities = []

        # Velocity tracking
        self.applied_velocity_x = []
        self.applied_velocity_y = []
        self.applied_velocity_z = []
        self.applied_velocity_roll = []
        self.applied_velocity_pitch = []
        self.applied_velocity_yaw = []

        # Initialize DINOv2 settings
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.desc = ViTExtractor('dinov2_vits14', stride=14, device=self.device)

        # Initialize EMA for velocity smoothing
        self.initialize_ema()

        # Initialize ROS
        if not rospy.get_node_uri():
            rospy.init_node('ibvs_controller', anonymous=True)

        # Initialize ROS communication
        self.bridge = CvBridge()
        self.setup_ros_communication()

        # Load goal image
        self.goal_image = self.load_goal_image(self.image_path)
        self.goal_image_array = np.array(self.goal_image)

        # Wait for first image
        rospy.loginfo("Waiting for the first image...")
        rospy.wait_for_message("/camera/color/image_raw", ImageMsg, timeout=10.0)
        rospy.loginfo("First image received!")

    def load_parameters(self):
        """Load parameters from a YAML configuration file."""
        with open(self.config_path, 'r') as file:
            config = yaml.safe_load(file)

        # Camera and image parameters
        self.u_max = config['u_max']  # Image width
        self.v_max = config['v_max']  # Image height
        self.f_x = config['f_x']  # Focal length x
        self.f_y = config['f_y']  # Focal length y
        self.c_x = self.u_max / 2  # Principal point x ONLY VALID FOR SIMULATION
        self.c_y = self.v_max / 2  # Principal point y ONLY VALID FOR SIMULATION

        # Control parameters
        self.lambda_ = config['lambda_']  # Control gain
        self.max_velocity = config.get('max_velocity', 1.0)
        self.min_error = config['min_error']
        self.max_error = config['max_error']
        self.num_pairs = config['num_pairs']  # Number of feature pairs to track

        # DINO feature detection parameters
        self.background_thresh = config['background_thresh']
        self.thresh_filter_keypoints = config['thresh_filter_keypoints']
        self.dino_input_size = config['dino_input_size']
        self.use_feature_binning = config['use_feature_binning']

        # Sampling parameters
        self.num_samples = config['num_samples']
        self.num_circles = config['num_circles']
        self.circle_radius_aug = config['circle_radius_aug']

        # Convergence parameters
        self.velocity_convergence_threshold = config['velocity_convergence_threshold']
        self.velocity_threshold_translation = config['velocity_threshold_translation']
        self.velocity_threshold_rotation = config['velocity_threshold_rotation']
        self.error_threshold_ratio = config['error_threshold_ratio']
        self.error_threshold_absolute_translation = config['error_threshold_absolute_translation']
        self.error_threshold_absolute_rotation = config['error_threshold_absolute_rotation']

        # Iteration control
        self.min_iterations = config['min_iterations']
        self.max_iterations = config['max_iterations']
        self.max_velocity_vector_history = config['max_velocity_vector_history']

        # EMA parameter
        self.ema_alpha = config.get('ema_alpha', 0.1)

        # Set the image path
        current_directory = os.path.dirname(__file__)
        self.image_path = os.path.join(current_directory, config['image_path'])

    def initialize_ema(self):
        """Initialize EMA for each velocity component."""
        self.ema_velocities = [None] * 6

    def update_ema(self, index, new_value):
        """Update EMA for a single velocity component."""
        if self.ema_velocities[index] is None:
            self.ema_velocities[index] = new_value
        else:
            self.ema_velocities[index] = self.ema_alpha * new_value + (1 - self.ema_alpha) * self.ema_velocities[index]
        return self.ema_velocities[index]

    def is_visual_servoing_done(self):
        """Check if visual servoing has converged or should stop."""
        window_size = 100  # Window for calculating velocity_mean_100
        trend_window = 200  # Window for analyzing the trend

        # Check minimum iterations
        if self.iteration_count < self.min_iterations:
            return False, False

        # Calculate current velocity mean
        current_velocity_mean = np.mean(np.abs(self.v_c))
        self.velocity_history.append(current_velocity_mean)

        # Calculate mean velocity for the window
        velocity_mean_100 = np.mean(self.velocity_history[-window_size:]) if len(
            self.velocity_history) >= window_size else None

        # Calculate current errors
        current_error_translation, current_error_rotation = self.calculate_end_error(self.desired_orientation)

        # Check if current error is more than twice the initial error
        if current_error_translation > 2 * self.initial_position_error:
            rospy.logerr("Position error exceeds twice the initial error. Aborting.")
            return True, False

        if velocity_mean_100 is not None:
            self.velocity_mean_history.append(velocity_mean_100)

            # Check if we have enough data to analyze trend
            if len(self.velocity_mean_history) >= trend_window:
                trend_increasing, _ = detect_trend(self.velocity_mean_history[-trend_window:])

                if trend_increasing:
                    error_reduced = (current_error_translation <= self.target_error_translation) and \
                                    (current_error_rotation <= self.target_error_rotation)
                    if error_reduced:
                        rospy.loginfo("Error reduction achieved. Stopping.")
                        return True, True

        # Check convergence criteria
        velocity_below_threshold = velocity_mean_100 is not None and \
                                   velocity_mean_100 < self.velocity_convergence_threshold
        error_reduced = (current_error_translation <= self.target_error_translation) and \
                        (current_error_rotation <= self.target_error_rotation)

        # Print status every 250 iterations
        if self.iteration_count % 250 == 0:
            rospy.logwarn(f"Translation error: {current_error_translation:.4f} cm "
                          f"(target: {self.target_error_translation:.4f} cm)")
            rospy.logwarn(f"Rotation error: {current_error_rotation:.4f} degrees "
                          f"(target: {self.target_error_rotation:.4f} degrees)")
            if velocity_mean_100 is not None:
                rospy.loginfo(f"Mean velocity (100 iterations): {velocity_mean_100:.6f}")

        converged = error_reduced and velocity_below_threshold

        # Check maximum iterations
        if self.iteration_count >= self.max_iterations:
            target_error_translation_90 = self.initial_error_translation * 0.1
            target_error_rotation_90 = self.initial_error_rotation * 0.1
            error_reduced_90_percent = (current_error_translation <= target_error_translation_90) and \
                                       (current_error_rotation <= target_error_rotation_90)

            if error_reduced_90_percent:
                rospy.logwarn("Max iterations reached with 90% error reduction. Marking as converged.")
                return True, True
            else:
                rospy.logwarn("Max iterations reached without sufficient error reduction.")
                return True, False

        return converged, converged

    def setup_ros_communication(self):
        """Setup ROS publishers and subscribers."""
        # Subscribers
        self.image_sub_rgb = rospy.Subscriber("/camera/color/image_raw",
                                              ImageMsg,
                                              self.image_callback_rgb,
                                              queue_size=10)
        self.image_sub_depth = rospy.Subscriber("/camera/depth/image_raw",
                                                ImageMsg,
                                                self.image_callback_depth,
                                                queue_size=10)

        # Publishers
        self.pub = rospy.Publisher('/camera_vel', Twist, queue_size=10)
        self.image_pub = rospy.Publisher('/camera/image_processed', ImageMsg, queue_size=10)
        self.goal_image_pub = rospy.Publisher('/goal_image_processed', ImageMsg, queue_size=10)
        self.current_image_pub = rospy.Publisher('/current_image_processed', ImageMsg, queue_size=10)
        self.correspondence_pub = rospy.Publisher('/correspondence_visualization', ImageMsg, queue_size=10)

        rospy.loginfo("ROS publishers and subscribers initialized")

    def load_goal_image(self, image_path):
        """Load the goal image from the specified path."""
        try:
            goal_image = Image.open(image_path)
            goal_image = goal_image.convert('RGB')  # Ensure the image is in RGB format
            rospy.loginfo(".... got goal image")
            return goal_image
        except Exception as e:
            rospy.logerr(f"Failed to load image at {image_path}: {e}")
            sys.exit(1)

    def image_callback_rgb(self, msg):
        """Callback for RGB image messages."""
        self.latest_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        self.latest_pil_image = Image.fromarray(cv2.cvtColor(self.latest_image, cv2.COLOR_BGR2RGB))

    def image_callback_depth(self, msg):
        """Callback for depth image messages."""
        self.latest_image_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

    def detect_features(self):
        """Detect features using DINOv2."""
        if self.latest_image is None:
            return None, None

        # Resize images to DINOv2's input size
        goal_image_resized = self.goal_image.resize((self.dino_input_size, self.dino_input_size))
        current_image_resized = self.latest_pil_image.resize((self.dino_input_size, self.dino_input_size))

        with torch.no_grad():
            # Process images
            goal_tensor = self.desc.preprocess_pil(goal_image_resized)
            current_tensor = self.desc.preprocess_pil(current_image_resized)

            desc1 = self.desc.extract_descriptors(
                goal_tensor.to(self.device),
                layer=11,
                facet='token',
                bin=self.use_feature_binning
            )
            desc2 = self.desc.extract_descriptors(
                current_tensor.to(self.device),
                layer=11,
                facet='token',
                bin=self.use_feature_binning
            )
            # Find correspondences
            points1, points2, sim_selected_12 = find_correspondences_batch(
                desc1, desc2,
                num_pairs=self.num_pairs
            )

            if points1 is None or points2 is None:
                return None, None

            # Scale points
            scale = self.dino_input_size / int(np.sqrt(desc1.size(-2)))
            points1_scaled = points1 * scale + scale / 2  # goal points
            points2_scaled = points2 * scale + scale / 2  # current points

            # Visualize correspondences
            self.visualize_correspondences_with_lines(
                goal_image_resized,
                current_image_resized,
                points1_scaled,
                points2_scaled
            )

            return self.calculate_uv(points1_scaled.tolist(), points2_scaled.tolist()), sim_selected_12

    def calculate_uv(self, goal_features, current_features):
        """Calculate feature points and scale them to the real image resolution."""
        num_pairs = self.num_pairs

        s_uv_star_resized = np.flip(np.asarray(goal_features), 1)
        s_uv_resized = np.flip(np.asarray(current_features), 1)

        # Calculate the scaled feature values
        s_uv = np.zeros([num_pairs, 2], dtype=int)
        s_uv_star = np.zeros([num_pairs, 2], dtype=int)

        if len(goal_features) != num_pairs:
            rospy.logwarn("Number of extracted features doesn't match desired number of pairs")
            num_pairs = len(goal_features)
            if num_pairs < 4:
                rospy.logwarn("Too few features detected (<4). Skipping processing.")
                return s_uv_star, s_uv

        # Scale factors to convert from DINO input size to original image size
        scale_x = self.u_max / self.dino_input_size
        scale_y = self.v_max / self.dino_input_size

        for count in range(num_pairs):
            s_uv_star[count, 0] = round(s_uv_star_resized[count, 0] * scale_x)
            s_uv_star[count, 1] = round(s_uv_star_resized[count, 1] * scale_y)
            s_uv[count, 0] = round(s_uv_resized[count, 0] * scale_x)
            s_uv[count, 1] = round(s_uv_resized[count, 1] * scale_y)

        return s_uv_star, s_uv

    def publish_figure(self, fig, publisher):
        """Convert a matplotlib figure to a ROS Image message and publish it."""
        fig.canvas.draw()
        width, height = fig.canvas.get_width_height()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8').reshape(height, width, 3)
        ros_image_msg = self.bridge.cv2_to_imgmsg(image, encoding="rgb8")
        publisher.publish(ros_image_msg)

        # Close the figure to free up memory
        plt.close(fig)

    def get_depth(self, current_points):
        """Get depth values for the current feature points."""
        if self.latest_image_depth is None:
            rospy.logwarn("No depth image received yet")
            return None

        z_values_meter = np.zeros((len(current_points), 1))
        height, width = self.latest_image_depth.shape

        for count, point in enumerate(current_points):
            x, y = int(point[0]), int(point[1])

            # Ensure the point is within the image bounds
            if 0 <= x < width and 0 <= y < height:
                depth_value = self.latest_image_depth[y, x]
                # Convert depth value to meters
                z_values_meter[count] = depth_value / 1000.0 if depth_value != 0 else 100
            else:
                z_values_meter[count] = 100  # Out of bounds point

        return z_values_meter

    def ibvs(self):
        """Image Based Visual Servoing Method."""
        if self.latest_image is None:
            print("No latest image available")
            return

        start_time = time.time()

        (s_uv_star, s_uv), _ = self.detect_features()
        if s_uv_star is None or s_uv is None:
            print("Feature detection failed")
            return

        self.draw_points(np.array(self.latest_pil_image), s_uv, s_uv_star)

        # Transform feature points to real-world coordinates
        s_xy, s_star_xy = self.transform_to_real_world(s_uv, s_uv_star)

        # Calculate error and interaction matrix
        e = s_xy - s_star_xy
        e = e.reshape((len(s_xy) * 2, 1))

        Z = self.get_depth(s_uv)
        L = self.calculate_interaction_matrix(s_xy, Z)

        # Calculate camera velocities using control law
        v_c = -self.lambda_ * np.linalg.pinv(L.astype('float')) @ e

        # Update EMA for each velocity component
        self.v_c = np.array([self.update_ema(i, v) for i, v in enumerate(v_c.flatten())])

        end_time = time.time()
        print(f"IBVS iteration executed in: {end_time - start_time:.2f} seconds ||| iteration count: {self.iteration_count}")

    def transform_to_real_world(self, s_uv, s_uv_star):
        """Transform pixel feature points to real-world coordinates."""
        s_xy = []
        s_star_xy = []

        for uv, uv_star in zip(s_uv, s_uv_star):
            x = (uv[0] - self.c_x) / self.f_x
            y = (uv[1] - self.c_y) / self.f_y
            s_xy.append([x, y])

            x_star = (uv_star[0] - self.c_x) / self.f_x
            y_star = (uv_star[1] - self.c_y) / self.f_y
            s_star_xy.append([x_star, y_star])

        return np.array(s_xy), np.array(s_star_xy)

    def calculate_interaction_matrix(self, s_xy, Z):
        """Calculate the interaction matrix for the feature points."""
        L = np.zeros([2 * len(s_xy), 6], dtype=float)

        for count in range(len(s_xy)):
            x, y, z = s_xy[count, 0], s_xy[count, 1], Z[count, 0]
            L[2 * count, :] = [-1 / z, 0, x / z, x * y, -(1 + x ** 2), y]
            L[2 * count + 1, :] = [0, -1 / z, y / z, 1 + y ** 2, -x * y, -x]

        return L

    def publish_twist(self, v_c):
        """Publish velocity commands and store history."""
        twist_msg = Twist()

        # Apply velocity limits
        twist_msg.linear.x = np.clip(v_c[2], -self.max_velocity, self.max_velocity)
        twist_msg.linear.y = np.clip(-v_c[0], -self.max_velocity, self.max_velocity)
        twist_msg.linear.z = np.clip(-v_c[1], -self.max_velocity, self.max_velocity)
        twist_msg.angular.x = np.clip(v_c[5], -self.max_velocity, self.max_velocity)
        twist_msg.angular.y = np.clip(-v_c[3], -self.max_velocity, self.max_velocity)
        twist_msg.angular.z = np.clip(-v_c[4], -self.max_velocity, self.max_velocity)

        # Store velocity history
        self.applied_velocity_x.append(twist_msg.linear.x)
        self.applied_velocity_y.append(twist_msg.linear.y)
        self.applied_velocity_z.append(twist_msg.linear.z)
        self.applied_velocity_roll.append(twist_msg.angular.x)
        self.applied_velocity_pitch.append(twist_msg.angular.y)
        self.applied_velocity_yaw.append(twist_msg.angular.z)

        if np.any(np.abs(v_c) > self.max_velocity):
            rospy.logwarn("Velocity capped due to exceeding maximum allowed value.")

        self.pub.publish(twist_msg)

    def draw_points(self, image, current_points, goal_points):
        """Draw current and goal feature points on the image."""
        for x, y in current_points:
            cv2.circle(image, (x, y), 1, (0, 255, 0), -1)  # Current points in green
        for x, y in goal_points:
            cv2.circle(image, (x, y), 1, (0, 0, 255), -1)  # Goal points in red

        ros_image = self.bridge.cv2_to_imgmsg(image, "bgr8")
        self.image_pub.publish(ros_image)

    def run(self):
        """Main visual servoing control loop."""
        rospy.loginfo("Starting visual servoing process...")
        self.iteration_count = 0

        # Initialize histories
        self.position_history = []
        self.orientation_history = []
        self.velocity_history = []
        self.average_velocities = []
        self.velocity_mean_100 = []
        self.velocity_mean_10 = []

        # Initialize velocity tracking
        self.applied_velocity_x = []
        self.applied_velocity_y = []
        self.applied_velocity_z = []
        self.applied_velocity_roll = []
        self.applied_velocity_pitch = []
        self.applied_velocity_yaw = []

        # Initialize tracking variables
        lowest_position_error = float('inf')
        lowest_orientation_error = float('inf')

        while not rospy.is_shutdown():
            if self.latest_image is None:
                rospy.logwarn_throttle(1, "Waiting for image...")
                continue

            # Perform IBVS
            self.ibvs()
            self.iteration_count += 1

            # Calculate and store average velocity
            avg_velocity = np.mean(np.abs(self.v_c))
            self.average_velocities.append(avg_velocity)
            self.velocity_history.append(avg_velocity)

            # Update velocity means
            if len(self.velocity_history) >= 100:
                self.velocity_mean_100.append(np.mean(self.velocity_history[-100:]))
            else:
                self.velocity_mean_100.append(np.mean(self.velocity_history))

            if len(self.velocity_history) >= 10:
                self.velocity_mean_10.append(np.mean(self.velocity_history[-10:]))
            else:
                self.velocity_mean_10.append(np.mean(self.velocity_history))

            # Apply control
            self.publish_twist(self.v_c)

            # Update position and orientation tracking
            self.camera_position, self.orientation_quaternion = self.get_current_camera_pose()
            self.position_history.append(self.camera_position)
            self.orientation_history.append(self.orientation_quaternion)

            # Calculate current errors
            current_position_error, current_orientation_error = self.calculate_end_error(self.desired_orientation)

            # Update the lowest errors
            lowest_position_error = min(lowest_position_error, current_position_error)
            lowest_orientation_error = min(lowest_orientation_error, current_orientation_error)

            # Check if servoing is done
            done, converged = self.is_visual_servoing_done()
            if done:
                rospy.loginfo(f"Visual servoing completed after {self.iteration_count} iterations.")
                rospy.loginfo(f"Converged: {converged}")

                # Calculate final errors
                final_position_error, final_orientation_error = self.calculate_end_error(self.desired_orientation)

                # Log final status
                rospy.loginfo(f"Final Position Error: {final_position_error:.2f} cm")
                rospy.loginfo(f"Final Orientation Error: {final_orientation_error:.2f} degrees")
                rospy.loginfo(f"Lowest Position Error: {lowest_position_error:.2f} cm")
                rospy.loginfo(f"Lowest Orientation Error: {lowest_orientation_error:.2f} degrees")

                return (self.camera_position, self.orientation_quaternion, converged,
                        final_position_error, final_orientation_error,
                        np.array(self.position_history), np.array(self.orientation_history),
                        self.iteration_count,
                        lowest_position_error, lowest_orientation_error,
                        np.array(self.average_velocities),
                        np.array(self.velocity_mean_100),
                        np.array(self.velocity_mean_10),
                        np.array(self.applied_velocity_x),
                        np.array(self.applied_velocity_y),
                        np.array(self.applied_velocity_z),
                        np.array(self.applied_velocity_roll),
                        np.array(self.applied_velocity_pitch),
                        np.array(self.applied_velocity_yaw))

        # If interrupted, send zero velocity
        self.publish_twist(np.zeros(6))
        final_position_error, final_orientation_error = self.calculate_end_error(self.desired_orientation)

        return (self.camera_position, self.orientation_quaternion, False,
                final_position_error, final_orientation_error,
                np.array(self.position_history), np.array(self.orientation_history),
                self.iteration_count,
                lowest_position_error, lowest_orientation_error,
                np.array(self.average_velocities),
                np.array(self.velocity_mean_100),
                np.array(self.velocity_mean_10),
                np.array(self.applied_velocity_x),
                np.array(self.applied_velocity_y),
                np.array(self.applied_velocity_z),
                np.array(self.applied_velocity_roll),
                np.array(self.applied_velocity_pitch),
                np.array(self.applied_velocity_yaw))

    def calculate_end_error(self, desired_orientation):
        """Calculate position and orientation errors."""
        # Calculate position error in centimeters
        position_error = np.linalg.norm(self.camera_position - self.desired_position) * 100

        # Calculate orientation error in degrees
        current_orientation = R.from_quat(self.orientation_quaternion)
        desired_orientation = R.from_quat(desired_orientation)
        orientation_error = (current_orientation.inv() * desired_orientation).magnitude() * (180 / np.pi)

        return position_error, orientation_error

    def visualize_correspondences_with_lines(self, goal_image, current_image, points1, points2):
        """Create and publish visualization of correspondences between two images."""
        fig = plt.figure(figsize=(12, 6))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

        ax1.imshow(goal_image)
        ax2.imshow(current_image)

        # Convert points to numpy if they're tensors
        points1_np = points1.cpu().numpy() if torch.is_tensor(points1) else np.array(points1)
        points2_np = points2.cpu().numpy() if torch.is_tensor(points2) else np.array(points2)

        # Plot correspondences with rainbow colors
        colors = plt.cm.rainbow(np.linspace(0, 1, len(points1_np)))
        for i, ((y1, x1), (y2, x2), color) in enumerate(zip(points1_np, points2_np, colors)):
            # Plot points and labels
            ax1.plot(x1, y1, 'o', color=color, markersize=8)
            ax1.text(x1 + 5, y1 + 5, str(i), color=color, fontsize=8)

            ax2.plot(x2, y2, 'o', color=color, markersize=8)
            ax2.text(x2 + 5, y2 + 5, str(i), color=color, fontsize=8)

            # Draw correspondence lines
            con = ConnectionPatch(
                xyA=(x1, y1), xyB=(x2, y2),
                coordsA="data", coordsB="data",
                axesA=ax1, axesB=ax2, color=color, alpha=0.5
            )
            fig.add_artist(con)

        ax1.set_title("Goal Image")
        ax2.set_title("Current Image")
        plt.tight_layout()

        # Convert figure to ROS message and publish
        fig.canvas.draw()
        img_data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img_data = img_data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        ros_image = self.bridge.cv2_to_imgmsg(img_data, encoding="rgb8")

        # Publish all visualizations
        self.correspondence_pub.publish(ros_image)
        self.goal_image_pub.publish(self.bridge.cv2_to_imgmsg(np.array(goal_image), encoding="rgb8"))
        self.current_image_pub.publish(self.bridge.cv2_to_imgmsg(np.array(current_image), encoding="rgb8"))

        plt.close(fig)

    def get_current_camera_pose(self):
        """Get current camera pose from Gazebo."""
        rospy.wait_for_service('/gazebo/get_model_state')
        try:
            get_model_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
            model_state = get_model_state('realsense2_camera', '')

            position = np.array([
                model_state.pose.position.x,
                model_state.pose.position.y,
                model_state.pose.position.z
            ])

            orientation = np.array([
                model_state.pose.orientation.x,
                model_state.pose.orientation.y,
                model_state.pose.orientation.z,
                model_state.pose.orientation.w
            ])

            return position, orientation
        except rospy.ServiceException as e:
            rospy.logerr(f"Failed to get camera pose: {e}")
            return None, None

def scale_points_direct(points, target_size, num_patches):
    """Scale points from patch coordinates to pixel coordinates"""
    scale = target_size / num_patches
    scaled_points = points * scale + scale/2
    scaled_points = torch.clamp(scaled_points, 0, target_size-1)
    return scaled_points

def sample_camera_positions(volume_dimensions, num_samples, desired_position):
    """
    Sample random camera positions within a specified volume.

    Args:
        volume_dimensions (np.ndarray): The dimensions of the volume for sampling (width, height, depth).
        num_samples (int): The number of samples to generate.
        desired_position (np.ndarray): The desired central position to offset the samples from.

    Returns:
        np.ndarray: An array of sampled camera positions.
    """
    # Offset the volume to be centered around the desired position
    half_dims = volume_dimensions / 2
    min_bounds = desired_position - half_dims
    max_bounds = desired_position + half_dims

    # Sample positions uniformly within the defined bounds
    positions = np.random.uniform(min_bounds, max_bounds, size=(num_samples, 3))
    return positions


def sample_focal_points_original(num_samples, reference_point, num_circles, circle_radius_aug):
    """
    Sample focal points based on the original implementation in PoseLookingAtSamePointWithNoiseAndRotationZGenerator.

    Args:
        num_samples (int): The total number of samples to generate.
        reference_point (np.ndarray): The reference point (3D vector: [x, y, z]).
        num_circles (int): Number of circles to generate points on.
        circle_radius_aug (float): Radius augmentation factor for circles.

    Returns:
        np.ndarray: An array of sampled focal points (shape: [num_samples, 3]).
    """
    samples_per_circle = num_samples // num_circles
    looked_at_points = np.empty((num_samples, 3))

    for cn in range(num_circles):
        radius = circle_radius_aug * (cn + 1)
        istart = cn * samples_per_circle

        # Sample points on a circle
        rand_theta = np.random.uniform(-np.pi, np.pi, size=samples_per_circle)
        x = np.cos(rand_theta) * radius + reference_point[0]
        y = np.sin(rand_theta) * radius + reference_point[1]
        z = np.repeat(reference_point[2], samples_per_circle)

        points = np.column_stack((x, y, z))
        looked_at_points[istart: istart + samples_per_circle] = points

    return looked_at_points


def calculate_position_error(positions, desired_position):
    """
    Calculate the position error as the Euclidean distance from the desired position.

    Args:
        positions (np.ndarray): The sampled camera positions.
        desired_position (np.ndarray): The desired central position.

    Returns:
        tuple: The average error and standard deviation of the error in centimeters.
    """
    errors = np.linalg.norm(positions - desired_position, axis=1)
    average_error = np.mean(errors) * 100  # Convert to centimeters
    std_deviation = np.std(errors) * 100  # Convert to centimeters
    return average_error, std_deviation


def detect_trend(data, window_size=100, consecutive_increases=5):
    """
    Detect if there's an increasing trend in the data.

    Args:
    data (array): Input data
    window_size (int): Size of the sliding window for linear regression
    consecutive_increases (int): Number of consecutive positive slopes required to confirm trend

    Returns:
    tuple: (bool indicating if trend is increasing, index where trend starts)
    """
    slopes = []
    for i in range(len(data) - window_size):
        y = data[i:i + window_size]
        x = np.arange(window_size)
        slope, _, _, _, _ = stats.linregress(x, y)
        slopes.append(slope)

    increasing_count = 0
    for i, slope in enumerate(slopes):
        if slope > 0:
            increasing_count += 1
            if increasing_count >= consecutive_increases:
                return True, i + window_size - consecutive_increases
        else:
            increasing_count = 0

    return False, -1


def calculate_orientation_error(quaternion_list, desired_orientation):
    # Ensure quaternion_list is a numpy array
    quaternion_list = np.array(quaternion_list)  # is x y z w

    # Convert desired_orientation to a Rotation object
    desired_rotation = R.from_quat(desired_orientation)

    errors = []

    for quaternion in quaternion_list:
        # Convert the current quaternion to a Rotation object
        current_rotation = R.from_quat(quaternion)

        # Calculate the relative rotation from current to desired
        relative_rotation = current_rotation.inv() * desired_rotation

        # Get the angle of rotation (in radians)
        angle = relative_rotation.magnitude()

        # Convert to degrees
        error_degrees = np.degrees(angle)

        errors.append(error_degrees)

    errors = np.array(errors)

    # Calculate mean and standard deviation of the errors
    mean_error = np.mean(errors)
    std_dev_error = np.std(errors)

    return mean_error, std_dev_error


def place_red_box_at_focal_point(x, y, z=0.01, counter=1):
    """
    Place a red box model at the given focal point in Gazebo.

    Args:
        x (float): The x-coordinate of the focal point.
        y (float): The y-coordinate of the focal point.
        z (float): The z-coordinate (height above the poster).
        counter (int): A counter to create unique model names.
    """
    rospy.wait_for_service('/gazebo/spawn_sdf_model')
    try:
        spawn_model = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)

        # Load the model XML from the file
        model_xml = open("/root/catkin_ws/src/ibvs/models/red_box/model.sdf", 'r').read()

        initial_pose = Pose()
        initial_pose.position.x = x
        initial_pose.position.y = y
        initial_pose.position.z = z

        # Create a unique model name using the counter
        model_name = f"red_box_{counter}"

        # Spawn the model in Gazebo
        spawn_model(model_name, model_xml, "", initial_pose, "world")
        print(f"Spawned {model_name} at position ({x}, {y}, {z})")
    except rospy.ServiceException as e:
        print(f"Service call failed: {e}")


def set_camera_pose(camera_position, orientation_quaternion):
    """
    Set the camera's pose in Gazebo with the given position and orientation.

    Args:
        camera_position (np.ndarray): The position of the camera.
        orientation_quaternion (np.ndarray): The orientation quaternion for the camera.
    """
    rospy.wait_for_service('/gazebo/set_model_state')
    try:
        set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)

        # Create a new state for the camera
        state = ModelState()
        state.model_name = 'realsense2_camera'
        state.pose.position.x = camera_position[0]
        state.pose.position.y = camera_position[1]
        state.pose.position.z = camera_position[2]
        state.pose.orientation.x = orientation_quaternion[0]
        state.pose.orientation.y = orientation_quaternion[1]
        state.pose.orientation.z = orientation_quaternion[2]
        state.pose.orientation.w = orientation_quaternion[3]
        state.reference_frame = 'world'

        # Set the new state in Gazebo
        set_state(state)
    except rospy.ServiceException as e:
        print(f"Service call failed: {e}")


def rotate_camera_x_axis(orientation_quaternion, angle_degrees):
    """
    Rotate the camera around its X-axis by the specified angle.

    Args:
        orientation_quaternion (np.ndarray): The original orientation quaternion.
        angle_degrees (float): The rotation angle in degrees.

    Returns:
        np.ndarray: The new orientation quaternion after rotation.
    """
    # Convert the original quaternion to a rotation object
    original_rotation = R.from_quat(orientation_quaternion)

    # Create a rotation around the X-axis
    x_rotation = R.from_euler('x', angle_degrees, degrees=True)

    # Combine the rotations
    new_rotation = original_rotation * x_rotation

    # Convert back to quaternion
    new_quaternion = new_rotation.as_quat()

    return new_quaternion


def find_and_set_best_pose(controller, camera_position, initial_quaternion):
    """
    Find the best pose by testing four different orientations and set the camera to that pose.

    Args:
        controller (Controller): The controller object.
        camera_position (np.ndarray): The initial camera position.
        initial_quaternion (np.ndarray): The initial orientation quaternion.

    Returns:
        tuple: A tuple containing the best camera position and orientation quaternion.
    """
    best_pose = None
    best_mean = float('-inf')
    current_mean = 0

    # Test four different orientations
    for angle in [0, 90, 180, 270]:
        if angle == 0:
            current_quaternion = initial_quaternion
        else:
            current_quaternion = rotate_camera_x_axis(initial_quaternion, angle)

        set_camera_pose(camera_position, current_quaternion)
        rospy.sleep(1)  # Wait for the pose to settle

        rospy.loginfo(f"Testing Camera Position ({angle}°): {camera_position}")
        rospy.loginfo(f"Testing Orientation Quaternion ({angle}°): {current_quaternion}")

        _, sim_selected_12 = controller.detect_features()
        current_mean = sim_selected_12.mean()
        rospy.loginfo(f"sim mean for {angle}° rotation: {current_mean}")

        # Update the best pose if current mean is higher
        if current_mean > best_mean:
            best_mean = current_mean
            best_pose = (camera_position, current_quaternion)

    # Set the camera to the best pose
    set_camera_pose(*best_pose)
    rospy.sleep(1)  # Wait for the pose to settle

    rospy.loginfo(f"Selected Best Camera Position: {best_pose[0]}")
    rospy.loginfo(f"Selected Best Orientation Quaternion: {best_pose[1]}")
    rospy.loginfo(f"Best sim mean: {best_mean}")

    return best_pose


def manage_gazebo_models(model_index):
    """
    Delete the current model and spawn a new perturbed model in Gazebo.

    Args:
        model_index (int): The index of the perturbed model to spawn (1-500).
    """
    # Delete the current model
    rospy.wait_for_service('/gazebo/delete_model')
    try:
        delete_model = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)

        # If it's the first iteration, delete the original "resized" model
        if model_index == 1:
            model_to_delete = "resized"
        else:
            model_to_delete = f"resized{model_index - 1}"

        delete_model(model_to_delete)
        rospy.loginfo(f"Deleted model: {model_to_delete}")
    except rospy.ServiceException as e:
        rospy.logerr(f"Failed to delete model {model_to_delete}: {e}")

    # Spawn the new perturbed model
    rospy.wait_for_service('/gazebo/spawn_sdf_model')
    try:
        spawn_model = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)

        # Construct the path to the new model
        model_path = f"/root/catkin_ws/src/ibvs/models/viso{model_index}/model.sdf"

        # Check if the model file exists
        if not os.path.exists(model_path):
            rospy.logerr(f"Model file not found: {model_path}")
            return

        with open(model_path, "r") as f:
            model_xml = f.read()

        # Set the pose for the new model
        initial_pose = Pose()
        initial_pose.position.x = 0
        initial_pose.position.y = 0
        initial_pose.position.z = 0.005

        # Convert Euler angles to quaternion
        quaternion = tf_conversions.transformations.quaternion_from_euler(1.5708, 0, 1.5708)
        initial_pose.orientation.x = quaternion[0]
        initial_pose.orientation.y = quaternion[1]
        initial_pose.orientation.z = quaternion[2]
        initial_pose.orientation.w = quaternion[3]

        # Spawn the model
        new_model_name = f"resized{model_index}"
        spawn_model(new_model_name, model_xml, "", initial_pose, "world")

        rospy.loginfo(f"Spawned perturbed model: {new_model_name}")
    except rospy.ServiceException as e:
        rospy.logerr(f"Failed to spawn model {new_model_name}: {e}")


def calculate_look_at_orientation(camera_positions, focal_points):
    """
    Calculate the rotation matrix and quaternion for the camera to look at the target position.

    Args:
        camera_positions (np.ndarray): The positions of the camera.
        focal_points (np.ndarray): The target positions (focal point).

    Returns:
        tuple: A tuple containing two numpy arrays:
               - An array of rotation matrices for each sample
               - An array of quaternions (w, x, y, z) for each sample
    """
    num_samples = len(camera_positions)
    rotation_matrices = np.empty((num_samples, 3, 3))
    quaternions = np.empty((num_samples, 4))

    for i in range(num_samples):
        # Calculate the forward vector (X-axis of camera)
        forward = focal_points[i] - camera_positions[i]
        forward = forward / np.linalg.norm(forward)

        # Calculate the right vector (negative Y-axis of camera)
        world_up = np.array([-1, 0, 0])  # Z is up in world space
        right = -np.cross(forward, world_up)
        right = right / np.linalg.norm(right)

        # Calculate the up vector (Z-axis of camera)
        up = np.cross(right, forward)

        # Construct the rotation matrix
        rotation_matrix = np.column_stack((forward, -right, up))
        rotation_matrices[i] = rotation_matrix

        # Convert rotation matrix to quaternion
        r = R.from_matrix(rotation_matrix)
        quat = r.as_quat()  # Returns in scalar-last format [x, y, z, w]
        quaternions[i] = r.as_quat()

    return rotation_matrices, quaternions


def apply_z_axis_rotation(rotation_matrices, num_circles, samples_per_circle, rz_max=np.radians(120)):
    """
    Apply a random rotation around the optical axis (z-axis) to the given rotation matrices.

    Args:
        rotation_matrices (np.ndarray): The initial rotation matrices.
        num_circles (int): Number of circles used in sampling.
        samples_per_circle (int): Number of samples per circle.
        rz_max (float): Maximum rotation angle around the optical axis in radians.

    Returns:
        np.ndarray: An array of quaternions (x, y, z, w) for each sample after z-axis rotation.
    """
    num_samples = len(rotation_matrices)
    quaternions = []

    for cn in range(num_circles):
        # Generate a sequence of rotation angles for this circle
        rz_values = np.linspace(-rz_max, rz_max, num=samples_per_circle)

        for i in range(samples_per_circle):
            idx = cn * samples_per_circle + i
            if idx >= num_samples:
                break

            # Create a rotation matrix for the optical axis rotation
            rz = rz_values[i]
            cos_rz = np.cos(rz)
            sin_rz = np.sin(rz)
            Rx = np.array([
                [1, 0, 0],
                [0, cos_rz, -sin_rz],
                [0, sin_rz, cos_rz]
            ])

            # Apply the optical axis rotation to the initial rotation matrix
            final_rotation_matrix = np.dot(rotation_matrices[idx], Rx)

            # Convert final rotation matrix to quaternion using scipy
            r = R.from_matrix(final_rotation_matrix)
            quaternion = r.as_quat()  # Returns in scalar-last format [x, y, z, w]

            # Reorder to [w, x, y, z] to match common conventions
            # quaternion = np.roll(quaternion, 1)

            quaternions.append(quaternion)

    return np.array(quaternions)


def main(args):
    # Initialize ROS node
    rospy.init_node('visual_servoing_inference', anonymous=True)
    start_time = time.time()

    # Initialize result storage lists
    results = {
        'final_positions': [],
        'final_quaternions': [],
        'convergence_flags': [],
        'position_errors': [],
        'orientation_errors': [],
        'best_poses': [],
        'position_histories': [],
        'orientation_histories': [],
        'iteration_histories': [],
        'lowest_position_errors': [],
        'lowest_orientation_errors': [],
        'average_velocities': [],
        'velocity_mean_100': [],
        'velocity_mean_10': [],
        'velocities': {
            'x': [], 'y': [], 'z': [],
            'roll': [], 'pitch': [], 'yaw': []
        }
    }

    # Load configuration
    current_directory = os.path.dirname(__file__)
    config_filename = args.config if args.config else 'config.yaml'
    config_path = os.path.join(current_directory, f'../config/{config_filename}')
    config_path = os.path.abspath(config_path)

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Set up simulation parameters
    desired_position = np.array([0, 0, 0.61])
    desired_orientation = np.array([0, 0.7071068, 0, 0.7071068])
    box_sample_size = np.array([1.2, 1.2, 0.3])
    reference_point = np.array([0.0, 0.0, 0.01])

    # Set random seeds for reproducibility
    torch.manual_seed(121)
    np.random.seed(41)

    # Sample camera positions and focal points
    camera_positions = sample_camera_positions(box_sample_size, config['num_samples'], desired_position)
    focal_points = sample_focal_points_original(config['num_samples'],
                                                reference_point,
                                                config['num_circles'],
                                                config['circle_radius_aug'])

    # Calculate initial orientations
    look_at_matrices, orientations = calculate_look_at_orientation(camera_positions, focal_points)

    # Initialize controller
    controller = Controller(desired_position, desired_orientation, config_path)

    # Run visual servoing for each sample
    for i in range(config['num_samples']):
        print(f"\nProcessing sample {i + 1}/{config['num_samples']}")

        if args.perturbation:
            manage_gazebo_models(i + 1)
            rospy.sleep(1)

        # Find best pose and run servoing
        best_pose = find_and_set_best_pose(controller, camera_positions[i], orientations[i])

        result = controller.run()

        # Unpack results
        (final_position, final_quaternion, converged,
         position_error, orientation_error,
         position_history, orientation_history, iteration_history,
         lowest_position_error, lowest_orientation_error,
         average_velocities, velocity_mean_100, velocity_mean_10,
         vel_x, vel_y, vel_z, vel_roll, vel_pitch, vel_yaw) = result

        # Store results
        results['final_positions'].append(final_position)
        results['final_quaternions'].append(final_quaternion)
        results['convergence_flags'].append(converged)
        results['position_errors'].append(position_error)
        results['orientation_errors'].append(orientation_error)
        results['best_poses'].append(best_pose)
        results['position_histories'].append(position_history)
        results['orientation_histories'].append(orientation_history)
        results['iteration_histories'].append(iteration_history)
        results['lowest_position_errors'].append(lowest_position_error)
        results['lowest_orientation_errors'].append(lowest_orientation_error)
        results['average_velocities'].append(average_velocities)
        results['velocity_mean_100'].append(velocity_mean_100)
        results['velocity_mean_10'].append(velocity_mean_10)
        results['velocities']['x'].append(vel_x)
        results['velocities']['y'].append(vel_y)
        results['velocities']['z'].append(vel_z)
        results['velocities']['roll'].append(vel_roll)
        results['velocities']['pitch'].append(vel_pitch)
        results['velocities']['yaw'].append(vel_yaw)

    # Calculate execution time
    end_time = time.time()
    total_execution_time = end_time - start_time

    # Save results
    config_name = os.path.splitext(os.path.basename(args.config))[0]
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_filename = f"results_{config_name}_{timestamp}.npz"

    # Create data dictionary for saving
    save_data = {
        'initial_positions': camera_positions,
        'initial_orientations': orientations,
        'final_positions': np.array(results['final_positions']),
        'final_quaternions': np.array(results['final_quaternions']),
        'convergence_flags': np.array(results['convergence_flags']),
        'position_errors': np.array(results['position_errors']),
        'orientation_errors': np.array(results['orientation_errors']),
        'best_poses': np.array(results['best_poses'], dtype=object),
        'position_histories': np.array(results['position_histories'], dtype=object),
        'orientation_histories': np.array(results['orientation_histories'], dtype=object),
        'iteration_histories': np.array(results['iteration_histories']),
        'lowest_position_errors': np.array(results['lowest_position_errors']),
        'lowest_orientation_errors': np.array(results['lowest_orientation_errors']),
        'average_velocities': np.array(results['average_velocities'], dtype=object),
        'velocity_mean_100': np.array(results['velocity_mean_100'], dtype=object),
        'velocity_mean_10': np.array(results['velocity_mean_10'], dtype=object),
        'velocities': {k: np.array(v, dtype=object) for k, v in results['velocities'].items()},
        'total_execution_time': total_execution_time,
        'config': config
    }

    np.savez(results_filename, **save_data)
    print(f"\nResults saved to: {results_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visual Servoing Inference")
    parser.add_argument("--config", type=str, help="Name of the configuration YAML file")
    parser.add_argument("--perturbation", action="store_true", help="Enable image perturbation")
    args = parser.parse_args()

    main(args)
