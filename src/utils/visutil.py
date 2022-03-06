import math
from typing import List, Optional

import matplotlib.cm
import numpy as np
import torch
from pytorch3d.renderer.cameras import CamerasBase
from pytorch3d.structures import Meshes
from pytorch3d.vis.plotly_vis import plot_scene
from visdom import Visdom

from ..nnutils.render import render_depths
from . import image as image_utils
from .misc import batchify_func


def visualize_initial_scene(
        viz : Visdom,
        rgbas : torch.Tensor,
        gt_cameras : Optional[List[CamerasBase]] = None,
        in_cameras : Optional[List[CamerasBase]] = None,
        gt_mesh : Optional[Meshes] = None,
        in_mesh : Optional[Meshes] = None,
        camera_scale : float = 0.2,
        win_size : int = 300,
    ) -> None:

    if viz is None:
        return

    N = len(rgbas)
    if not (None in (gt_cameras, gt_mesh)):
        depths = render_depths(gt_mesh, gt_cameras)
        depths = batchify_func(depths, visualize_depth)
        viz.images(
            depths,
            nrow=round(math.sqrt(N)),
            win="depths: gt_mesh x gt_camera",
            opts=dict(
                title= "depths: gt_mesh x gt_camera",
                width = win_size,
                height = win_size,
            ),
        )
    viz.images(
        rgbas[:,:3,:,:],
        nrow=round(math.sqrt(N)),
        win="input images",
        opts=dict(
            title= "input images",
            width = win_size,
            height = win_size,
        ),
    )
    camera_trace = {}
    if gt_cameras is not None:
        camera_trace.update({f"gt_cam_{i:03d}": gt_cameras[i].to('cpu') for i in range(N)})
    if in_cameras is not None:
        camera_trace.update({f"in_cam_{i:03d}": in_cameras[i].to('cpu') for i in range(N)})

    shape_trace = {}
    if gt_mesh is not None: shape_trace.update({'gt_shape': gt_mesh.cpu()})
    if in_mesh is not None: shape_trace.update({'in_shape': in_mesh.cpu()})
    plotly_plot = plot_scene({
            "initial scene": {
                **shape_trace,
                **camera_trace,
            },
        },
        camera_scale = camera_scale,
    )
    viz.plotlyplot(plotly_plot, win="initial scene")

def num_channels(img):
    Cdim = image_utils.get_img_format(img).find('C')
    return 1 if Cdim==-1 else img.shape[Cdim]

def visualize_depth(img, alpha_exists=None, force_consistent=False):
    """ normalize depths to (0,1) for visualization """
    if alpha_exists is None: alpha_exists = num_channels(img)>1
    img_format = image_utils.get_img_format(img)
    if not alpha_exists:
        if force_consistent or not img_format.startswith('N'):
            img = torch.where(img<0, img.max()*1.05, img)
            img = (img - img.min())/(img.max() - img.min() + 1e-12)
            return img.clamp(min=0, max=1)
        else:
            return torch.stack([
                visualize_depth(img[i], alpha_exists=False)
                for i in range(len(img))
            ])
    else:
        img, alpha, Cdim = image_utils.split_alpha(img)
        img = visualize_depth(img, alpha_exists=False)
        img = torch.cat([img, alpha], dim=Cdim)
        return img.clamp(min=0, max=1)

def visualize_normals(img, alpha_exists=None):
    """ normalize normals to (0,1) for visualization """
    if alpha_exists is None: alpha_exists = num_channels(img)>3
    if not alpha_exists:
        return ((img + 1)/2).clamp(min=0, max=1)
    else:
        img, alpha, Cdim = image_utils.split_alpha(img)
        img = visualize_normals(img, alpha_exists=False)
        img = torch.cat([img, alpha], dim=Cdim)
        return img.clamp(min=0, max=1)

def uv2bgr(UV):
    """
    UV: ...,2 in [-1,1]
    returns ...,3
    converts UV values to RGB color
    """
    orig_shape = UV.shape
    UV = UV.reshape(-1,2)
    hue = (UV[:,0]+1)/2 * 179
    light = (UV[:,1]+1)/2
    sat = np.where(light>0.5,(1-light)*2,1) * 255   # [1 -> 1 -> 0]
    val = np.where(light<0.5,light*2,1) * 255       # [0 -> 1 -> 1]
    import cv2
    input_image = np.stack((hue,sat,val),axis=-1)
    output_image = cv2.cvtColor(input_image[None,...].astype(np.uint8), cv2.COLOR_HSV2BGR)
    BGR = output_image.reshape(orig_shape[:-1]+(3,))
    return BGR

def gray_to_colormap(img, cmap='magma', device=None, out_format=None, alpha=False):
    """ grayscale image to cmap-colored image."""
    if device is None: device = img.device
    img = img.detach().cpu().numpy()

    if out_format is None: out_format = image_utils.get_img_format(img)
    img = image_utils.change_img_format(img, 'HW')
    img = np.clip(img, 0, 1)
    if isinstance(cmap, str):
        cmap = matplotlib.cm.get_cmap(cmap)
    img = cmap(img)
    if not alpha:
        img = img[:,:,:3]
    img = image_utils.change_img_format(img, out_format)

    img = torch.as_tensor(img, device=device)
    return img
