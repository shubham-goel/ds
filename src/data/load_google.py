import logging
import os
from pathlib import Path
from typing import Optional

import cv2
import imageio
import numpy as np
import torch
from dotmap import DotMap
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from pytorch3d import transforms
from pytorch3d.renderer.cameras import look_at_view_transform
from pytorch3d.renderer.utils import convert_to_tensors_and_broadcast
from pytorch3d.structures.meshes import Meshes
from pytorch3d.transforms import rotation_conversions
from pytorch3d.utils import ico_sphere

from ..nnutils import geom_utils
from ..nnutils.render import render_pcl_masks
from ..utils import image as image_utils
from ..utils import mesh as mesh_utils
from ..utils import misc as misc_utils
from .load_blender import load_all_blender_data
from .load_t2 import load_all_t2_data


def generate_360_RTs(elev: float, dist: float, num_frames: int,
        at: torch.Tensor = torch.zeros(3), device: torch.device = torch.device('cpu')):
    """ Google dataset has a different coordinate system than pytorch3d.
        Align model before rendering 360-view
    """
    at=at[None].float().to(device)
    azim = torch.linspace(-180, 180, num_frames, device=device) + 180.0
    dist, elev, azim, at = convert_to_tensors_and_broadcast(dist, elev, azim, at, device=device)
    Rs, Ts = look_at_view_transform(dist=dist, elev=elev, azim=azim, at=at, device=device)

    # Without adding any extra rotation, rendered images look down on object from the top.
    # So we first rotate the object by 90 degrees about X (left) before applying Rs.
    # Transformation:
    #       R@x + T -> R@(X90@(x-at) + at) + T
    #    =>  R -> R@X90
    #    =>  T -> T + R@at - R@X90@at
    X90 = rotation_conversions._axis_angle_rotation('X', torch.full((num_frames,), np.pi/2, device=device))
    Rs, Ts = geom_utils.rotate_at(Rs.transpose(1,2), Ts, at, X90.transpose(1,2), premultiply=False)
    Rs = Rs.transpose(1,2)
    # Ts = Ts + (at.unsqueeze(-2) @ Rs).squeeze(-2)
    # Rs = X90 @ Rs
    # Ts = Ts - (at.unsqueeze(-2) @ Rs).squeeze(-2)
    return Rs, Ts

def load_t2_data(cfg: DictConfig, num_train: Optional[int] = None):
    assert(cfg.type == 'tanks_and_temples')

    rgb, poses, hfovs, pcl_gt = load_all_t2_data(
        images_dir = to_absolute_path(cfg.images_dir),
        colmap_sfm_log = to_absolute_path(cfg.colmap_sfm_log),
        colmap2gt_trans = to_absolute_path(cfg.colmap2gt_trans),
        gt_ply = to_absolute_path(cfg.gt_ply),
        colmap_ply = to_absolute_path(cfg.colmap_ply),
    )

    ids = list(range(rgb.shape[0]))
    if num_train is not None and num_train<len(ids):
        ids = [b[0] for b in misc_utils.chunk_items_into_baskets(ids, num_train)]
    logging.info(f'Using ids {ids}')

    rgb= rgb[ids]
    poses = poses[ids]
    hfovs = hfovs[ids]

    if cfg.mask_generator == 'render':
        # Load alpha maps from disk, or render them (only renders the first time this code is run)
        pcl_gt_cuda = pcl_gt.cuda()
        poses_cuda = poses.cuda()
        hfovs_cuda = hfovs.cuda()
        masks = []
        for i,id in enumerate(ids):
            Path(cfg.masks_dir).mkdir(parents=True, exist_ok=True)  # make directory if it doesn't exist
            mask_file = f'{cfg.masks_dir}/{id:06d}.jpg'
            if os.path.isfile(mask_file):
                mask = imageio.imread(mask_file)
                mask = torch.as_tensor(mask).float()[None]/255
            else:
                logging.info(f'Rendering mask {i} (id {id}) from point could...')
                _,_,H,W = rgb.shape
                mask = render_pcl_masks(pcl_gt_cuda, poses_cuda[None,i], hfovs_cuda[None,i], (H,W),
                                        downsample=cfg.render_masks_downsample_factor)[0]
                mask = mask.cpu()
                imageio.imwrite(mask_file, mask[0].numpy())
            masks.append(mask)
        masks = torch.stack(masks, dim=0)
    elif cfg.mask_generator == 'pointrend':
        from ..preprocess.run_pointrend import get_category_masks
        masks = get_category_masks(
                    rgb,
                    category=cfg.mask_coco_category,
                    detectron2_repo=cfg.detectron2_repo_path,
                    debug=cfg.get('mask_debug', False),
                    area_threshold=cfg.mask_area_threshold
                )
    elif cfg.mask_generator is None:
        # Downstream algo doesn't meed masks. Can skip
        masks = torch.ones_like(rgb[:,:1,:,:])
    else:
        raise ValueError(cfg.mask_generator)

    if (masks.sum(dim=(1,2,3)) == 0).any():
        zeromaskids = (masks.sum(dim=(1,2,3)) == 0).nonzero()
        raise ValueError(f'Masks {zeromaskids} are all zeros')

    imgs = torch.cat([rgb, masks], dim=1)

    if pcl_gt is not None:
        verts = pcl_gt.points_packed()
        mesh_vcen = (verts.min(0).values + verts.max(0).values)/2
        mesh_radius = cfg.get('mesh_radius', (verts-mesh_vcen).norm(dim=-1).max().item())   # TODO: remove cfg.get
    else:
        camera_list = geom_utils.poses_to_cameras_list(poses, hfovs)
        mesh_vcen, mesh_radius = geom_utils.get_centre_radius(imgs[:,3:,:,:], camera_list)

    if pcl_gt is None:
        # Create a fake GT mesh
        mesh_gt = ico_sphere(3).scale_verts_(mesh_radius)
        mesh_gt.offset_verts_(mesh_vcen.expand_as(mesh_gt.verts_packed()))
    else:
        # Create GT mesh from pointcloud
        voxels, xyz_min, xyz_max = mesh_utils.pcl_to_voxel(pcl_gt.points_packed())
        verts, faces = mesh_utils.marching_cubes(
                            voxels, 0.01,
                            xyz_min = xyz_min,
                            xyz_max = xyz_max,
                        )
        mesh_gt = Meshes(verts[None], faces[None])
        mesh_gt = mesh_utils.decimate_mesh(mesh_gt, 20000)

    # Scale poses, mesh_gt, mesh_vcen, mesh_radius
    factor = cfg.target_mesh_radius/mesh_radius
    poses[:,:3,3] *= factor
    mesh_vcen *= factor
    mesh_radius *= factor
    mesh_gt.scale_verts_(factor)
    if pcl_gt is not None: pcl_gt.scale_(factor)

    if pcl_gt is not None:
        near, far = get_nearfar(pcl_gt.points_packed(), poses)
    else:
        near, far = get_nearfar(mesh_gt.verts_packed(), poses)

    data = DotMap({
        'pcl_gt': pcl_gt,
        'mesh_gt': mesh_gt,
        'mesh_centre': mesh_vcen,
        'mesh_radius': mesh_radius,
        'render_poses': None,
        'render_hfov': None,
    })
    data.train = DotMap({
        'rgba': imgs,
        'poses': poses,
        'hfovs': hfovs,
        'near': near,
        'far': far,
    })
    val_start = len(imgs)
    data.val = DotMap({
        'rgba':  imgs[val_start:],
        'poses': poses[val_start:],
        'hfovs': hfovs[val_start:],
        'near':  near[val_start:],
        'far':   far[val_start:],
    })
    return data

def load_google_data(cfg: DictConfig, num_train: int = -1):
    assert(cfg.type == 'google')
    if num_train>0 and num_train > cfg.val_start:
        raise ValueError(f'Passed config results in train/test overlap.'
                    + f' num_train ({num_train} must be <= {cfg.val_start}')

    imgs, poses, hfovs, mesh_gt, render_poses, render_hfov = load_all_blender_data(
        to_absolute_path(cfg.root_dir), to_absolute_path(cfg.metadata_path), max_views=cfg.val_start+cfg.num_val, load_mesh=True
    )

    poses = geom_utils.invert_poses(poses) # Change from blender's c2w format to w2c
    poses[:,[0,2],:] *= -1      # Flip X,Z to switch from blender's (y up, x right) to py3d (y up, x left) coordinate system
    render_poses = geom_utils.invert_poses(render_poses) # Change from blender's c2w format to w2c
    render_poses[:,[0,2],:] *= -1      # Flip X,Z to switch from blender's (y up, x right) to py3d (y up, x left) coordinate system

    # Find mesh radius from gt mesh
    verts = mesh_gt.verts_packed()
    mesh_vcen = (verts.min(0)[0] + verts.max(0)[0])/2
    mesh_radius = cfg.get('mesh_radius', (verts-mesh_vcen).norm(dim=-1).max().item())   # TODO: remove cfg.get

    if cfg.target_mesh_radius is not None:
        # Scale-normalize poses, mesh_gt, mesh_vcen, mesh_radius
        factor = cfg.target_mesh_radius/mesh_radius
        poses[:,:3,3] *= factor
        render_poses[:,:3,3] *= factor
        mesh_vcen *= factor
        mesh_radius *= factor
        mesh_gt.scale_verts_(factor)

    near, far = get_nearfar(mesh_gt.verts_packed(), poses)

    data = DotMap({
        'mesh_gt': mesh_gt,
        'mesh_centre': mesh_vcen,
        'mesh_radius': mesh_radius,
        'render_poses': render_poses,
        'render_hfov': render_hfov,
    })
    data.train = DotMap({
        'rgba': imgs,
        'poses': poses,
        'hfovs': hfovs,
        'near': near,
        'far': far,
    })
    data.val = DotMap({
        'rgba':  imgs[cfg.val_start: cfg.val_start+cfg.num_val],
        'poses': poses[cfg.val_start: cfg.val_start+cfg.num_val],
        'hfovs': hfovs[cfg.val_start: cfg.val_start+cfg.num_val],
        'near':  near[cfg.val_start: cfg.val_start+cfg.num_val],
        'far':   far[cfg.val_start: cfg.val_start+cfg.num_val],
    })
    return data

def load_demo_data(cfg: DictConfig, num_train: int = -1):
    assert(cfg.type == 'demo')

    imgs, poses, hfovs, _, _, _ = load_all_blender_data(
        cfg.images_dir, cfg.metadata_path, max_views=num_train, load_mesh=False
    )

    poses = geom_utils.invert_poses(poses) # Change from blender's c2w format to w2c
    poses[:,[0,2],:] *= -1      # Flip X,Z to switch from blender's (y up, x right) to py3d (y up, x left) coordinate system

    # Find mesh centre/radius from camera poses + masks
    cameras_list = geom_utils.poses_to_cameras_list(poses, hfovs)
    mesh_vcen, mesh_radius = geom_utils.get_centre_radius(imgs[:,3:,:,:], cameras_list)

    # Create a fake GT mesh
    mesh_gt = ico_sphere(3).scale_verts_(mesh_radius)
    mesh_gt.offset_verts_(mesh_vcen.expand_as(mesh_gt.verts_packed()))

    # Scale-normalize poses, mesh_gt, mesh_vcen, mesh_radius
    factor = cfg.target_mesh_radius/mesh_radius
    poses[:,:3,3] *= factor
    mesh_vcen *= factor
    mesh_radius *= factor
    mesh_gt.scale_verts_(factor)

    near, far = get_nearfar(mesh_gt.verts_packed(), poses)

    data = DotMap({
        'mesh_gt': mesh_gt,
        'mesh_centre': mesh_vcen,
        'mesh_radius': mesh_radius,
        'render_poses': None,
        'render_hfov': None,
    })
    data.train = DotMap({
        'rgba': imgs,
        'poses': poses,
        'hfovs': hfovs,
        'near': near,
        'far': far,
    })
    val_start = len(imgs)
    data.val = DotMap({
        'rgba':  imgs[val_start:],
        'poses': poses[val_start:],
        'hfovs': hfovs[val_start:],
        'near':  near[val_start:],
        'far':   far[val_start:],
    })
    return data

def load_data(cfg: DictConfig):
    if cfg.source.type == 'demo':
        data = load_demo_data(cfg.source, num_train=cfg.num_views)
    elif cfg.source.type == 'google':
        data = load_google_data(cfg.source, num_train=cfg.num_views)
    elif cfg.source.type == 'tanks_and_temples':
        data = load_t2_data(cfg.source, num_train=cfg.num_views)
    else:
        raise ValueError

    # make data static to catch bugs quickly
    data._dynamic = False
    data.train._dynamic = False
    data.val._dynamic = False

    ## --- Trim views
    for k in ['rgba', 'poses', 'hfovs', 'near', 'far']:
        data.train[k] = data.train[k][:cfg.num_views]
    logging.info(f'#train views {len(data.train.rgba)}')
    logging.info(f'#val   views {len(data.val.rgba)}')

    ## --- Preprocess RGBA
    # Resize rgba
    def resize_rgba(rgbas, H, W):
        if rgbas.shape[0] == 0:
            newsize = (rgbas.shape[0],rgbas.shape[1],H,W)
            return torch.zeros(newsize, dtype=rgbas.dtype, device=rgbas.device)
        rgba = rgbas.permute(0,2,3,1).numpy()
        assert(image_utils.get_img_format(rgba)=='NHWC')
        # assert(rgba.shape[1]==rgba.shape[2])
        rgba = np.stack([cv2.resize(im, (W, H), interpolation=cv2.INTER_AREA) for im in rgba])
        return torch.as_tensor(rgba).float().permute(0,3,1,2)

    N,_,H,W = data.train.rgba.shape
    nH,nW = cfg.image_size
    if (nH/nW != H/W):
        logging.warning(f'RGBA resize ({[H,W]} -> {[nH,nW]}) does not maintain aspect ratio')
    logging.info(f'Resizing RGBA ({[H,W]} -> {[nH,nW]})')
    data.train.rgba = resize_rgba(data.train.rgba, nH, nW)
    data.val.rgba = resize_rgba(data.val.rgba, nH, nW)
    assert(data.train.rgba.shape == (N,4,nH,nW))
    assert(data.val.rgba.shape[1:] == (4,nH,nW))

    ## --- Add noise to cameras
    for k in ['poses', 'hfovs', 'near', 'far']:
        data.train[f'{k}_gt'] = data.train[k]
    data.train.poses, data.train.hfovs = add_gaussian_camera_noise(
        data.train.poses,
        data.train.hfovs,
        cfg.cam_noise.std_hfov * np.pi/180,
        cfg.cam_noise.std_trans,
        cfg.cam_noise.std_rot * np.pi/180,
        skip_first=cfg.cam_noise.skip_first,
        vcen=data.mesh_centre,
        random_seed=cfg.random_seed
    )

    return data

def add_gaussian_camera_noise(poses, hfovs,
        noise_std_hfov, noise_std_trans, noise_std_rot,
        vcen=torch.tensor([0,0,0], dtype=torch.float32),
        random_seed=0xaaaa_aaaa_aaaa_aaaa, skip_first=True):
    """
    """
    rng = torch.Generator(device=poses.device)
    rng.manual_seed(random_seed)

    R = poses[:,:3,:3]
    T = poses[:,:3,3] + R @ vcen    # T about vcen
    Tx, Ty, Tz = T.unbind(-1)

    # Need to define this because torch.randn_like doesn't accept a generator
    def randn_like(t):
        return torch.randn(
            t.shape, dtype=t.dtype, layout=t.layout,
            device=t.device, generator=rng
        )

    # Change hfov
    delta_hfov = randn_like(hfovs) * noise_std_hfov
    if skip_first:
        delta_hfov[0] = 0
    new_hfovs = (hfovs + delta_hfov).clamp(min=1e-4, max=np.pi/2)

    # Change R
    R_noise_axis = torch.nn.functional.normalize(randn_like(T))
    R_noise_angle = randn_like(hfovs) * noise_std_rot
    if skip_first:
        R_noise_angle[0] = 0
    R_noise = transforms.quaternion_to_matrix(geom_utils.axisangle2quat(R_noise_axis, R_noise_angle))
    new_R = R_noise @ R

    # Change T: scale Tz for hfov change, add delta_T, move world centre to 0
    delta_T = randn_like(T) * noise_std_trans
    if skip_first:
        delta_T[0] = 0
    Tz = Tz * torch.tan(hfovs) / torch.tan(new_hfovs)
    new_T = torch.stack([Tx,Ty,Tz], dim=-1) + delta_T
    new_T = new_T - new_R @ vcen

    logging.info(f'Camera noise: R_angle {R_noise_angle * 180/np.pi}')
    logging.info(f'Camera noise: delta_hfov {delta_hfov}')
    logging.info(f'Camera noise: delta_T {delta_T}')

    poses = poses.clone()
    poses[:,:3,:3] = new_R
    poses[:,:3,3] = new_T
    poses[:,3,:3] = 0
    poses[:,3,3] = 1

    return poses, new_hfovs

def get_nearfar(verts, poses, pad=0.1):
    _verts_proj = torch.einsum('vc,pac->vpa', verts, poses[:,:3,:3]) + poses[:,:3,3]
    _zmin = _verts_proj.min(0)[0][:,2]
    _zmax = _verts_proj.max(0)[0][:,2]
    _zrange = _zmax-_zmin
    _near = (_zmin - pad*_zrange).clamp(min=1e-12)
    _far =  (_zmax + pad*_zrange).clamp(max=100)
    return _near, _far
