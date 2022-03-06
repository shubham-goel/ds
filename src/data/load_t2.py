import logging
import os
from typing import Optional, Tuple

import imageio
import numpy as np
import torch
from pytorch3d.io.ply_io import load_ply
from pytorch3d.structures.pointclouds import Pointclouds

from ..nnutils import geom_utils
from ..utils.misc import read_file_as_list


def str_to_floats(s):
    return [float(f) for f in s.split(' ') if len(f)>0]

def read_log_file_as_poses(fpath):
    """ Reads and returns cam2world poses: Nx4x4
        Logfile file format explained here (http://redwood-data.org/indoor/fileformat.html)
    """
    logfile_raw = read_file_as_list(fpath)
    assert(len(logfile_raw)%5==0)
    num_poses = int(len(logfile_raw)/5)

    poses = []
    for i in range(num_poses):
        header = str_to_floats(logfile_raw[5*i])
        assert(header[0] == header[1] == i)
        pose = torch.tensor([
            str_to_floats(logfile_raw[5*i+1]),
            str_to_floats(logfile_raw[5*i+2]),
            str_to_floats(logfile_raw[5*i+3]),
            str_to_floats(logfile_raw[5*i+4]),
        ]).float()
        poses.append(pose)
    poses = torch.stack(poses, dim=0)
    assert(poses.shape == (num_poses,4,4))
    return poses

def load_all_t2_data(
        images_dir,
        colmap_sfm_log,
        colmap2gt_trans,
        gt_ply,
        colmap_ply = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[Pointclouds]]:
    # Read colmap poses
    poses_c2w = read_log_file_as_poses(colmap_sfm_log) # contains cam2world
    poses = geom_utils.invert_poses(poses_c2w)
    assert((torch.det(poses[:,:3,:3])-1).abs().max() < 1e-4)
    logging.info(f'Loaded tanks_and_temples poses: {poses.shape}')

    # Transform camera poses from colmap -> gt
    if colmap2gt_trans is not None:
        try:
            colmap2gt_pose = torch.tensor([str_to_floats(l) for l in read_file_as_list(colmap2gt_trans)])
        except Exception as e:
            logging.error(e)
            logging.error(f'Error loading colmap2gt_pose from {colmap2gt_trans}. Proceeding without it.')
            colmap2gt_pose = torch.eye(4)
    else:
        colmap2gt_pose = torch.eye(4)
    assert(colmap2gt_pose.shape==(4,4))
    colmap2gt_pose_scale = torch.det(colmap2gt_pose[:3,:3]).pow(1/3)
    colmap2gt_pose_R = colmap2gt_pose[:3,:3]/colmap2gt_pose_scale
    colmap2gt_pose_T = colmap2gt_pose[:3,3]/colmap2gt_pose_scale
    colmap2gt_pose_unscaled = geom_utils.RT_to_poses(colmap2gt_pose_R[None], colmap2gt_pose_T[None])[0]
    poses = poses @ geom_utils.invert_poses(colmap2gt_pose_unscaled[None])

    # Colmap cameras have x to right, y bottom, z front. Change signs of x/y to match py3d
    poses[:,[0,1],:] *= -1

    # Are images 0-indexed or 1-indexed?
    zero_indexed = False
    for _xx in range(1,10):
            fname = f'{images_dir}/{0:0{_xx}d}.jpg'
            if os.path.exists(fname):
                zero_indexed = True
                break

    # Read rgb images
    N = poses.shape[0]
    imgs = []
    for i in range(N):
        for _xx in range(1,10):
            if zero_indexed:
                fname = f'{images_dir}/{i:0{_xx}d}.jpg'      # For data from https://github.com/YoYo000/MVSNet
            else:
                fname = f'{images_dir}/{i+1:0{_xx}d}.jpg'    # For data from https://www.tanksandtemples.org/download/
            if os.path.exists(fname):
                break
        imgs.append(imageio.imread(fname))
    imgs =  torch.as_tensor(np.array(imgs)).float().permute(0,3,1,2) / 255
    assert(imgs.shape[0]==N)
    assert(imgs.shape[1]==3)    # only rgb
    logging.info(f'Loaded tanks_and_temples images: {imgs.shape}')

    # Hfovs
    # From the section on camera calibration at https://tanksandtemples.org/download/
    # f = (0.7 * W) / (W/2)
    # But pytorch3d assumes f will scale smaller of (H,W) to [-1,1], not larger.
    # So f = (0.7 * W) / (W/2) * (W/min(H,W))
    _,_,H,W = imgs.shape
    # f = torch.full((N,), 0.7 * 2)
    f = torch.full((N,), 0.5904 * 2)           # Using 0.5904 because it gave accurate mask-reprojections
    # print('f before', f)
    f = f * W / min(H,W)
    # print('f after', f)
    hfovs = torch.atan(1/f)
    print('tanks_and_temples hfovs', hfovs[0] * 180 / np.pi)

    # Load GT point-cloud
    if gt_ply is not None:
        try:
            verts, _ = load_ply(gt_ply)
            verts = verts/colmap2gt_pose_scale  # Scale GT mesh by 1/colmap2gt_pose_scale
            # verts = geom_utils.poses_to_transform(
            #         colmap2gt_pose[None].inverse()
            #         # geom_utils.invert_poses(colmap2gt_pose[None])
            #     ).transform_points(verts)
            gt_pcl = Pointclouds(verts[None])
            logging.info(f'Loaded tanks_and_temples gt pcl: {verts.shape}')
        except Exception as e:
            logging.error(e)
            logging.error(f'Error loading gt_pcl from {gt_ply}. Proceeding without it.')
            gt_pcl = None
    else:
        gt_pcl = None

    # # Test alignment btw gt/colmap ply
    # cverts, _ = load_ply(colmap_ply)
    # pytorch3d.io.save_ply('colmap_ply.ply', cverts)
    # # breakpoint()
    # pytorch3d.io.save_ply('colmap_trans_ply.ply',
    #     geom_utils.poses_to_transform(colmap2gt_pose_unscaled[None]).transform_points(cverts)
    # )
    # pytorch3d.io.save_ply('gt_ply.ply', verts)

    # # Test camera poses by rendering alpha masks
    # from pytorch3d.renderer import PointsRasterizer, PointsRasterizationSettings
    # from pytorch3d.renderer.points.rasterizer import PointFragments
    # from ..nnutils.cameras import dollyParamCameras
    # gt_pcl = gt_pcl.cuda()
    # # Colmap cameras have x to right, y bottom, z front
    # poses[:,[0,1],:] *= -1
    # # camgen = dollyParamCameras(poses.cuda(), hfovs.cuda())
    # # _,_,H,W = imgs.shape
    # # H = H//4; W = W//4
    # # bin_size = int(2 ** max(np.ceil(np.log2(max(H,W))) - 4, 4))
    # # rasterizer = PointsRasterizer(raster_settings=PointsRasterizationSettings(image_size=[H,W], bin_size=bin_size))
    # # # rasterizer = PointsRasterizer()
    # # for i in [20]:
    # #     # breakpoint()
    # #     cam = camgen.create_cameras(id=i)
    # #     frag:PointFragments = rasterizer(gt_pcl, cameras=cam)
    # #     mask = frag.zbuf[:,:,:,0] >= 0
    # #     imageio.imwrite(f'mask{i:04d}_f{cam.focal_length.item()}_sq.jpg', mask[0].float().cpu().numpy())
    # #     imageio.imwrite(f'rgb{i:04d}.jpg', imgs[i].permute(1,2,0).float().cpu().numpy())

    # imgs = imgs.cuda()

    # # Rectangle/Square PCL rendering
    # i=20
    # _,_,H,W = imgs.shape
    # H = H//4; W = W//4
    # print(H,W)      # 270, 480
    # cam_kwargs = dict(R=poses[i,:3,:3].t()[None].cuda(),
    #                 T=poses[i,:3,3][None].cuda(),
    #                 device='cuda')
    # for ff in np.linspace(0.57,0.62,50):
    #     f0 = ff * 2
    #     f1 = ff * 2 * W/min(H,W)   # 2.49
    #     cam0 = PerspectiveCameras(focal_length=f0, **cam_kwargs)
    #     cam1 = PerspectiveCameras(focal_length=f1, **cam_kwargs)
    #     # cam0 = FoVPerspectiveCameras(fov=math.atan(1/f0)*2*180/np.pi, **cam_kwargs)
    #     # cam1 = FoVPerspectiveCameras(fov=math.atan(1/f1)*2*180/np.pi, **cam_kwargs)

    #     bin_size = int(2 ** max(np.ceil(np.log2(max(H,W))) - 4, 4))
    #     rasterizer_sq = PointsRasterizer()
    #     rasterizer_rect = PointsRasterizer(raster_settings=PointsRasterizationSettings(image_size=(H,W), bin_size=bin_size))

    #     # mask0_sq = rasterizer_sq(gt_pcl, cameras=cam0).zbuf[0,:,:,0]>=0
    #     # mask1_sq = rasterizer_sq(gt_pcl, cameras=cam1).zbuf[0,:,:,0]>=0
    #     mask1_rect = rasterizer_rect(gt_pcl, cameras=cam1).zbuf[0,:,:,0]>=0
    #     # imageio.imwrite(f'mask0_sq.jpg', mask0_sq.float().cpu().numpy())
    #     # imageio.imwrite(f'mask1_sq.jpg', mask1_sq.float().cpu().numpy())
    #     imageio.imwrite(f'mask{i}_f{ff:.04f}_rect.jpg', mask1_rect.float().cpu().numpy())

    #     mask1_rect = F.interpolate(mask1_rect[None,None].float(), scale_factor=4, mode='bilinear')[0]
    #     rgba = torch.cat((imgs[i],mask1_rect), dim=0)
    #     imageio.imwrite(f'rgba{i}_f{ff:.04f}.png', rgba.permute(1,2,0).float().cpu().numpy())

    return imgs, poses, hfovs, gt_pcl
