from __future__ import annotations

import logging
import os
import pickle
import random
from pathlib import Path

import hydra
import imageio
import numpy as np
import torch
from dotmap import DotMap
from omegaconf import DictConfig

from ..data.load_google import load_data
from ..utils.colmap import Colmap
from ..utils.misc import symlink_submitit, to_tensor

CONFIG_DIR='../../configs'
@hydra.main(config_path=CONFIG_DIR, config_name="colmap")
def main(cfg: DictConfig):

    # Symlink stderr/stdout for ease
    logging.info(f'CWD {os.getcwd()}')
    symlink_submitit()

    random.seed(cfg.random_seed)
    np.random.seed(cfg.random_seed)
    torch.manual_seed(cfg.random_seed)
    torch.cuda.manual_seed(cfg.random_seed)

    if cfg.resume and os.path.isfile(cfg.init_data_path):
        logging.info(f'Loading data from {cfg.init_data_path}')
        with open(cfg.init_data_path, 'rb') as handle:
            data = pickle.load(handle)
    else:
        data = load_data(cfg.data)
        data = to_tensor(data, _dynamic=False)
        logging.info(f'Saving data to {cfg.init_data_path}')
        with open(cfg.init_data_path, 'wb') as handle:
            pickle.dump(data, handle)

    run_colmap(cfg, data)

def run_colmap(cfg: DictConfig, data: DotMap):

    frames_dir = os.path.abspath(cfg.frames_dir)
    working_dir = os.path.abspath(cfg.working_dir)
    colmap_path = os.path.abspath(cfg.colmap_path)

    # if cfg.resume == False:
    #     # Clear colmap working directory
    #     logging.info(f'Deleteing working directory {working_dir} because resume=False')
    #     if Path(working_dir).is_dir():
    #         shutil.rmtree(working_dir, ignore_errors=True)

    # Save images to disk
    N, _, H, W = data.train.rgba.shape
    Path(frames_dir).mkdir(exist_ok=True)
    img_paths = [Path(f'{frames_dir}/{i}.png') for i in range(N)]
    for i,f in enumerate(img_paths):
        img = data.train.rgba[i].permute(1,2,0).cpu().numpy()
        imageio.imwrite(str(f), img)

    colmap = Colmap(colmap_path, working_dir, img_paths, colmap_envs=cfg.colmap_envs, camera_model=cfg.camera_model, mapper_args=cfg.mapper_args)


    logging.info('Running feature_extractor')
    colmap.feature_extractor()

    if cfg.use_gt_cameras:
        # raise NotImplementedError
        logging.info('Running write_calib')
        poses = data.train.poses_gt.cpu()
        poses[:,[0,1],:] *= -1              # Colmap cameras have x to right, y bottom, z front. Change signs of x/y to match py3d
        # poses = geom_utils.invert_poses(poses)
        Ts = poses[:,:3,:4]

        f = 1/torch.tan(data.train.hfovs_gt) * min(H,W) * 0.5
        Ks = torch.zeros((N, 3, 3))
        Ks[:,0,0] = f
        Ks[:,1,1] = f
        Ks[:,2,2] = 1
        Ks[:,0,2] = W/2
        Ks[:,1,2] = H/2
        imdims = torch.tensor([W,H])[None].expand(N,-1)
        colmap.write_calib(Ks.numpy(), Ts.numpy(), imdims.numpy())

    logging.info('Running feature_matcher')
    colmap.feature_matcher()

    if cfg.use_gt_cameras:
        logging.info('Running point_triangulator')
        colmap.point_triangulator()
    else:
        logging.info('Running mapper')
        colmap.mapper()

    if cfg.resume and os.path.isfile(colmap._fusion_output_path()):
        logging.info(f'Skipping dense_reconstruction. Saved pointcloud exists at {colmap._fusion_output_path()}')
    else:
        logging.info('Running dense_reconstruction')
        colmap.dense_reconstruction()
        logging.info(f'Saved pointcloud to {colmap._fusion_output_path()}')

    if cfg.resume and os.path.isfile(colmap._poisson_output_path()):
        logging.info(f'Skipping poisson_meshing. Saved mesh exists at {colmap._poisson_output_path()}')
    else:
        logging.info('Running poisson_meshing')
        colmap.poisson_meshing()
        logging.info(f'Saved poisson mesh to {colmap._poisson_output_path()}')

    if cfg.resume and os.path.isfile(colmap._delaunay_output_path()):
        logging.info(f'Skipping delaunay_meshing. Saved mesh exists at {colmap._delaunay_output_path()}')
    else:
        logging.info('Running delaunay_meshing')
        colmap.delaunay_meshing()
        logging.info(f'Saved delaunay mesh to {colmap._delaunay_output_path()}')

if __name__ == "__main__":
    print('pid', os.getpid())
    main()
