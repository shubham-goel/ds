import logging
import os
import pickle
import random
from pathlib import Path

import hydra
import imageio
import numpy as np
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from ..data.load_google import load_data
from ..utils.misc import to_tensor

CONFIG_DIR='../../configs'
@hydra.main(config_path=CONFIG_DIR, config_name="data_to_idr_format")
def main(cfg: DictConfig):
    # Symlink stderr/stdout for ease
    logging.info(f'CWD {os.getcwd()}')
    # symlink_submitit()

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
        # logging.info(f'Saving data to {cfg.init_data_path}')
        # with open(cfg.init_data_path, 'wb') as handle:
        #     pickle.dump(data, handle)

    ## Save images, masks and camera-poses to {cfg.out_dir}/{cfg.data.source.type}/{suffix}/{cfg.data.source.asin}
    # Img + Mask
    suffix = f'r{cfg.data.cam_noise.std_rot}t{cfg.data.cam_noise.std_trans}h{cfg.data.cam_noise.std_hfov}/v{cfg.data.num_views}/'
    out_dir = Path(f'{to_absolute_path(cfg.out_dir)}/{cfg.data.source.type}/{suffix}/{cfg.data.source.asin}/')
    img_dir = out_dir / 'image'
    mask_dir = out_dir / 'mask'
    img_dir.mkdir(exist_ok=True, parents=True)
    mask_dir.mkdir(exist_ok=True)
    for i in range(len(data.train.rgba)):
        imageio.imwrite(img_dir / f'{i:03d}.png', data.train.rgba[i,:3,:,:].permute(1,2,0).numpy())
        imageio.imwrite(mask_dir / f'{i:03d}.png', data.train.rgba[i,3,:,:,None].expand(-1,-1,3).numpy())

    # Camera poses
    gt_cameras_fname = out_dir / 'cameras_old.npz'
    noisy_cameras_fname = out_dir / 'cameras_linear_init_old.npz'
    gt_Ps = {}
    noisy_Ps = {}
    H, W = cfg.data.image_size
    for i in range(len(data.train.rgba)):
        gt_Ps['world_mat_%d'%i] = (hfov_to_K(data.train.hfovs_gt[i], H, W) @ data.train.poses_gt[i][:3,:]).numpy()
        noisy_Ps['world_mat_%d'%i] = (hfov_to_K(data.train.hfovs[i], H, W) @ data.train.poses[i][:3,:]).numpy()
    np.savez(gt_cameras_fname, **gt_Ps)
    np.savez(noisy_cameras_fname, **noisy_Ps)
    logging.info(f'Saved to {out_dir}')

def hfov_to_K(hfov, img_H, img_W):
    f = 1/torch.tan(hfov)
    K = torch.diag(torch.tensor([-f,-f,1]))  # -ve sign changes from py3d -> opencv coordinate system

    K[:2,:2] *= min(img_W, img_H) / 2
    K[0,2] = img_W / 2
    K[1,2] = img_H / 2
    return K

if __name__ == "__main__":
    print('pid', os.getpid())
    main()
