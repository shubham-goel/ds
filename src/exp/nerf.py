"""
Adapted from the pytorch3D NeRF implementation
"""
import logging
import os
import pickle
import random

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from visdom import Visdom

from ..data.load_google import load_data
from ..nerfPy3D.train_wrapper import fit_nerf
from ..nnutils.cameras import dollyParamCameras
from ..utils.misc import symlink_submitit, to_tensor
from ..utils.tb_visualizer import TBVisualizer
from ..utils.visutil import visualize_initial_scene

CONFIG_DIR='../../configs'
@hydra.main(config_path=CONFIG_DIR, config_name="nerf")
def main(cfg: DictConfig):

    # Symlink stderr/stdout for ease
    logging.info(f'CWD {os.getcwd()}')
    symlink_submitit()

    random.seed(cfg.random_seed)
    np.random.seed(cfg.random_seed)
    torch.manual_seed(cfg.random_seed)
    torch.cuda.manual_seed(cfg.random_seed)

    # Init the visualisation visdom env.
    if cfg.viz.server is not None:
        viz = Visdom(
            server=cfg.viz.server,
            port=cfg.viz.port,
            env=cfg.viz.env
        )
    else:
        logging.info(f'Skipping visdom connection because server=None')
        viz = None
    tb_visualizer = TBVisualizer('summary/')


    if cfg.train.resume and os.path.isfile(cfg.init_data_path):
        logging.info(f'Loading data from {cfg.init_data_path}')
        with open(cfg.init_data_path, 'rb') as handle:
            data = pickle.load(handle)
    else:
        data = load_data(cfg.data)
        data = to_tensor(data)

        logging.info(f'Saving data to {cfg.init_data_path}')
        with open(cfg.init_data_path, 'wb') as handle:
            pickle.dump(data, handle)

    # --- Visualize gt mesh and camera poses
    visualize_initial_scene(
        viz,
        rgbas=data.train.rgba,
        gt_mesh=data.mesh_gt,
        in_cameras=dollyParamCameras(
                        data.train.poses,
                        data.train.hfovs,
                        optimize_cam=False
                    ).create_cameras_list(),
        camera_scale=0.2 * data.mesh_radius,
    )

    # --- Optimize Nerf
    device = torch.device('cuda:0')
    neural_radiance_field = fit_nerf(
        cfg.train,
        data,
        cfg.viz,
        lr=cfg.train.optim.lr,
        n_iter=cfg.train.optim.num_iter,
        optim_eps=cfg.train.optim.eps,
        lr_scheduler_step_size=cfg.train.optim.lr_scheduler_step_size,
        lr_scheduler_gamma=cfg.train.optim.lr_scheduler_gamma,
        visualizer=viz,
        tb_visualizer=tb_visualizer,
        device=device
    )

if __name__ == "__main__":
    main()

