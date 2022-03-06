from __future__ import annotations

import os
from pathlib import Path
from pprint import pprint

import numpy as np
import pytorch3d
import pytorch3d.io
import torch
from aggregate_results import BAD_SHAPE_METRIC

from ..utils import metrics as metric_utils
from .evaluate_base import eval_base


class eval_colmap(eval_base):
    def load_checkpoint(self, iter):
        return None

    def __init__(self, expdir, device='cuda') -> None:
        super().__init__(expdir, device=device)

        loaded_data = self.loaded_data
        cfg = self.cfg
        data = self.data

        # Load predicted pointclouds/meshes from colmap prediction dir
        poisson_ply_path = Path(self.newpath(cfg.working_dir)) / 'dense' / 'poisson_photometric.ply'
        fused_ply_path = Path(self.newpath(cfg.working_dir)) / 'dense' / 'fused_photometric.ply'

        io = pytorch3d.io.IO()
        if poisson_ply_path.is_file():
            self.poisson_ply = io.load_pointcloud(poisson_ply_path, device=device)
            print('poisson pcl size', self.poisson_ply.num_points_per_cloud())
        else:
            self.poisson_ply = None
        if fused_ply_path.is_file():
            self.fused_ply = io.load_pointcloud(fused_ply_path, device=device)
            print('fused pcl size', self.fused_ply.num_points_per_cloud())
        else:
            self.fused_ply = None

        # breakpoint()
        if False: #debug
            # Save ply to file
            io.save_pointcloud(self.fused_ply, 'debug/fused_photometric.ply')
            io.save_pointcloud(self.poisson_ply, 'debug/poisson_photometric.ply')
            io.save_mesh(data.mesh_gt, 'debug/mesh_gt.obj')

    def get_alignment(self, use_icp=True, align_scale_view0=False, align_depth_kwargs=...):
        # Assume that there is no noise in cameras
        if (self.cfg.data.std_hfov==0 or
            self.cfg.data.std_trans==0 or
            self.cfg.data.std_rot==0):
            raise NotImplementedError('COLMAP eval assumes cameras are noise-free')

        # NOTE: No additional alignment is required. Predictions are already aligned with GT mesh!

        self.aligned = True
        return self.align_w2c

    def compute_shape_metrics(self, align=True, **align_kwargs):
        # Return if colmap failed to reconstruct any points
        if self.fused_ply is None:
            return {'fused_ply_size':0, **BAD_SHAPE_METRIC}

        # Computing shape metrics between point-clouds. We can only compute Chamfer,F1,Precision,Recall
        pcl_pred = self.fused_ply
        mesh_gt = self.data.mesh_gt

        # Skipping alignment because meshes are already aligned

        shape_metrics = metric_utils.compare_pointclouds(pcl_pred, mesh_gt, align_icp=False)
        shape_metrics['fused_ply_size'] = pcl_pred.num_points_per_cloud().item()
        return shape_metrics

def aggregate_all(views=8, recompute=False):
    base_dir = Path(f'{os.environ["SHARED_HOME"]}/dump_facebook/google/__colmap_highres_map2view__camera_model=PINHOLE,use_gt_cameras=True/r0t0h0/')
    base_dir = base_dir / f'v{views}'

    all_shape_metrics_path = Path(__file__).parent.parent.parent / f'colmap_shape_metrics_v{views}.pt'
    if all_shape_metrics_path.is_file() and not recompute:
        all_shape_metrics = torch.load(all_shape_metrics_path)
    else:
        all_shape_metrics = {}
        for i, exp_dir in enumerate(base_dir.iterdir()):
            print(i, exp_dir.name)
            evaluator = eval_colmap(exp_dir)
            all_shape_metrics[exp_dir.name] = evaluator.compute_shape_metrics()
        torch.save(all_shape_metrics, all_shape_metrics_path)

    # Collate results
    keys = list(all_shape_metrics.values())[0].keys()
    asins = sorted(list(all_shape_metrics.keys()))
    shape_metrics_collated = {k: [all_shape_metrics[asin][k] for asin in asins] for k in keys}
    for k in keys:
        data = np.array(shape_metrics_collated[k])
        # Find 25th, 50th, 75th percentile
        print(f'{k:20s} \t25% {np.percentile(data, 25):.4f} \t50% {np.percentile(data, 50):.4f} \t75% {np.percentile(data, 75):.4f}')

def main():
    exp_dir = '/shared/shubham/dump_facebook/google/__colmap_highres_map2view__camera_model=PINHOLE,use_gt_cameras=True/r0t0h0/v8/US_Army_Stash_Lunch_Bag/'
    exp_dir = '/shared/shubham/dump_facebook/google/__colmap_highres_map2view__camera_model=PINHOLE,use_gt_cameras=True/r0t0h0/v8/Diamond_Visions_Scissors_Red/'
    evaluator = eval_colmap(exp_dir)
    shape_metrics = evaluator.compute_shape_metrics()
    pprint('shape_metrics')
    pprint(shape_metrics)

if __name__ == '__main__':
    main()
    aggregate_all()
