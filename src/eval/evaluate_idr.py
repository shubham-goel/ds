
"""
Run as
## Extract metrics
export GOOGLE_ASINS_INTERESTING_TOPO=`tr '\n' ',' < all_benchmark_google_asins.txt`
python -m src.eval.evaluate_idr asin=$GOOGLE_ASINS_INTERESTING_TOPO recompute=True \
    hydra/launcher=submitit_slurm hydra.launcher.qos=low hydra.launcher.gpus_per_node=1 -m

## Aggregate
python -m src.eval.evaluate_idr aggregate=True
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from pprint import pprint
from typing import Optional

import hydra
import numpy as np
import pandas as pd
import torch
from hydra.core.config_store import ConfigStore
from hydra.utils import to_absolute_path
from pytorch3d.io import load_objs_as_meshes

from ..nnutils.cameras import dollyParamCameras
from ..utils import metrics as metric_utils
from ..utils.mesh import (invert_RTs, save_mesh, transform_cameras,
                          transform_mesh)
from ..utils.misc import (Metrics, apply_dict_rec, read_file_as_list,
                          try_move_device)
from .evaluate_base import find_alignment
from .plot_results import combine_results_from_alignments, flatten_results

# Evaluation path uses environment variable SHARED_HOME
IDR_evals_dir = Path(os.environ['SHARED_HOME']) / 'code' / 'idr' / 'evals'
GSO_home_dir = Path(os.environ['SHARED_HOME']) / '.ignition/fuel/fuel.ignitionrobotics.org/GoogleResearch/directdl/'


def K_to_hfov(K_all, img_size=None):
    if img_size is not None:
        K_all[:,0,2] += -img_size[0]/2
        K_all[:,1,2] += -img_size[1]/2
        K_all[:, :2] /= min(img_size)/2
        hfov_idx = 0 if img_size[0] < img_size[1] else 1
    else:
        hfov_idx = 0
    assert(torch.isclose(K_all[:,0,0], K_all[:,1,1]).all())
    assert(torch.isclose(K_all[:,:2,2], torch.zeros(K_all.shape[0],2), atol=1e-5).all())
    hfovs = torch.atan(1/K_all[:,hfov_idx,hfov_idx])
    return hfovs

def wierd_pose_fix(poses):
    # Convert to sensible Rt. Opencv is wierd. (https://stackoverflow.com/questions/62686618/opencv-decompose-projection-matrix)
    new_poses = []
    for pose in poses:
        R = pose[:3, :3].t()
        t = R @ -pose[:3, 3]
        pose = torch.cat([R, t[:,None]], dim=1)
        new_poses.append(pose)
    return torch.stack(new_poses, dim=0)

def load_idr_predictions(asin, eval_dir):
    finished_iter = int(next(eval_dir.glob('cameras_gt_*.npz')).stem.split('_')[-1])
    gt_cameras = np.load(eval_dir/f'cameras_gt_{finished_iter}.npz')
    pred_cameras = np.load(eval_dir/f'cameras_pred_{finished_iter}.npz')
    pred_mesh = load_objs_as_meshes([str(eval_dir / f'surface_untransformed_unclean_{finished_iter}.obj')])
    gt_mesh = load_objs_as_meshes([str(GSO_home_dir / asin / '1/meshes/model.obj')])

    convert_fn = lambda x: torch.from_numpy(x).float()
    gt_R, gt_t, gt_K = convert_fn(gt_cameras['R']),  convert_fn(gt_cameras['t']),  convert_fn(gt_cameras['K'])
    pred_R, pred_t, pred_K = convert_fn(pred_cameras['R']),  convert_fn(pred_cameras['t']),  convert_fn(pred_cameras['K'])
    gt_poses = wierd_pose_fix(torch.cat([gt_R, gt_t[:,:,None]], dim=2))
    pred_poses = wierd_pose_fix(torch.cat([pred_R, pred_t[:,:,None]], dim=2))

    # Check K validity, get hfov
    # Get image size assuming principal point in K is image center
    img_size = gt_K[0, 0, 2] * 2, gt_K[0, 1, 2] * 2
    if img_size[0] < 20:
        # Assume K is in NDC space
        img_size = None
    hfovs = K_to_hfov(gt_K, img_size=img_size)
    assert(torch.isclose(hfovs, K_to_hfov(pred_K, img_size=img_size)).all())

    # Convert opencv camera to pytorch camera
    gt_poses[:,:2,:] *= -1
    pred_poses[:,:2,:] *= -1

    return finished_iter, gt_mesh, pred_mesh, gt_poses, pred_poses, hfovs

def gather_results_IDR(asin, setting, recompute=False, iter='latest', save_if_iter_exceeds=1000, use_icp=True, icp_type='g2p_uncentered'):
    eval_dir = IDR_evals_dir / f'gso_trained_cameras_{setting}_{asin}'

    icp_str = 'noicp' if not use_icp else f'icp{icp_type}'
    save_metric_basename = f'___aggregation_shapecam_metrics_{icp_str}'
    save_metric_path = eval_dir / f'{save_metric_basename}_i{iter}.pth'
    if not recompute and save_metric_path.is_file():
        logging.info(f'Loading metrics from {save_metric_path}')
        metrics = torch.load(str(save_metric_path))
        return metrics

    finished_iters, mesh_gt, mesh_pred, gt_poses, pred_poses, hfovs = load_idr_predictions(asin, eval_dir)

    # To cuda
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mesh_gt = mesh_gt.to(device)
    mesh_pred = mesh_pred.to(device)
    gt_poses = gt_poses.to(device)
    pred_poses = pred_poses.to(device)
    hfovs = hfovs.to(device)

    # Find mesh radius from gt mesh
    verts = mesh_gt.verts_packed()
    mesh_vcen = (verts.min(0)[0] + verts.max(0)[0])/2
    mesh_radius = (verts-mesh_vcen).norm(dim=-1).max().item()

    # Normalize gt_poses and mesh_gt for target_mesh_radius=0.1. Forgot while creating IDR data.
    target_mesh_radius=0.1
    factor = target_mesh_radius/mesh_radius
    gt_poses[:,:3,3] *= factor
    mesh_vcen *= factor
    mesh_radius *= factor
    mesh_gt.scale_verts_(factor)
    data_mesh_centre = mesh_vcen

    gt_cam_gen = dollyParamCameras(gt_poses, hfovs, optimize_cam=False)
    pred_cam_gen = dollyParamCameras(pred_poses, hfovs, optimize_cam=False)

    # align_w2c = find_alignment2(mesh_gt, mesh_pred, gt_cam_gen, pred_cam_gen, use_icp=use_icp, icp_type=icp_type, align_scale_view0=True)
    align_w2c = find_alignment(mesh_pred, mesh_gt, pred_poses[0,:3,:4], gt_poses[0,:3,:4], use_icp=use_icp, icp_type=icp_type, align_scale_view0=True)

    # Compute shape metrics
    align_c2w = invert_RTs(align_w2c)
    mesh_pred_transformed = transform_mesh(mesh_pred, align_c2w)
    # shape_metrics = metric_utils.compare_meshes(mesh_pred_transformed, mesh_gt, align_icp=False)
    uni_chamfer = True # Compute uni-directional chamfer distances also
    shape_metrics = metric_utils.compare_meshes(mesh_pred_transformed, mesh_gt, align_icp=False,
                        num_samples=100_000,
                        return_per_point_metrics=uni_chamfer)

    if uni_chamfer:
        shape_metrics, (_,_,metrics_p2g), (_,_,metrics_g2p) = shape_metrics
        shape_metrics.update({
            "Chamfer-L2-p2g": metrics_p2g["Chamfer-L2"].mean().item(),
            "Chamfer-L2-g2p": metrics_g2p["Chamfer-L2"].mean().item(),
        })

    # Compute camera metrics
    cam_pred = pred_cam_gen.create_cameras()
    cam_gt = gt_cam_gen.create_cameras()
    cam_pred = transform_cameras(cam_pred, align_c2w)
    camera_metrics = metric_utils.compare_cameras(cam_pred, cam_gt, centre=data_mesh_centre, per_camera=True)


    if False: # Debug: visualize alignment
        # save_mesh(to_absolute_path(f'debug/{setting}_{asin}_pred.obj'), mesh_pred)
        Path(to_absolute_path(f'debug/idr_align3')).mkdir(exist_ok=True)
        save_mesh(to_absolute_path(f'debug/idr_align3/{setting}_{asin}_gt.obj'), mesh_gt)
        save_mesh(to_absolute_path(f'debug/idr_align3/{setting}_{asin}_{icp_str}_pred_tr.obj'), mesh_pred_transformed)

    pprint('shape_metrics')
    pprint(shape_metrics)

    metrics = Metrics(
            shape_metrics=apply_dict_rec(shape_metrics, fv = lambda x: try_move_device(x, 'cpu')),
            camera_metrics=apply_dict_rec(camera_metrics, fv = lambda x: try_move_device(x, 'cpu')),
            finished_iters=finished_iters,
            image_metrics=None,
        )

    # Save metrics
    if finished_iters >= save_if_iter_exceeds:
        logging.info(f'Saving metrics to {save_metric_path}')
        torch.save(metrics, str(save_metric_path))
        if iter=='latest':
            torch.save(metrics, str(eval_dir/f'{save_metric_basename}_i{finished_iters-1}.pth'))
    return metrics

@dataclass
class MyConfig:
    asin: Optional[str] = None
    r_noise: int = 30
    num_views: int = 8
    recompute: bool = False
    use_icp: bool = True
    icp_type: str = 'g2p_noscale_centered'

    aggregate: bool = False

cs = ConfigStore.instance()
# Registering the Config class with the name 'config'.
cs.store(name="config", node=MyConfig)

@hydra.main(config_name="config")
def main(cfg: MyConfig):
    setting = f'r{cfg.r_noise}v{cfg.num_views}'

    if cfg.asin is not None:
        # asin = 'Sonny_School_Bus'
        # setting = 'r5v8'
        metrics = gather_results_IDR(cfg.asin, setting, use_icp=cfg.use_icp, icp_type=cfg.icp_type, recompute=cfg.recompute)
        print(metrics)
        return

    if cfg.aggregate:
        asin_list = read_file_as_list(to_absolute_path('all_benchmark_google_asins.txt'))
        def get_metrics_pd(use_icp, icp_type):
            metrics_all = {
                asin: gather_results_IDR(asin, setting, use_icp=use_icp, icp_type=icp_type, recompute=cfg.recompute)
                for asin in asin_list
            }
            big_results_dict = {f'r{cfg.r_noise}t0h0':{f'v{cfg.num_views}': metrics_all}}
            idr_pd = pd.DataFrame.from_dict(flatten_results(big_results_dict), orient='index')
            idr_pd.index.set_names(['noise', 'views', 'asin'], inplace=True)
            return idr_pd
        idr_noicp = get_metrics_pd(False, 'none')
        idr_g2p = get_metrics_pd(True, 'g2p_noscale_centered')
        idr_p2g = get_metrics_pd(True, 'p2g_noscale_centered')
        idr_best = combine_results_from_alignments([idr_g2p, idr_p2g, idr_noicp], idr_noicp)
        # idr_q50 = idr_noicp.groupby(['noise', 'views']).quantile().sort_index(axis=1).rename(columns={'R/avg': 'R'})
        idr_q50 = idr_best.groupby(['noise', 'views']).quantile().sort_index(axis=1).rename(columns={'R/avg': 'R'})
        print(idr_q50.T.to_string())

if __name__ == '__main__':
    main()
