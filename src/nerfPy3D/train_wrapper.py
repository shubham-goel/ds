import collections
import itertools
import logging
import math
import os
import pickle
import time
from typing import List, Optional

import imageio
import numpy as np
import pytorch3d.io
import torch
import torch.nn.functional as F
from dotmap import DotMap
from omegaconf import DictConfig
from pytorch3d.renderer import PerspectiveCameras, ray_bundle_to_ray_points
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.vis.plotly_vis import plot_scene
from visdom import Visdom

from ..data.load_blender import pose_spherical
from ..data.load_google import generate_360_RTs
from ..nnutils import geom_utils
from ..nnutils.cameras import dollyParamCameras
from ..utils import metrics as metric_utils
from ..utils.mesh import decimate_mesh, marching_cubes
from ..utils.misc import add_suffix_to_path, apply_dict_rec, flatten_dict, try_move_device
from ..utils.tb_visualizer import TBVisualizer
from .nerf.implicit_function import NeuralRadianceField
from .nerf.nerf_renderer import RadianceFieldRenderer
from .nerf.stats import Stats


@torch.no_grad()
def show_rotating_nerf(neural_radiance_field_model,
        hfov, dist, elevations, centre,
        visualizer: Optional[Visdom], iteration,
        min_depth=0, max_depth=1,
        num_frames = 50, device=torch.device('cuda:0'),
        write_to_file = True,
    ):

    # Nerf to eval mode
    nerf_was_training = neural_radiance_field_model.training
    neural_radiance_field_model.eval()

    viz_videos = {}

    for elevation in elevations:
        rgba_cont_dict = {'coarse':[], 'fine':[]}
        Rs, Ts = generate_360_RTs(elev=elevation, dist=dist, num_frames=num_frames, device=device, at=centre)

        # Render cameras in chunks
        for idx in range(num_frames):
            cameras = PerspectiveCameras(
                device=device,
                R=Rs[idx:idx+1],
                T=Ts[idx:idx+1],
                focal_length=1/np.tan(hfov),
            )
            # Render rgba
            # rend_output = self.render_novel_cameras(cameras, use_cached=True)
            dummy_rgba = torch.zeros(neural_radiance_field_model._image_size + (4,), device=device)
            val_nerf_out, val_metrics = neural_radiance_field_model(
                    None,
                    cameras,
                    dummy_rgba[None],        # add dim for batchsize
                    min_depth = min_depth,
                    max_depth = max_depth,
                    mode = "val",
                )
            for cf in ['coarse', 'fine']:
                rgba = val_nerf_out[f'rgba_{cf}'][0].permute(2, 0, 1).detach().cpu().clamp(0.0, 1.0)
                rgba_cont_dict[cf].append(rgba)

        # Merge rendered chunks
        for cf in ['coarse', 'fine']:
            rgba_cont = torch.stack(rgba_cont_dict[cf], dim=0)
            rgb_cont = rgba_cont[:, :3, :, :]
            alpha_cont = rgba_cont[:, 3:, :, :].expand(-1,3,-1,-1)
            viz_videos[f'cont_{elevation}_{cf}/tex'] = rgb_cont.cpu().numpy()
            viz_videos[f'cont_{elevation}_{cf}/alpha'] = alpha_cont.cpu().numpy()

            name = f'cont_{elevation}_{cf}'
            # Save to video file, plot to visdom
            if write_to_file:
                rgba_cont_np = rgba_cont.permute(0,2,3,1).numpy()
                imageio.mimwrite(f'{name}_{iteration:08d}.mp4',
                                (255*np.clip(rgba_cont_np, 0, 1)).astype(np.uint8),
                                fps=int(num_frames/12), quality=8)
            if visualizer is not None:
                if write_to_file:
                    visualizer.video(videofile=f'{name}_{iteration:08d}.mp4', win=name,
                                opts=dict(fps=int(num_frames/12), title=name)
                    )
                else:
                    visualizer.video(rgb_cont, win=name,
                                opts=dict(fps=int(num_frames/12), title=name)
                    )

    # Back to train mode
    if nerf_was_training:
        neural_radiance_field_model.train()

    return viz_videos

def fit_nerf(
    nerf_cfg: DictConfig,
    data: DotMap,
    viz_cfg: DictConfig,
    lr: float = 0.0005,
    n_iter: int = 3e6,
    optim_eps: float = 1e-7,
    lr_scheduler_step_size: int = 5000,
    lr_scheduler_gamma: float = 0.1,
    visualizer: Optional[Visdom] = None,
    tb_visualizer: Optional[TBVisualizer] = None,
    device: torch.device = torch.device('cuda')
) -> RadianceFieldRenderer:

    # Move data to the correct device
    data = apply_dict_rec(data, fv = lambda x: try_move_device(x, device))

    # Original code source requires channel dimension to be at the end
    train_rgba = data.train.rgba.permute(0,2,3,1)
    val_rgba = data.val.rgba.permute(0,2,3,1)

    if nerf_cfg.get('set_img_bg_black', False):
        train_rgba[:,:,:,:3] = train_rgba[:,:,:,:3] * train_rgba[:,:,:,3:]
        val_rgba[:,:,:,:3] = val_rgba[:,:,:,:3] * val_rgba[:,:,:,3:]

    logging.info(f'Running NeRF in dir {os.getcwd()}')
    logging.debug(f'NeRF near {list(data.train.near)}')
    logging.debug(f'NeRF far  {list(data.train.far)}')

    # Initialize camera generators
    train_cam_gen = dollyParamCameras(data.train.poses, data.train.hfovs,
                            centre=data.mesh_centre,
                            param_R_axis_angle=nerf_cfg.param_R_axis_angle,
                            optimize_cam=nerf_cfg.optimize_cam,
                            optimize_first=nerf_cfg.optimize_first,
                            optimize_R=nerf_cfg.optimize_R,
                            optimize_T=nerf_cfg.optimize_T,
                            optimize_Fov=nerf_cfg.optimize_Fov,
                            device=device
                        )
    val_cam_gen = dollyParamCameras(data.val.poses, data.val.hfovs, optimize_cam=False, device=device)
    gt_cam_gen = dollyParamCameras(data.train.poses_gt, data.train.hfovs_gt, optimize_cam=False, device=device)

    # Move stuff to correct device
    train_cam_gen: dollyParamCameras = train_cam_gen.to(device)
    val_cam_gen: dollyParamCameras = val_cam_gen.to(device)
    gt_cam_gen: dollyParamCameras = gt_cam_gen.to(device)
    train_rgba = train_rgba.to(device)
    val_rgba = val_rgba.to(device)
    data.train.near = data.train.near.to(device)
    data.train.far  = data.train.far.to(device)
    data.val.near   = data.val.near.to(device)
    data.val.far    = data.val.far.to(device)

    xyz_min = data.mesh_centre - 1.1 * data.mesh_radius
    xyz_max = data.mesh_centre + 1.1 * data.mesh_radius

    # Instantiate the radiance field model.
    neural_radiance_field_model = RadianceFieldRenderer(
        # Components
        cfg_mask_loss = nerf_cfg.mask_loss,
        cfg_implicit = nerf_cfg.implicit_function,
        cfg_raysampler = nerf_cfg.raysampler,
        image_size = train_rgba.shape[1:3],
        harmonic_xyz_omega0 = 2. / data.mesh_radius,     # match omega0 to legos (which have radius 2)
        xyz_min = xyz_min,
        xyz_max = xyz_max,

        # Options for NerfRenderer
        use_single_network = nerf_cfg.use_single_network,
        chunk_size_test=nerf_cfg.chunk_size_test,
    )
    neural_radiance_field_model.to(device)

    # Instantiate the Adam optimizer. We set its master learning rate.
    optimizer = torch.optim.Adam(
        itertools.chain(
            neural_radiance_field_model.parameters(),
            train_cam_gen.parameters()
        ),
        lr=lr,
        eps=optim_eps
    )

    # Learning rate scheduler setup.
    # Following the original code, we use exponential decay of the
    # learning rate: current_lr = base_lr * gamma ** (epoch / step_size)
    def lr_lambda(epoch):
        return lr_scheduler_gamma ** (
            epoch / lr_scheduler_step_size
        )
    # The learning rate scheduling is implemented with LambdaLR PyTorch scheduler.
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda, verbose=False
    )

    # Initialize the cache for storing variables needed for visulisation.
    output_cache = collections.deque(maxlen=viz_cfg.history_size)

    # Init the stats object.
    stats = Stats(
        ['loss', 'mse_coarse', 'mse_fine', 'mask_coarse', 'mask_fine', 'psnr_coarse', 'psnr_fine', 'sec/it'],
    )
    shape_stat_keys = [
        'Chamfer-L2', 'NormalConsistency', 'AbsNormalConsistency', 'Precision', 'Recall', 'F1',
    ]
    shape_stat_keys += [f'Aligned{k}' for k in shape_stat_keys]
    shape_stats = Stats(shape_stat_keys)
    camera_stats = Stats(metric_utils.camera_metric_keys(per_camera=True, N=len(data.train.poses)))

    # Load from checkpoint
    start_iter = 0
    if len(nerf_cfg.checkpoint_path) > 0:
        # Make the root of the experiment directory.
        checkpoint_dir = os.path.split(nerf_cfg.checkpoint_path)[0]
        if checkpoint_dir: os.makedirs(checkpoint_dir, exist_ok=True)

        # Resume training if requested.
        if nerf_cfg.resume:
            if not os.path.isfile(nerf_cfg.checkpoint_path):
                logging.warn(f'Requested resume but checkpoint not found: {nerf_cfg.checkpoint_path}')
            else:
                logging.info(f"Resuming from checkpoint {nerf_cfg.checkpoint_path}.")
                loaded_data = torch.load(nerf_cfg.checkpoint_path)
                neural_radiance_field_model.load_state_dict(loaded_data["model"])
                train_cam_gen.load_state_dict(loaded_data["camera_params"])
                stats = pickle.loads(loaded_data["stats"])
                shape_stats = pickle.loads(loaded_data["shape_stats"])
                camera_stats = pickle.loads(loaded_data["camera_stats"])
                optimizer.load_state_dict(loaded_data["optimizer"])
                lr_scheduler.load_state_dict(loaded_data["lr_scheduler"])
                start_iter = loaded_data["finished_iter"] + 1
                torch.random.set_rng_state(loaded_data['torch_rng_state'])
                logging.info(f"   => resuming from epoch {stats.epoch} iter {start_iter}.")

    if nerf_cfg.precache_rays:
        if nerf_cfg.optimize_cam:
            raise ValueError('Cannot set both precache_rays and optimize_cam')
        # Precache the projection rays.
        neural_radiance_field_model.eval()
        with torch.no_grad():
            for (cam_prefix, cam_gen, min_depths, max_depths) in zip(
                    ['t', 'v'],
                    [train_cam_gen, val_cam_gen],
                    [data.train.near, data.val.near],
                    [data.train.far, data.val.far],
                ):
                cache_cameras = cam_gen.create_cameras_list()
                cache_camera_hashes = [f'{cam_prefix}{i}' for i in range(len(cache_cameras))]
                neural_radiance_field_model.precache_rays(
                    cache_cameras,
                    cache_camera_hashes,
                    min_depths = min_depths,
                    max_depths = max_depths,
                )

    # Set the right training mode.
    neural_radiance_field_model.train()

    # The main optimization loop.
    time_start = time.time()
    for iteration in range(start_iter, n_iter):

        if (iteration % len(train_rgba))==0:
            time_start = time.time()  # Get the epoch start timestamp.
            stats.new_epoch()  # New epoch.
            shape_stats.new_epoch()  # New epoch.
            camera_stats.new_epoch()  # New epoch.

        # Sample random batch indices.
        train_idx = torch.randint(low=0, high=len(train_rgba), size=[1]).to(device)

        image = train_rgba[train_idx]
        mask = train_rgba[train_idx, :, :, 3]
        camera = train_cam_gen.create_cameras(id=train_idx)
        camera_near = data.train.near[train_idx]
        camera_far = data.train.far[train_idx]

        # Zero the optimizer gradient.
        optimizer.zero_grad()

        # Run network
        nerf_out, metrics = neural_radiance_field_model(
            None,
            camera,
            image,
            min_depth = camera_near,
            max_depth = camera_far,
            masks = mask
        )
        # The loss is a sum of coarse and fine MSEs
        loss = metrics['mse_coarse']  * nerf_cfg.rgb_loss_wt
        loss += metrics['mask_coarse'] * nerf_cfg.mask_loss_wt
        if nerf_cfg.raysampler.n_pts_per_ray_fine>0:
            loss += metrics['mse_fine']  * nerf_cfg.rgb_loss_wt
            loss += metrics['mask_fine'] * nerf_cfg.mask_loss_wt

        # Take the training step.
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        # Update stats with the current metrics.
        stats.update(
            {'loss': float(loss), **metrics},
            stat_set='train',
            time_start=time_start,
        )

        # Every iprint iterations, print the current values of the losses.
        if iteration % viz_cfg.iprint == 0:
            stats.print(stat_set='train')

            # Evaluate cameras, plot camera_metrics
            with torch.no_grad():
                pred_cameras = train_cam_gen.create_cameras(device=device)
                gt_cameras = gt_cam_gen.create_cameras(device=device)
                camera_metrics = metric_utils.compare_cameras(pred_cameras, gt_cameras,
                                    centre=data.mesh_centre,
                                    per_camera=True, wrt_cam0=nerf_cfg.cammetric_wrt_cam0
                                )
                camera_stats.update(
                    flatten_dict(camera_metrics),
                    stat_set='train',
                    time_start=time_start,
                )
            camera_stats.plot_stats(viz=visualizer)
            tb_visualizer.plot_current_scalars(
                dict(camera_metrics=camera_metrics),
                iteration
            )
            if not nerf_cfg.benchmark_fast:
                tb_visualizer.flush()

        output_cache.append(
            {
                'camera': camera.cpu(),
                'camera_idx': None,
                'rgba': image[0].cpu().detach(),
                'rgba_fine': nerf_out['rgba_fine'].cpu().detach(),
                'rgba_coarse': nerf_out['rgba_coarse'].cpu().detach(),
                'rgba_gt': nerf_out['rgba_gt'].cpu().detach(),
                'coarse_ray_bundle': nerf_out['coarse_ray_bundle'],
            }
        )

        # Visualize the full renders every idisplay iterations.
        if iteration>0 and (
            iteration % viz_cfg.idisplay == 0
            or iteration == n_iter-1
        ):
            logging.info(f'Displaying results (iter {iteration})...')
            neural_radiance_field_model.eval()
            if len(val_rgba) > 0:
                with torch.no_grad():
                    val_idx = torch.randint(low=0, high=len(val_rgba), size=[1]).to(device)
                    val_nerf_out, val_metrics = neural_radiance_field_model(
                        None,
                        val_cam_gen.create_cameras(id=val_idx),
                        val_rgba[val_idx],        # add dim for batchsize
                        min_depth = data.val.near[val_idx],
                        max_depth = data.val.far[val_idx],
                        mode = "val",
                    )

                # Update and print stats
                stats.update(
                    {'loss': float(loss), **val_metrics},
                    stat_set='val',
                    time_start=time_start,
                )
                stats.print(stat_set='val')

                # Plot the metrics.
                stats.plot_stats(viz=visualizer)
                tb_visualizer.plot_current_scalars(
                    dict(
                        val_metrics=val_metrics,
                        loss = float(loss)
                    ),
                    iteration
                )
                if not nerf_cfg.benchmark_fast:
                    tb_visualizer.flush()

                # Visualize the intermediate results.
                visualize_nerf_outputs(
                    val_nerf_out,
                    output_cache,
                    visualizer,
                    gtshape=data.mesh_gt
                )

            # Visualize training image
            with torch.no_grad():
                # Visualize training rendering
                train_nerf_out, _ = neural_radiance_field_model(
                    None,
                    camera,
                    image,        # add dim for batchsize
                    min_depth = camera_near,
                    max_depth = camera_far,
                    mode = "val",
                )
            visualize_nerf_outputs(
                train_nerf_out,
                None,
                visualizer,
                prefix='train '
            )

            neural_radiance_field_model.train()  # Do not forget this!

        # # Visualize the full renders every i360 iterations.
        if (iteration > 0) and (iteration % viz_cfg.i360 == 0):
            logging.info(f'Visualizing 360 (iter {iteration})')
            elevations = [-30,30]
            hfov = 25 *np.pi/180
            dist = data.mesh_radius / np.sin(hfov)     # r = d sin(hfov)
            min_depth = dist - data.mesh_radius
            max_depth = dist + data.mesh_radius
            show_rotating_nerf(neural_radiance_field_model,
                hfov, dist, elevations, data.mesh_centre,
                visualizer, iteration,
                num_frames=viz_cfg.i360_num_frames,
                device=device,
                min_depth = min_depth,
                max_depth = max_depth,
            )


        # Visualize the extracted shape every ishape iterations.
        if iteration>0 and (
            iteration % viz_cfg.ishape == 0
            or iteration == n_iter-1
        ):
            logging.info(f"Evaluating NeRF shape (iter {iteration}).")
            neural_radiance_field_model.eval()
            evaluate_NeRF_shape(
                data,
                neural_radiance_field_model,
                visualizer,
                tb_visualizer,
                shape_stats,
                voxel_res = nerf_cfg.shape.voxel_res,
                voxel_thresh = nerf_cfg.shape.voxel_thresh,
                camera_scale = 0.2 * data.mesh_radius,
                iteration=iteration
            )
            neural_radiance_field_model.train()

            if not nerf_cfg.benchmark_fast:
                tb_visualizer.flush()

        # Checkpoint.
        if (
            (
                iteration % nerf_cfg.checkpoint_iter_interval == 0
                or iteration==(n_iter-1)
            )
            and len(nerf_cfg.checkpoint_path) > 0
            and iteration > 0
        ):
            logging.info(f"Storing checkpoint (iter {iteration}) {nerf_cfg.checkpoint_path}.")
            data_to_store = {
                "model": neural_radiance_field_model.state_dict(),
                "camera_params": train_cam_gen.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "stats": pickle.dumps(stats),
                "shape_stats": pickle.dumps(shape_stats),
                "camera_stats": pickle.dumps(camera_stats),
                "finished_iter": iteration,
                "torch_rng_state": torch.random.get_rng_state()
            }
            torch.save(data_to_store, nerf_cfg.checkpoint_path)
            if nerf_cfg.checkpoint_each_iter:
                fpath = add_suffix_to_path(nerf_cfg.checkpoint_path, f'_{iteration:08d}')
                logging.info(f"Copying checkpoint w/o stats (iter {iteration}) to {fpath}.")
                if os.path.isfile(fpath): logging.warn('Overwriting existing file')
                data_to_store = {k:v for k,v in data_to_store.items() if k not in ['stats', 'stats_sparse', 'shape_stats', 'camera_stats']}
                torch.save(data_to_store, fpath)

    # Bring nerf to evaluation mode before returning
    neural_radiance_field_model.eval()

    return neural_radiance_field_model

@torch.no_grad()
def NeRF_to_voxels(
    neural_radiance_field: NeuralRadianceField,
    xyz_min: List[float] = [-0.2,-0.2,-0.2],
    xyz_max: List[float] = [0.2,0.2,0.2],
    voxel_res: int = 256,
    chunk_size: int = 50000,
) -> torch.Tensor:
    """
    Extract shape from NeRF, return as occupancy-voxel-grid of shape [VxVxV]
    """
    device = neural_radiance_field.density_layer.bias.device
    xyz_min = torch.as_tensor(xyz_min).float().to(device)
    xyz_max = torch.as_tensor(xyz_max).float().to(device)

    # Create voxel grid to evaluate NeRF at
    kwargs = {'steps':voxel_res, 'dtype':torch.float32, 'device':device}
    xs = torch.linspace(xyz_min[0], xyz_max[0], **kwargs)
    ys = torch.linspace(xyz_min[1], xyz_max[1], **kwargs)
    zs = torch.linspace(xyz_min[2], xyz_max[2], **kwargs)

    Wx,Wy,Wz = torch.meshgrid(xs,ys,zs)
    Wxyz = torch.stack([Wx,Wy,Wz], dim=-1)  # V x V x V x 3
    Wxyz = Wxyz.view(-1, 3)

    # Evaluate NeRF densities at voxel-grid in chunks to prevent OOM
    voxels = []
    for i in range(0, Wxyz.shape[0], chunk_size):
        voxels.append(
            neural_radiance_field.points_to_raw_densities(
                Wxyz[i:i+chunk_size][None]
            ).squeeze(0)
        )
    voxels = torch.cat(voxels, dim=0)
    voxels = voxels.view(voxel_res, voxel_res, voxel_res)

    return voxels


@torch.no_grad()
def NeRF_to_mesh(
    neural_radiance_field: NeuralRadianceField,
    xyz_min: List[float] = [-0.2,-0.2,-0.2],
    xyz_max: List[float] = [0.2,0.2,0.2],
    voxel_res: int = 256,
    voxel_thresh: float = 50,
) -> Meshes:
    """
    Extract shape from NeRF, return as pytorch3D mesh
    """
    voxels = NeRF_to_voxels(
        neural_radiance_field,
        xyz_min=xyz_min,
        xyz_max=xyz_max,
        voxel_res=voxel_res
    )
    verts, faces = marching_cubes(
        voxels,
        voxel_thresh,
        xyz_min=xyz_min,
        xyz_max=xyz_max
    )
    return Meshes(verts[None], faces[None])

def evaluate_NeRF_shape(
    data: DotMap,
    neural_radiance_field_model: RadianceFieldRenderer,
    visualizer: Visdom,
    tb_visualizer: TBVisualizer,
    shape_stats: Stats,
    voxel_res: int = 256,
    voxel_thresh: float = 50,
    camera_scale: float = 0.1,
    iteration: int = 0,
) -> None:

    # Extract Meshes
    xyz_min = data.mesh_centre - 1.1 * data.mesh_radius
    xyz_max = data.mesh_centre + 1.1 * data.mesh_radius
    c_mesh = NeRF_to_mesh(
        neural_radiance_field_model._implicit_function['coarse'],
        xyz_min=xyz_min,
        xyz_max=xyz_max,
        voxel_res=voxel_res,
        voxel_thresh=voxel_thresh,
    )
    f_mesh = NeRF_to_mesh(
        neural_radiance_field_model._implicit_function['fine'],
        xyz_min=xyz_min,
        xyz_max=xyz_max,
        voxel_res=voxel_res,
        voxel_thresh=voxel_thresh,
    )

    # Decimate meshes down to a million faces if they're too high res
    def is_valid(m: Meshes) -> bool: return len(m.num_verts_per_mesh())>0
    if is_valid(c_mesh): c_mesh = decimate_mesh(c_mesh, numF_target=int(1e6))
    if is_valid(f_mesh): f_mesh = decimate_mesh(f_mesh, numF_target=int(1e6))

    # Compute metrics
    c_metrics = metric_utils.compare_meshes_align_notalign(c_mesh, data.mesh_gt) if is_valid(c_mesh) else {}
    f_metrics = metric_utils.compare_meshes_align_notalign(f_mesh, data.mesh_gt) if is_valid(f_mesh) else {}

    # Update, print and plot stats
    for prefix, _metrics in zip(['coarse', 'fine'], [c_metrics, f_metrics]):
        for k,v in _metrics.items():
            ks = k.split('@')
            if len(ks)==1:
                shape_stats.update({k:v}, stat_set=prefix)
            else:
                assert(len(ks)==2)
                shape_stats.update({ks[0]:v}, stat_set=f'{prefix}@{ks[1]}')
    for k in shape_stats.stats.keys():
        shape_stats.print(stat_set=k)
    shape_stats.plot_stats(viz=visualizer)
    tb_visualizer.plot_current_scalars(
        dict(
            coarse_shape_metrics=c_metrics,
            fine_shape_metrics=f_metrics,
        ),
        iteration
    )

    # Visualize shapes with cameras in visdom
    shapes_dict = {
        'gt': data.mesh_gt,
        'coarse': c_mesh,
        'fine': f_mesh,
    }
    camera_trace = {
        f"cam_{i:03d}": cam.cpu()
        for i, cam in enumerate(dollyParamCameras(
                        data.train.poses,
                        data.train.hfovs,
                        optimize_cam=False
                    ).create_cameras_list())
    }
    # Plot shapes in different scenes because visdom doesn't let you switch meshes on/off
    for k,v in shapes_dict.items():
        if is_valid(v):
            plotly_plot = plot_scene(
                {f"{k} shape scene": {k:v, **camera_trace}},
                camera_scale = camera_scale,
            )
            visualizer.plotlyplot(plotly_plot, win=f"{k} shape scene")

    # Save meshes to file
    for k,v in shapes_dict.items():
        if is_valid(v):
            pytorch3d.io.save_obj(f'nerf-exp-{k}.obj', v.verts_packed(), v.faces_packed(), decimal_places=10)


def visualize_nerf_outputs(
    nerf_out: dict,
    output_cache: Optional[List],
    viz: Visdom,
    visdom_env: Optional[str] = None,
    gtshape: Optional[Meshes] = None,
    win_size: int = 100,
    prefix=''
):
    """
    @nocommit
    TODO
    """
    if output_cache is not None:
        ims = torch.stack([o["rgba"] for o in output_cache])
        num_ims = len(ims)
        ims = torch.cat(list(ims), dim=1)
        viz.image(
            ims.permute(2, 0, 1)[:3],
            env=visdom_env,
            win=f"{prefix}images",
            opts=dict(
                title = f"{prefix}train_images",
                height = win_size,
                width = win_size * num_ims,
            ),
        )

    if nerf_out is not None:
        ims_full = torch.cat(
            [
                nerf_out[imvar][0].permute(2, 0, 1).detach().cpu().clamp(0.0, 1.0)
                for imvar in ("rgba_coarse", "rgba_fine", "rgba_gt")
            ],
            dim=2,
        )
        rgb_full = ims_full[:3,:,:]
        alpha_full = ims_full[3:,:,:].expand(3,-1,-1)
        rgba_full = torch.cat((rgb_full, alpha_full), dim=1)
        viz.image(
            rgba_full[:3,:,:],
            env=visdom_env,
            win=f"{prefix}rgba_full",
            opts=dict(
                title = f"{prefix}coarse | fine | target",
                height = 2 * win_size,
                width = 3 * win_size
            ),
        )

    if output_cache is not None:
        camera_trace = {
            f"camera_{ci:03d}": o["camera"].cpu() for ci, o in enumerate(output_cache)
        }

        ray_pts_trace = {
            f"ray_pts_{ci:03d}": Pointclouds(
                ray_bundle_to_ray_points(o["coarse_ray_bundle"])
                .detach()
                .cpu()
                .view(1, -1, 3)
            )
            for ci, o in enumerate(output_cache)
        }

        gtshape_trace = {'gtshape': gtshape} if gtshape is not None else {}
        plotly_plot = plot_scene(
            {
                f"{prefix}training_scene": {
                    **gtshape_trace,
                    **camera_trace,
                    **ray_pts_trace,
                },
            },
            pointcloud_max_points=5000,
            pointcloud_marker_size=1,
            camera_scale=0.1,
        )
        viz.plotlyplot(plotly_plot, env=visdom_env, win=f"{prefix}scenes")
