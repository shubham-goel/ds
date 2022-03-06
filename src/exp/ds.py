from __future__ import annotations

import copy
import logging
import math
import os
import pickle
import random
from collections import OrderedDict
from shutil import copyfile
from typing import List, Optional, Tuple, Union

import hydra
import lpips
import numpy as np
import torch
import torch.nn.functional as F
from dotmap import DotMap
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch3d.loss import mesh_edge_loss
from pytorch3d.ops.subdivide_meshes import SubdivideMeshes
from pytorch3d.renderer import (BlendParams, PerspectiveCameras, TexturesUV)
from pytorch3d.renderer.cameras import CamerasBase
from pytorch3d.renderer.mesh.rasterizer import Fragments
from pytorch3d.renderer.points.compositor import AlphaCompositor
from pytorch3d.renderer.points.rasterizer import (PointsRasterizationSettings,
                                                  PointsRasterizer)
from pytorch3d.renderer.points.renderer import PointsRenderer
from pytorch3d.structures import Meshes, Pointclouds
from torch.optim.lr_scheduler import _LRScheduler
from visdom import Visdom

from ..nerfPy3D.nerf.stats import SimpleStats
from ..data.init_shape import initialize_shape, prep_blender_uvunwrap
from ..data.load_google import generate_360_RTs, load_data
from ..nnutils import geom_utils, loss_utils
from ..nnutils.cameras import dollyParamCameras
from ..nnutils.mask_bidt_loss import maskbidt_loss
from ..nnutils.mesh_laplacian_smoothing_voronoi import mesh_laplacian_smoothing
from ..nnutils.render import (SimpleDepthShader, SimpleNormalShader,
                              TexOptRenderer, TexTransferRenderer, TRANS_EPS)
from ..utils import image as im_utils
from ..utils import metrics as metric_utils
from ..utils import visutil
from ..utils.mesh import (refine_shape_topology_voxelize, save_mesh)
from ..utils.misc import (EmptyContext, add_prefix_to_keys, add_suffix_to_path,
                          apply_dict, apply_dict_rec, batchify_func,
                          flatten_dict, symlink_submitit, to_tensor,
                          try_move_device)
from ..utils.tb_visualizer import TBVisualizer
from ..utils.visutil import (visualize_depth, visualize_initial_scene,
                             visualize_normals)

torch.backends.cudnn.benchmark = True

class overfit_single(torch.nn.Module):
    def __init__(self,
        data: DotMap,
        mesh_init: Meshes,
        shader: DictConfig,
        raster: DictConfig,
        loss: DictConfig,
        optimize_cam: bool,
        optimize_R: bool,
        param_R_axis_angle: bool,
        optimize_T: bool,
        optimize_Fov: bool,
        optimize_first: bool,
        optimize_shape: bool,
        device: torch.device = torch.device('cuda:0'),
        load_lpips = True,
        **kwargs
    ):
        super().__init__()

        self.camera_generator = dollyParamCameras(
            data.train.poses, data.train.hfovs,
            centre=data.mesh_centre,
            param_R_axis_angle=param_R_axis_angle,
            optimize_cam=optimize_cam, optimize_first=optimize_first,
            optimize_R=optimize_R, optimize_T=optimize_T, optimize_Fov=optimize_Fov,
            device=device
        )
        self.gt_camera_generator = dollyParamCameras(
            data.train.poses_gt, data.train.hfovs_gt, optimize_cam=False,
            device=device
        )
        self.val_camera_generator = dollyParamCameras(
            data.val.poses, data.val.hfovs, optimize_cam=False, device=device
        )

        self.cfg_shader = shader
        self.cfg_loss = loss
        self.cfg_raster = raster
        self.data = data
        self.device = device
        self.N = data.train.rgba.shape[0]
        self._cache = DotMap()
        self.mesh_init = mesh_init.clone()

        # Need to save these options as they'll be used during reinitialization
        self.optimize_cam = optimize_cam
        self.optimize_R = optimize_R
        self.param_R_axis_angle = param_R_axis_angle
        self.optimize_T = optimize_T
        self.optimize_Fov = optimize_Fov
        self.optimize_first = optimize_first
        self.optimize_shape = optimize_shape

        # parameters
        zero_verts = torch.zeros_like(self.mesh_init.verts_packed())
        if optimize_shape:
            logging.info('Optimizing shape, set shape parameters.')
            self.register_parameter('deltaV', torch.nn.Parameter(zero_verts))
        else:
            logging.info('Not optimizing shape, Set shape buffers.')
            self.register_buffer('deltaV', zero_verts)

        ### -- Loss function stuffs -- ###
        # mean edge length for edge loss target
        edges_packed = self.mesh_init.edges_packed()  # (sum(E_n), 3)
        verts_packed = self.mesh_init.verts_packed()  # (sum(V_n), 3)
        verts_edges = verts_packed[edges_packed]
        v0, v1 = verts_edges.unbind(1)
        self.edgelen_mean_init = (v0 - v1).norm(dim=1, p=2).mean().item()
        logging.info(f'Average initial edge length: {self.edgelen_mean_init}')

        # loss functions
        if load_lpips:
            self.texLpips_loss_fn = lpips.LPIPS(net='vgg')


    def clone_with_new_initshape(self, new_mesh_init: Meshes) -> overfit_single:
        """ Clone current module preserving camera generator modules, but changing base shape"""
        new_model: overfit_single = self.__class__(
            self.data,
            new_mesh_init, # Use new mesh instead of old self.mesh_init
            self.cfg_shader,
            self.cfg_raster,
            self.cfg_loss,
            self.optimize_cam,
            self.optimize_R,
            self.param_R_axis_angle,
            self.optimize_T,
            self.optimize_Fov,
            self.optimize_first,
            self.optimize_shape,
            self.device,
        )
        new_model = new_model.to(self.device)
        # Preserve training camera poses
        new_model.camera_generator.load_state_dict(self.camera_generator.state_dict())
        return new_model

    def to(self, device: torch.device) -> overfit_single:
        super().to(device)
        self.data = apply_dict_rec(self.data, fv = lambda x: try_move_device(x, device))
        self.mesh_init = self.mesh_init.to(device)
        return self

    def state_dict(self, *args, **kwargs):
        dd = super().state_dict(*args, **kwargs)
        dd = dd.__class__({k:v for k,v in dd.items() if not k.startswith('texLpips_loss_fn')})
        return dd

    @property
    def mesh_gt(self) -> Meshes:
        return self.data.mesh_gt

    @staticmethod
    def get_znear_zfar(cameras: CamerasBase) -> Tuple[float, float]:
        zmin = cameras.T[:, 2].min().item()
        zmax = cameras.T[:, 2].max().item()
        if ((zmax-zmin) > 90 ):
            print("too much difference in z-translations")
            import ipdb; ipdb.set_trace()
        zmid = (zmin+zmax)/2
        return (zmid-50, zmid+50)

    def idx_to_data(self, idx: List[int], detach_cameras=False) -> DotMap:
        context = torch.no_grad() if detach_cameras else EmptyContext()
        with context:
            cameras = self.camera_generator.create_cameras(
                id=idx, device=self.device
            )
        gt_cameras = self.gt_camera_generator.create_cameras(
            id=idx, device=self.device
        )
        rgba = self.data.train.rgba[idx]
        return DotMap(cameras=cameras, gt_cameras=gt_cameras,
                    rgba=rgba,
                    idx=idx,
                    is_train=True, _dynamic=False)

    def idx_to_valdata(self, idx: List[int]) -> DotMap:
        cameras = self.val_camera_generator.create_cameras(
            id=idx, device=self.device
        )
        rgba = self.data.val.rgba[idx]
        return DotMap(cameras=cameras, rgba=rgba,
                    idx=idx, is_train=False, _dynamic=False)

    def get_texture_masks(self, rend_out:DotMap, batch:DotMap):
        _, rend_mask = rend_out.rgba.split([3,1], dim=1)
        _, tgt_mask = batch.rgba.split([3,1], dim=1)
        rend_tmask, tgt_tmask = loss_utils.texture_masks(rend_mask, tgt_mask, masking=self.cfg_loss.tex_mask)
        return rend_tmask, tgt_tmask

    def compute_image_losses(
        self, mesh_pred:Meshes, rend_out:DotMap, batch:DotMap
    ) -> Tuple[DotMap, DotMap]:
        cfg = self.cfg_loss
        weights = self.cfg_loss.wt

        rend_rgb, rend_mask = rend_out.rgba.split([3,1], dim=1)
        tgt_rgb, tgt_mask = batch.rgba.split([3,1], dim=1)
        cameras = batch.cameras

        # Texture masks and losses
        rend_tex_mask, tgt_tex_mask = self.get_texture_masks(rend_out, batch)
        texL1_loss_mp = F.l1_loss(rend_rgb * rend_tex_mask, tgt_rgb * tgt_tex_mask,
                                reduction='none').mean(dim=1, keepdim=True) if self.cfg_loss.wt.texL1>0 else torch.zeros_like(rend_rgb[:,:1,:,:])
        range_11 = lambda x: (2*x-1).clamp(min=-1, max=1)
        texLpips_loss = self.texLpips_loss_fn(range_11(rend_rgb) * rend_tex_mask,
                                                range_11(tgt_rgb) * tgt_tex_mask) if self.cfg_loss.wt.texLpips>0 else torch.tensor(0.)
        mask_loss_mp = F.mse_loss(rend_mask, tgt_mask, reduction='none') if weights.mask>0 else torch.tensor(0.)

        maskbidt_loss_mp = maskbidt_loss(
            rend_mask, tgt_mask, mask0_xy=rend_out.rgba_pxy,
            K = cfg.bidt_K,
            margin_pix = cfg.bidt_margin_pix,
            max_dist = cfg.bidt_maxdist,
        )

        # compute losses. keys should match those in cfg.loss.wt
        losses = DotMap()
        losses.texL1 = texL1_loss_mp.mean()
        losses.texLpips = texLpips_loss.mean()
        losses.mask = mask_loss_mp.mean()
        losses.maskbidt = maskbidt_loss_mp.mean()

        # Intermediate outputs
        outputs = DotMap(
            texL1_loss_mp = texL1_loss_mp.detach(),
            mask_loss_mp = mask_loss_mp.detach(),
            maskbidt_loss_mp = maskbidt_loss_mp.detach(),
            _dynamic=False
        )

        return outputs, losses

    def compute_shape_prior(self, mesh_pred:Meshes) -> Tuple[DotMap, DotMap]:
        weights = self.cfg_loss.wt
        shape_losses = DotMap()
        shape_losses.laplacian = mesh_laplacian_smoothing(mesh_pred, method=self.cfg_loss.laplacian_method) if weights.laplacian>0 else torch.tensor(0.)
        # shape_losses.normal    = mesh_normal_consistency(mesh_pred) if weights.normal>0 else torch.tensor(0.)

        # Mean edge length
        with torch.no_grad():
            mesh_bdb = mesh_pred.get_bounding_boxes().detach()
            mesh_size = (mesh_bdb[...,1] - mesh_bdb[...,0]).mean() / 2

        # Edge loss target computation:
        shape_losses.edge = mesh_edge_loss(mesh_pred, target_length=self.edgelen_mean_init) if weights.edge>0 else torch.tensor(0.)

        # Multiplier for laplacian/edge loss
        def shape_loss_multiplier(mult: Union[str,float]) -> float:
            if isinstance(mult, float):
                pass
            elif isinstance(mult, int):
                mult = float(mult)
            elif isinstance(mult, str) and mult.startswith('edge_'):
                mult = math.pow(self.edgelen_mean_init, float(mult[5:]))
            elif isinstance(mult, str) and mult.startswith('size_'):
                mult = math.pow(mesh_size, float(mult[5:]))
            else:
                raise ValueError(f'Unknown multiplier {mult} of type {type(mult)}. Options: <float>, "edge_<pow>", "size_<pow>"')
            return mult

        shape_losses.laplacian *= shape_loss_multiplier(self.cfg_loss.laplacian_multiplier)
        shape_losses.edge *= shape_loss_multiplier(self.cfg_loss.edgeloss_multiplier)

        # Intermediate outputs
        outputs = DotMap(_dynamic=False)

        return outputs, shape_losses

    def compute_metrics(self, batch:DotMap, outputs:DotMap, per_camera: bool = False, icp_align: bool = True, wrt_cam0=False) -> DotMap:
        shape_metrics = metric_utils.compare_meshes_align_notalign(outputs.mesh_pred, self.mesh_gt, icp_align=icp_align)
        camera_metrics = self.compute_camera_metrics(per_camera=per_camera, wrt_cam0=wrt_cam0)
        metrics = DotMap(shape_metrics=shape_metrics,
                        camera_metrics=camera_metrics, _dynamic=False)
        return metrics

    def compute_camera_metrics(self, per_camera: bool = False, wrt_cam0 = False) -> dict:
        # Compute metrics over all training cameras
        pred_cameras = self.camera_generator.create_cameras(device=self.device)
        gt_cameras = self.gt_camera_generator.create_cameras(device=self.device)
        camera_metrics = metric_utils.compare_cameras(pred_cameras, gt_cameras,
                            centre=self.data.mesh_centre,
                            per_camera=per_camera, wrt_cam0=wrt_cam0)
        return camera_metrics

    def run_single(self, **kwargs) -> Tuple[DotMap, DotMap, DotMap]:
        """ Returns a tuple containing (minibatch, predicted outputs, losses)"""
        raise NotImplementedError('Not implemented by child class')

    def forward(self, no_texture_loss: bool = False, **kwargs) -> Tuple[torch.Tensor, Tuple[DotMap, DotMap, DotMap]]:
        """ Returns total_loss """
        batch, outputs, losses = self.run_single(**kwargs)

        # Total loss
        # Filter out any loss containing 'tex' if no_texture_loss=True
        total_loss = sum([self.cfg_loss.wt[loss_name] * loss for loss_name, loss in losses.items()
                            if not (no_texture_loss and ('tex' in loss_name))])

        return total_loss, (batch, outputs, losses)

    @torch.no_grad()
    def get_current_scalars(self, batch: DotMap, outputs: DotMap, losses: DotMap, **kwargs) -> OrderedDict:
        total_loss = sum([self.cfg_loss.wt[loss_name] * loss for loss_name, loss in losses.items()])
        weighted_loss = {
            f'{name}_loss': float(loss) * self.cfg_loss.wt[name]
            for name, loss in losses.items()
        }
        percent_contrib = {
            f'{name}_loss': float(wloss) / total_loss.item()
            for name, wloss in weighted_loss.items()
        }
        sc_dict = OrderedDict(
            [
                ('total_loss', total_loss.item()),
                ('weighted_loss',weighted_loss),
                ('percent_contrib',percent_contrib),
            ],
            losses = apply_dict(losses, fv = lambda x: float(x)),
            **self.compute_metrics(batch, outputs, **kwargs)
        )
        return sc_dict

    def render_depth_normals(self,
        meshes: Meshes, cameras: CamerasBase,
        fragments: Optional[Fragments] = None,
        normal: bool = True, depth: bool = True,
        return_alpha: bool = False, **kwargs
    ) -> DotMap:

        # Rasterize meshes
        if fragments is None:
            fragments = self.rasterizer(meshes)

        # Near/Far
        znear, zfar = self.get_znear_zfar(cameras)

        # Render and store outputs
        outputs = DotMap()
        blend_bgx = lambda x: BlendParams(background_color=x)
        if normal:
            outputs.normal = SimpleNormalShader(cameras, blend_bgx(1))(
                fragments, meshes, return_alpha=return_alpha,
                znear=znear, zfar=zfar, **kwargs
            )
        if depth:
            outputs.depth = SimpleDepthShader(cameras, blend_bgx(-1))(
                fragments, meshes, return_alpha=return_alpha,
                znear=znear, zfar=zfar, **kwargs
            )
        return outputs

    def build_mesh(self, **kwargs) -> Meshes:
        """ Build mesh from parameters, or pick from cache"""
        mesh_pred = self.mesh_init.offset_verts(self.deltaV)
        return mesh_pred

    def render_novel_cameras(self, cameras: CamerasBase) -> DotMap:
        """ Render novel camera view using this class's texture model"""
        raise NotImplementedError

    def generate_visuals(self, outputs: DotMap, batch: DotMap) -> DotMap:
        """ Generate CPU images for visualization: rendered/gt image/mask/depth/normals """
        rend_rgb, rend_mask = outputs.rgba.split([3,1], dim=1)
        tgt_rgb, tgt_mask = batch.rgba.split([3,1], dim=1)
        dn = self.render_depth_normals(outputs.mesh_pred.extend(len(batch.cameras)),
                    batch.cameras, outputs.fragments, return_alpha=True)

        # Losses can be visualized:
        render_img = im_utils.concatenate_images2d([
                        [
                            outputs.rgba, rend_mask,
                            visualize_depth(dn.depth),
                            visualize_normals(dn.normal),
                        ],
                        [
                            batch.rgba, tgt_mask,
                            outputs.texL1_loss_mp,
                            outputs.mask_loss_mp,
                            outputs.maskbidt_loss_mp,
                        ]
                    ]).detach().cpu()

        visuals = DotMap(_dynamic=False)
        visuals.update({f'{i}/render':render_img[i] for i in range(len(outputs.rgba))})

        return visuals

    def populate_cache(self) -> None:
        # Caches important data to speed up 360-renderings
        pass

    @torch.no_grad()
    def get_current_visuals(self, prefix: str = '') -> DotMap:
        vis_dict = DotMap(img={}, mesh={}, video={}, text={}, p3d_mesh={})

        batch, outputs, losses = self.run_single(return_weights=True)
        vis_dict.image.update(self.generate_visuals(outputs, batch))

        # Save mesh
        lights = [
            {
            'cls': 'AmbientLight',
            'color': '#ffffff',
            'intensity': 0.75,
            }, {
            'cls': 'DirectionalLight',
            'color': '#ffffff',
            'intensity': 0.75,
            'position': [0, -1, 2],
            },{
            'cls': 'DirectionalLight',
            'color': '#ffffff',
            'intensity': 0.75,
            'position': [5, 1, -2],
            },
        ]
        material = {'cls': 'MeshStandardMaterial', 'side': 2}
        config = {'material':material, 'lights':lights}
        vis_dict.mesh[f'{prefix}mesh'] = {
                'v': outputs.mesh_pred.verts_padded().detach().cpu().contiguous(),
                'f': outputs.mesh_pred.faces_padded().detach().cpu().contiguous(),
                'cfg':config
            }

        return vis_dict

    @torch.no_grad()
    def get_shapemetric_visuals(self,
            num_frames: int = 24, prefix: str = '', chunk_size: int = 1, align_icp: Optional[bool] = None
        ) -> DotMap:

        if align_icp is None:
            # Compute metrics for both align_icp = True/False
            vis_dict0 = self.get_shapemetric_visuals(num_frames=num_frames, prefix=prefix, chunk_size=chunk_size,align_icp=False)
            vis_dict1 = self.get_shapemetric_visuals(num_frames=num_frames, prefix=prefix, chunk_size=chunk_size,align_icp=True)
            vis_dict0.video.update(vis_dict1.video)
            return vis_dict0

        assert(isinstance(align_icp, bool))
        metric_name_prefix = 'Aligned' if align_icp else ''

        vis_dict = DotMap(video={})

        # Build mesh, compute metrics
        mesh_pred = self.build_mesh()
        (
            shape_metrics,
            (pred_points, pred_normals, metrics_predpt),
            (gt_points, gt_normals, metrics_gtpt),
        ) = metric_utils.compare_meshes(mesh_pred, self.mesh_gt,
                                    return_per_point_metrics=True,
                                    unscale_returned_points=True,
                                    align_icp=align_icp)

        ### Modify metrics for better visualization
        # No change on schmafer distance
        metrics_predpt['Chamfer-L2']
        # Maximum normal inconsistency (-1) should show as 1; minimum (1) should show as 0.
        metrics_predpt['NormalConsistency'] = (1 - metrics_predpt['NormalConsistency'])/2
        metrics_gtpt['NormalConsistency'] = (1 - metrics_gtpt['NormalConsistency'])/2
        # Maximum normal inconsistency (0) should show as 1; minimum (1) should show as 0.
        metrics_predpt['AbsNormalConsistency'] = 1-metrics_predpt['AbsNormalConsistency']
        metrics_gtpt['AbsNormalConsistency'] = 1-metrics_gtpt['AbsNormalConsistency']

        # Stack metrics into a feature vector. Last channel is alpha=1
        device = pred_points.device
        metric_names = list(metrics_predpt.keys())
        pred_points_feat = torch.stack([metrics_predpt[k] for k in metric_names], dim=-1).to(device)
        gt_points_feat = torch.stack([metrics_gtpt[k] for k in metric_names], dim=-1).to(device)
        alpha = torch.ones_like(gt_points_feat[...,-1:])
        pred_points_feat = torch.cat([pred_points_feat, alpha], dim=-1)
        gt_points_feat = torch.cat([gt_points_feat, alpha], dim=-1)

        pred_pcl = Pointclouds(pred_points, normals=pred_normals, features=pred_points_feat)
        gt_pcl = Pointclouds(gt_points, normals=gt_normals, features=gt_points_feat)

        # PCL renderer
        img_size = self.data.train.rgba.shape[2:4]
        raster_settings=PointsRasterizationSettings(image_size=img_size, radius=0.02)
        rasterizer = PointsRasterizer(raster_settings=raster_settings)
        compositor = AlphaCompositor(background_color=torch.zeros(len(metric_names)+1, dtype=torch.float, device=device))
        renderer = PointsRenderer(rasterizer=rasterizer, compositor=compositor)

        # Render continuum
        hfov = 25 *np.pi/180
        dist = self.data.mesh_radius / np.sin(hfov)     # d = r sin(hfov)

        for elevation in [-30, 30]:
            predpcl_cont = []
            gtpcl_cont = []
            Rs, Ts = generate_360_RTs(elev=elevation, dist=dist, num_frames=num_frames, at=self.data.mesh_centre)

            # Render cameras in chunks
            for idx in range(0, num_frames, chunk_size):
                cameras = PerspectiveCameras(
                    device=self.device,
                    R=Rs[idx:idx+chunk_size],
                    T=Ts[idx:idx+chunk_size],
                    focal_length=1/np.tan(hfov),
                )
                # Render PCL
                rend_predpcl = renderer(pred_pcl.extend(len(cameras)), cameras=cameras, eps=TRANS_EPS) # NxHxWxF+1
                rend_gtpcl = renderer(gt_pcl.extend(len(cameras)), cameras=cameras, eps=TRANS_EPS)     # NxHxWxF+1

                predpcl_cont.append(rend_predpcl)
                gtpcl_cont.append(rend_gtpcl)
                assert(rend_predpcl.isfinite().all())
                assert(rend_gtpcl.isfinite().all())

            # Merge rendered chunks
            predpcl_cont = torch.cat(predpcl_cont, dim=0)   # NxHxWxF+1
            gtpcl_cont = torch.cat(gtpcl_cont, dim=0)       # NxHxWxF+1

            for i, metric_name in enumerate(metric_names):
                pred_vid = batchify_func(predpcl_cont[...,i:i+1], lambda x: visutil.gray_to_colormap(x))
                gt_vid = batchify_func(gtpcl_cont[...,i:i+1], lambda x: visutil.gray_to_colormap(x))
                _green = torch.tensor([0,1,0], device=device, dtype=pred_vid.dtype)
                pred_vid = pred_vid * predpcl_cont[...,-1:] + _green*(1-predpcl_cont[...,-1:])
                gt_vid = gt_vid * gtpcl_cont[...,-1:] + _green*(1-gtpcl_cont[...,-1:])

                # Change format to NCHW because that's what tensorboard expects for videos
                pred_vid = im_utils.change_img_format(pred_vid, out_format="NCHW", inp_format="NHWC")
                gt_vid = im_utils.change_img_format(gt_vid, out_format="NCHW", inp_format="NHWC")

                vis_dict.video[f'{prefix}cont_{elevation}/{metric_name_prefix}{metric_name}/predpcl'] = pred_vid.cpu().numpy()
                vis_dict.video[f'{prefix}cont_{elevation}/{metric_name_prefix}{metric_name}/gtpcl'] = gt_vid.cpu().numpy()

        vis_dict.video_fps = num_frames / 12

        return vis_dict

    @torch.no_grad()
    def get_360_visual(self, num_frames: int = 24, prefix: str = '', chunk_size: int = 1, elevations = [-30, 30], **kwargs) -> DotMap:
        vis_dict = DotMap(video={})

        # Cache fragments to speed up 360-renderings
        self.populate_cache()

        # Render continuum
        hfov = kwargs.get('hfov', 25 * np.pi/180)
        mesh_radius = kwargs.get('mesh_radius', self.data.mesh_radius)
        mesh_centre = kwargs.get('mesh_centre', self.data.mesh_centre)
        dist = mesh_radius / np.sin(hfov)     # r = d sin(hfov)

        for elevation in elevations:
            rgba_cont = []
            depth_cont = []
            normal_cont = []
            Rs, Ts = generate_360_RTs(elev=elevation, dist=dist, num_frames=num_frames, device=self.device, at=mesh_centre)

            # Render cameras in chunks
            for idx in range(0, num_frames, chunk_size):
                cameras = PerspectiveCameras(
                    device=self.device,
                    R=Rs[idx:idx+chunk_size],
                    T=Ts[idx:idx+chunk_size],
                    focal_length=1/np.tan(hfov),
                )
                # Render rgba
                rend_output = self.render_novel_cameras(cameras, use_cached=True)
                rgba_cont.append(rend_output.rgba)
                assert(rend_output.rgba.isfinite().all())

                # Render Depth/Normals
                dn = self.render_depth_normals(rend_output.mesh_pred.extend(len(cameras)),
                                        cameras, rend_output.fragments, return_alpha=True,
                                        world_coordinates=True)
                depth_cont.append(dn.depth)
                normal_cont.append(dn.normal)
                assert(dn.depth.isfinite().all())
                assert(dn.normal.isfinite().all())

            # Merge rendered chunks
            rgba_cont = torch.cat(rgba_cont, dim=0)
            depth_cont = torch.cat(depth_cont, dim=0)
            normal_cont = torch.cat(normal_cont, dim=0)
            dn_cont = im_utils.concatenate_images1d([
                                visualize_depth(depth_cont, force_consistent=True),
                                visualize_normals(normal_cont)
                            ])

            # Change format to NCHW because that's what tensorboard expects for videos
            rgba_cont = im_utils.change_img_format(rgba_cont, out_format="NCHW")
            dn_cont = im_utils.change_img_format(dn_cont, out_format="NCHW")
            vis_dict.video[f'{prefix}cont_{elevation}/tex'] = im_utils.split_alpha(rgba_cont)[0].cpu().numpy()
            vis_dict.video[f'{prefix}cont_{elevation}/dn'] = im_utils.split_alpha(dn_cont)[0].cpu().numpy()

        vis_dict.video_fps = num_frames / 12

        # Don't forget to clear cache
        self._cache.clear()

        return vis_dict

    def display_visuals(self, viz: Visdom, prefix: str = '') -> None:
        raise NotImplementedError

class TexOptUV(overfit_single):
    def __init__(self,
        data: DotMap,
        mesh_init: Meshes,
        teximg_size: int,
        shader: DictConfig,
        raster: DictConfig,
        loss: DictConfig,
        *args,
        **kwargs
    ):
        super().__init__(
            data, mesh_init,
            shader,
            raster, loss,
            *args, **kwargs
        )

        # Parameters
        self.teximg_size = teximg_size
        self.textureImg = torch.nn.Parameter(
            torch.zeros(3,teximg_size,teximg_size)
        )

        # Store checkerboard uvImg for debugging
        hrange = torch.arange(teximg_size).float()/teximg_size * 2 - 1    # [-1,1]
        wrange = torch.arange(teximg_size).float()/teximg_size * 2 - 1    # [-1,1]
        grid_h, grid_w = torch.meshgrid(hrange, wrange)     # h,w
        pointsGT_ij = torch.stack([grid_w, grid_h],dim=-1)  # h,w,2
        pointsGT_ij = pointsGT_ij.float()
        uvImg = visutil.uv2bgr(pointsGT_ij.cpu().numpy())
        uvImg = np.transpose(uvImg, (2,0,1))
        uvImg = torch.as_tensor(uvImg).float() / 255
        self.register_buffer('uvImg', uvImg)
        self.register_buffer('verts_uv', kwargs.get('verts_uv', data.mesh_init_verts_uv))
        self.register_buffer('fuv_idx', kwargs.get('fuv_idx', data.mesh_init_fuv_idx))

        # Renderer
        self.renderer = TexOptRenderer(
            image_size = data.train.rgba.shape[2:4],
            cfg_raster = raster,
            cfg_shader = shader,
        )

    def clone_with_new_initshape(self, new_mesh_init: Meshes) -> overfit_single:
        """ Clone current module preserving camera generator modules, but changing base shape"""

        # Recompute verts_uv, faces_uv
        new_mesh_init, verts_uv, faces_uv_idx = prep_blender_uvunwrap(
            new_mesh_init.verts_packed(),
            new_mesh_init.faces_packed(),
            simplify=False
        )

        # Re-initialize base class
        new_model: overfit_single = self.__class__(
            self.data,
            new_mesh_init, # Use new mesh instead of old self.mesh_init
            self.teximg_size,
            self.cfg_shader,
            self.cfg_raster,
            self.cfg_loss,
            self.optimize_cam,
            self.optimize_R,
            self.param_R_axis_angle,
            self.optimize_T,
            self.optimize_Fov,
            self.optimize_first,
            self.optimize_shape,
            self.device,
            verts_uv = verts_uv,
            fuv_idx = faces_uv_idx
        )
        new_model = new_model.to(self.device)
        # Preserve training camera poses
        new_model.camera_generator.load_state_dict(self.camera_generator.state_dict())
        return new_model

    def run_single(self, detach_cameras=False, **kwargs) -> Tuple[DotMap, DotMap, DotMap]:
        # Runs 1 iteration of the forward pass.
        # Returns dicts of losses

        # Get minibatch of target data
        target_idx = kwargs.get('target_idx', list(range(self.N)))
        batch = self.idx_to_data(target_idx, detach_cameras=detach_cameras)

        # Render camera viewpoints
        outputs = self.render_novel_cameras(batch.cameras)

        # Losses
        imloss_out, img_losses = self.compute_image_losses(outputs.mesh_pred, outputs, batch)
        shloss_out, shape_losses = self.compute_shape_prior(outputs.mesh_pred)
        losses = DotMap(**img_losses, **shape_losses, _dynamic=False)
        outputs = DotMap(**outputs, **imloss_out, **shloss_out, _dynamic=False)
        return batch, outputs, losses

    def build_mesh(self, **kwargs) -> Meshes:
        # Build textured mesh
        tex_img = self.get_texture_image().permute((1,2,0)).contiguous()
        tex = TexturesUV(
                verts_uvs=self.verts_uv[None],
                faces_uvs=self.fuv_idx[None],
                maps=tex_img[None]
        )
        mesh_pred = super().build_mesh()
        mesh_pred.textures = tex
        return mesh_pred

    def render_novel_cameras(self, cameras: CamerasBase, **kwargs) -> DotMap:
        """ Render novel camera view """
        # Build mesh
        mesh_pred = self.build_mesh()

        # Render images
        znear, zfar = self.get_znear_zfar(cameras)
        rend_out = self.renderer(
            mesh_pred,
            cameras,
            znear=znear, zfar=zfar,
            eps=TRANS_EPS
        )
        rend_out.rgba = rend_out.rgba.permute(0,3,1,2)
        rend_out.rgba_pxy = rend_out.rgba_pxy.permute(0,3,1,2)
        assert(im_utils.get_img_format(rend_out.rgba) == 'NCHW')
        return DotMap(mesh_pred=mesh_pred, **rend_out, _dynamic=False)

    def get_texture_image(self, use_checkerboard: bool = False) -> torch.Tensor:
        if use_checkerboard:
            return self.uvImg
        else:
            # Nasty hack to accelerate texture optimizaiton.
            # Essentially increases learning rate by 1000x
            return torch.sigmoid(1e3 * self.textureImg)

    def get_current_visuals(self, prefix: str = '') -> DotMap:
        vis_dict = super().get_current_visuals(prefix=prefix)

        # Add texture image
        vis_dict.img[f'{prefix}texImg'] = self.get_texture_image().detach().cpu()

        return vis_dict

class TexTransfer(overfit_single):
    def __init__(self,
        data: DotMap,
        mesh_init: Meshes,
        shader: DictConfig,
        raster: DictConfig,
        loss: DictConfig,
        *args,
        **kwargs
    ):
        super().__init__(
            data, mesh_init,
            shader,
            raster, loss,
            *args, **kwargs
        )

        # No additional Parameters

        # Renderer
        self.renderer = TexTransferRenderer(
            image_size = data.train.rgba.shape[2:4],
            cfg_raster = raster,
            cfg_shader = shader,
        )

    def run_single(self, detach_cameras=False, **kwargs) -> Tuple[DotMap, DotMap, DotMap]:
        # Runs 1 iteration of the forward pass.
        # Returns dicts of losses

        # Get minibatch of target data
        target_idx = kwargs.get('target_idx', list(range(self.N)))
        batch = self.idx_to_data(target_idx, detach_cameras=detach_cameras)

        # Get minibatch of texture source data
        source_idx = kwargs.get('source_idx', list(range(self.N)))
        # To prevent rasterizing the same camera poses twice during training
        if source_idx==target_idx:
            source_batch = batch
        else:
            source_batch = self.idx_to_data(source_idx, detach_cameras=detach_cameras)

        outputs = self.render_novel_cameras(
            batch.cameras,
            source_batch=source_batch,
            sample_from_target = False,
            **kwargs
        )

        # Losses
        imloss_out, img_losses = self.compute_image_losses(outputs.mesh_pred, outputs, batch)
        shloss_out, shape_losses = self.compute_shape_prior(outputs.mesh_pred)
        losses = DotMap(**img_losses, **shape_losses, _dynamic=False)
        outputs = DotMap(**outputs, **imloss_out, **shloss_out, _dynamic=False)
        return batch, outputs, losses

    def build_mesh(self, use_cached: bool = False) -> Meshes:
        # Build mesh, or pick from cache
        if use_cached and self._cache.mesh_pred:
            mesh_pred = self._cache.mesh_pred
        else:
            mesh_pred = super().build_mesh()
        return mesh_pred

    def render_novel_cameras(self,
        cameras: CamerasBase, use_cached: bool = False,
        source_idx: Optional[list] = None,
        source_batch: Optional[DotMap] = None,
        **kwargs
    ) -> DotMap:
        """ Render novel camera view """
        # Build mesh
        mesh_pred = self.build_mesh(use_cached=use_cached)

        # Pick ref_fragments from cache
        if use_cached and self._cache.ref_fragments:
            ref_fragments = self._cache.ref_fragments
        else:
            ref_fragments = None

        # Get minibatch of texture source data
        if source_idx is None:
            source_idx = list(range(self.N))
        if source_batch is None:
            source_batch = self.idx_to_data(source_idx)

        # Render images
        znear, zfar = self.get_znear_zfar(cameras)
        rend_out = self.renderer(
            mesh_pred,
            cameras,
            ref_rgba = source_batch.rgba,
            ref_cameras = source_batch.cameras,
            ref_fragments = ref_fragments,
            znear=znear, zfar=zfar,
            eps=TRANS_EPS,   # eps for perspective transform
            **kwargs
        )
        rend_out.rgba = rend_out.rgba.permute(0,3,1,2)
        rend_out.rgb_valid = rend_out.rgb_valid.permute(0,3,1,2)
        rend_out.rgba_pxy = rend_out.rgba_pxy.permute(0,3,1,2)
        assert(im_utils.get_img_format(rend_out.rgba) == 'NCHW')
        return DotMap(mesh_pred=mesh_pred, **rend_out, _dynamic=False)

    def get_texture_masks(self, rend_out:DotMap, batch:DotMap):
        rend_tmask, tgt_tmask = super().get_texture_masks(rend_out, batch)

        # Compute texture losses only where rendered rgb is valid
        THRESH = 0.1
        rend_tmask = rend_tmask * (rend_out.rgb_valid>THRESH)
        tgt_tmask = tgt_tmask * (rend_out.rgb_valid>THRESH)

        return rend_tmask, tgt_tmask

    def populate_cache(self) -> None:
        # Caches important data to speed up 360-renderings
        batch = self.idx_to_data(list(range(self.N)))
        self._cache.mesh_pred = self.build_mesh()
        self._cache.ref_fragments = self.renderer.rasterizer(
            self._cache.mesh_pred.extend(len(batch.cameras)), cameras=batch.cameras, eps=TRANS_EPS
        )

    def generate_visuals(self, outputs: DotMap, batch: DotMap) -> DotMap:
        visuals = super().generate_visuals(outputs, batch)
        if 'prob_weights' in outputs:
            prob_imgs = im_utils.concat_images_grid(
                    im_utils.change_img_format(outputs.prob_weights,
                                            out_format='PNHWC', inp_format='NHWP')
                ).detach().cpu()
            visuals.update({f'{i}/prob':prob_imgs[i] for i in range(len(outputs.rgba))})
        if 'rgb_valid' in outputs:
            rgb_valid = im_utils.change_img_format(outputs.rgb_valid,
                                            out_format='NHWC').detach().cpu()
            visuals.update({f'{i}/rgb_valid':rgb_valid[i] for i in range(len(outputs.rgba))})
        return visuals

CONFIG_DIR='../../configs'
@hydra.main(config_path=CONFIG_DIR, config_name="ds")
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

    if os.path.isfile(cfg.init_data_path):
        logging.info(f'Loading data from {cfg.init_data_path}')
        with open(cfg.init_data_path, 'rb') as handle:
            data = pickle.load(handle)
    else:
        data = load_data(cfg.data)
        data = initialize_shape(data, ico_sphere_level=cfg.init_shape.level)
        data = to_tensor(data, _dynamic=False)

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
    tb_visualizer.plot_images(
        {'train_imgs': im_utils.concat_images_grid(data.train.rgba.cpu())}, 0
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = overfit_single_instance(cfg, data, tb_visualizer, device=device)
    return

def overfit_single_instance(
        cfg: DictConfig, data: DotMap,
        tb_visualizer: TBVisualizer,
        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    ) -> overfit_single:
    data = apply_dict_rec(data, fv = lambda x: try_move_device(x, device))

    # Model, optimizer and scheduler
    mesh_init = data.mesh_init
    model : overfit_single = instantiate(
        cfg.texture_model,
        data,
        mesh_init,
        device=device,
        _recursive_=False
    )
    model = model.to(device)
    def get_optimizer(_model) -> torch.optim.Optimizer:
        try:
            _optim = instantiate(cfg.optim, _model.parameters())
        except ValueError:
            _optim = instantiate(cfg.optim, [torch.nn.Parameter(torch.zeros(1))])
        return _optim
    optim = get_optimizer(model)
    lr_scheduler: _LRScheduler = instantiate(cfg.lr_scheduler, optim)

    def recreate_model_optim_scheduler(_new_mesh, _model, _optim, _lr_scheduler):
        """ Recreate model, optimizer, lrscheduler. Preserve state dicts """
        optim_sdict = _optim.state_dict()
        optim_sdict['state'].clear() # Clear optimizer state which contains deltaV
        lr_scheduler_sdict = _lr_scheduler.state_dict()
        _model = _model.clone_with_new_initshape(_new_mesh)
        _optim = get_optimizer(_model)
        _lr_scheduler: _LRScheduler = instantiate(cfg.lr_scheduler, _optim)
        _optim.load_state_dict(optim_sdict)
        _lr_scheduler.load_state_dict(lr_scheduler_sdict)
        return _model, _optim, _lr_scheduler

    # Simple losses/metrics tracker
    stats = SimpleStats()
    stats_sparse = SimpleStats()        # save stats less frequently (only on i360, for saving shapes at best metrics)

    # Load from checkpoint
    state = DotMap(
        smooth_loss = 0,
        premptive_terminate = False,
        best_loss = float('inf'),
        best_iter = -1,
        _dynamic = False
    )
    start_iter = 0
    best_model_params = None
    best_model_meshinit = None
    if len(cfg.checkpoint_path) > 0:
        # Make the root of the experiment directory.
        checkpoint_dir = os.path.split(cfg.checkpoint_path)[0]
        if checkpoint_dir: os.makedirs(checkpoint_dir, exist_ok=True)

        # Resume training if requested.
        if cfg.resume:
            if not os.path.isfile(cfg.checkpoint_path):
                logging.warn(f'Requested resume but checkpoint not found: {cfg.checkpoint_path}')
            else:
                logging.info(f"Resuming from checkpoint {cfg.checkpoint_path}.")
                loaded_data = torch.load(cfg.checkpoint_path)
                mesh_init = Meshes(loaded_data["mesh_init_verts"], loaded_data["mesh_init_faces"])
                model, optim, lr_scheduler = recreate_model_optim_scheduler(mesh_init, model, optim, lr_scheduler)
                missing_keys, unexpected_keys = model.load_state_dict(loaded_data["model"], strict=False)
                for k in missing_keys:
                    assert any([k.startswith(kk) for kk in [
                        'texLpips_loss_fn',
                    ]])
                # assert(len(missing_keys)==0)
                assert(len(unexpected_keys)==0)
                stats = pickle.loads(loaded_data["stats"])
                stats_sparse = pickle.loads(loaded_data["stats_sparse"])
                optim.load_state_dict(loaded_data["optimizer"])
                lr_scheduler.load_state_dict(loaded_data["lr_scheduler"])
                start_iter = loaded_data["finished_iter"] + 1
                state = loaded_data["state_vars"]
                best_model_params = loaded_data["best_model_params"]
                best_model_meshinit = Meshes(loaded_data["best_model_mesh_init_verts"], loaded_data["best_model_mesh_init_faces"])
                torch.random.set_rng_state(loaded_data['torch_rng_state'])
                logging.info(f"   => resuming from iter {start_iter}.")


    if start_iter==0 and not cfg.benchmark_fast:
        logging.info('Benchmarking initial shape, cameras')
        shape_metric_init = metric_utils.compare_meshes_align_notalign(data.mesh_init, data.mesh_gt)
        camera_metric_init = model.compute_camera_metrics(per_camera=True, wrt_cam0 = cfg.cammetric_wrt_cam0)
        scalars = dict(
                camera_metrics = camera_metric_init,
                shape_metrics = shape_metric_init,
                shape_metric_init = shape_metric_init,
            )
        tb_visualizer.plot_current_scalars(scalars, 0)
        stats.update(flatten_dict(scalars), 0)
        logging.info(f'Visualizing initial texture')
        tb_visualizer.display_current_results(model.get_current_visuals(prefix='init_'), 0, save_meshes=False)
        tb_visualizer.display_current_results(model.get_360_visual(prefix='init_', num_frames=cfg.viz.i360_num_frames), 0)
        tb_visualizer.flush()


    logging.info('Starting training loop')
    grad_norm_shape = -1
    grad_norm_camR = -1
    grad_norm_camT = -1
    grad_norm_camF = -1
    for i in range(start_iter, cfg.num_iter):

        if state.premptive_terminate:
            break

        # Re-initializing meshes
        reinitialize_model = False
        if i in list(cfg.int_subdivide_iters):
            logging.info(f'Reinitializing mesh (iter {i}) using subdivision')
            reinitialize_model = True
            with torch.no_grad():
                mesh_init = SubdivideMeshes()(model.build_mesh())

        if i in list(cfg.int_remesh.iters):
            logging.info(f'Reinitializing mesh (iter {i}) using {cfg.int_remesh.type}')
            reinitialize_model = True
            with torch.no_grad():
                curr_R, curr_T, curr_hfovs, _ = model.camera_generator.get_RTfovF()
                save_mesh(f'remesh_{i:08d}_before.obj', model.build_mesh())
                if cfg.int_remesh.type == 'voxelize':
                    mesh_init = refine_shape_topology_voxelize(
                                            model.build_mesh(),
                                            geom_utils.RT_to_poses(curr_R, curr_T),
                                            curr_hfovs,
                                            data.train.rgba,
                                            **cfg.int_remesh,
                                        )
                else:
                    raise ValueError(f'{cfg.int_remesh.type} {type(cfg.int_remesh.type)}')
                save_mesh(f'remesh_{i:08d}_after.obj', mesh_init)
        if reinitialize_model:
            model, optim, lr_scheduler = recreate_model_optim_scheduler(mesh_init, model, optim, lr_scheduler)

        # raster.blur_radius and shader.sigma schedules
        assert cfg.blur_radius_schedule.decay == 'exp'
        assert cfg.sigma_schedule.decay == 'exp'
        blur_radius = cfg.texture_model.raster.blur_radius
        sigma = cfg.texture_model.shader.blend_params.sigma
        blur_radius = blur_radius * (cfg.blur_radius_schedule.min/blur_radius) ** (float(i)/cfg.num_iter)
        sigma = sigma * (cfg.sigma_schedule.min/sigma) ** (float(i)/cfg.num_iter)
        blend_params = model.renderer.shader.blend_params._replace(sigma=sigma)
        raster_settings = copy.copy(model.renderer.rasterizer.raster_settings)
        raster_settings.blur_radius = blur_radius

        # Forward and backward model
        is_warmup = i<cfg.warmup_shape_iter
        optim.zero_grad()
        loss, (batch, outputs, losses) = model(
                                            detach_cameras = is_warmup,
                                            no_texture_loss = is_warmup,
                                            raster_settings = raster_settings,
                                            blend_params = blend_params,
                                        )
        if loss.requires_grad:
            loss.backward()
            # First clip very high shape gradients by value
            if cfg.grad_norm_clip > 0:
                grad_norm_shape = torch.nn.utils.clip_grad_norm_(model.deltaV, cfg.grad_norm_clip)
                grad_norm_camR = torch.nn.utils.clip_grad_norm_(model.camera_generator.rel_quats, cfg.grad_norm_clip)
                grad_norm_camT = torch.nn.utils.clip_grad_norm_(model.camera_generator.rel_trans, cfg.grad_norm_clip)
                grad_norm_camF = torch.nn.utils.clip_grad_norm_(model.camera_generator.rel_hfovs, cfg.grad_norm_clip)
            optim.step()
        else:
            logging.warn('Not backpropagating, loss.requires_grad=False')

        # Keep track of best model weights
        if loss.item() <= state.best_loss:
            state.best_iter = i
            state.best_loss = loss.item()
            best_model_params = model.state_dict()
            best_model_meshinit = mesh_init

        # LR Scheduler
        state.smooth_loss = loss.item() if i==0 else 0.8*state.smooth_loss + 0.2*loss.item()
        lr_scheduler.step() # No arguments to step()
        # Terminate when warm restart just happened, and there aren't enough iterations
        # left to finish another cycle.
        iter_left = cfg.num_iter-i-1
        iter_needed = lr_scheduler.T_i - lr_scheduler.T_cur
        state.premptive_terminate = (lr_scheduler.T_cur==0 and (iter_needed > iter_left))

        if state.premptive_terminate:
            logging.warn(f'Terminating preemptively because of {type(lr_scheduler).__name__}')

        # Print+plot scalars/metrics
        if  i>0 and i%cfg.viz.iprint == 0:
            logging.info(f'{i}, {loss.item()}')
            scalars = model.get_current_scalars(batch, outputs, losses, per_camera=True, icp_align= (not cfg.benchmark_fast), wrt_cam0 = cfg.cammetric_wrt_cam0)
            scalars.update(grad_norm_shape=grad_norm_shape,
                grad_norm_camR=grad_norm_camR, grad_norm_camT=grad_norm_camT, grad_norm_camF=grad_norm_camF,
                smooth_loss=state.smooth_loss, lr=lr_scheduler._last_lr[0],
                )
            if isinstance(model, TexOptUV):
                scalars.update(teximg_max = model.get_texture_image().max().item())    # debugging texture image only
            tb_visualizer.plot_current_scalars(scalars, i)
            if not cfg.benchmark_fast:
                tb_visualizer.flush()

            # Save scalars to stats object that maintains average/min
            flat_scalars = flatten_dict(scalars)
            flat_scalars = apply_dict_rec(flat_scalars, fv = lambda x: try_move_device(x, 'cpu'))
            stats.update(flat_scalars, i)

        # Visualize renderings to tensorboard
        if  i>0 and i%cfg.viz.idisplay == 0:
            logging.info('Displaying results...')
            tb_visualizer.display_current_results(model.get_current_visuals(), i, save_meshes=True)
            if not cfg.benchmark_fast:
                tb_visualizer.flush()

        # Visualize 360-video
        if  i>0 and i%cfg.viz.i360 == 0:
            v360 = model.get_360_visual(num_frames=cfg.viz.i360_num_frames)
            tb_visualizer.display_current_results(v360, i)
            if not cfg.benchmark_fast:
                vshape = model.get_shapemetric_visuals(num_frames=cfg.viz.i360_num_frames)
                tb_visualizer.display_current_results(vshape, i)

                # Compare metrics to best metrics, save + visuzlize mesh
                scalars = model.get_current_scalars(batch, outputs, losses, per_camera=False, wrt_cam0 = cfg.cammetric_wrt_cam0)
                flat_scalars = flatten_dict(scalars)
                stats_sparse.update(flat_scalars, i)
                metrics_save_best_shape = [
                    'shape_metrics/Chamfer-L2',
                    'shape_metrics/NormalConsistency',
                    'shape_metrics/AbsNormalConsistency',
                    'shape_metrics/AlignedChamfer-L2',
                    'shape_metrics/AlignedNormalConsistency',
                    'shape_metrics/AlignedAbsNormalConsistency',
                ]
                prefixes = []
                fpaths = []
                for k in metrics_save_best_shape:
                    if flat_scalars[k] <= stats_sparse.meter_dict[k].min:
                        logging.info(f"Found best {k} (value {float(flat_scalars[k]):.6f}) mesh at iter {i}")
                        prefixes.append(f'best/{k}/')
                        fpaths.append(add_suffix_to_path(cfg.mesh_checkpoint_path, f'_best{k.split("/")[-1]}'))
                for prefix in prefixes:
                    nv360 = dict(video=add_prefix_to_keys(v360.video, prefix), video_fps=v360.video_fps)
                    tb_visualizer.display_current_results(nv360, i)
                    if not cfg.benchmark_fast:
                        nvshape = dict(video=add_prefix_to_keys(vshape.video, prefix), video_fps=vshape.video_fps)
                        tb_visualizer.display_current_results(nvshape, i)
                for fpath in fpaths:
                    logging.info(f"Saving mesh (iter {i}) to {fpath}.")
                    save_mesh(fpath, model.build_mesh())

            # Flush TB
            if not cfg.benchmark_fast:
                tb_visualizer.flush()


        # Checkpoint.
        if (
            (
                i % cfg.checkpoint_iter_interval == 0
                or i==(cfg.num_iter-1)
                or state.premptive_terminate
            )
            and len(cfg.checkpoint_path) > 0
            and i > 0
        ):
            logging.info(f"Storing checkpoint (iter {i}) to {cfg.checkpoint_path}.")
            data_to_store = {
                "mesh_init_verts": mesh_init.verts_padded(),
                "mesh_init_faces": mesh_init.faces_padded(),
                "model": model.state_dict(),
                "optimizer": optim.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "stats": pickle.dumps(stats),
                "stats_sparse": pickle.dumps(stats_sparse),
                "finished_iter": i,
                "state_vars": state,
                "best_model_params": best_model_params,
                "best_model_mesh_init_verts": best_model_meshinit.verts_padded(),
                "best_model_mesh_init_faces": best_model_meshinit.faces_padded(),
                "torch_rng_state": torch.random.get_rng_state()
            }
            torch.save(data_to_store, cfg.checkpoint_path)
            if cfg.checkpoint_each_iter:
                fpath = add_suffix_to_path(cfg.checkpoint_path, f'_{i:08d}')
                logging.info(f"Copying checkpoint w/o stats (iter {i}) to {fpath}.")
                if os.path.isfile(fpath): logging.warn('Overwriting existing file')
                data_to_store = {k:v for k,v in data_to_store.items() if k not in ['stats', 'stats_sparse']}
                torch.save(data_to_store, fpath)


            logging.info(f"Storing mesh (iter {i}) to {cfg.mesh_checkpoint_path}.")
            save_mesh(cfg.mesh_checkpoint_path, model.build_mesh())
            if cfg.checkpoint_each_iter:
                fpath = add_suffix_to_path(cfg.mesh_checkpoint_path, f'_{i:08d}')
                logging.info(f"Copying mesh (iter {i}) to {fpath}.")
                if os.path.isfile(fpath): logging.warn('Overwriting existing file')
                copyfile(cfg.mesh_checkpoint_path, fpath)

    # Load best shape into model for final visualization
    model, optim, lr_scheduler = recreate_model_optim_scheduler(best_model_meshinit, model, optim, lr_scheduler)
    model.load_state_dict(best_model_params, strict=False)
    fpath = add_suffix_to_path(cfg.mesh_checkpoint_path, f'_final')
    logging.info(f"Storing mesh (final) to {fpath}.")
    save_mesh(fpath, model.build_mesh())

    logging.info('Benchmarking final shape, cameras')
    shape_metric_final = metric_utils.compare_meshes_align_notalign(model.build_mesh(), data.mesh_gt)
    camera_metric_final = model.compute_camera_metrics(per_camera=True, wrt_cam0 = cfg.cammetric_wrt_cam0)
    scalars = dict(
            camera_metrics = camera_metric_final,
            shape_metrics = shape_metric_final,
            shape_metric_final = shape_metric_final,
        )
    tb_visualizer.plot_current_scalars(scalars, cfg.num_iter)
    stats.update(flatten_dict(scalars), cfg.num_iter)

    logging.info(f'Visualizing final texture')
    # Visualize final shape
    tb_visualizer.display_current_results(model.get_current_visuals(), cfg.num_iter, save_meshes=True)
    tb_visualizer.display_current_results(model.get_360_visual(num_frames=cfg.viz.i360_num_frames), cfg.num_iter)
    if not cfg.benchmark_fast:
        tb_visualizer.display_current_results(model.get_shapemetric_visuals(num_frames=cfg.viz.i360_num_frames), cfg.num_iter)
    tb_visualizer.flush()
    return model

if __name__ == "__main__":
    print('pid', os.getpid())
    main()
