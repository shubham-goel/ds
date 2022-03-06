from __future__ import annotations

import logging
import math
import pickle

import lpips
import omegaconf
import pytorch3d
import pytorch3d.io
import torch
from pytorch3d import transforms
from pytorch3d.ops.points_alignment import SimilarityTransform
from pytorch3d.renderer.cameras import CamerasBase
from pytorch3d.transforms import Transform3d
from pytorch3d.vis.plotly_vis import plot_scene
from visdom import Visdom

from ..nnutils.cameras import dollyParamCameras
from ..utils import metrics as metric_utils
from ..utils.align_meshes import align_depth
from ..utils.mesh import (RTs_to_transform, align_shapes, invert_RTs,
                          transform_cameras, transform_mesh, transform_to_RTs)
from ..utils.misc import add_suffix_to_path, apply_dict_rec, try_move_device


def find_alignment(mesh_pred, mesh_gt, pose_pred0, pose_gt0, use_icp=True, align_scale_view0=False, align_depth_kwargs={}, icp_type='g2p_uncentered',):

    # First align predcam0 with gtcam0
    pR,pT = pose_pred0.view(3, 4)[None, :,:3], pose_pred0.view(3, 4)[None, :,3]
    gR,gT = pose_gt0.view(3, 4)[None, :,:3], pose_gt0.view(3, 4)[None, :,3]

    pTr = RTs_to_transform(SimilarityTransform(pR.transpose(1,2),pT,1))
    gTr = RTs_to_transform(SimilarityTransform(gR.transpose(1,2),gT,1))

    if align_scale_view0:
        mesh_pred_view = transform_mesh(mesh_pred, pTr)
        mesh_gt_view = transform_mesh(mesh_gt, gTr)
        _, scalingTr_pred2gt = align_depth(mesh_pred_view, mesh_gt_view, **align_depth_kwargs)
        g2pTr = gTr.compose(scalingTr_pred2gt.inverse(), pTr.inverse())
    else:
        g2pTr = gTr.compose(pTr.inverse())

    if use_icp:
        logging.info(f'Aligning with ICP (type={icp_type})')
        assert icp_type in [
            'g2p_uncentered', 'g2p_centered', 'p2g_uncentered', 'p2g_centered',
            'g2p_noscale_uncentered', 'g2p_noscale_centered', 'p2g_noscale_uncentered', 'p2g_noscale_centered',
        ]

        mesh_gt = transform_mesh(mesh_gt, g2pTr)

        if icp_type.endswith('_centered'):
            # Centre both mesh_gt and mesh_pred to mesh_gt's centroid
            # First, find mesh radius from gt mesh, and create a centering transform
            verts = mesh_gt.verts_packed()
            mesh_vcen = (verts.min(0)[0] + verts.max(0)[0])/2
            mesh_radius = (verts-mesh_vcen).norm(dim=-1).max().item()
            centering_transform = Transform3d(device=mesh_gt.device).translate(-mesh_vcen[None]).scale(1/mesh_radius)

            # Then, transform accordingly.
            mesh_gt = transform_mesh(mesh_gt, centering_transform)
            mesh_pred = transform_mesh(mesh_pred, centering_transform)
        else:
            centering_transform = Transform3d(device=mesh_gt.device)

        estimate_scale = 'noscale' not in icp_type
        if icp_type.startswith('g2p'):
            align_w2c = align_shapes(mesh_gt, mesh_pred, estimate_scale=estimate_scale, verbose=True, num_samples=100_000)
        else:
            align_w2c = invert_RTs(align_shapes(mesh_pred, mesh_gt, estimate_scale=estimate_scale, verbose=True, num_samples=100_000))
        logging.debug(f'Alignment: {align_w2c}')

        if not (1e-3 <= align_w2c.s <= 1e3):
            logging.warning(f'ICP failed (scale {align_w2c.s}), using identity')
            device = align_w2c.R.device
            align_w2c = SimilarityTransform(
                torch.eye(3, device=device, dtype=torch.float)[None],
                torch.zeros(3, device=device, dtype=torch.float)[None],
                torch.ones(1, device=device, dtype=torch.float),
            )
        tr_w2c = g2pTr.compose(centering_transform, RTs_to_transform(align_w2c), centering_transform.inverse())
    else:
        tr_w2c = g2pTr

    align_w2c = transform_to_RTs(tr_w2c)
    return align_w2c


class eval_base():
    def newpath(self, path):
        if path[0]=='/':
            return path
        else:
            return f'{self.expdir}/{path}'

    def get_checkpoint_path(self):
        raise NotImplementedError

    def load_checkpoint(self, iter):
        chkpt_path = self.newpath(self.get_checkpoint_path())
        if iter=='latest':
            pass # No suffix
        else:
            chkpt_path = add_suffix_to_path(chkpt_path, f'_{iter:08d}')
        loaded_data = torch.load(chkpt_path)
        return loaded_data

    def __init__(self, expdir, iter='latest', device='cuda') -> None:

        # Load config
        self.expdir = expdir
        self.cfg = cfg = omegaconf.OmegaConf.load(self.newpath('.hydra/config.yaml'))

        # Load input data
        with open(self.newpath(cfg.init_data_path), 'rb') as handle:
            self.data = data = pickle.load(handle)
        self.data = apply_dict_rec(self.data, fv = lambda x: try_move_device(x, device))

        # Load checkpoint
        self.loaded_data = self.load_checkpoint(iter)

        # Camera generators
        self.pred_camera_generator: dollyParamCameras
        self.gt_camera_generator = dollyParamCameras(
            data.train.poses_gt, data.train.hfovs_gt, optimize_cam=False, device=device
        )
        self.val_camera_generator = dollyParamCameras(
            data.val.poses, data.val.hfovs, optimize_cam=False, device=device
        )

        # Misc
        self.iter = iter
        self.device = device
        self.aligned = False
        self.align_w2c = transforms.Transform3d()
        self._lpips_fn = None    # populate when needed

    @property
    def lpips_fn(self):
        if self._lpips_fn is None:
            self._lpips_fn = lpips.LPIPS(net='alex').to(self.device)
        return self._lpips_fn

    def finished_iters(self):
        return self.loaded_data["finished_iter"] + 1

    def build_mesh(self):
        raise NotImplementedError

    @torch.no_grad()
    def get_alignment(self, use_icp=True, align_scale_view0=False, align_depth_kwargs={}, icp_type='g2p_uncentered',):
        if not self.aligned:
            # First align predcam0 with gtcam0
            pR,pT,_,_ = self.pred_camera_generator.get_RTfovF(id=0)
            gR,gT,_,_ = self.gt_camera_generator.get_RTfovF(id=0)
            p_pose = torch.cat([pR, pT[:,:,None]], dim=2)
            g_pose = torch.cat([gR, gT[:,:,None]], dim=2)

            mesh_pred = self.build_mesh().to(self.device)
            mesh_gt = self.data.mesh_gt

            self.align_w2c = find_alignment(mesh_pred, mesh_gt, p_pose, g_pose,
                                use_icp=use_icp, align_scale_view0=align_scale_view0,
                                icp_type=icp_type, **align_depth_kwargs)
            self.aligned = True
        return self.align_w2c

    @torch.no_grad()
    def compute_shape_metrics(self, align=True, uni_chamfer=False, **align_kwargs):
        mesh_pred = self.build_mesh()
        mesh_gt = self.data.mesh_gt

        # Aligning pred shape to gt
        if align:
            align_c2w = invert_RTs(self.get_alignment(**align_kwargs))
            mesh_pred = transform_mesh(mesh_pred, align_c2w)

        shape_metrics = metric_utils.compare_meshes(mesh_pred, mesh_gt, align_icp=False,
                            num_samples=100_000, return_per_point_metrics=uni_chamfer)

        if uni_chamfer:
            shape_metrics, (_,_,metrics_p2g), (_,_,metrics_g2p) = shape_metrics
            shape_metrics.update({
                "Chamfer-L2-p2g": metrics_p2g["Chamfer-L2"].mean().item(),
                "Chamfer-L2-g2p": metrics_g2p["Chamfer-L2"].mean().item(),
            })

        return shape_metrics


    @torch.no_grad()
    def compute_camera_metrics(self, align=True, **align_kwargs):
        cam_pred = self.pred_camera_generator.create_cameras()
        cam_gt = self.gt_camera_generator.create_cameras()
        centre = self.data.mesh_centre

        # Aligning cameras
        if align:
            align_c2w = invert_RTs(self.get_alignment(**align_kwargs))
            cam_pred = transform_cameras(cam_pred, align_c2w)

        camera_metrics = metric_utils.compare_cameras(cam_pred, cam_gt, centre=centre, per_camera=True)

        return camera_metrics

    def render(self, cameras: CamerasBase, **kwargs):
        raise NotImplementedError


    def output_to_rgba(self, rend_out):
        raise NotImplementedError

    @torch.no_grad()
    def visualize(self, viz:Visdom, align=True, gt_cam=False, init_cam=True):
        """
            Visualize predicted/GT cameras, pred mesh, rendered images
        """
        mesh_pred = self.build_mesh()
        cam_pred_list = self.pred_camera_generator.create_cameras_list()
        cam_gt_list = self.gt_camera_generator.create_cameras_list()
        cam_init_list = dollyParamCameras(self.data.train.poses, self.data.train.hfovs).create_cameras_list()
        centre = self.data.mesh_centre

        # Render training images (before aligning cameras)
        rend_rgba = []
        assert(len(cam_pred_list) == len(self.data.train.rgba))
        for camera in cam_pred_list:
            rend_out = self.render(camera)
            rend_rgba.append(rend_out.rgba)
        rend_rgba = torch.cat(rend_rgba, dim=0)

        # Align pred cameras to gt
        if align:
            align_c2w = invert_RTs(self.get_alignment(align_scale_view0=True, use_icp=False))
            cam_pred_list = [transform_cameras(c, align_c2w) for c in cam_pred_list]
            mesh_pred = transform_mesh(mesh_pred, align_c2w)

        if viz is not None:
            # Visualize images
            N = len(rend_rgba)
            img_kwargs = dict(
                nrow=round(math.sqrt(N)),
                opts=dict(width = 800, height = 400, title="input | rend")
            )
            inp_rend_rgba = torch.cat([self.data.train.rgba,rend_rgba], dim=3)
            viz.images(inp_rend_rgba[:,:3,:,:]*255, win="input | rend", **img_kwargs)

            # Visualize cameras and shape
            shape_trace = {'mesh_pred': mesh_pred.cpu()}
            pred_camera_trace = {f'pred_cam_{i:03d}': c.to('cpu') for i,c in enumerate(cam_pred_list)}
            def plot_shape_cams(strace, ctrace, win='scene'):
                plotly_plot = plot_scene({
                        win: {**strace, **ctrace,},
                    },
                    camera_scale = 0.05,
                )
                viz.plotlyplot(plotly_plot, win=win)
            plot_shape_cams(shape_trace, pred_camera_trace, 'pred cam scene')
            if gt_cam:
                gt_camera_trace = {f'gt_cam_{i:03d}': c.to('cpu') for i,c in enumerate(cam_gt_list)}
                plot_shape_cams(shape_trace, gt_camera_trace, 'gt cam scene')
            if init_cam:
                init_camera_trace = {f'init_cam_{i:03d}': c.to('cpu') for i,c in enumerate(cam_init_list)}
                plot_shape_cams(shape_trace, init_camera_trace, 'init cam scene')

        return dict(
            rend_rgba=rend_rgba,
        )

    @torch.no_grad()
    def generate_360_visuals(self, out_dir_prefix='', num_frames=36, elevations=[-30,30]):
        raise NotImplementedError
