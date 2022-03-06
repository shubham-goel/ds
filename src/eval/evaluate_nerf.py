from __future__ import annotations

import itertools
import logging
import os
from pathlib import Path

import hydra
import imageio
import numpy as np
import pytorch3d
import pytorch3d.io
import torch
from dotmap import DotMap
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from pytorch3d.renderer.cameras import CamerasBase
from pytorch3d.structures import Meshes
from visdom import Visdom

from ..nerfPy3D.nerf.nerf_renderer import RadianceFieldRenderer
from ..nerfPy3D.train_wrapper import NeRF_to_mesh, show_rotating_nerf
from ..nnutils.cameras import dollyParamCameras
from ..utils import image as im_utils
from ..utils.mesh import decimate_mesh, save_mesh
from .evaluate_base import eval_base

EMPTY_MESH = Meshes(torch.zeros((1,0,3)).float(), torch.zeros((1,0,3)).long())
class eval_nerf(eval_base):
    def __init__(self, expdir, iter='latest', device='cuda') -> None:

        super().__init__(expdir, iter=iter, device=device)

        loaded_data = self.loaded_data
        cfg = self.cfg
        data = self.data
        nerf_cfg = self.cfg.train

        ############################################
        ### Follow Nerf training code to initialize models and lod from checkpoint
        ############################################

        # Initialize camera generators
        self.pred_camera_generator = dollyParamCameras(data.train.poses, data.train.hfovs,
                                centre=data.mesh_centre,
                                param_R_axis_angle=nerf_cfg.param_R_axis_angle,
                                optimize_cam=nerf_cfg.optimize_cam,
                                optimize_first=nerf_cfg.optimize_first,
                                optimize_R=nerf_cfg.optimize_R,
                                optimize_T=nerf_cfg.optimize_T,
                                optimize_Fov=nerf_cfg.optimize_Fov,
                                device=device
                            )

        # Original code source requires channel dimension to be at the end
        train_rgba = data.train.rgba.permute(0,2,3,1)
        val_rgba = data.val.rgba.permute(0,2,3,1)

        # Instantiate the radiance field model.
        xyz_min = torch.full((3,), -1.1 * data.mesh_radius).float().to(device) + data.mesh_centre
        xyz_max = torch.full((3,),  1.1 * data.mesh_radius).float().to(device) + data.mesh_centre
        self.neural_radiance_field_model = RadianceFieldRenderer(
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
        self.neural_radiance_field_model.to(device)
        # Turn off gradient computation while evaluation
        for param in itertools.chain(self.neural_radiance_field_model.parameters(),
                                    self.pred_camera_generator.parameters()):
            param.requires_grad = False

        # Read data from loaded model
        self.neural_radiance_field_model.load_state_dict(loaded_data["model"])
        self.pred_camera_generator.load_state_dict(loaded_data["camera_params"])

        # Turn on eval mode.
        self.neural_radiance_field_model.eval()

        # Misc
        self._mesh = None
        self.dummy_rgba = torch.zeros(self.neural_radiance_field_model._image_size + (4,), device=device)

    def get_checkpoint_path(self):
        return self.cfg.train.checkpoint_path

    def build_mesh(self, voxel_res=256, voxel_thresh=50):
        if self._mesh is None:
            # TODO: Extract mesh from Nerf model

            mesh_file = self.newpath(f'___mesh_{self.finished_iters()}.obj')
            mesh_file_isinvalid = mesh_file + '.invalid'
            if mesh_file is not None and Path(mesh_file).is_file():
                logging.info(f'Loading mesh from {mesh_file}')
                f_mesh = pytorch3d.io.IO().load_mesh(mesh_file).to(self.device)
                # f_mesh, _, _ = load_mesh_from_file(mesh_file, device=self.device)
                self._mesh = f_mesh
                return self._mesh

            if Path(mesh_file_isinvalid).is_file():
                logging.info(f'Found empty token from {mesh_file_isinvalid}. Returning empty mesh')
                return EMPTY_MESH

            # Extract only fine meshes
            xyz_min = torch.full((3,), -1.1 * self.data.mesh_radius).to(self.device) + self.data.mesh_centre
            xyz_max = torch.full((3,),  1.1 * self.data.mesh_radius).to(self.device) + self.data.mesh_centre
            # c_mesh = NeRF_to_mesh(
            #     self.neural_radiance_field_model._implicit_function['coarse'],
            #     xyz_min=xyz_min,
            #     xyz_max=xyz_max,
            #     voxel_res=voxel_res,
            #     voxel_thresh=voxel_thresh,
            # )
            f_mesh = NeRF_to_mesh(
                self.neural_radiance_field_model._implicit_function['fine'],
                xyz_min=xyz_min,
                xyz_max=xyz_max,
                voxel_res=voxel_res,
                voxel_thresh=voxel_thresh,
            )

            # Decimate meshes down to a million faces if they're too high res
            def is_valid(m: Meshes) -> bool: return len(m.num_verts_per_mesh())>0
            # if is_valid(c_mesh): c_mesh = decimate_mesh(c_mesh, numF_target=int(1e6))
            if is_valid(f_mesh): f_mesh = decimate_mesh(f_mesh, numF_target=int(1e6))

            # Save mesh for future use
            if mesh_file is not None:
                if is_valid(f_mesh):
                    logging.info(f'Saving mesh to {mesh_file}')
                    save_mesh(mesh_file, f_mesh)
                else:
                    logging.info(f'Saving empty token to {mesh_file_isinvalid}')
                    Path(mesh_file_isinvalid).touch()

            self._mesh = f_mesh

        return self._mesh

    def render(self, cameras: CamerasBase, near = None, far = None, **kwargs):
        if near is None:
            near = self.data.train.near.min()[None]
        if far is None:
            far = self.data.train.far.max()[None]

        # breakpoint()
        val_nerf_out, val_metrics = self.neural_radiance_field_model(
            None,
            cameras,
            self.dummy_rgba[None],
            min_depth = near,
            max_depth = far,
            mode = "val",
        )

        ## Convert to DotMap and add 'rgba' key to use self.visualize(...)
        val_nerf_out = DotMap(val_nerf_out, _dynamic=False)
        val_nerf_out.update(rgba=im_utils.change_img_format(val_nerf_out.rgba_fine, "BCHW", inp_format="BHWC"))
        return val_nerf_out

    def output_to_rgba(self, rend_out):
        return rend_out['rgba_fine'].permute(0,3,1,2)

    @torch.no_grad()
    def generate_360_visuals(self, out_dir_prefix='', num_frames=36, elevations=[-30,30]):

        hfov = 25 *np.pi/180
        dist = self.data.mesh_radius / np.sin(hfov)     # r = d sin(hfov)
        min_depth = dist - self.data.mesh_radius
        max_depth = dist + self.data.mesh_radius
        visuals_360 = show_rotating_nerf(self.neural_radiance_field_model,
            hfov, dist, elevations, self.data.mesh_centre,
            None, 0,
            num_frames=num_frames,
            device=self.device,
            min_depth = min_depth,
            max_depth = max_depth,
            write_to_file=False
        )

        if out_dir_prefix:
            raise NotImplementedError

        return visuals_360

CONFIG_DIR='../../configs'
@hydra.main(config_path=CONFIG_DIR, config_name="eval")
def main(cfg: DictConfig):

    # Symlink stderr/stdout for ease
    logging.info(f'CWD {os.getcwd()}')

    logging.info(f'Loading model from {to_absolute_path(cfg.exp_dir)}')
    eval_model = eval_nerf(to_absolute_path(cfg.exp_dir), iter=cfg.iter)
    # shape_metrics = eval_model.compute_shape_metrics(align=cfg.align)
    # camera_metrics = eval_model.compute_camera_metrics(align=cfg.align)
    # image_metrics = eval_model.compute_image_metrics(align=cfg.align)
    # logging.info(f'done w/ metrics')

    # Create output directory
    out_dir = Path(to_absolute_path(cfg.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f'Visualizing to Visdom...')
    if cfg.viz.server is not None:
        viz = Visdom(
            server=cfg.viz.server,
            port=cfg.viz.port,
            env=cfg.viz.env
        )
        viz.text(cfg.asin, win='asin', opts=dict(width = 100, height = 20, title="asin"))
        viz.text(cfg.exp_dir, win='expdir', opts=dict(width = 100, height = 20, title="expdir"))
    else:
        logging.info(f'Skipping visdom connection because server=None')
        viz = None
    rend_out = eval_model.visualize(viz, align=cfg.align)

    logging.info(f'Saving renderings...')
    # Save rgba gt|rend
    inp_rend_rgba = torch.cat([eval_model.data.train.rgba,rend_out['rend_rgba']], dim=3)
    inp_rend_rgba_grid = im_utils.change_img_format(im_utils.concat_images_grid(inp_rend_rgba), 'HWC')
    imageio.imwrite(out_dir/f'{cfg.asin}_gt_rend.png', (255*np.clip(inp_rend_rgba_grid.cpu().numpy(), 0, 1)).astype(np.uint8))

    logging.info(f'360 Visualizations...')
    # Save 360-visualizations
    visuals_360 = eval_model.generate_360_visuals(elevations=[-30,0,30])
    dns = []
    coarse_texss = []
    fine_texss = []
    for el in [-30,0,30]:
        coarse_texss.append(visuals_360[f'cont_{el}_coarse/tex'])
        fine_texss.append(visuals_360[f'cont_{el}_fine/tex'])
    coarse_texss = np.concatenate(coarse_texss, axis=2)
    fine_texss = np.concatenate(fine_texss, axis=2)
    tex_dns = np.concatenate([coarse_texss, fine_texss], axis=3)
    tex_dns = im_utils.change_img_format(tex_dns, 'NHWC')
    num_frames = tex_dns.shape[0]
    out_file = out_dir / f'{cfg.asin}_texCoarseFine.mp4'
    imageio.mimwrite(out_file,
                    (255*np.clip(tex_dns, 0, 1)).astype(np.uint8),
                    fps=int(num_frames/12), quality=8)

    logging.info(f'done')

if __name__ == "__main__":
    print('pid', os.getpid())
    main()
