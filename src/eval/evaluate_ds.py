from __future__ import annotations

import copy
import logging
import os
from pathlib import Path

import hydra
import imageio
import numpy as np
import pytorch3d
import pytorch3d.io
import torch
from hydra.utils import instantiate, to_absolute_path
from omegaconf import DictConfig
from pytorch3d.renderer.cameras import CamerasBase
from pytorch3d.structures import Meshes
from visdom import Visdom

from ..nnutils import geom_utils
from ..utils import image as im_utils
from ..utils.mesh import (invert_RTs, transform_mesh)
from .evaluate_base import eval_base
from ..exp.ds import overfit_single


class eval_overfit_single(eval_base):
    def __init__(self, expdir, iter='latest', best=False, device='cuda') -> None:

        super().__init__(expdir, iter=iter, device=device)

        loaded_data = self.loaded_data
        cfg = self.cfg
        data = self.data

        # Create initial mesh
        if best:
            mesh_init = Meshes(loaded_data["best_model_mesh_init_verts"], loaded_data["best_model_mesh_init_faces"])
        else:
            mesh_init = Meshes(loaded_data["mesh_init_verts"], loaded_data["mesh_init_faces"])

        # Create texture model
        self.model : overfit_single = instantiate(
            cfg.texture_model,
            data,
            mesh_init,
            device=device,
            load_lpips=False,
            _recursive_=False
        )
        self.model = self.model.to(device)
        self.model = self.model.clone_with_new_initshape(mesh_init) # added to handle TexOptUV

        # Turn off gradient computation while evaluation
        for param in self.model.parameters():
            param.requires_grad = False

        # Read data from loaded model
        if best:
            missing_keys, unexpected_keys = self.model.load_state_dict(loaded_data["best_model_params"], strict=False)
        else:
            missing_keys, unexpected_keys = self.model.load_state_dict(loaded_data["model"], strict=False)
        for k in missing_keys:
            assert any([k.startswith(kk) for kk in [
                'texLpips_loss_fn',
            ]])
        # assert(len(missing_keys)==0)
        assert(len(unexpected_keys)==0)

        # Camera generators
        self.pred_camera_generator = self.model.camera_generator

    def get_checkpoint_path(self):
        return self.cfg.checkpoint_path

    def build_mesh(self):
        return self.model.build_mesh()

    def output_to_rgba(self, rend_out):
        return rend_out.rgba

    def render(self, cameras: CamerasBase, **kwargs):
        raster_settings = copy.copy(self.model.renderer.rasterizer.raster_settings)
        raster_settings.blur_radius = 1e-6
        blend_params = self.model.renderer.shader.blend_params._replace(sigma = 1e-6)

        return self.model.render_novel_cameras(cameras, use_cached=True,
                    raster_settings=raster_settings, blend_params=blend_params,
                    **kwargs
                )

    @torch.no_grad()
    def generate_360_visuals(self, out_dir_prefix='', num_frames=36, elevations=[-30,30]):
        cameras_list = self.model.camera_generator.create_cameras_list()
        mesh_centre, mesh_radius = geom_utils.get_centre_radius(self.data.train.rgba[:,3:,:,:], cameras_list)
        visuals_360 = self.model.get_360_visual(num_frames=num_frames, mesh_centre=mesh_centre, mesh_radius=mesh_radius, elevations=elevations)

        if out_dir_prefix:
            for k,v in visuals_360.video.items():
                out_file = f'{out_dir_prefix}{k.replace("/", "_")}.mp4'
                rgba_cont_np = torch.as_tensor(v).permute(0,2,3,1).numpy()
                imageio.mimwrite(out_file,
                                (255*np.clip(rgba_cont_np, 0, 1)).astype(np.uint8),
                                fps=int(num_frames/12), quality=8)

        return visuals_360

CONFIG_DIR='../../configs'
@hydra.main(config_path=CONFIG_DIR, config_name="eval")
def main(cfg: DictConfig):

    # Symlink stderr/stdout for ease
    logging.info(f'CWD {os.getcwd()}')

    logging.info(f'Loading model from {to_absolute_path(cfg.exp_dir)}')
    eval_model = eval_overfit_single(to_absolute_path(cfg.exp_dir), iter=cfg.iter, best=cfg.best)
    # shape_metrics = eval_model.compute_shape_metrics(align=cfg.align)
    # camera_metrics = eval_model.compute_camera_metrics(align=cfg.align)
    # image_metrics = eval_model.compute_image_metrics(align=cfg.align)

    # Create output directory
    out_dir = Path(to_absolute_path(cfg.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f'Extracting mesh to file...')
    mesh_pred = eval_model.build_mesh()
    align_c2w = invert_RTs(eval_model.get_alignment(align_scale_view0=True, use_icp=False))
    mesh_pred = transform_mesh(mesh_pred, align_c2w)
    pytorch3d.io.IO().save_mesh(mesh_pred, out_dir / f'{cfg.asin}_mesh_pred_noicp.obj')

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
    texss = []
    for el in [-30,0,30]:
        dns.append(visuals_360.video[f'cont_{el}/dn'])
        texss.append(visuals_360.video[f'cont_{el}/tex'])
    dns = np.concatenate(dns, axis=2)
    texss = np.concatenate(texss, axis=2)
    tex_dns = np.concatenate([texss, dns], axis=3)
    tex_dns = im_utils.change_img_format(tex_dns, 'NHWC')
    num_frames = tex_dns.shape[0]
    imageio.mimwrite(out_dir / f'{cfg.asin}_texDepthNormal.mp4',
                    (255*np.clip(tex_dns, 0, 1)).astype(np.uint8),
                    fps=int(num_frames/12), quality=8)


    logging.info(f'done')

if __name__ == "__main__":
    print('pid', os.getpid())
    main()
