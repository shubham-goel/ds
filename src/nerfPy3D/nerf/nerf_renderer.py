# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
from typing import List, Optional, Tuple

import hydra
import torch
import torch.nn.functional as F
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch3d import renderer as pt3r
from pytorch3d.renderer.cameras import CamerasBase
from pytorch3d.structures import Pointclouds
from pytorch3d.structures.meshes import Meshes
from pytorch3d.vis.plotly_vis import plot_scene
from visdom import Visdom

from .implicit_function import NeuralRadianceField
from .raymarcher import EmissionAbsorptionNerfRaymarcher
from .raysampler import NerfRaysampler, ProbabilisticRaysampler
from .utils import calc_mse, calc_psnr, sample_images_at_mc_locs


class RadianceFieldRenderer(torch.nn.Module):

    def __init__(
        self,
        cfg_mask_loss : DictConfig,
        cfg_implicit : DictConfig,
        cfg_raysampler : DictConfig,
        image_size: Tuple[int, int],
        chunk_size_test: int,
        harmonic_xyz_omega0: float = 1.,
        use_single_network: bool = False,
        xyz_min: torch.FloatTensor = torch.tensor([-1,-1,-1], dtype=torch.float),
        xyz_max: torch.FloatTensor = torch.tensor([ 1, 1, 1], dtype=torch.float),
    ):

        super().__init__()
        self._renderer = torch.nn.ModuleDict()
        self._implicit_function = torch.nn.ModuleDict()

        self.cfg_mask_loss = cfg_mask_loss

        # Init the EA raymarcher used by both passes.
        raymarcher = EmissionAbsorptionNerfRaymarcher()

        # Parse out imge dimensions.
        image_height, image_width = image_size

        # Check for non-square image.
        if image_height == image_width:
            s_h = s_w = 1
        elif image_height > image_width:
            s_h = image_height / image_width
            s_w = 1
        else:
            s_h = 1
            s_w = image_width / image_height
        self.s_h = s_h
        self.s_w = s_w

        for render_pass in ("coarse", "fine"):
            if render_pass == "coarse":
                raysampler = NerfRaysampler(
                    min_x= -s_w + 1/image_width,
                    max_x=  s_w - 1/image_width,
                    min_y= -s_h + 1/image_height,
                    max_y=  s_h - 1/image_height,
                    image_height=image_height,
                    image_width=image_width,
                    **cfg_raysampler,
                )
            elif render_pass == "fine":
                raysampler = ProbabilisticRaysampler(
                    n_pts_per_ray=cfg_raysampler.n_pts_per_ray_fine,
                    stratified=cfg_raysampler.stratified,
                    stratified_test=cfg_raysampler.stratified_test,
                )
            else:
                raise ValueError(render_pass)

            self._renderer[render_pass] = pt3r.ImplicitRenderer(
                raysampler=raysampler,
                raymarcher=raymarcher,
            )

            if use_single_network and len(self._implicit_function)>0:
                self._implicit_function[render_pass] = list(self._implicit_function.values())[0]
            else:
                self._implicit_function[render_pass] = instantiate(
                    cfg_implicit,
                    harmonic_xyz_omega0=harmonic_xyz_omega0,
                    xyz_min = xyz_min,
                    xyz_max = xyz_max,
                )

        self._density_noise_std = cfg_implicit.density_noise_std
        self._chunk_size_test = chunk_size_test
        self._image_size = image_size

    def precache_rays(
        self,
        cache_cameras: List[CamerasBase],
        cache_camera_hashes: List,
        **kwargs,
    ):
        """
        Precaches the rays emitted from the list of cameras `cache_cameras`,
        where each camera is uniquely identified with the corresponding hash
        from `cache_camera_hashes`.

        The cached rays are moved to cpu and stored in
        `self._renderer['coarse']._ray_cache`.

        Raises `ValueError` when caching two cameras with the same hash.

        Args:
            cache_cameras: A list of `N` cameras for which the rays are pre-cached.
            cache_camera_hashes: A list of `N` unique identifiers for each
                camera from `cameras`.
        """
        self._renderer["coarse"].raysampler.precache_rays(
            cache_cameras,
            cache_camera_hashes,
            **kwargs,
        )

    def _process_ray_chunk(
        self,
        camera_hash,
        camera,
        image,
        chunk_idx,
        **kwargs
    ):
        coarse_ray_bundle = None
        coarse_weights = None

        for renderer_pass in ("coarse", "fine"):
            (rgb, weights), ray_bundle_out = self._renderer[renderer_pass](
                cameras=camera,
                volumetric_function=self._implicit_function[renderer_pass],
                chunksize=self._chunk_size_test,
                chunk_idx=chunk_idx,
                density_noise_std=(self._density_noise_std if self.training else 0.0),
                input_ray_bundle=coarse_ray_bundle,
                ray_weights=coarse_weights,
                camera_hash=camera_hash,
                **kwargs
            )

            alpha = weights.sum(dim=-1, keepdim=True)
            rgba = torch.cat((rgb, alpha), dim=-1)

            if renderer_pass == "coarse":
                rgba_coarse = rgba
                coarse_ray_bundle = ray_bundle_out
                coarse_weights = weights
                rgba_gt = sample_images_at_mc_locs(
                    image[..., :4],
                    ray_bundle_out.xys / torch.tensor([self.s_w, self.s_h],
                                                    dtype=ray_bundle_out.xys.dtype,
                                                    device=ray_bundle_out.xys.device),
                )

            elif renderer_pass == "fine":
                rgba_fine = rgba

            else:
                raise ValueError(renderer_pass)

        return {
            "rgba_fine": rgba_fine,
            "rgba_coarse": rgba_coarse,
            "rgba_gt": rgba_gt,
            # Store the coarse rays/weights only for visualization purposes.
            "coarse_ray_bundle": type(coarse_ray_bundle)(
                *[v.detach().cpu() for k, v in coarse_ray_bundle._asdict().items()]
            ),
            "coarse_weights": coarse_weights.detach().cpu(),
        }

    def forward(
        self,
        camera_hash,
        camera: CamerasBase,
        image: torch.Tensor,
        **kwargs
    ):
        if not self.training:
            # Full evaluation pass.
            n_chunks = self._renderer["coarse"].raysampler.get_n_chunks(
                self._chunk_size_test,
                camera.R.shape[0],
            )
        else:
            # MonteCarlo ray sampling.
            n_chunks = 1

        # Process the chunks of rays.
        chunk_outputs = [
            self._process_ray_chunk(
                camera_hash,
                camera,
                image,
                chunk_idx,
                **kwargs
            )
            for chunk_idx in range(n_chunks)
        ]

        if not self.training:
            # For a full render pass concatenate the output chunks,
            # and reshape to image size.
            out = {
                k: torch.cat(
                    [ch_o[k] for ch_o in chunk_outputs],
                    dim=1,
                ).view(-1, *self._image_size, 4)
                for k in ("rgba_fine", "rgba_coarse", "rgba_gt")
            }
        else:
            out = chunk_outputs[0]

        for render_pass in ("coarse", "fine", "gt"):
            out[f'rgb_{render_pass}'] = out[f'rgba_{render_pass}'][..., :3]
            out[f'alpha_{render_pass}'] = out[f'rgba_{render_pass}'][..., 3:]

        # Calc the error metrics.
        metrics = {}
        for render_pass in ("coarse", "fine"):
            for metric_name, metric_fun in zip(("mse", "psnr"), (calc_mse, calc_psnr)):
                metrics[f"{metric_name}_{render_pass}"] = metric_fun(
                    out["rgb_" + render_pass][..., :3],
                    out["rgb_gt"][..., :3],
                )

            metrics[f"mask_{render_pass}"] = hydra.utils.instantiate(
                    self.cfg_mask_loss,
                    out["alpha_" + render_pass][..., :3],
                    out["alpha_gt"][..., :3],
            )

        return out, metrics
