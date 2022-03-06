import sys
import traceback
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from dotmap import DotMap
from omegaconf.dictconfig import DictConfig
from pytorch3d.renderer import (BlendParams, MeshRasterizer, MeshRenderer,
                                PointsRasterizationSettings, PointsRasterizer,
                                RasterizationSettings)
from pytorch3d.renderer.blending import BlendParams
from pytorch3d.renderer.cameras import CamerasBase
from pytorch3d.renderer.mesh.rasterizer import Fragments
from pytorch3d.renderer.mesh.utils import interpolate_face_attributes
from pytorch3d.structures import Meshes
from hydra.utils import instantiate

from .cameras import dollyParamCameras

# Epsilon for all denomiantors in transform_points()
TRANS_EPS = 1e-4

# Shaders
def softmax_ccc_blend(
    colors, fragments, blend_params,
    znear: float = 1.0, zfar: float = 100,
    return_alpha: bool = True, return_all: bool = False,
) -> torch.Tensor:
    """
    Generalizes pytorch3d.renderer.blending.softmax_rgb_blend to c-channel colors
    """
    C = colors.shape[-1]
    N, H, W, K = fragments.pix_to_face.shape
    device = fragments.pix_to_face.device
    CO = (C+1 if return_alpha else C)
    pixel_colors = torch.ones((N, H, W, CO), dtype=colors.dtype, device=colors.device)
    background = blend_params.background_color
    if not torch.is_tensor(background):
        background = torch.tensor(background, dtype=torch.float32, device=device)

    # Weight for background color
    eps = 1e-10

    # Mask for padded pixels.
    mask = fragments.pix_to_face >= 0

    # Sigmoid probability map based on the distance of the pixel to the face.
    prob_map = torch.sigmoid(-fragments.dists / blend_params.sigma) * mask

    # The cumulative product ensures that alpha will be 0.0 if at least 1
    # face fully covers the pixel as for that face, prob will be 1.0.
    # This results in a multiplication by 0.0 because of the (1.0 - prob)
    # term. Therefore 1.0 - alpha will be 1.0.
    alpha = torch.prod((1.0 - prob_map), dim=-1)

    # Weights for each face. Adjust the exponential by the max z to prevent
    # overflow. zbuf shape (N, H, W, K), find max over K.
    # TODO: there may still be some instability in the exponent calculation.

    if not ((zfar >= fragments.zbuf).all()):
        print("zfar too small")
        import ipdb; ipdb.set_trace()
    z_inv = (zfar - fragments.zbuf) / (zfar - znear) * mask
    # pyre-fixme[16]: `Tuple` has no attribute `values`.
    # pyre-fixme[6]: Expected `Tensor` for 1st param but got `float`.
    z_inv_max = torch.max(z_inv, dim=-1).values[..., None].clamp(min=eps)
    # pyre-fixme[6]: Expected `Tensor` for 1st param but got `float`.
    weights_num = prob_map * torch.exp((z_inv - z_inv_max) / blend_params.gamma)

    # Also apply exp normalize trick for the background color weight.
    # Clamp to ensure delta is never 0.
    # pyre-fixme[20]: Argument `max` expected.
    # pyre-fixme[6]: Expected `Tensor` for 1st param but got `float`.
    delta = torch.exp((eps - z_inv_max) / blend_params.gamma).clamp(min=eps)

    # Normalize weights.
    # weights_num shape: (N, H, W, K). Sum over K and divide through by the sum.
    denom = weights_num.sum(dim=-1)[..., None] + delta

    # Sum: weights * textures + background color
    weighted_colors = (weights_num[..., None] * colors).sum(dim=-2)
    weighted_background = delta * background
    pixel_colors[..., :C] = (weighted_colors + weighted_background) / denom
    if return_alpha:
        pixel_colors[..., C] = 1.0 - alpha

    if not return_all:
        return pixel_colors
    else:
        return (pixel_colors, alpha, prob_map, z_inv, z_inv_max, weights_num, delta, denom)

class SimpleShader(torch.nn.Module):
    def __init__(
        self, cameras=None, blend_params=None, return_alpha=True, return_weights=False,
    ):
        super().__init__()
        self.cameras = cameras
        self.return_alpha = return_alpha
        self.return_weights = return_weights
        self.blend_params = blend_params if blend_params is not None else BlendParams()

    def get_texels(self, fragments, meshes, **kwargs) -> torch.Tensor:
        # Child class must define texture sampling method
        raise NotImplementedError

    def forward(self, fragments, meshes, **kwargs) -> torch.Tensor:
        cameras = kwargs.get("cameras", self.cameras)
        if cameras is None:
            msg = "Cameras must be specified either at initialization \
                or in the forward pass of SimpleTextureShader"
            raise ValueError(msg)

        blend_params = kwargs.get("blend_params", self.blend_params)
        return_alpha = kwargs.get("return_alpha", self.return_alpha)
        znear = kwargs.get("znear", getattr(cameras, "znear", 1.0))
        zfar = kwargs.get("zfar", getattr(cameras, "zfar", 100.0))

        texels = self.get_texels(fragments, meshes, **kwargs)

        if not self.return_weights:
            images = softmax_ccc_blend(
                texels, fragments, blend_params,
                znear=znear, zfar=zfar, return_alpha=return_alpha
            )
            return images
        else:
            (images, _, _, _, _, weights_num, _, _) = softmax_ccc_blend(
                texels, fragments, blend_params,
                znear=znear, zfar=zfar, return_alpha=return_alpha, return_all=True
            )
            return images, weights_num

class SimpleDepthShader(SimpleShader):
    def get_texels(self, fragments, meshes, **kwargs) -> torch.Tensor:
        return fragments.zbuf[:,:,:,:,None]

class SimpleNormalShader(SimpleShader):
    def get_texels(self, fragments, meshes,
            world_coordinates=False, **kwargs
        ) -> torch.Tensor:
        """ If world_coordinates is False, transform normals to camera space"""
        if len(meshes) != len(fragments.zbuf): raise ValueError
        normals_pd = meshes.verts_normals_padded()
        if not world_coordinates:
            cameras = kwargs.get('cameras', self.cameras)
            if cameras is None:
                msg = "Cameras must be specified either at initialization \
                    or in the forward pass of SimpleNormalShader when \
                    world_coordinates = False"
                raise ValueError(msg)
            w2v = cameras.get_world_to_view_transform()
            normals_pd = w2v.transform_normals(normals_pd)
        pd2pk = meshes.verts_padded_to_packed_idx()
        normals_pk = normals_pd.reshape(-1, 3)[pd2pk]
        fnormals_pk = normals_pk[meshes.faces_packed()]
        fnormals_texels = interpolate_face_attributes(
            fragments.pix_to_face, fragments.bary_coords, fnormals_pk
        )
        fnormals_texels = F.normalize(fnormals_texels, dim=-1)
        return fnormals_texels

class SimpleTextureShader(SimpleShader):
    def get_texels(self, fragments:Fragments, meshes:Meshes, **kwargs) -> torch.Tensor:
        try:
            texels = meshes.sample_textures(fragments)
        except RuntimeError:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            print("*** print_tb:")
            traceback.print_tb(exc_traceback, limit=1, file=sys.stdout)
            print("*** print_exception:")
            # exc_type below is ignored on 3.5 and later
            traceback.print_exception(exc_type, exc_value, exc_traceback,
                                    limit=2, file=sys.stdout)

            import ipdb; ipdb.set_trace()
            texels = meshes.sample_textures(fragments)
        return texels

class SimpleTextureShader_RGBA_pxy(SimpleTextureShader):
    def __init__(self, cameras=None, blend_params=None, return_alpha=True):
        super().__init__(cameras, blend_params, return_alpha, return_weights=True)

    def forward(self, fragments, meshes, **kwargs) -> torch.Tensor:
        rgba, weights_num = super().forward(fragments, meshes, **kwargs)

        ## Rest is pixel_xy computation
        # Find XYZ positions of ray-intersections as pixel_XYZ: (Nc, H, W, K, 3)
        Nc, H, W, K = fragments.pix_to_face.shape
        # meshes = meshes_world.extend(len(cameras))
        faces = meshes.faces_packed()   # nF,3
        verts = meshes.verts_packed()   # nV,3
        faces_verts = verts[faces]
        pixel_XYZ = interpolate_face_attributes(
            fragments.pix_to_face, fragments.bary_coords, faces_verts
        )

        # Compute weighted average of projected pixel_XYZ for maskbidt loss
        weights_num = weights_num/(weights_num.sum(dim=-1,keepdim=True) + 1e-12)

        cameras = kwargs.get('cameras', self.cameras)
        pixel_xy = cameras.transform_points(pixel_XYZ.view(Nc, H*W*K, 3), eps=TRANS_EPS)[:,:,:2]
        pixel_xy = pixel_xy.view(Nc, H, W, K, 2)
        pixel_xy = (weights_num[..., None] * pixel_xy).sum(dim=-2)      # Nc,H,W,2
        return DotMap(
            rgba = rgba,
            rgba_pxy=pixel_xy,
            _dynamic=False
        )

class TextureSampleShader(torch.nn.Module):
    """
    Sample texture by projecting vertices onto img0
    texture_images: NxCxHxW gt images to project vertices and sample texture from
    rasterizer: rasterizer for transforming vertices to screen space
    """
    def __init__(
        self, texture_sampler: DictConfig,
        view_dep_normals=True, smooth_normals=True, sample_from_target=False,
        device="cpu", cameras=None, blend_params=None,
        texture_images=None, texture_cameras=None, texture_fragments=None,
        rgba_pxy_weighted=True,
        rasterizer=None,
    ):
        super().__init__()
        self.cameras = cameras
        # self.view_dep_normals = view_dep_normals
        # self.smooth_normals = smooth_normals
        # self.sample_from_target = sample_from_target
        # self.rgba_pxy_weighted = rgba_pxy_weighted
        self.blend_params = blend_params if blend_params is not None else BlendParams()
        self.texture_images = texture_images.to(device) if texture_images is not None else None
        self.texture_cameras = texture_cameras.to(device) if texture_cameras is not None else None
        self.texture_fragments = texture_fragments.to(device) if texture_fragments is not None else None
        self.rasterizer = rasterizer
        self.tex_sampler_cfg = texture_sampler

    def forward(self, fragments, meshes, return_weights=False, sample_from_target=True, **kwargs) -> torch.Tensor:
        """
        cameras: cameras views to render
        fragments: rasterizing out from these cameras
        texture_images: images to sample texture from
        texture_cameras: cameras for corresponding texture_images
        texture_fragments: rasterizing out from texture_cameras (used for depth/visibility)
        """
        cameras: CamerasBase = kwargs.get("cameras", self.cameras)
        texture_images = kwargs.get("texture_images", self.texture_images)
        texture_cameras = kwargs.get("texture_cameras", self.texture_cameras)
        texture_fragments = kwargs.get("texture_fragments", self.texture_fragments)
        assert(cameras is not None)
        assert(texture_images is not None)
        assert(texture_cameras is not None)
        assert(texture_fragments is not None)

        blend_params = kwargs.get("blend_params", self.blend_params)
        znear = kwargs.get("znear", getattr(cameras, "znear", 1.0))
        zfar = kwargs.get("zfar", getattr(cameras, "zfar", 100.0))
        device = fragments.pix_to_face.device

        assert(len(cameras) == fragments.zbuf.shape[0])
        assert(len(texture_images) == len(texture_cameras))

        # Find XYZ positions of ray-intersections as pixel_XYZ: (Nc, H, W, K, 3)
        Nc, H, W, K = fragments.pix_to_face.shape
        faces = meshes.faces_packed()   # nF,3
        verts = meshes.verts_packed()   # nV,3
        faces_verts = verts[faces]
        pixel_XYZ = interpolate_face_attributes(
            fragments.pix_to_face, fragments.bary_coords, faces_verts
        )
        # Face normals at these ray-intersections as pixel_normals: (Nc, H, W, K, 3)
        pixel_normals = SimpleNormalShader().get_texels(fragments, meshes, world_coordinates=True)

        # Flip normals to have negative dot product with line joining camera-centre to pixel
        camera_centres = cameras.get_camera_center()    # Nc,3
        f_to_meshid = meshes.faces_packed_to_mesh_idx() # nF
        camera_centres = camera_centres[f_to_meshid]    # nF,3
        camera_centres = camera_centres[fragments.pix_to_face]  # Nc,H,W,K,3
        pix_to_cam = F.normalize(camera_centres - pixel_XYZ, dim=-1)   # Nc,H,W,K,3
        pix_to_cam_dot_n = (pixel_normals * pix_to_cam).sum(dim=-1, keepdims=True)    # Nc,H,W,K,1
        pixel_normals = torch.where(pix_to_cam_dot_n >= 0, 1, -1) * pixel_normals

        # Don't sample from target image, only if texture_cameras == cameras
        Nt = len(texture_cameras)
        if not sample_from_target:
            assert texture_cameras == cameras
            can_sample_mask_flat = ~torch.eye(Nc, dtype=torch.bool, device=device)
            can_sample_mask_flat = can_sample_mask_flat[...,None].expand(-1,-1,H*W*K) # Nt,Nc,H,W,K
            can_sample_mask_flat = can_sample_mask_flat.reshape(Nt, Nc*H*W*K) # Nt,Nc*H*W*K
        else:
            can_sample_mask_flat = torch.ones((Nt, Nc*H*W*K), dtype=torch.bool, device=device)

        # Prune out invalid points (non-intersections)
        mask = (fragments.pix_to_face >= 0).view(-1)
        pixel_XYZ_flat = pixel_XYZ.view(-1, 3)
        pixel_normals_flat = pixel_normals.view(-1, 3)
        pixel_XYZ_pruned = pixel_XYZ_flat[mask, :]
        pixel_normals_pruned = pixel_normals_flat[mask, :]
        can_sample_mask_pruned = can_sample_mask_flat[:, mask]

        # Estimate color of 3d points pixel_XYZ with normals pixel_normals
        # by projecting onto each image, and sampling color from most head-on image.
        # pixel_pickidx_pruned tells which image the color was picked from
        texel_colors_pruned, texel_valid_pruned, texel_weights_pruned = sample_texture_3dpoints(
            pixel_XYZ_pruned,
            pixel_normals_pruned,
            texture_images,
            texture_cameras,
            texture_fragments,
            can_sample_mask_pruned,
            blend_params=blend_params,
            znear=znear,
            zfar=zfar,
            **self.tex_sampler_cfg,
        )
        assert(texel_colors_pruned.isfinite().all())
        assert(texel_colors_pruned.shape[0] == mask.sum())

        # Reshape pixel_colors into texels
        Nt = len(texture_cameras)
        texel_colors_flat = torch.zeros((Nc*H*W*K, 3), dtype=torch.float, device=device)
        texel_valid_flat = torch.zeros((Nc*H*W*K, 1), dtype=torch.bool, device=device)
        texel_weights_flat = torch.zeros((Nt, Nc*H*W*K), dtype=torch.float, device=device)
        texel_colors_flat[mask, :] = texel_colors_pruned
        texel_valid_flat[mask, :] = texel_valid_pruned
        texel_weights_flat[:, mask] = texel_weights_pruned
        texel_colors = texel_colors_flat.view(Nc, H, W, K, 3)
        texel_valid = texel_valid_flat.view(Nc, H, W, K, 1)
        texel_weights = texel_weights_flat.view(Nt, Nc, H, W, K)

        # Blend texels into image.
        (images, _, _, _, _, weights_num, _, _) = softmax_ccc_blend(
            texel_colors, fragments, blend_params, znear=znear, zfar=zfar, return_all=True
        )

        # Compute weighted average of projected pixel_XYZ for maskbidt loss
        weights_num = weights_num/(weights_num.sum(dim=-1,keepdim=True) + 1e-12)
        pixel_xy = cameras.transform_points(pixel_XYZ.view(Nc, H*W*K, 3), eps=TRANS_EPS)[:,:,:2]
        pixel_xy = pixel_xy.view(Nc, H, W, K, 2)
        pixel_xy = (weights_num[..., None] * pixel_xy).sum(dim=-2)      # Nc,H,W,2

        with torch.no_grad():
            images_valid = softmax_ccc_blend(
                texel_valid.float(), fragments,
                blend_params._replace(background_color=0),
                znear=znear, zfar=zfar, return_alpha=False
            )

        output = DotMap(rgba=images, rgb_valid=images_valid, rgba_pxy=pixel_xy, _dynamic=False)
        if return_weights:
            # Render texel_weights into an image (Nc,H,W,Nt) containing
            # probabilities of picking texture from different target images
            weights_img = softmax_ccc_blend(
                texel_weights.permute(1,2,3,4,0), fragments,
                blend_params._replace(background_color=1),
                znear=znear, zfar=zfar, return_alpha=False,
            )
            output.update(prob_weights=weights_img)
        return output

def grid_sample_ndc(input, grid_ndc, *args, return_whether_in_bounds = False, **kwargs):
    """ input: NxCxHxW
        grid_ndc: NxAxBx2 grid of NDC points, telling where to sample input from

        Flip Sign on grid_ndc because topleft is [1,1] (not [-1,-1]) in NDC space
        Scale larger (of H,W) to [-1,1] since ndc coodrinate can be >1 for rectangular images
    """
    N,C,H,W = input.shape
    assert(grid_ndc.shape[0]==N)
    assert(grid_ndc.shape[3]==2)
    if H!=W:
        x,y = grid_ndc.unbind(dim=3)
        if H < W:
            x = x * H/W
        else:
            y = y * W/H
        grid_ndc = torch.stack([x,y], dim=3)
    samples = F.grid_sample(input, -grid_ndc, *args, **kwargs)
    if not return_whether_in_bounds:
        return samples
    else:
        assert(kwargs.get('align_corners')==False)
        in_bounds = (grid_ndc < 1).all(dim=3) & (grid_ndc > -1).all(dim=3)
        return samples, in_bounds[:,None]

def find_visibility_depth_exp(
        points3d_ndc: torch.Tensor, points3d_view:torch.Tensor,
        texture_cameras: CamerasBase, texture_fragments: Fragments,
        vis_gamma: float = 1, depth_smooth: bool = False, eps_vis: float = 0,
        blend_params: BlendParams = BlendParams(),
        **kwargs
    ) -> torch.Tensor:
    ## Estimate point visibility by comparing view-z to depthmap from zbuf
    # Render depth map
    blend_params = blend_params._replace(gamma=vis_gamma, background_color=-1)
    znear = kwargs.get("znear", getattr(texture_cameras, "znear", 1.0))
    zfar = kwargs.get("zfar", getattr(texture_cameras, "zfar", 100.0))
    texture_depth = softmax_ccc_blend(
        texture_fragments.zbuf[...,None],
        texture_fragments,
        blend_params,
        znear=znear,
        zfar=zfar,
        return_alpha=False
    )

    # Sample depth map at points3d_ndc[...,:2]
    points3d_dz = grid_sample_ndc(
        texture_depth[:,None,:,:,0], points3d_ndc[:,:,None,:2], align_corners=False
    )
    points3d_dz = points3d_dz.squeeze(-1).squeeze(1)  # (N,1,P,1) -> (N,P)

    # Compare points3d_view and points3d_dz to estimate visibility. Pick gamma used
    # for depth rendering in softmax_ccc_blend
    zdiff = (points3d_dz - points3d_view[..., 2]) / (zfar-znear)
    zdiff = zdiff.clamp(max=0)
    points3d_visprob = torch.exp(zdiff/vis_gamma)
    return points3d_visprob.clamp(min=0, max=1)

def sample_texture_3dpoints(points3d_world, normals3d_world,
        texture_images, texture_cameras, texture_fragments, can_sample_mask,
        vis_gamma=1, cos_gamma=1, eps: float = 1e-10,
        vis_thresh: float = 0.1, weight_thresh: float = 1e-6,
        **kwargs):
    """
    Sample colors for 3D points, from texture_images
    points3d_world: Find color for these 3D points
    normals3d_world:  corresponding normals
    texture_fragments: raster out when rendering texture_cameras.
                        Use depth for finding visibility
    can_sample_mask: NtxP bool tensor specifying what texture_images we can sample from
    """
    assert(points3d_world.shape == normals3d_world.shape)
    input_shape = points3d_world.shape
    N = len(texture_cameras)
    assert(can_sample_mask.shape[0] == N)

    # Return if input is empty
    if points3d_world.shape[0] == 0:
        C = texture_images.shape[1]
        dtype = points3d_world.dtype
        device = points3d_world.device
        points3d_color = torch.zeros((*input_shape[:-1],C), dtype=dtype, device=device)
        points3d_valid = torch.zeros((*input_shape[:-1],1), dtype=bool, device=device)
        sampling_weights = torch.ones((N,*input_shape[:-1]), dtype=dtype, device=device)
        return points3d_color, points3d_valid, sampling_weights

    # Reshape and batch points/normals
    points3d_world = points3d_world.view(-1, 3)
    normals3d_world = normals3d_world.view(-1, 3)
    points3d_world = points3d_world[None].expand(N, -1, -1)
    normals3d_world = normals3d_world[None].expand(N, -1, -1)

    # Transform points + normals into each camera's ndc and view space
    view_transform = texture_cameras.get_world_to_view_transform()
    points3d_view = view_transform.transform_points(points3d_world, eps=TRANS_EPS)
    points3d_ndc = texture_cameras.transform_points(points3d_world, eps=TRANS_EPS)
    normals3d_view = view_transform.transform_normals(normals3d_world)

    with torch.no_grad():
        #######
        ## We will now estimate point visibility by comparing view-z to depthmap from zbuf
        # Render depth map
        points3d_visprob = find_visibility_depth_exp(
                points3d_ndc, points3d_view, texture_cameras, texture_fragments,
                vis_gamma=vis_gamma, **kwargs
                )

        # Zero out visibility of points that are barely visible
        points3d_visprob = torch.where(points3d_visprob<vis_thresh, torch.zeros_like(points3d_visprob), points3d_visprob)
        #######

        # Estimate foreshortening using normals.
        normals3d_z = -1 * normals3d_view[..., 2]
        away_normals = (normals3d_z<0)
        normals3d_z = (normals3d_z - 1 - eps).clamp(max=0)
        foreshorten_prob = torch.exp(normals3d_z/cos_gamma)
        # Some points aren't visible because their normals point away
        foreshorten_prob = torch.where(away_normals, torch.zeros_like(foreshorten_prob), foreshorten_prob)

        # Compute weights for which image is used for sampling.
        # Combine colors from diff images by these weights.
        sampling_weights = points3d_visprob * foreshorten_prob

        # zero out weight wherever can_sample_mask==False
        # zero out weight if too small
        sampling_weight_valid = can_sample_mask & (sampling_weights > weight_thresh)
        sampling_weights = torch.where(~sampling_weight_valid, torch.zeros_like(sampling_weights), sampling_weights)
        sampling_weights = sampling_weights/(sampling_weights.sum(dim=0, keepdims=True) + eps)

    # Sample color at points3d_ndc[...,:2]
    points3d_color = grid_sample_ndc(
        texture_images, points3d_ndc[:,:,None,:2], align_corners=False
    )
    points3d_color = points3d_color.squeeze(-1).transpose(1,2)  # (N,C,P,1) -> (N,C,P) -> (N,P,C)

    # Weighted sum to get each point's color
    points3d_color = (points3d_color * sampling_weights[..., None]).sum(0)
    points3d_color_valid = sampling_weight_valid.any(dim=0)

    points3d_color = points3d_color.view(input_shape[:-1] + (-1,))
    points3d_color_valid = points3d_color_valid.view(input_shape[:-1] + (-1,))
    sampling_weights = sampling_weights.view((N,) + input_shape[:-1])

    # assert(points3d_view.isfinite().all())
    # assert(points3d_ndc.isfinite().all())
    # assert(texture_depth.isfinite().all())
    # assert(points3d_dz.isfinite().all())
    # assert(points3d_visprob.isfinite().all())
    # assert(normals3d_z.isfinite().all())
    # assert(sampling_weights.isfinite().all())
    # assert(points3d_color.isfinite().all())

    return points3d_color, points3d_color_valid, sampling_weights

# Renderers
def _get_rasterizer(
        image_size: Tuple[int,int],
        cfg: DictConfig,
        cameras: Optional[CamerasBase] = None,
    ) -> MeshRasterizer :
    raster_settings = RasterizationSettings(
        image_size = image_size,
        **cfg
    )
    return MeshRasterizer(cameras=cameras, raster_settings=raster_settings)

class TexOptRenderer(torch.nn.Module):
    def __init__(self, image_size: Tuple[int,int],
            cfg_raster: DictConfig, cfg_shader: DictConfig) -> None:
        super().__init__()
        self.rasterizer = _get_rasterizer(image_size, cfg_raster)
        self.shader =  instantiate(cfg_shader)
    def forward(
            self,
            meshes_world: Meshes,
            cameras: CamerasBase,
            **kwargs
        ) -> torch.Tensor:
        if len(meshes_world) > 1:
            raise ValueError(f'Got {len(meshes_world)} (>1) meshes in batch')

        fragments = self.rasterizer(
            meshes_world.extend(len(cameras)),
            cameras=cameras,
            **kwargs
        )
        out = self.shader(
            fragments,
            meshes_world.extend(len(cameras)),
            cameras=cameras,
            **kwargs
        )
        out.update(fragments = fragments)
        return out

class TexTransferRenderer(torch.nn.Module):
    def __init__(self, image_size: Tuple[int,int],
            cfg_raster: DictConfig, cfg_shader: DictConfig,
        ) -> None:
        super().__init__()

        # Rasterizer
        self.rasterizer = _get_rasterizer(image_size, cfg_raster)

        # Shader
        self.shader = instantiate(cfg_shader)

    def to(self, device):
        # Rasterizer and shader have submodules which are not of type nn.Module
        self.rasterizer.to(device)
        self.shader.to(device)

    def forward(
            self,
            meshes_world: Meshes,
            cameras: CamerasBase,
            ref_rgba: torch.Tensor,
            ref_cameras: CamerasBase,
            ref_fragments: Optional[Fragments] = None,
            return_weights: bool = False,
            **kwargs
        ) -> Dict[str, torch.Tensor]:
        if len(meshes_world) > 1:
            raise ValueError(f'Got {len(meshes_world)} (>1) meshes in batch')

        # Rasterize
        fragments = self.rasterizer(
            meshes_world.extend(len(cameras)),
            cameras=cameras,
            **kwargs
        )
        if fragments.zbuf.isnan().any():
            raise ValueError(f'fragments zbuf got nan at {fragments.zbuf.isnan().nonzero()}')
        if ref_fragments is None:
            if ref_cameras == cameras:
                # TODO: Cache fragments in rasterizer by camera_hash
                ref_fragments = fragments
            else:
                ref_fragments = self.rasterizer(
                    meshes_world.extend(len(ref_cameras)),
                    cameras=ref_cameras,
                    **kwargs
                )

        # Shade
        out = self.shader(
            fragments,
            meshes_world.extend(len(cameras)),
            cameras=cameras,
            texture_images=ref_rgba[:,:3],
            texture_cameras=ref_cameras,
            texture_fragments=ref_fragments,
            return_weights=return_weights,
            **kwargs
        )
        out.update(
            fragments = fragments,
            ref_fragments = ref_fragments,
        )
        return out


# Utility things
def render_depths(
        mesh: Meshes,
        camera_list:List[CamerasBase],
        image_size=128,
        device=torch.device('cuda'),
        background=-1,
    ):
    """Returns depth maps as Bx1xHxW """
    depth_renderer = MeshRenderer(
        rasterizer = MeshRasterizer(
            raster_settings = RasterizationSettings(
                image_size = image_size
        )),
        shader = SimpleDepthShader(
            blend_params=BlendParams(background_color=background)
        )
    )
    mesh = mesh.to(device)
    depths = torch.cat([
            depth_renderer(mesh.to(device), cameras=cam.to(device))
            for cam in camera_list
        ], dim=0
    )
    depths = depths.permute(0,3,1,2)[:,:-1,:,:]
    return depths

def render_pcl_masks(gt_pcl, poses, hfovs, img_size, downsample=4):
    """ Returns Nx1xHxW sized masks """
    H,W = img_size
    Hd, Wd = int(H/downsample), int(W/downsample)

    camgen = dollyParamCameras(poses, hfovs, img_size=(Hd,Wd))
    bin_size = int(2 ** max(np.ceil(np.log2(max(Hd,Wd))) - 4, 4))
    rasterizer_rect = PointsRasterizer(raster_settings=PointsRasterizationSettings(image_size=(Hd,Wd), bin_size=bin_size))

    masks = []
    for i in range(poses.shape[0]):
        cam = camgen.create_cameras(id=i)
        mask = rasterizer_rect(gt_pcl, cameras=cam).zbuf[0,:,:,0]>=0
        masks.append(mask)
    masks = torch.stack(masks)
    return F.interpolate(masks[:,None].float(), size=(H,W), mode='bilinear')
