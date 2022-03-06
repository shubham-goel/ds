# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
import math
from typing import List, Union

import torch
from pytorch3d.renderer import RayBundle
from pytorch3d.renderer.cameras import CamerasBase

from .utils import sample_pdf


def _xy_to_ray_bundle(
    cameras: CamerasBase,
    xy_grid: torch.Tensor,
    min_depth: Union[float,torch.Tensor],
    max_depth: Union[float,torch.Tensor],
    n_pts_per_ray: int,
) -> RayBundle:
    """
    Extends the `xy_grid` input of shape `(batch_size, ..., 2)` to rays.
    This adds to each xy location in the grid a vector of `n_pts_per_ray` depths
    uniformly spaced between `min_depth` and `max_depth`.

    The extended grid is then unprojected with `cameras` to yield
    ray origins, directions and depths.
    """
    batch_size = xy_grid.shape[0]
    spatial_size = xy_grid.shape[1:-1]
    n_rays_per_image = spatial_size.numel()  # pyre-ignore

    # ray z-coords
    if isinstance(min_depth, float):
        min_depth = torch.full((batch_size,), min_depth, dtype=xy_grid.dtype, device=xy_grid.device)
    if isinstance(max_depth, float):
        max_depth = torch.full((batch_size,), max_depth, dtype=xy_grid.dtype, device=xy_grid.device)
    depths = min_depth[:, None] + torch.linspace(
        0, 1, n_pts_per_ray, dtype=xy_grid.dtype, device=xy_grid.device
    ) * (max_depth[:, None] - min_depth[:, None])
    rays_zs = depths[:, None, :].expand(-1, n_rays_per_image, -1)

    # pytorch xy coordinates -> pytorch3d NDC space
    xy_grid_ndc = -1 * xy_grid

    # make two sets of points at a constant depth=1 and 2
    to_unproject = torch.cat(
        (
            xy_grid_ndc.view(batch_size, 1, n_rays_per_image, 2)
            .expand(batch_size, 2, n_rays_per_image, 2)
            .reshape(batch_size, n_rays_per_image * 2, 2),
            torch.cat(
                (
                    xy_grid.new_ones(batch_size, n_rays_per_image, 1),  # pyre-ignore
                    2.0 * xy_grid.new_ones(batch_size, n_rays_per_image, 1),
                ),
                dim=1,
            ),
        ),
        dim=-1,
    )

    # unproject the points
    unprojected = cameras.unproject_points(to_unproject)  # pyre-ignore

    # split the two planes back
    rays_plane_1_world = unprojected[:, :n_rays_per_image]
    rays_plane_2_world = unprojected[:, n_rays_per_image:]

    # directions are the differences between the two planes of points
    rays_directions_world = rays_plane_2_world - rays_plane_1_world

    # origins are given by subtracting the ray directions from the first plane
    rays_origins_world = rays_plane_1_world - rays_directions_world

    # Convert zs to ray lengths
    rays_zs = rays_zs * rays_directions_world.norm(dim=-1, keepdim=True)

    return RayBundle(
        rays_origins_world.view(batch_size, *spatial_size, 3),
        rays_directions_world.view(batch_size, *spatial_size, 3),
        rays_zs.view(batch_size, *spatial_size, n_pts_per_ray),
        xy_grid,
    )

class CustomGridRaysampler(torch.nn.Module):
    """
    Samples a fixed number of points along rays which are regulary distributed
    in a batch of rectangular image grids. Points along each ray
    have uniformly-spaced z-coordinates between a predefined minimum and maximum depth.
    """

    def __init__(
        self,
        min_x: float,
        max_x: float,
        min_y: float,
        max_y: float,
        image_width: int,
        image_height: int,
        n_pts_per_ray: int = None,
        min_depth: float = None,
        max_depth: float = None,
    ):
        """
        Args:
            min_x: The leftmost x-coordinate of each ray's
                source pixel's center in screen space units.
            max_x: The rightmost x-coordinate of each ray's
                source pixel's center in screen space units.
            min_y: The topmost y-coordinate of each ray's
                source pixel's center in screen space units.
            max_y: The bottommost y-coordinate of each ray's
                source pixel's center in screen space units.
            image_width: The horizontal size of the image grid.
            image_height: The vertical size of the image grid.
            n_pts_per_ray: The number of points sampled along each ray.
            min_depth: The minimum depth of a ray-point.
            max_depth: The maximum depth of a ray-point.
        """
        super().__init__()
        self._n_pts_per_ray = n_pts_per_ray
        self._min_depth = min_depth
        self._max_depth = max_depth

        # get the initial grid of image xy coords
        _xy_grid = torch.stack(
            tuple(
                reversed(
                    torch.meshgrid(
                        torch.linspace(min_y, max_y, image_height, dtype=torch.float32),
                        torch.linspace(min_x, max_x, image_width, dtype=torch.float32),
                    )
                )
            ),
            dim=-1,
        )
        self.register_buffer("_xy_grid", _xy_grid)

    def forward(self, cameras: CamerasBase, **kwargs) -> RayBundle:
        """
        Args:
            cameras: A batch of `batch_size` cameras from which the rays are emitted.
        Returns:
            A named tuple RayBundle with the following fields:
            origins: A tensor of shape
                `(batch_size, image_height, image_width, 3)`
                denoting the locations of ray origins in the world coordinates.
            directions: A tensor of shape
                `(batch_size, image_height, image_width, 3)`
                denoting the directions of each ray in the world coordinates.
            lengths: A tensor of shape
                `(batch_size, image_height, image_width, n_pts_per_ray)`
                containing the z-coordinate (=depth) of each ray in world units.
            xys: A tensor of shape
                `(batch_size, image_height, image_width, 2)`
                containing the 2D image coordinates of each ray.
        """

        batch_size = cameras.R.shape[0]  # pyre-ignore

        device = cameras.device

        # expand the (H, W, 2) grid batch_size-times to (B, H, W, 2)
        xy_grid = self._xy_grid.to(device)[None].expand(  # pyre-ignore
            batch_size, *self._xy_grid.shape
        )

        min_depth = kwargs.get('min_depth', self._min_depth)
        max_depth = kwargs.get('max_depth', self._max_depth)
        n_pts_per_ray = kwargs.get('n_pts_per_ray', self._n_pts_per_ray)

        return _xy_to_ray_bundle(
            cameras, xy_grid, min_depth, max_depth, n_pts_per_ray
        )


class CustomMonteCarloRaysampler(torch.nn.Module):
    """
    Samples a fixed number of pixels within denoted xy bounds uniformly at random.
    For each pixel, a fixed number of points is sampled along its ray at uniformly-spaced
    z-coordinates such that the z-coordinates range between a predefined minimum
    and maximum depth.
    """

    def __init__(
        self,
        min_x: float,
        max_x: float,
        min_y: float,
        max_y: float,
        n_rays_per_image: int,
        n_pts_per_ray: int = None,
        min_depth: float = None,
        max_depth: float = None,
    ):
        """
        Args:
            min_x: The smallest x-coordinate of each ray's source pixel in screen space units.
            max_x: The largest x-coordinate of each ray's source pixel in screen space units.
            min_y: The smallest y-coordinate of each ray's source pixel in screen space units.
            max_y: The largest y-coordinate of each ray's source pixel in screen space units.
            n_rays_per_image: The number of rays randomly sampled in each camera.
            n_pts_per_ray: The number of points sampled along each ray.
            min_depth: The minimum depth of each ray-point.
            max_depth: The maximum depth of each ray-point.
        """
        super().__init__()
        self._min_x = min_x
        self._max_x = max_x
        self._min_y = min_y
        self._max_y = max_y
        self._n_rays_per_image = n_rays_per_image
        self._n_pts_per_ray = n_pts_per_ray
        self._min_depth = min_depth
        self._max_depth = max_depth

    def forward(self, cameras: CamerasBase, **kwargs) -> RayBundle:
        """
        Args:
            cameras: A batch of `batch_size` cameras from which the rays are emitted.
        Returns:
            A named tuple RayBundle with the following fields:
            origins: A tensor of shape
                `(batch_size, n_rays_per_image, 3)`
                denoting the locations of ray origins in the world coordinates.
            directions: A tensor of shape
                `(batch_size, n_rays_per_image, 3)`
                denoting the directions of each ray in the world coordinates.
            lengths: A tensor of shape
                `(batch_size, n_rays_per_image, n_pts_per_ray)`
                containing the z-coordinate (=depth) of each ray in world units.
            xys: A tensor of shape
                `(batch_size, n_rays_per_image, 2)`
                containing the 2D image coordinates of each ray.
        """

        batch_size = cameras.R.shape[0]  # pyre-ignore

        device = cameras.device

        # get the initial grid of image xy coords
        # of shape (batch_size, n_rays_per_image, 2)
        rays_xy = torch.cat(
            [
                torch.rand(
                    size=(batch_size, self._n_rays_per_image, 1),
                    dtype=torch.float32,
                    device=device,
                )
                * (high - low)
                + low
                for low, high in (
                    (self._min_x, self._max_x),
                    (self._min_y, self._max_y),
                )
            ],
            dim=2,
        )

        min_depth = kwargs.get('min_depth', self._min_depth)
        max_depth = kwargs.get('max_depth', self._max_depth)
        n_pts_per_ray = kwargs.get('n_pts_per_ray', self._n_pts_per_ray)

        return _xy_to_ray_bundle(
            cameras, rays_xy, min_depth, max_depth, n_pts_per_ray
        )


def sample_xy_inside(mask, n_rays_per_image):
    """ Sample n_rays_per_image uniform random xy locations from within the mask
    """
    h,w = mask.shape
    gridintpts_Px2 = (mask>0.5).nonzero()
    gridintpts_Px2 = gridintpts_Px2[torch.randint(len(gridintpts_Px2), (n_rays_per_image,))]
    gridintpts_Px2 = gridintpts_Px2.float()
    gridsubpts_Px2 = gridintpts_Px2 + torch.rand_like(gridintpts_Px2)
    gridsubpts_Px2 = 2 * gridsubpts_Px2 / torch.tensor([h,w], device=mask.device) - 1
    gridsubpts_Px2 = gridsubpts_Px2[:,[1,0]]
    return gridsubpts_Px2

class CustomMonteCarloSilhouetteRaysampler(torch.nn.Module):
    """
    Samples a fixed number of pixels uniformly at random from inside/outside the silhouette.
    For each pixel, a fixed number of points is sampled along its ray at uniformly-spaced
    z-coordinates such that the z-coordinates range between a predefined minimum
    and maximum depth.
    """

    def __init__(
        self,
        n_rays_per_image: int,
        frac_inside: float,
        n_pts_per_ray: int = None,
        min_depth: float = None,
        max_depth: float = None,
    ):
        """
        Args:
            n_rays_per_image: The number of rays randomly sampled in each camera.
            n_pts_per_ray: The number of points sampled along each ray.
            frac_inside: The fraction of `n_rays_per_image` sampled from inside the silhouette
            min_depth: The minimum depth of each ray-point.
            max_depth: The maximum depth of each ray-point.
        """
        super().__init__()
        self._n_rays_per_image = n_rays_per_image
        self._n_pts_per_ray = n_pts_per_ray
        self._frac_inside = frac_inside
        self._min_depth = min_depth
        self._max_depth = max_depth



    def forward(self, cameras: CamerasBase, **kwargs) -> RayBundle:
        """
        Args:
            cameras: A batch of `batch_size` cameras from which the rays are emitted.
            masks: Corresponding silhouettes of shape HxW for specifying regions to sample from.
        Returns:
            A named tuple RayBundle with the following fields:
            origins: A tensor of shape
                `(batch_size, n_rays_per_image, 3)`
                denoting the locations of ray origins in the world coordinates.
            directions: A tensor of shape
                `(batch_size, n_rays_per_image, 3)`
                denoting the directions of each ray in the world coordinates.
            lengths: A tensor of shape
                `(batch_size, n_rays_per_image, n_pts_per_ray)`
                containing the z-coordinate (=depth) of each ray in world units.
            xys: A tensor of shape
                `(batch_size, n_rays_per_image, 2)`
                containing the 2D image coordinates of each ray.
        """

        batch_size = cameras.R.shape[0]  # pyre-ignore

        masks = kwargs.get('masks', None)
        if masks is None:
            raise ValueError

        num_inside = int(self._frac_inside * self._n_rays_per_image)
        rays_xy_inside = torch.stack([
                            sample_xy_inside(masks[i], num_inside)      # ni,2
                            for i in range(batch_size)
                        ], dim=0)

        num_outside = self._n_rays_per_image - num_inside
        rays_xy_outside = torch.stack([
                            sample_xy_inside(1-masks[i], num_outside)   # no,2
                            for i in range(batch_size)
                        ], dim=0)

        rays_xy = torch.cat((rays_xy_inside, rays_xy_outside), dim=1)   # b,(ni+no),2

        min_depth = kwargs.get('min_depth', self._min_depth)
        max_depth = kwargs.get('max_depth', self._max_depth)
        n_pts_per_ray = kwargs.get('n_pts_per_ray', self._n_pts_per_ray)

        return _xy_to_ray_bundle(
            cameras, rays_xy, min_depth, max_depth, n_pts_per_ray
        )


class ProbabilisticRaysampler(torch.nn.Module):
    """
    Implements the importance sampling of points along rays.
    The input is a `RayBundle` object with a `ray_weights` tensor
    which specifies the probabilities of sampling a point along each ray.
    """

    def __init__(
        self,
        n_pts_per_ray: int,
        stratified: bool,
        stratified_test: bool,
        add_input_samples: bool = True,
    ):
        """
        Args:
            n_pts_per_ray: The number of points to sample along each ray.
            stratified: If `True`, the input `ray_weights` are assumed to be
                sampled at equidistant intervals.
            stratified_test: Same as `stratified` with the difference that this
                setting is applied when the module is in the `eval` mode
                (`self.training==False`).
            add_input_samples: Concatenates and returns the sampled values
                together with the input samples.
        """
        super().__init__()
        self._n_pts_per_ray = n_pts_per_ray
        self._stratified = stratified
        self._stratified_test = stratified_test
        self._add_input_samples = add_input_samples

    def forward(
        self,
        input_ray_bundle: RayBundle,
        ray_weights: torch.Tensor,
        **kwargs,
    ) -> RayBundle:
        """
        Args:
            input_ray_bundle: An instance of `RayBundle` specifying the
                source rays for sampling of the probability distribution.
            ray_weights: A tensor of shape
                `(..., input_ray_bundle.legths.shape[-1])` with non-negative
                elements defining the probability distribution to sample
                ray points from.

        Returns:
            ray_bundle: A new `RayBundle` instance containing the input ray
                points together with `n_pts_per_ray` additional sampled
                points per ray.
        """

        # Calculate the mid-points between the ray depths.
        z_vals = input_ray_bundle.lengths
        batch_size = z_vals.shape[0]
        z_vals_mid = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])

        # Carry out the importance sampling.
        z_samples = (
            sample_pdf(
                z_vals_mid.view(-1, z_vals_mid.shape[-1]),
                ray_weights.view(-1, ray_weights.shape[-1])[..., 1:-1],
                self._n_pts_per_ray,
                det=not (
                    (self._stratified and self.training)
                    or (self._stratified_test and not self.training)
                ),
            )
            .detach()
            .view(batch_size, z_vals.shape[1], self._n_pts_per_ray)
        )

        if self._add_input_samples:
            # Add the new samples to the input ones.
            z_vals = torch.cat((z_vals, z_samples), dim=-1)
        else:
            z_vals = z_samples
        # Resort by depth.
        z_vals, _ = torch.sort(z_vals, dim=-1)

        return RayBundle(
            origins=input_ray_bundle.origins,
            directions=input_ray_bundle.directions,
            lengths=z_vals,
            xys=input_ray_bundle.xys,
        )


class NerfRaysampler(torch.nn.Module):
    """
    Implements the raysampler of NeRF.

    Depending on the `self.training` flag, the raysampler either samples
    a chunk of random rays (`self.training==True`), or returns a subset of rays
    of the full image grid (`self.training==False`).
    The chunking of rays allows for efficient evaluation of the NeRF implicit
    surface function without encountering out-of-GPU-memory errors.

    Additionally, this raysampler supports pre-caching of the ray bundles
    for a set of input cameras (`self.precache_rays`).
    Pre-caching the rays before training greatly speeds-up the ensuing
    raysampling step of the training NeRF iterations.
    """

    def __init__(
        self,
        min_x: float,
        max_x: float,
        min_y: float,
        max_y: float,
        n_pts_per_ray: int,
        image_width: int,
        image_height: int,
        stratified: bool = False,
        stratified_test: bool = False,
        invert_directions: bool = False,
        min_depth: float = None,
        max_depth: float = None,
        n_rays_per_image: int = None,
        sample_frac_inside_mask: float = -1,
        **kwargs
    ):
        super().__init__()
        self._stratified = stratified
        self._stratified_test = stratified_test
        self._invert_directions = invert_directions

        self._grid_raysampler = CustomGridRaysampler(
            min_x=min_x,
            max_x=max_x,
            min_y=min_y,
            max_y=max_y,
            image_width=image_width,
            image_height=image_height,
            n_pts_per_ray=n_pts_per_ray,
            min_depth=min_depth,
            max_depth=max_depth,
        )

        if sample_frac_inside_mask == -1:
            self._mc_raysampler = CustomMonteCarloRaysampler(
                min_x=min_x,
                max_x=max_x,
                min_y=min_y,
                max_y=max_y,
                n_rays_per_image=n_rays_per_image,
                n_pts_per_ray=n_pts_per_ray,
                min_depth=min_depth,
                max_depth=max_depth,
            )
        else:
            self._mc_raysampler = CustomMonteCarloSilhouetteRaysampler(
                n_rays_per_image=n_rays_per_image,
                n_pts_per_ray=n_pts_per_ray,
                min_depth=min_depth,
                max_depth=max_depth,
                frac_inside=sample_frac_inside_mask
            )


        # create empty ray cache
        self._ray_cache = {}

    def get_n_chunks(self, chunksize: int, batch_size: int):
        """
        Returns the total number of `chunksize`-sized chunks
        of the raysampler's rays.

        Args:
            chunksize: The number of rays per chunk.
            batch_size: The size of the batch of the raysampler.

        Returns:
            n_chunks: The total number of chunks.
        """
        return int(
            math.ceil(
                (self._grid_raysampler._xy_grid.numel() * 0.5 * batch_size) / chunksize
            )
        )

    def precache_rays(self, cameras: List[CamerasBase], camera_hashes: List, **kwargs):
        """
        Precaches the rays emitted from the list of cameras `cameras`,
        where each camera is uniquely identified with the corresponding hash
        from `camera_hashes`.

        The cached rays are moved to cpu and stored in `self._ray_cache`.
        Raises `ValueError` when caching two cameras with the same hash.

        Args:
            cameras: A list of `N` cameras for which the rays are pre-cached.
            camera_hashes: A list of `N` unique identifiers of each
                camera from `cameras`.
        """
        print("Precaching rays ...")
        full_chunksize = (
            self._grid_raysampler._xy_grid.numel()
            // 2
            * self._grid_raysampler._n_pts_per_ray
        )
        if self.get_n_chunks(full_chunksize, 1) != 1:
            raise ValueError("There has to be one chunk for precaching rays!")

        kwarg_list = [
            {k[:-1] : float(kwargs[k][i]) for k in ['min_depths', 'max_depths'] if k in kwargs}
            for i in range(len(cameras))
        ]
        for camera, camera_hash, elem_kwargs in zip(cameras, camera_hashes, kwarg_list):
            ray_bundle = self.forward(
                camera,
                caching=True,
                chunksize=full_chunksize,
                **elem_kwargs,
            )
            if camera_hash in self._ray_cache:
                raise ValueError("There are redundant cameras!")
            self._ray_cache[camera_hash] = RayBundle(
                *[v.to("cpu").detach() for v in ray_bundle]
            )

    def _stratify_ray_bundle(self, ray_bundle: RayBundle):
        """
        Stratifies the lengths of the input `ray_bundle`.

        More specifically, the stratification replaces each ray points' depth `z`
        with a sample from a uniform random distribution on
        `[z - delta_depth, z+delta_depth]`, where `delta_depth` is the difference
        of depths of the consecutive ray depth values.

        Args:
            `ray_bundle`: The input `RayBundle`.

        Returns:
            `stratified_ray_bundle`: `ray_bundle` whose `lengths` field is replaced
                with the stratified samples.
        """
        z_vals = ray_bundle.lengths
        # Get intervals between samples.
        mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat((mids, z_vals[..., -1:]), dim=-1)
        lower = torch.cat((z_vals[..., :1], mids), dim=-1)
        # Stratified samples in those intervals.
        z_vals = lower + (upper - lower) * torch.rand_like(lower)
        return ray_bundle._replace(lengths=z_vals)

    def _normalize_raybundle(self, ray_bundle: RayBundle):
        """
        Normalizes the ray directions of the input `RayBundle` to unit norm and,
        if `self._invert_directions==True`, inverts the ray directions.
        """
        ray_sign = 1 - 2 * float(self._invert_directions)
        ray_bundle = ray_bundle._replace(
            directions=ray_sign
            * torch.nn.functional.normalize(ray_bundle.directions, dim=-1)
        )
        return ray_bundle

    def forward(
        self,
        cameras: CamerasBase,
        chunksize: int = None,
        chunk_idx: int = 0,
        camera_hash: str = None,
        caching: bool = False,
        **kwargs,
    ) -> RayBundle:
        """
        Args:
            cameras: A batch of `batch_size` cameras from which the rays are emitted.
        Returns:
            A named tuple RayBundle with the following fields:
            origins: A tensor of shape
                `(batch_size, n_rays_per_image, 3)`
                denoting the locations of ray origins in the world coordinates.
            directions: A tensor of shape
                `(batch_size, n_rays_per_image, 3)`
                denoting the directions of each ray in the world coordinates.
            lengths: A tensor of shape
                `(batch_size, n_rays_per_image, n_pts_per_ray)`
                containing the z-coordinate (=depth) of each ray in world units.
            xys: A tensor of shape
                `(batch_size, n_rays_per_image, 2)`
                containing the 2D image coordinates of each ray.
        """

        batch_size = cameras.R.shape[0]  # pyre-ignore
        device = cameras.device

        if (camera_hash is None) and (not caching) and self.training:
            # Sample random rays from scratch.
            ray_bundle = self._mc_raysampler(cameras, **kwargs)
            ray_bundle = self._normalize_raybundle(ray_bundle)
        else:
            if camera_hash is not None:
                # The case where we retrieve a camera from cache.
                if batch_size != 1:
                    raise NotImplementedError(
                        "Ray caching works only for batches with a single camera!"
                    )
                full_ray_bundle = self._ray_cache[camera_hash]
            else:
                # We generate a full ray grid from scratch.
                full_ray_bundle = self._grid_raysampler(cameras, **kwargs)
                full_ray_bundle = self._normalize_raybundle(full_ray_bundle)

            n_pixels = full_ray_bundle.directions.shape[:-1].numel()

            if self.training:
                # During training we randomly subsample rays.
                sel_rays = torch.randperm(n_pixels, device=device)[
                    : self._mc_raysampler._n_rays_per_image
                ]
            else:
                # In case we test, we take only the requested chunk.
                if chunksize is None:
                    chunksize = n_pixels * batch_size
                start = chunk_idx * chunksize * batch_size
                end = min(start + chunksize, n_pixels)
                sel_rays = torch.arange(
                    start,
                    end,
                    dtype=torch.long,
                    device=full_ray_bundle.lengths.device,
                )

            # Take the "sel_rays" rays from the full ray bundle.
            ray_bundle = RayBundle(
                *[
                    v.view(n_pixels, -1)[sel_rays]
                    .view(batch_size, sel_rays.numel() // batch_size, -1)
                    .to(device)
                    for v in full_ray_bundle
                ]
            )

        if (
            (self._stratified and self.training)
            or (self._stratified_test and not self.training)
        ) and not caching:  # Make sure not to stratify when caching!
            ray_bundle = self._stratify_ray_bundle(ray_bundle)

        return ray_bundle
