from typing import List

import torch
import torch.nn.functional as F
from pytorch3d.renderer import RayBundle, ray_bundle_to_ray_points
import logging

from .harmonic_embedding import HarmonicEmbedding

# Define all modules to import
__all__ = [
    'NeuralRadianceField',
]

class NeuralRadianceField(torch.nn.Module):
    def __init__(
        self,
        n_harmonic_functions_xyz: int = 6,
        n_harmonic_functions_dir: int = 4,
        n_hidden_neurons_xyz: int = 256,
        n_hidden_neurons_dir: int = 128,
        n_layers_xyz: int = 8,
        append_xyz: List[int] = (5,),
        harmonic_xyz_omega0: float = 1.0,
        harmonic_dir_omega0: float = 1.0,
        **kwargs,
    ):
        """
        Args:
            n_harmonic_functions: The number of harmonic functions
                used to form the harmonic embedding of each point.
            n_hidden_neurons: The number of hidden units in the
                fully connected layers of the MLPs of the model.
        """
        super().__init__()

        # The harmonic embedding layer converts input 3D coordinates
        # to a representation that is more suitable for
        # processing with a deep neural network.
        if n_harmonic_functions_xyz>0: self.harmonic_embedding_xyz = HarmonicEmbedding(n_harmonic_functions_xyz, omega0=harmonic_xyz_omega0)
        if n_harmonic_functions_dir>0: self.harmonic_embedding_dir = HarmonicEmbedding(n_harmonic_functions_dir, omega0=harmonic_dir_omega0)
        embedding_dim_xyz = n_harmonic_functions_xyz * 2 * 3 + 3
        embedding_dim_dir = n_harmonic_functions_dir * 2 * 3 + 3

        if n_layers_xyz>0:
            self.mlp_xyz = MLPWithInputSkips(
                n_layers_xyz,
                embedding_dim_xyz,
                n_hidden_neurons_xyz,
                embedding_dim_xyz,
                n_hidden_neurons_xyz,
                input_skips=append_xyz,
            )

        if n_hidden_neurons_xyz>0 and n_hidden_neurons_dir>0:
            # Output of this layer is passed to self.color_layer
            # Don't create if we're not going to create color layer
            self.intermediate_linear = torch.nn.Linear(
                n_hidden_neurons_xyz, n_hidden_neurons_xyz
            )

        if n_hidden_neurons_xyz>0:
            self.density_layer = torch.nn.Linear(n_hidden_neurons_xyz, 1)

        if n_hidden_neurons_dir>0:
            self.color_layer = torch.nn.Sequential(
                torch.nn.Linear(
                    n_hidden_neurons_xyz + embedding_dim_dir, n_hidden_neurons_dir
                ),
                torch.nn.ReLU(True),
                torch.nn.Linear(n_hidden_neurons_dir, 3),
                torch.nn.Sigmoid(),
            )

        # Use same initialization as original tensorflow implementation
        def weights_init(m):
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                torch.nn.init.zeros_(m.bias.data)
        self.apply(weights_init)

    def _get_raw_densities(self, features: torch.Tensor):
        """ features.shape = [minibatch x ... x channel] """
        return self.density_layer(features)

    def _get_densities(
        self,
        features: torch.Tensor,
        depth_values: torch.Tensor,
        density_noise_std: float,
    ):
        """
        This function takes `features` predicted by `self.mlp_xyz`
        and converts them to `raw_densities` with `self.density_layer`.
        `raw_densities` are later re-weighted using the depth step sizes
        and mapped to [0-1] range with 1 - inverse exponential of `raw_densities`.
        """
        raw_densities = self._get_raw_densities(features)
        deltas = torch.cat(
            (
                depth_values[..., 1:] - depth_values[..., :-1],
                1e10 * torch.ones_like(depth_values[..., :1]),
            ),
            dim=-1,
        )[..., None]
        if density_noise_std > 0.0:
            raw_densities = (
                raw_densities + torch.randn_like(raw_densities) * density_noise_std
            )
        densities = 1 - (-deltas * torch.relu(raw_densities)).exp()
        return densities

    def _get_colors(self, features: torch.Tensor, rays_directions: torch.Tensor):
        """
        This function takes per-point `features` predicted by `self.mlp_xyz`
        and evaluates the color model in order to attach to each
        point a 3D vector of its RGB color.
        """
        spatial_size = features.shape[:-1]
        # Normalize the ray_directions to unit l2 norm.
        rays_directions_normed = F.normalize(rays_directions, dim=-1)

        # Obtain the harmonic embedding of the normalized ray directions.
        rays_embedding = torch.cat(
            (
                self.harmonic_embedding_dir(rays_directions_normed),
                rays_directions_normed,
            ),
            dim=-1,
        )

        # Expand the ray directions tensor so that its spatial size
        # is equal to the size of features.
        rays_embedding_expand = rays_embedding[..., None, :].expand(
            *spatial_size, rays_embedding.shape[-1]
        )

        # Concatenate ray direction embeddings with
        # features and evaluate the color model.
        color_layer_input = torch.cat(
            (self.intermediate_linear(features), rays_embedding_expand), dim=-1
        )
        return self.color_layer(color_layer_input)

    def points_to_features(self, rays_points_world: torch.Tensor) -> torch.Tensor:
        # For each 3D world coordinate, we obtain its harmonic embedding.
        embeds_xyz = torch.cat(
            (self.harmonic_embedding_xyz(rays_points_world), rays_points_world),
            dim=-1,
        )
        # embeds_xyz.shape = [minibatch x ... x self.n_harmonic_functions*6 + 3]

        # self.mlp maps each harmonic embedding to a latent feature space.
        features = self.mlp_xyz(embeds_xyz, embeds_xyz)
        # features.shape = [minibatch x ... x self.n_hidden_neurons_xyz]

        return features

    def forward(
        self,
        ray_bundle: RayBundle,
        density_noise_std: float = 0.0,
        **kwargs,
    ):
        """
        The forward function accepts the parametrizations of
        3D points sampled along projection rays. The forward
        pass is responsible for attaching a 3D vector
        and a 1D scalar representing the point's
        RGB color and opacity respectively.

        Args:
            ray_bundle: A RayBundle object containing the following variables:
                origins: A tensor of shape `(minibatch, ..., 3)` denoting the
                    origins of the sampling rays in world coords.
                directions: A tensor of shape `(minibatch, ..., 3)`
                    containing the direction vectors of sampling rays in world coords.
                lengths: A tensor of shape `(minibatch, ..., num_points_per_ray)`
                    containing the lengths at which the rays are sampled.
            density_noise_std: A floating point value representing the
                variance of the random normal noise added to the output of
                the opacity function. This can prevent floating artifacts.

        Returns:
            rays_densities: A tensor of shape `(minibatch, ..., num_points_per_ray, 1)`
                denoting the opacitiy of each ray point.
            rays_colors: A tensor of shape `(minibatch, ..., num_points_per_ray, 3)`
                denoting the color of each ray point.
        """
        # We first convert the ray parametrizations to world
        # coordinates with `ray_bundle_to_ray_points`.
        rays_points_world = ray_bundle_to_ray_points(ray_bundle)
        # rays_points_world.shape = [minibatch x ... x 3]

        # Harmonic-embed points and get hidden features
        features = self.points_to_features(rays_points_world)
        # features.shape = [minibatch x ... x self.n_hidden_neurons_xyz]

        rays_densities = self._get_densities(
            features, ray_bundle.lengths, density_noise_std
        )
        # rays_densities.shape = [minibatch x ... x 1] in [0-1]

        rays_colors = self._get_colors(features, ray_bundle.directions)
        # rays_colors.shape = [minibatch x ... x 3] in [0-1]

        return rays_densities, rays_colors

    def points_to_raw_densities(self, rays_points_world:torch.Tensor):
        """
            Compute and return raw densities at points specified by
            rays_points_world (minibatch x ... x 3)
        """
        # Harmonic-embed points and get hidden features
        features = self.points_to_features(rays_points_world)
        # features.shape = [minibatch x ... x self.n_hidden_neurons_xyz]

        # Get raw densities
        raw_densities = self._get_raw_densities(features)
        # raw_densities.shape = [minibatch x ... x 1]

        return raw_densities

class MLPWithInputSkips(torch.nn.Module):
    def __init__(
        self,
        n_layers: int,
        input_dim: int,
        output_dim: int,
        skip_dim: int,
        hidden_dim: int,
        input_skips: List[int] = (),
    ):
        super().__init__()
        layers = []
        for layeri in range(n_layers):
            if layeri == 0:
                dimin = input_dim
                dimout = hidden_dim
            elif layeri in input_skips:
                dimin = hidden_dim + skip_dim
                dimout = hidden_dim
            else:
                dimin = hidden_dim
                dimout = hidden_dim
            layers.append(
                torch.nn.Sequential(torch.nn.Linear(dimin, dimout), torch.nn.ReLU(True))
            )
        self.mlp = torch.nn.ModuleList(layers)
        self._input_skips = set(input_skips)

    def forward(self, x, z):
        y = x
        for li, layer in enumerate(self.mlp):
            if li in self._input_skips:
                y = torch.cat((y, z), dim=-1)
            y = layer(y)
        return y
