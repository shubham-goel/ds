import logging
import os
from typing import Optional, Tuple, Union

import mcubes
import numpy as np
import pytorch3d
import pytorch3d.io
import scipy
import torch
from hydra.utils import to_absolute_path
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.ops.points_alignment import (SimilarityTransform,
                                            iterative_closest_point)
from pytorch3d.renderer.cameras import CamerasBase
from pytorch3d.structures import Pointclouds, Volumes, Meshes
from pytorch3d.transforms import Transform3d

from ..nnutils.render import TRANS_EPS
from . import binvox_rw, spacecarve
from .save_textured_mesh import save_obj

BINVOX_PATH = to_absolute_path('./binvox')               # sib


### Topology refinement algos
@torch.no_grad()
def refine_shape_topology_voxelize(
    mesh: Meshes, poses: torch.Tensor, hfovs: torch.Tensor,
    rgba: torch.Tensor, voxelizer='BINVOX', decimate=True, thresh=0.5,
    voxel_postprocessing=[],
    # dilate_iters=1, min_island_size=0,
    **kwargs) -> Meshes:

    # Voxelize mesh surface
    assert voxelizer == 'BINVOX'
    voxels, xyz_min, xyz_max = binvox_mesh_to_voxel(mesh, **kwargs)
    res = voxels.shape[0]
    assert(voxels.shape == (res,res,res))

    # Fill interior of voxels with 1
    logging.info('Filling voxel holes')
    voxels_np = scipy.ndimage.binary_fill_holes(voxels.detach().cpu().numpy())
    voxels = torch.as_tensor(voxels_np, dtype=voxels.dtype, device=voxels.device)
    logging.info(f'voxel mean: {voxels.float().mean().item()}; occupied: {(voxels.float() > 0.5).float().mean().item()}')

    # Check voxels for occupancy
    logging.info('Checking voxels for occupancy')
    voxels_idx = voxels.nonzero()
    voxels_xyz = xyz_min + voxels_idx/(res-1) * (xyz_max - xyz_min)
    voxels_idx_occupied = spacecarve.check_occupancy(voxels_xyz, rgba[:,3,:,:], poses, hfovs)
    voxels_idx_bad = voxels_idx_occupied < 1e-12
    xbad, ybad, zbad = voxels_idx[voxels_idx_bad].unbind(dim=1)
    voxels[xbad, ybad, zbad] = 0
    logging.info(f'voxel mean: {voxels.float().mean().item()}; occupied: {(voxels.float() > 0.5).float().mean().item()}')

    # Post-process/smoothen/dilate voxels before marching-cubes
    voxels_np = voxels.detach().cpu().numpy()
    for method in voxel_postprocessing:
        logging.info(f'Processing voxels: {method}')
        voxels_np = process_binary_ndimage(voxels_np, method)
        logging.info(f'voxel mean: {voxels_np.mean()}; occupied: {(voxels_np > 0.5).mean()}')
    voxels = torch.as_tensor(voxels_np, dtype=voxels.dtype, device=voxels.device)

    # Meshify
    verts, faces = marching_cubes(
                        voxels, thresh,
                        xyz_min = xyz_min,
                        xyz_max = xyz_max,
                        **kwargs
                    )
    verts = torch.as_tensor(verts).float().to(voxels.device)
    faces = torch.as_tensor(faces).long().to(voxels.device)
    out_mesh = Meshes(verts=[verts], faces=[faces])

    if decimate:
        numF_target = mesh.num_faces_per_mesh()[0]
        logging.info(f'Decimating mesh from f{out_mesh.num_faces_per_mesh()[0]} -> f{numF_target}')
        out_mesh = decimate_mesh(out_mesh, numF_target=int(numF_target))

    return out_mesh


##### General Utility #####
### Mesh -> Voxel algos
def binvox_mesh_to_voxel(mesh: Meshes, res=256, pad=0.1, **kwargs):
    assert(len(mesh)==1)
    verts, faces = mesh.get_mesh_verts_faces(0)

    # Compute bounds of voxel grid
    xyz_min = verts.min(dim=0).values
    xyz_max = verts.max(dim=0).values
    xyz_centre = (xyz_min+xyz_max)/2
    xyz_min = xyz_centre + (1+pad)*(xyz_min-xyz_centre)
    xyz_max = xyz_centre + (1+pad)*(xyz_max-xyz_centre)
    minx,miny,minz = [float(x) for x in xyz_min]
    maxx,maxy,maxz = [float(x) for x in xyz_max]

    BINVOX_INP_FILE = 'binvox-input.obj'
    BINVOX_OUT_FILE = 'binvox-input.binvox'

    if not os.path.isfile(BINVOX_PATH):
        raise ValueError(f'binvox not found at {BINVOX_PATH}')
    if os.path.isfile(BINVOX_OUT_FILE):
        logging.warning('binvox output file {} already exists, removing it.')
        os.remove(BINVOX_OUT_FILE)

    # Call Binvox
    save_mesh(BINVOX_INP_FILE, mesh)
    binvox_cmd = f'{BINVOX_PATH} -pb -e -t binvox -d {res} -bb {minx} {miny} {minz} {maxx} {maxy} {maxz} {BINVOX_INP_FILE}'
    logging.info(f'Calling binvox: {binvox_cmd}')
    if os.system(binvox_cmd)!=0:
        raise ChildProcessError('binvox falied')
    logging.info(f'done.')

    # Read binvox data
    voxels, xyz_min, xyz_max = read_binvox(BINVOX_OUT_FILE)

    voxels, xyz_min, xyz_max = (
        torch.as_tensor(voxels).to(verts.device),
        torch.as_tensor(xyz_min).float().to(verts.device),
        torch.as_tensor(xyz_max).float().to(verts.device)
    )

    logging.info(f'voxel mean: {voxels.float().mean()}; occupied: {(voxels.float() > 0.5).float().mean()}')
    return voxels, xyz_min, xyz_max

### Point Cloud -> Voxel algos
def pcl_to_voxel(points: torch.Tensor, res=256, pad=0.1, **kwargs):
    if (singleton := (points.dim()==2)):
        points = points[None]

    assert(points.dim()==3)
    assert(points.shape[2]==3)
    N,P,_ = points.shape
    pointclouds = Pointclouds(
        points=points, features=points.new_zeros((N,P,0))
    )

    # Voxel position/boundaries
    xyz_min = points.min(dim=1).values  # N,3
    xyz_max = points.max(dim=1).values  # N,3
    xyz_centre = (xyz_min+xyz_max)/2    # N,3
    voxel_size = (1+pad) * (xyz_max-xyz_min)/(res-1)

    initial_volumes = Volumes(
        features = points.new_zeros(N, 0, res, res, res),
        densities = points.new_zeros(N, 1, res, res, res),
        volume_translation = -xyz_centre,       # Note the - sign
        voxel_size = voxel_size,
    )
    # add the pointcloud to the 'initial_volumes' buffer using
    # trilinear splatting
    updated_volumes = pytorch3d.ops.add_pointclouds_to_volumes(
        pointclouds=pointclouds,
        initial_volumes=initial_volumes,
        mode="trilinear",
    )

    # Voxels are stored internally in py3d volumes as ZYX. Permute accordingly
    densities = updated_volumes.densities().squeeze(1).permute(0,3,2,1)
    xyz0 = updated_volumes.local_to_world_coords(-points.new_ones((N,3))) # world coordinates for voxel at [-1,-1,-1]
    xyz1 = updated_volumes.local_to_world_coords(points.new_ones((N,3)))  # world coordinates for voxel at [1,1,1]

    if singleton:
        densities = densities[0]
        xyz0 = xyz0[0]
        xyz1 = xyz1[0]

    return densities, xyz0, xyz1

### Misc
def remove_small_voxel_islands(voxels_np, min_island_size):
    # Removed connected components that are smaller than min_island_size
    if min_island_size > 1:
        voxels_np = voxels_np.copy()
        labeled_array, num_features = scipy.ndimage.label(voxels_np)
        for cid in range(num_features+1):
            csize = (labeled_array==cid).sum()
            if csize < min_island_size:
                logging.info(f'Removing small voxel island of size {csize}')
                voxels_np[labeled_array==cid] = 0
    return voxels_np

def process_binary_ndimage(voxels_np, method):
    method = method.lower()
    if method.startswith('close'):
        voxels_np = scipy.ndimage.binary_closing(voxels_np, iterations=int(method[len('close'):]))
    elif method.startswith('dilate'):
        voxels_np = scipy.ndimage.binary_dilation(voxels_np, iterations=int(method[len('dilate'):]))
    elif method.startswith('minisland'):
        voxels_np = remove_small_voxel_islands(voxels_np, int(method[len('minisland'):]))
    else:
        raise ValueError(f'Unkown method {method} of type {type(method)}')
    return voxels_np

@torch.no_grad()
def decimate_mesh(mesh:Meshes, numF_target='x1') -> Meshes:
    assert(len(mesh)==1)
    if isinstance(numF_target, str) and numF_target[0]=='x':
        numF_target = int(float(numF_target[1:]) * mesh.num_faces_per_mesh()[0])
    elif isinstance(numF_target, int):
        pass
    else:
        raise ValueError(f'{numF_target} {type(numF_target)}')

    verts_in, faces_in = mesh.get_mesh_verts_faces(0)
    o3d_mesh_in = _get_o3d_mesh(verts_in.detach().cpu().numpy(), faces_in.detach().cpu().numpy())
    o3d_mesh_smp = o3d_mesh_in.simplify_quadric_decimation(target_number_of_triangles=numF_target)
    verts, faces = np.asarray(o3d_mesh_smp.vertices), np.asarray(o3d_mesh_smp.triangles)
    verts = torch.as_tensor(verts).float().to(verts_in.device)
    faces = torch.as_tensor(faces).long().to(verts_in.device)

    return Meshes([verts], [faces])

def read_binvox(fpath):
    # Read binvox data
    with open(fpath, 'rb') as f:
        voxels = binvox_rw.read_as_3d_array(f)
    logging.info(f'Read 3d voxel grid sized {list(voxels.data.shape)} from {fpath}')
    zero_5 = np.array([0.5,0.5,0.5])
    xyz_min = zero_5/voxels.dims * voxels.scale + voxels.translate
    xyz_max = (voxels.dims - zero_5)/voxels.dims * voxels.scale + voxels.translate
    return voxels.data, xyz_min, xyz_max

def _get_o3d_mesh(verts: np.ndarray, faces: np.ndarray):
    import open3d as o3d
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(verts),
        o3d.utility.Vector3iVector(faces),
    )
    assert(mesh.has_vertices())
    return mesh

def align_shapes(mesh_w: Meshes, mesh_c: Meshes, estimate_scale=True, verbose=False, num_samples=10000) -> SimilarityTransform:
    """ Returns w2c transformation aligning mesh_w and mesh_c"""
    points_w = sample_points_from_meshes(mesh_w, num_samples=num_samples)
    points_c = sample_points_from_meshes(mesh_c, num_samples=num_samples)
    icpsol = iterative_closest_point(points_w, points_c, estimate_scale=estimate_scale, max_iterations=500, verbose=verbose)
    if not icpsol.converged:
        logging.warn(f'compare_meshes ICP did not converge. rmse {float(icpsol.rmse)}')
    else:
        logging.debug(f'compare_meshes ICP converged. rmse {float(icpsol.rmse)}')
    return icpsol.RTs

def transform_mesh(meshes: Meshes, transforms: Union[Transform3d,SimilarityTransform]):
    """ Transforms a batch of meshes, by a batch of transform """
    if isinstance(transforms, SimilarityTransform):
        transforms = RTs_to_transform(transforms)
    mesh_verts = meshes.verts_padded()
    mesh_verts_new = transforms.transform_points(mesh_verts, eps=TRANS_EPS)
    meshes_new = meshes.update_padded(new_verts_padded=mesh_verts_new)
    return meshes_new

def RTs_to_transform(RTs: SimilarityTransform) -> Transform3d:
    return Transform3d(device=RTs.R.device).scale(RTs.s).rotate(RTs.R).translate(RTs.T)

def transform_to_RTs(tr: Transform3d) -> SimilarityTransform:
    mat = tr.get_matrix()
    assert(mat[:,3,3].isclose(torch.ones_like(mat[:,3,3])).all())
    s = torch.det(mat[:,:3,:3]).pow(1/3)
    R = mat[:,:3,:3]/s[:,None,None]
    T = mat[:,3,:3]
    return SimilarityTransform(R,T,s)

def invert_RTs(RTs: SimilarityTransform) -> SimilarityTransform:
    R = RTs.R.transpose(1,2)
    s = 1/(RTs.s+1e-12)
    T = -s * (RTs.T[:,None,:] @ R)[:,0,:]
    return SimilarityTransform(R, T, s)

def transform_cameras(cameras: CamerasBase, RTs: SimilarityTransform):
    cameras = cameras.clone()
    cameras.R = RTs.R.transpose(1,2) @ cameras.R
    cameras.T = RTs.s * cameras.T - (RTs.R @ RTs.T.unsqueeze(2)).squeeze(2)
    return cameras

@torch.no_grad()
def save_mesh(fpath: str, mesh: Meshes, decimal_places: int = 8, verts_uvs: Optional[list] = None,
             faces_uvs: Optional[list] = None, texture_map: Optional[list] = None) -> None:
    save_obj(
        fpath,
        mesh.verts_packed(),
        mesh.faces_packed(),
        decimal_places=decimal_places,
        verts_uvs = verts_uvs,
        faces_uvs = faces_uvs,
        texture_map = texture_map
    )

def load_mesh_from_file(file_path: str, device = 'cpu') -> Tuple[Meshes, torch.Tensor, torch.Tensor]:
    verts, faces, aux = pytorch3d.io.load_obj(file_path)
    logging.info(f'loaded mesh: v{list(verts.shape)} f{list(faces.verts_idx.shape)}')
    return (
        Meshes(verts[None].to(device), faces.verts_idx[None].to(device)),
        aux.verts_uvs.to(device) if aux.verts_uvs is not None else None,
        faces.textures_idx.to(device) if faces.textures_idx is not None else None
    )

def marching_cubes(
    voxels: torch.Tensor,
    voxel_thresh: float,
    xyz_min: torch.Tensor = torch.tensor([-1,-1,-1]).float(),
    xyz_max: torch.Tensor = torch.tensor([1,1,1]).float(),
    smooth: bool = False,
    **kwargs
) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Run mcubes marching cubes and return """
    # Marching Cubes
    logging.info('Running marching cubes')
    logging.info(f'voxel_mean: {torch.mean((voxels).float())}')
    logging.info(f'fraction of voxels occupied: {torch.mean((voxels > voxel_thresh).float())}')
    voxels_np = voxels.cpu().numpy()
    if smooth:
        voxels_np = voxels_np - voxel_thresh + 0.5
        voxel_thresh = 0
        smooth_kwargs = kwargs.get("smooth_kwargs", {})
        logging.info(f'smoothing voxels with kwargs: {smooth_kwargs}')
        voxels_np = mcubes.smooth(voxels_np, **smooth_kwargs)
    verts, faces = mcubes.marching_cubes(voxels_np, voxel_thresh)
    logging.info(f'done. v{list(verts.shape)} f{list(faces.shape)}')

    device = voxels.device
    xyz_min = xyz_min.to(device)
    xyz_max = xyz_max.to(device)
    voxel_res = voxels.shape[0]
    verts = torch.tensor(verts, dtype=torch.float32, device=device)
    faces = torch.tensor(faces.astype(np.int64), dtype=torch.long, device=device)
    verts = xyz_min + verts/(voxel_res-1) * (xyz_max-xyz_min)

    return verts, faces

