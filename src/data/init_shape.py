import logging
import os
from typing import Tuple

import pytorch3d
import pytorch3d.io
import torch
from dotmap import DotMap
from pytorch3d.structures import Meshes
from pytorch3d.utils import ico_sphere

from ..nnutils import geom_utils
from ..utils import mesh as mesh_util


def initialize_shape(
    data: DotMap,
    ico_sphere_level: int = 2,
    init_shape_path: str = 'init_shape.obj'
):
    ## --- Initialize Shape to a sphere
    cameras_list = geom_utils.poses_to_cameras_list(data.train.poses, data.train.hfovs)
    mesh_centre, mesh_radius = geom_utils.get_centre_radius(data.train.rgba[:,3:,:,:], cameras_list)
    mesh_init = ico_sphere(ico_sphere_level).scale_verts_(mesh_radius/2)
    mesh_init.offset_verts_(mesh_centre[None].expand_as(mesh_init.verts_packed()))
    mesh_init_fuv_idx = mesh_init.faces_packed()
    mesh_init_verts_uv = geom_utils.convert_3d_to_uv_coordinates(mesh_init.verts_packed())

    # Save mesh to file
    mesh_util.save_mesh(init_shape_path, mesh_init, verts_uvs=mesh_init_verts_uv, faces_uvs=mesh_init_fuv_idx)

    data.update(
        mesh_init = mesh_init,
        mesh_init_fuv_idx = mesh_init_fuv_idx,
        mesh_init_verts_uv = mesh_init_verts_uv,
    )

    if data.mesh_init is not None:
        logging.info(f'InitMesh:   {data.mesh_init.verts_packed().shape[0]:6d}verts {data.mesh_init.faces_packed().shape[0]:6d}faces')
    if data.mesh_gt is not None:
        logging.info(f'GTMesh:     {data.mesh_gt.verts_packed().shape[0]:6d}verts {data.mesh_gt.faces_packed().shape[0]:6d}faces')

    return data

def prep_blender_uvunwrap(
    verts: torch.Tensor,
    faces: torch.Tensor,
    simplify: bool = True,
) -> Tuple[Meshes, torch.Tensor, torch.Tensor]:
    """ Preprocess marching-cubed mesh with blender to reduce resolution, find uv-mapping"""
    if verts.shape[0]==0 or faces.shape[0]==0:
        logging.warn(f'Got zero sized inputs: {list(verts.shape)} {list(faces.shape)}')

    finp = os.path.join(os.getcwd(), f'blender-uvunwrap.obj')
    fout = os.path.join(os.getcwd(), f'blender-uvunwrap-uv.obj')
    logging.info(f'Saving mesh to {finp}')
    pytorch3d.io.save_obj(finp, verts, faces, decimal_places=10)

    blender_unwrap_file = os.path.join(
        os.path.dirname(__file__),
        'blender-preprocess-uvunwrap.py'
    )
    blender_call_cmd = f'blender -b -P {blender_unwrap_file} -- {finp} {fout} {simplify} > prep_blender_uvunwrap.out 2>&1'
    logging.info(f'Calling blender: {blender_call_cmd}')
    if os.system(blender_call_cmd)!=0:
        raise ChildProcessError('Blender preprocessing falied')
    logging.info(f'done.')
    return mesh_util.load_mesh_from_file(fout, device=verts.device)
