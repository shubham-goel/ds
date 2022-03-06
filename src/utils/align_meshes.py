import itertools
import logging
import random
from typing import Tuple

import numpy as np
import torch
from pytorch3d.ops import knn_points, sample_points_from_meshes
from pytorch3d.ops.points_alignment import iterative_closest_point
from pytorch3d.structures.meshes import Meshes
from pytorch3d.transforms import Rotate, Scale, Transform3d, Translate
from pytorch3d.transforms.rotation_conversions import euler_angles_to_matrix

from .mesh import (RTs_to_transform, load_mesh_from_file, save_mesh,
                   transform_mesh)
from .metrics import compare_meshes

logger = logging.getLogger(__name__)

def chamfer_pcl(pred_points, gt_points):
    lengths_pred = torch.full(
        (pred_points.shape[0],), pred_points.shape[1], dtype=torch.int64, device=pred_points.device
    )
    lengths_gt = torch.full(
        (gt_points.shape[0],), gt_points.shape[1], dtype=torch.int64, device=gt_points.device
    )

    # For each predicted point, find its neareast-neighbor GT point
    knn_pred = knn_points(pred_points, gt_points, lengths1=lengths_pred, lengths2=lengths_gt, K=1)
    # Compute L1 and L2 distances between each pred point and its nearest GT
    pred_to_gt_dists2 = knn_pred.dists[..., 0]  # (N, S)

    # For each GT point, find its nearest-neighbor predicted point
    knn_gt = knn_points(gt_points, pred_points, lengths1=lengths_gt, lengths2=lengths_pred, K=1)
    # Compute L1 and L2 dists between each GT point and its nearest pred point
    gt_to_pred_dists2 = knn_gt.dists[..., 0]  # (N, S)

    # Compute L2 chamfer distances
    chamfer_l2 = pred_to_gt_dists2.mean(dim=1) + gt_to_pred_dists2.mean(dim=1)
    return chamfer_l2

@torch.no_grad()
def align_meshes_c2f(
        mesh_w: Meshes, mesh_c: Meshes,
        az_coarse=np.linspace(0, 360, num=4, endpoint=False, dtype=np.float),   # in degrees
        el_coarse=np.linspace(-90, 90, num=3, endpoint=True, dtype=np.float),  # in degrees
        cr_coarse=np.linspace(0, 360, num=4, endpoint=False, dtype=np.float),   # in degrees
        s_coarse=[0.7,1,1.3],
        t_coarse=[-0.3,0,0.3],
        max_icp_iterations=500,
    ) -> Tuple[Transform3d, bool]:
    """ Returns w2c transformation aligning mesh_w and mesh_c, and bool indicating whether alignment succeeded """

    assert(len(mesh_w)==1)
    assert(len(mesh_c)==1)

    # Sample points from meshes
    points_w = sample_points_from_meshes(mesh_w)[0]
    points_c = sample_points_from_meshes(mesh_c)[0]

    transforms_w = []
    transforms_c = []
    device = points_w.device

    ## Zero-centre and scale meshes to lie within unit cube
    # Match pcl centres
    def get_centre(points):
        """ Returns centre of Px3 points as their mean """
        return points.mean(dim=0, keepdim=True)
    centre_w = get_centre(points_w)
    centre_c = get_centre(points_c)
    transforms_w.append(Translate(-centre_w, device=device))
    transforms_c.append(Translate(-centre_c, device=device))
    points_w = transforms_w[-1].transform_points(points_w)
    points_c = transforms_c[-1].transform_points(points_c)

    # Match pcl scale
    def get_scale(points):
        """ Returns size of Px3 mean-centered points """
        return points.abs().max()
    scale_w = get_scale(points_w)
    scale_c = get_scale(points_c)
    transforms_w.append(Scale(1/(scale_w+1e-6), device=device))
    transforms_c.append(Scale(1/(scale_c+1e-6), device=device))
    points_w = transforms_w[-1].transform_points(points_w)
    points_c = transforms_c[-1].transform_points(points_c)

    ## Coarse 2 Fine alignment
    # Coarse rotation initialization
    azelcr_list = []
    for az in az_coarse:
        for el in el_coarse:
            for cr in ([0] if (el==90 or el==-90) else cr_coarse):
               azelcr_list.append((az,el,cr))

    best_chamfer = float('inf')
    best_coarse_transform = None
    best_fine_RTs = None
    iteration = 0
    for azelcr_np in azelcr_list:
        for s in s_coarse:
            for T in itertools.product(t_coarse,t_coarse,t_coarse):
                azelcr = torch.as_tensor(azelcr_np, device=device, dtype=torch.float)[None] * np.pi/180
                R = euler_angles_to_matrix(azelcr, "YXZ")
                T = torch.as_tensor(T, device=device, dtype=torch.float)[None]
                coarse_transforms_list = [
                    Rotate(R, device=device),
                    Scale(s, device=device),
                    Translate(T, device=device),
                ]
                coarse_transforms = Transform3d(device=device).compose(*coarse_transforms_list)

                # Coarse init
                points_w_init = coarse_transforms.transform_points(points_w.clone())

                # Fine refinement
                icpsol = iterative_closest_point(points_w_init[None], points_c[None], estimate_scale=True, max_iterations=max_icp_iterations)
                if not icpsol.converged:
                    pp = lambda x: list(x.detach().cpu().numpy())
                    logger.warn(f'iter {iteration} (azelcr{pp(azelcr[0])}, s{s}, T{pp(T[0])}) align_meshes_c2f ICP did not converge. rmse {float(icpsol.rmse.item())}')
                else:
                    logger.debug(f'align_meshes_c2f ICP converged. rmse {float(icpsol.rmse.item())}')

                # Compute chamfer loss on point-clouds
                chamfer_l2 = chamfer_pcl(icpsol.Xt, points_c[None])

                # Is this the best pose?
                iteration += 1
                # if icpsol.converged and (icpsol.rmse.item() < best_chamfer):
                if chamfer_l2.item() < best_chamfer:
                    pp = lambda x: list(x.detach().cpu().numpy())
                    logger.warn(f'iter {iteration} (azelcr{pp(azelcr[0])}, s{s}, T{pp(T[0])}) found better chamfer {float(chamfer_l2.item())}')
                    best_chamfer = chamfer_l2.item()
                    best_coarse_transform = coarse_transforms
                    best_fine_RTs = icpsol.RTs

    ## Compose all transforms together
    if best_coarse_transform is not None:
        assert best_fine_RTs is not None
        transforms_w.append(best_coarse_transform)
        transforms_w.append(RTs_to_transform(best_fine_RTs))
    transforms_c = Transform3d(device=device).compose(*transforms_c)
    transforms_w = Transform3d(device=device).compose(*transforms_w)
    transforms_w2c = transforms_w.compose(transforms_c.inverse())

    return transforms_w2c, (best_coarse_transform is not None)

@torch.no_grad()
def align_and_compare_meshes(mesh_pred: Meshes, mesh_gt: Meshes, rot_align_euler=(0,0,0), **kwargs):
    """ First align meshes, then compute metrics """
    if rot_align_euler is not None:
        # Rotations are already aligned
        # Don't brute-force over possible rotations
        kwargs.update(
            az_coarse=[rot_align_euler[0]],
            el_coarse=[rot_align_euler[1]],
            cr_coarse=[rot_align_euler[2]]
        )
    # kwargs.update(s_coarse=[1], t_coarse=[0])
    logger.info('Aligning meshes...')
    transform_pred2gt, success = align_meshes_c2f(mesh_pred, mesh_gt, **kwargs)

    # Transform pred mesh
    mesh_pred = transform_mesh(mesh_pred, transform_pred2gt)

    # Compute metrics
    logger.info('Computing metrics...')
    metrics = compare_meshes(mesh_pred, mesh_gt)

    return success, transform_pred2gt, mesh_pred, metrics


def align_depth(mesh_pred, mesh_gt,
        possible_scales = np.logspace(-0.5,0.5,base=2,num=51),
    ):
    """
        Inputs:
            mesh_pred: predicted mesh in camera coordinate system
            mesh_gt: GT mesh in camera coordinate system

        Assume camera is at origin pointing towards +Z

        Returns:
            scale: Factor by which mesh_pred should be scaled to align (minimize chamfer distance) to mesh_gt
            transform_pred2gt: Transform to mesh_pred aligning (minimize chamfer distance) to mesh_gt.
                                Is a scale transform
    """
    assert(len(mesh_pred)==1)
    assert(len(mesh_gt)==1)
    points_pred = sample_points_from_meshes(mesh_pred)[0]
    points_gt = sample_points_from_meshes(mesh_gt)[0]

    z_mean0 = points_gt.mean(dim=0)[2].item()
    z_mean1 = points_pred.mean(dim=0)[2].item()
    # print('z_mean', z_mean0, z_mean1)

    s_init = z_mean0/(z_mean1 + 1e-6)
    points_pred = points_pred * s_init
    # print('s_init', s_init)

    z_std0 = points_gt.std(dim=0)[2].item()
    z_std1 = points_pred.std(dim=0)[2].item()
    z_std = max([z_std0, z_std1])
    # print('z_std', z_std0, z_std1)

    best_chamfer = float('inf')
    best_s = None
    for s in possible_scales:
        # print(s)
        points_pred_aligned = s * points_pred
        chamfer = chamfer_pcl(points_pred_aligned[None], points_gt[None]).item()
        if chamfer < best_chamfer:
            prefix_str = f'align_depth scale {s}'
            print(f'{prefix_str} found better chamfer {chamfer}')
            best_chamfer = chamfer
            best_s = s

    best_s = best_s * s_init
    # print('final best_s', best_s)
    return best_s, Transform3d(device=points_pred.device).scale(best_s)

if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    # id = '9e19e66686ed019f811a574d57de824a'
    # f_pred = f'sandbox/content/ShapeNetOBJ/{id}-pred.obj'
    # f_gt = f'sandbox/content/ShapeNetOBJ/{id}-gt.obj'
    # f_out = f'sandbox/content/ShapeNetOBJ/{id}-pred-aligned.obj'

    f_pred = f'sandbox/content/content/3d_sample_data/B07B4MD9X1/91yGIDH7YBL.obj'
    f_gt = f'sandbox/content/content/3d_sample_data/B07B4MD9X1/B07B4MD9X1.obj'
    f_out = f'sandbox/content/content/3d_sample_data/B07B4MD9X1/91yGIDH7YBL-aligned.obj'

    # f_pred = f'sandbox/content/content/3d_sample_data/B07GFL6RF5/81Y8+Ykk4BL.obj'
    # f_gt = f'sandbox/content/content/3d_sample_data/B07GFL6RF5/B07GFL6RF5.obj'
    # f_out = f'sandbox/content/content/3d_sample_data/B07GFL6RF5/81Y8+Ykk4BL-aligned.obj'

    mesh_pred,_,_ = load_mesh_from_file(f_pred)
    mesh_gt,_,_ = load_mesh_from_file(f_gt)

    mesh_pred = mesh_pred.to('cuda')
    mesh_gt = mesh_gt.to('cuda')

    rot_align_euler=(0,0,  0) # For aligning R2N2 pred -> ShapeNet GT voxels
    rot_align_euler=(0,0,-90) # For aligning R2N2 pred -> Amazon GT mesh

    (success, transform_pred2gt, mesh_pred_transformed, metrics) = align_and_compare_meshes(mesh_pred, mesh_gt,
                                                                                rot_align_euler=rot_align_euler)

    print('success', success)
    print(metrics)
    save_mesh(f_out, mesh_pred_transformed)
