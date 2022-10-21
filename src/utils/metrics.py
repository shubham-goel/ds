import logging
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from pytorch3d import transforms
from pytorch3d.ops import (iterative_closest_point, knn_gather, knn_points,
                           sample_points_from_meshes)
from pytorch3d.renderer.cameras import CamerasBase
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.transforms.rotation_conversions import matrix_to_axis_angle

from .misc import add_prefix_to_keys


def camera_metric_keys(per_camera: bool = False, N=0):
    keys = []
    for prefix in ['R/', 'T/', 'Tx/', 'Ty/', 'Tz/', 'fov/']:
        keys.append(f'{prefix}avg')
        if per_camera:
            keys += [f'{prefix}cam{i}' for i in range(N)]
    return keys

@torch.no_grad()
def compare_cameras(pred_cameras: CamerasBase, gt_cameras: CamerasBase,
    centre: torch.Tensor = torch.zeros(3), per_camera: bool = False,
    wrt_cam0 = False) -> dict:
    cam_metrics = {}

    # We compute and store metrics independently for each camera
    def loss_tensor_to_dict(t, prefix=''):
        d = {}
        if per_camera:
            d.update({f'{prefix}cam{i}':float(v) for i,v in enumerate(t)})
        d[f'{prefix}avg'] = float(t.mean())
        return d

    centre = centre.to(pred_cameras.R.device)
    pred_R, pred_T = pred_cameras.R.transpose(1,2), pred_cameras.T
    gt_R, gt_T = gt_cameras.R.transpose(1,2), gt_cameras.T

    if wrt_cam0:
        # Move cameras to coordinate system of first camera
        pred_R0, pred_T0 = pred_R[0:1], pred_T[0:1]
        gt_R0, gt_T0 = gt_R[0:1], gt_T[0:1]

        pred_R = pred_R @ pred_R0.transpose(1,2)
        pred_T = pred_T - (pred_R0.transpose(1,2)@pred_T0[:,:,None])[:,:,0]
        gt_R = gt_R @ gt_R0.transpose(1,2)
        gt_T = gt_T - (gt_R0.transpose(1,2)@gt_T0[:,:,None])[:,:,0]
        centre = (gt_R0 @ centre + gt_T0)[0]

    # T about centre
    pred_T = pred_T + pred_R@centre
    gt_T = gt_T + gt_R@centre

    rel_R = pred_R @ gt_R.transpose(-1,-2)
    R_err = matrix_to_axis_angle(rel_R).norm(dim=-1) * 180 / np.pi
    cam_metrics.update(loss_tensor_to_dict(R_err, prefix='R/'))

    T_l1 = F.l1_loss(pred_T, gt_T, reduction='none')
    cam_metrics.update(loss_tensor_to_dict(T_l1.pow(2).mean(dim=-1).sqrt(), prefix='T/'))
    cam_metrics.update(loss_tensor_to_dict(T_l1[..., 0], prefix='Tx/'))
    cam_metrics.update(loss_tensor_to_dict(T_l1[..., 1], prefix='Ty/'))
    cam_metrics.update(loss_tensor_to_dict(T_l1[..., 2], prefix='Tz/'))

    pred_fov = 1/torch.atan(pred_cameras.focal_length) * 180 / np.pi
    gt_fov = 1/torch.atan(gt_cameras.focal_length) * 180 / np.pi
    fov_l1 = F.l1_loss(pred_fov, gt_fov, reduction='none')
    if fov_l1.dim() == 2:
        fov_l1= fov_l1.mean(axis=1)
    cam_metrics.update(loss_tensor_to_dict(fov_l1, prefix='fov/'))

    return cam_metrics

def compare_meshes_align_notalign(mesh_pred, mesh_gt, icp_align=True, **kwargs):
    """ Compute both aligned unaligned"""
    if kwargs.get('return_per_point_metrics', False):
        raise ValueError('Call compare_meshes directly to get per-point metrics')
    shape_metrics = compare_meshes(mesh_pred, mesh_gt, align_icp=False, **kwargs)
    if icp_align:
        aligned_shape_metrics = compare_meshes(mesh_pred, mesh_gt, align_icp=True, **kwargs)
        aligned_shape_metrics = add_prefix_to_keys(aligned_shape_metrics, prefix='Aligned')
        shape_metrics.update(aligned_shape_metrics)
    return shape_metrics

@torch.no_grad()
def compare_meshes(
    pred_meshes, gt_meshes, num_samples=10000, scale="gt-10", thresholds=None, reduce=True, eps=1e-8,
    return_per_point_metrics = False, unscale_returned_points = False, align_icp = False,
):
    """
    Compute evaluation metrics to compare meshes and pointclouds. We currently compute the
    following metrics:

    - L2 Chamfer distance
    - Normal consistency
    - Absolute normal consistency
    - Precision at various thresholds
    - Recall at various thresholds
    - F1 score at various thresholds

    Inputs:
        - pred_meshes (Meshes): Contains N predicted meshes
        - gt_meshes (Meshes): Contains 1 or N ground-truth meshes. If gt_meshes
          contains 1 mesh, it is replicated N times.
        - num_samples: The number of samples to take on the surface of each mesh.
          This can be one of the following:
            - (int): Take that many uniform samples from the surface of the mesh
            - 'verts': Use the vertex positions as samples for each mesh
            - A tuple of length 2: To use different sampling strategies for the
              predicted and ground-truth meshes (respectively).
        - scale: How to scale the predicted and ground-truth meshes before comparing.
          This can be one of the following:
            - (float): Multiply the vertex positions of both meshes by this value
            - A tuple of two floats: Multiply the vertex positions of the predicted
              and ground-truth meshes by these two different values
            - A string of the form 'gt-[SCALE]', where [SCALE] is a float literal.
              In this case, each (predicted, ground-truth) pair is scaled differently,
              so that bounding box of the (rescaled) ground-truth mesh has longest
              edge length [SCALE].
        - thresholds: The distance thresholds to use when computing precision, recall,
          and F1 scores.
        - reduce: If True, then return the average of each metric over the batch;
          otherwise return the value of each metric between each predicted and
          ground-truth mesh.
        - eps: Small constant for numeric stability when computing F1 scores.
        - return_per_point_metrics: If True, additionally returns dict containing
          per-point shape metrics like Chamfer-L2/NormalConsistency and the sampled
          point clouds.
        - unscale_returned_points: If True, unscale sampled pointclouds before returning
        - align_icp: If True, align pred_mesh's sampled point-cloud to gt_mesh using ICP

    Returns:
        - metrics: A dictionary mapping metric names to their values. If reduce is
          True then the values are the average value of the metric over the batch;
          otherwise the values are Tensors of shape (N,).
    """
    if thresholds is None:
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]

    pred_meshes, gt_meshes, (pred_scale, gt_scale) = _scale_meshes(pred_meshes, gt_meshes, scale)

    if isinstance(num_samples, tuple):
        num_samples_pred, num_samples_gt = num_samples
    else:
        num_samples_pred = num_samples_gt = num_samples

    if isinstance(pred_meshes, Meshes):
        pred_points, pred_normals = _sample_meshes(pred_meshes, num_samples_pred)
    else:
        pred_points, pred_normals = _sample_pointclouds(pred_meshes, num_samples_pred)
    if isinstance(gt_meshes, Meshes):
        gt_points, gt_normals = _sample_meshes(gt_meshes, num_samples_gt)
    else:
        gt_points, gt_normals = _sample_pointclouds(gt_meshes, num_samples_gt)
    if pred_points is None:
        logging.warn("Sampling predictions failed during eval")
        return None
    elif gt_points is None:
        logging.warn("Sampling GT failed during eval")
        return None

    if len(gt_meshes) == 1:
        # (1, S, 3) to (N, S, 3)
        gt_points = gt_points.expand(len(pred_meshes), -1, -1)
        gt_normals = gt_normals.expand(len(pred_meshes), -1, -1)

    # Align pred_points and pred_normals to GT
    if align_icp:
        logging.debug(f'compare_meshes running ICP')
        try:
            icpsol = iterative_closest_point(pred_points, gt_points, estimate_scale=True, max_iterations=500)
            if not icpsol.converged:
                logging.warn(f'compare_meshes ICP did not converge. rmse {float(icpsol.rmse)}')
            else:
                logging.debug(f'compare_meshes ICP converged. rmse {float(icpsol.rmse)}')
            if icpsol.RTs.s <= 1e-3:
                raise ValueError(f'ICP collapsed; scale {icpsol.RTs.s} too low')
            if (torch.det(icpsol.RTs.R) - 1).abs().max().item() > 1e-3:
                raise ValueError(f'ICP R invalid with det {torch.det(icpsol.RTs.R)}')
            sRT_transform = transforms.Transform3d(device=pred_points.device).scale(icpsol.RTs.s).rotate(icpsol.RTs.R).translate(icpsol.RTs.T)
            pred_points_new = sRT_transform.transform_points(pred_points)
            pred_normals_new = sRT_transform.transform_normals(pred_normals)
        except Exception as e:
            logging.warn(f'compare_meshes ICP failed')
            logging.error(e)
            pred_points_new = pred_points
            pred_normals_new = pred_normals
        pred_points = pred_points_new
        pred_normals = pred_normals_new

    if torch.is_tensor(pred_points) and torch.is_tensor(gt_points):
        # We can compute all metrics at once in this case
        metrics, metrics_predpt, metrics_gtpt = _compute_sampling_metrics(
            pred_points, pred_normals, gt_points, gt_normals, thresholds, eps
        )
    else:
        # Slow path when taking vert samples from non-equisized meshes; we need
        # to iterate over the batch
        metrics = defaultdict(list)
        metrics_predpt = defaultdict(list)
        metrics_gtpt = defaultdict(list)
        for cur_points_pred, cur_points_gt in zip(pred_points, gt_points):
            cur_metrics, curr_metrics_predpt, curr_metrics_gtpt = _compute_sampling_metrics(
                cur_points_pred[None], None, cur_points_gt[None], None, thresholds, eps
            )
            for k, v in cur_metrics.items():
                metrics[k].append(v.item())
            for k, v in curr_metrics_predpt.items():
                metrics_predpt[k].append(v)
            for k, v in curr_metrics_gtpt.items():
                metrics_gtpt[k].append(v)
        metrics = {k: torch.tensor(vs) for k, vs in metrics.items()}

    if reduce:
        # Average each metric over the batch
        metrics = {k: v.mean().item() for k, v in metrics.items()}

    if return_per_point_metrics:
        if unscale_returned_points:
            pred_points = pred_points / pred_scale
            gt_points = gt_points / gt_scale
        return (
            metrics,
            (pred_points, pred_normals, metrics_predpt),
            (gt_points, gt_normals, metrics_gtpt),
        )
    else:
        return metrics
compare_pointclouds = compare_meshes

def _scale_meshes_or_pcls(pred_meshes, gt_meshes, scale):
    if isinstance(scale, float):
        # Assume scale is a single scalar to use for both preds and GT
        pred_scale = gt_scale = scale
    elif isinstance(scale, tuple):
        # Rescale preds and GT with different scalars
        pred_scale, gt_scale = scale
    elif scale.startswith("gt-"):
        # Rescale both preds and GT so that the largest edge length of each GT
        # mesh is target
        target = float(scale[3:])
        bbox = gt_meshes.get_bounding_boxes()  # (N, 3, 2)
        long_edge = (bbox[:, :, 1] - bbox[:, :, 0]).max(dim=1)[0]  # (N,)
        scale = target / long_edge
        if scale.numel() == 1:
            scale = scale.expand(len(pred_meshes))
        pred_scale, gt_scale = scale, scale
    else:
        raise ValueError("Invalid scale: %r" % scale)
    if isinstance(pred_meshes, Meshes):
        pred_meshes = pred_meshes.scale_verts(pred_scale)
    else:
        pred_meshes = pred_meshes.scale(pred_scale)
    if isinstance(gt_meshes, Meshes):
        gt_meshes = gt_meshes.scale_verts(gt_scale)
    else:
        gt_meshes = gt_meshes.scale(gt_scale)
    return pred_meshes, gt_meshes, (pred_scale, gt_scale)
_scale_meshes = _scale_meshes_or_pcls

def _sample_meshes(meshes, num_samples):
    """
    Helper to either sample points uniformly from the surface of a mesh
    (with normals), or take the verts of the mesh as samples.

    Inputs:
        - meshes: A MeshList
        - num_samples: An integer, or the string 'verts'

    Outputs:
        - verts: Either a Tensor of shape (N, S, 3) if we take the same number of
          samples from each mesh; otherwise a list of length N, whose ith element
          is a Tensor of shape (S_i, 3)
        - normals: Either a Tensor of shape (N, S, 3) or None if we take verts
          as samples.
    """
    if num_samples == "verts":
        normals = None
        if meshes.equisized:
            verts = meshes.verts_batch
        else:
            verts = meshes.verts_list
    else:
        verts, normals = sample_points_from_meshes(meshes, num_samples, return_normals=True)
    return verts, normals

def _sample_pointclouds(pointclouds: Pointclouds, num_samples):
    """
    Helper to either sample points uniformly randomly from a pointcloud
    (with normals), or take the points of the mesh as samples.

    Inputs:
        - meshes: A Pointclouds object
        - num_samples: An integer, or the string 'verts'

    Outputs:
        - verts: Either a Tensor of shape (N, S, 3) if we take the same number of
          samples from each pointcloud; otherwise a list of length N, whose ith element
          is a Tensor of shape (S_i, 3)
        - normals: Either a Tensor of shape (N, S, 3) or None if we take verts
          as samples. The normals are are too expensive to compute, always set to [0,0,1]
    """
    if num_samples == "verts":
        normals = None
        if pointclouds.equisized:
            verts = pointclouds.points_padded()
        else:
            verts = pointclouds.points_list()
    else:
        # Sample points from pointclouds
        verts = _sample_points_from_pointclouds(pointclouds, num_samples, return_normals=False)

        # Not sampling normals because the operation is too expensive
        normals = torch.zeros_like(verts)
        normals[:, :, 2] = 1.0

    return verts, normals

def _sample_points_from_pointclouds(pointclouds: Pointclouds, num_samples, return_normals=False):
    num_pcl = len(pointclouds)

    # Initialize samples tensor with fill value 0 for empty pointclouds.
    samples = torch.zeros((num_pcl, num_samples, 3), device=pointclouds.device)

    if return_normals:
        pointclouds.estimate_normals()
        normal_samples = torch.zeros((num_pcl, num_samples, 3), device=pointclouds.device)

    for i, (points, normals) in enumerate(zip(pointclouds.points_list(), pointclouds.normals_list())):
        if len(points) > 0:
            # Randomly sample num_samples rows from points
            idx = torch.randint(0, len(points), (num_samples,), device=points.device)
            samples[i, :, :] = points[idx, :]
            if return_normals:
                normal_samples[i, :, :] = normals[idx, :]

    if return_normals:
        return samples, normal_samples
    else:
        return samples

def _compute_sampling_metrics(pred_points, pred_normals, gt_points, gt_normals, thresholds, eps):
    """
    Compute metrics that are based on sampling points and normals:

    - L2 Chamfer distance
    - Precision at various thresholds
    - Recall at various thresholds
    - F1 score at various thresholds
    - Normal consistency (if normals are provided)
    - Absolute normal consistency (if normals are provided)

    Inputs:
        - pred_points: Tensor of shape (N, S, 3) giving coordinates of sampled points
          for each predicted mesh
        - pred_normals: Tensor of shape (N, S, 3) giving normals of points sampled
          from the predicted mesh, or None if such normals are not available
        - gt_points: Tensor of shape (N, S, 3) giving coordinates of sampled points
          for each ground-truth mesh
        - gt_normals: Tensor of shape (N, S, 3) giving normals of points sampled from
          the ground-truth verts, or None of such normals are not available
        - thresholds: Distance thresholds to use for precision / recall / F1
        - eps: epsilon value to handle numerically unstable F1 computation

    Returns:
        - metrics: A dictionary where keys are metric names and values are Tensors of
          shape (N,) giving the value of the metric for the batch
        - metrics_predpt: A dictionary where keys are metric names and values are Tensors of
          shape (N,S) giving the value of the un-directional metric for each pred_point
        - metrics_gtpt: A dictionary where keys are metric names and values are Tensors of
          shape (N,S) giving the value of the un-directional metric for each gt_point
    """
    metrics = {}
    metrics_predpt = {}
    metrics_gtpt = {}
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
    pred_to_gt_dists = pred_to_gt_dists2.sqrt()  # (N, S)
    if gt_normals is not None:
        pred_normals_near = knn_gather(gt_normals, knn_pred.idx, lengths_gt)[..., 0, :]  # (N, S, 3)
    else:
        pred_normals_near = None

    # For each GT point, find its nearest-neighbor predicted point
    knn_gt = knn_points(gt_points, pred_points, lengths1=lengths_gt, lengths2=lengths_pred, K=1)
    # Compute L1 and L2 dists between each GT point and its nearest pred point
    gt_to_pred_dists2 = knn_gt.dists[..., 0]  # (N, S)
    gt_to_pred_dists = gt_to_pred_dists2.sqrt()  # (N, S)

    if pred_normals is not None:
        gt_normals_near = knn_gather(pred_normals, knn_gt.idx, lengths_pred)[..., 0, :]  # (N, S, 3)
    else:
        gt_normals_near = None

    # Compute L2 chamfer distances
    chamfer_l2 = pred_to_gt_dists2.mean(dim=1) + gt_to_pred_dists2.mean(dim=1)
    metrics["Chamfer-L2"] = chamfer_l2
    metrics_predpt["Chamfer-L2"] = pred_to_gt_dists2
    metrics_gtpt["Chamfer-L2"] = gt_to_pred_dists2

    # Compute normal consistency and absolute normal consistance only if
    # we actually got normals for both meshes
    if pred_normals is not None and gt_normals is not None:
        pred_to_gt_cos = F.cosine_similarity(pred_normals, pred_normals_near, dim=2)
        gt_to_pred_cos = F.cosine_similarity(gt_normals, gt_normals_near, dim=2)

        pred_to_gt_cos_sim = pred_to_gt_cos.mean(dim=1)
        pred_to_gt_abs_cos_sim = pred_to_gt_cos.abs().mean(dim=1)
        gt_to_pred_cos_sim = gt_to_pred_cos.mean(dim=1)
        gt_to_pred_abs_cos_sim = gt_to_pred_cos.abs().mean(dim=1)
        normal_dist = 0.5 * (pred_to_gt_cos_sim + gt_to_pred_cos_sim)
        abs_normal_dist = 0.5 * (pred_to_gt_abs_cos_sim + gt_to_pred_abs_cos_sim)
        metrics["NormalConsistency"] = normal_dist
        metrics["AbsNormalConsistency"] = abs_normal_dist
        metrics_predpt["NormalConsistency"] = pred_to_gt_cos
        metrics_predpt["AbsNormalConsistency"] = pred_to_gt_cos.abs()
        metrics_gtpt["NormalConsistency"] = gt_to_pred_cos
        metrics_gtpt["AbsNormalConsistency"] = gt_to_pred_cos.abs()

    # Compute precision, recall, and F1 based on L2 distances
    for t in thresholds:
        precision = 100.0 * (pred_to_gt_dists < t).float().mean(dim=1)
        recall = 100.0 * (gt_to_pred_dists < t).float().mean(dim=1)
        f1 = (2.0 * precision * recall) / (precision + recall + eps)
        metrics["Precision@%f" % t] = precision
        metrics["Recall@%f" % t] = recall
        metrics["F1@%f" % t] = f1

    # Move all metrics to CPU
    metrics = {k: v.cpu() for k, v in metrics.items()}
    metrics_predpt = {k: v.cpu() for k, v in metrics_predpt.items()}
    metrics_gtpt = {k: v.cpu() for k, v in metrics_gtpt.items()}
    return metrics, metrics_predpt, metrics_gtpt
