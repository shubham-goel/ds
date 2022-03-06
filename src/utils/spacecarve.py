import torch

from ..nnutils.render import grid_sample_ndc


def carve(alphas, poses, hfovs,
        coarse_to_fine_iter=3,
        xyz_min = [-0.2,-0.2,-0.2],
        xyz_max = [0.2,0.2,0.2],
        voxel_res = 256
    ):
    """
    alphas: NxHxW
    poses: Nx4x4
    hfovs: N
    """
    xyz_min = torch.as_tensor(xyz_min).float().to(alphas.device)
    xyz_max = torch.as_tensor(xyz_max).float().to(alphas.device)
    for _ in range(coarse_to_fine_iter):
        xyz_min -= (xyz_max - xyz_min)*0.03
        xyz_max += (xyz_max - xyz_min)*0.03
        xyz_min_old, xyz_max_old = xyz_min, xyz_max
        voxels, xyz_min, xyz_max = carve_region(
            alphas, poses, hfovs,
            xyz_min, xyz_max, voxel_res
        )
        print('space-carving', xyz_min_old, xyz_max_old, voxels.sum().float()/(voxel_res**3))
    return voxels, xyz_min_old, xyz_max_old

def carve_region(alphas, poses, hfovs,
        xyz_min, xyz_max, voxel_res):
    """
    Carve space defined by xyz-ranges using a voxel of resolution voxel_res
    """
    kwargs = {'steps':voxel_res, 'dtype':torch.float32, 'device':alphas.device}
    xs = torch.linspace(xyz_min[0], xyz_max[0], **kwargs)
    ys = torch.linspace(xyz_min[1], xyz_max[1], **kwargs)
    zs = torch.linspace(xyz_min[2], xyz_max[2], **kwargs)

    Wx,Wy,Wz = torch.meshgrid(xs,ys,zs)
    Wxyz = torch.stack([Wx,Wy,Wz], dim=-1)  # V x V x V x 3
    Wxyz = Wxyz.view(-1, 3)                 # V^3 x 3

    Wxyz_occupancy = check_occupancy(Wxyz, alphas, poses, hfovs)
    voxels = Wxyz_occupancy.view(voxel_res,voxel_res,voxel_res)

    occupied = voxels.nonzero()
    imin = (occupied.min(dim=0).values-1).clamp(min=0)
    imax = (occupied.max(dim=0).values+1).clamp(max=voxel_res-1)
    xyz_min_new = torch.stack([xs[imin[0]], ys[imin[1]], zs[imin[2]]])
    xyz_max_new = torch.stack([xs[imax[0]], ys[imax[1]], zs[imax[2]]])

    return voxels, xyz_min_new, xyz_max_new

def check_occupancy(points, alphas, poses, hfovs):
    # Project points onto each image
    assert((poses[:,3,:3]==0).all())
    assert((poses[:,3,3]==1).all())
    R = poses[:,:3,:3]
    t = poses[:,:3,3]
    Cxyz = torch.einsum('pmn,vn->vpm', R, points) + t

    focal = 1 / torch.tan(hfovs)
    Pxy = focal[None,:,None] * Cxyz[:,:,:2] / Cxyz[:,:,2:3] # V^3 x P x 2
    Pxy = Pxy.permute(1,0,2)                                # P x V^3 x 2
    # Pxy = Pxy * -1              # Py3D camera-coordinates, +x is left, +y is up

    # assert(alphas.shape[1]==alphas.shape[2])
    # voxels = F.grid_sample(alphas[:,None], Pxy[:,:,None,:], align_corners=False) # P x 1 x V^3 x 1
    voxels, in_bounds = grid_sample_ndc(alphas[:,None], Pxy[:,:,None,:], align_corners=False, return_whether_in_bounds=True) # P x 1 x V^3 x 1
    voxels = voxels.squeeze(1).squeeze(-1)                      # P x V^3
    in_bounds = in_bounds.squeeze(1).squeeze(-1)                      # P x V^3

    # Points that lay outside image boundaries are assumed occupied
    voxels[~in_bounds] = 1

    # Unoccupied if it lies outside image boundaries of all images
    in_bounds_any = in_bounds.any(dim=0, keepdim=True).expand_as(voxels)
    voxels[~in_bounds_any] = 0

    return voxels.min(dim=0).values
