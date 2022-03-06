"""
Utils related to geometry like projection
"""
from typing import List

import numpy as np
import torch
import torch.nn.functional as F

from ..nerfPy3D.nerf.raysampler import _xy_to_ray_bundle

Tensor = torch.Tensor

def rotate_at(R, T, at, R_adj, premultiply=False):
    if at.dim()==1:
        at = at[None]
    at = at.expand_as(T)
    R_adj = R_adj.expand_as(R)

    N = R.shape[0]
    assert(R.shape==(N,3,3))
    assert(T.shape==(N,3))
    assert(at.shape==(N,3))
    assert(R_adj.shape==(N,3,3))
    T = T + (R @ at.unsqueeze(2)).squeeze(2)
    if premultiply:
        R = R_adj @ R
    else:
        R = R @ R_adj
    T = T - (R @ at.unsqueeze(2)).squeeze(2)
    return R,T

def rotateposes_at(poses, *args, **kwargs):
    R,T = rotate_at(poses[:,:3,:3], poses[:,:3,3], *args, **kwargs)
    return RT_to_poses(R,T)

def RT_to_poses(R, T):
    N = R.shape[0]
    assert(R.shape==(N,3,3))
    assert(T.shape==(N,3))
    return poses34_to_44(torch.cat([R,T[:,:,None]], dim=2))

def poses34_to_44(poses34):
    N, _, _ = poses34.shape
    assert(poses34.shape == (N,3,4))
    lastrow = torch.tensor([0,0,0,1], dtype=poses34.dtype, device=poses34.device)
    lastrow = lastrow[None, None, :].expand_as(poses34[:,:1,:])
    poses44 = torch.cat((poses34, lastrow), dim=1)
    assert(poses44.shape == (N,4,4))
    return poses44

def poses_to_transform(poses):
    from pytorch3d.transforms import Transform3d
    return (
        Transform3d(device=poses.device)
        .rotate(poses[:,:3,:3].transpose(1,2))
        .translate(poses[:,:3,3])
    )

def poses_to_cameras_list(poses, hfovs):
    """
    poses: Nx4x4
    hfovs: N
    """
    from pytorch3d.renderer.cameras import PerspectiveCameras
    N = hfovs.shape[0]
    assert(hfovs.shape==(N,))
    assert(poses.shape==(N,4,4))
    cameras = [
        PerspectiveCameras(
            focal_length=1/torch.tan(hfovs[i])[None],
            R=poses[i,:3,:3].t()[None],
            T=poses[i,:3,3][None],
            device=poses.device
        ) for i in range(len(poses))
    ]
    return cameras

def get_centre_radius(masks, cameras):
    """
    Input:
        masks: Nx1xHxW mask of images
        cameras: N list of pytorch3d cameras

    Shoot rays from wherever mask==1.
    Compute centre as point in 3D space that minimizes sum of squared distances to rays.
    Compute radius as distance from centre to farthest ray.

    Return:
        centre: 3
        radius: float
    """
    N,_,H,W = masks.shape
    # Shoot out rays
    def mask_to_xy(mask):
        h,w = mask.shape
        gridintpts_Px2 = (mask>0.5).nonzero()
        gridintpts_Px2 = gridintpts_Px2.float()
        gridsubpts_Px2 = 2 * gridintpts_Px2 / torch.tensor([h,w], device=mask.device) - 1
        gridsubpts_Px2 = gridsubpts_Px2[:,[1,0]]
        return gridsubpts_Px2

    origins = []
    directions = []
    for i in range(N):
        rays_xy = mask_to_xy(masks[i,0])[None]
        rays_bundle = _xy_to_ray_bundle(cameras[i], rays_xy, 1., 1., 1)
        origins.append(rays_bundle.origins)
        directions.append(rays_bundle.directions)
    origins = torch.cat(origins, dim=1)[0]
    directions = torch.cat(directions, dim=1)[0]

    centre, t, dists = get_centre(origins, directions)
    radius = dists.max().item()
    return centre, radius

def get_centre(origins, directions):
    """
    Get interesction (distance-minimizing point) of skew rays.

    origins: Nx3
    directions: Nx3

    origins, directions parametrize rays as ray_i = (o_i + t_i * d_i).
    Objective function that gives distance from a point p to points t_i on rays:
    L(p,t_1..N) = \sum | p - (o_i + t_i * d_i) |^2

    We minimize L w.r.t p and t. Doing the math gives us close form solutions for t_i and p:
    t_i = (p - o_i).d_i
    A @ p = b; where
        A = \sum (I - d_i @ d_i')
        b = \sum (I - d_i @ d_i') @ o_i

    Returns:
        p: 3
        t: N
        dist: N distances to each ray
    """
    dtype = origins.dtype
    device = origins.device

    directions = F.normalize(directions, p=2, dim=-1)
    ddt = directions[:,:,None] @ directions[:,None,:]
    I_ddt = (torch.eye(3, dtype=dtype, device=device)[None] - ddt)
    A = I_ddt.mean(0)
    b = (I_ddt@origins[:,:,None])[:,:,0].mean(0)

    p = torch.pinverse(A) @ b
    t = ((p[None] - origins)*directions).sum(dim=-1)
    dists = (p[None] - (origins + t[:,None]*directions)).norm(dim=-1)
    return p, t, dists

def sample_points(img_CxHxW:Tensor, kpts_Px2:Tensor) -> Tensor:
    """
        @param img_CxHxW: input image
        @param kpts_Px2: [-1,1] normalized points (x,y) to compute loss
        returns texpts_PxC of img_CxHxW sampled at kpts_Px2
    """
    kpts_1x1xPx2 = kpts_Px2[None,None,:,:]
    texpts_1xCx1xP = F.grid_sample(img_CxHxW[None], kpts_1x1xPx2, align_corners=False)
    texpts_PxC = texpts_1xCx1xP.squeeze(2).squeeze(0).permute(1,0)
    return texpts_PxC

def sample_points_batch(imgs:Tensor, pts_list:List[Tensor]) -> Tensor:
    """
        @param imgs: NxCxHxW input image
        @param pts_list: [ Px2 ] list of length N, of points (x,y) to sample imgs at
        returns NPxC tensor of sampled points
    """
    assert(imgs.shape[0] == len(pts_list))
    texpts_NPxC = []
    for _i in range(len(pts_list)):
        texpts_PxC = sample_points(imgs[_i], pts_list[_i])
        texpts_NPxC.append(texpts_PxC)
    texpts_NPxC = torch.cat(texpts_NPxC, dim=0)
    return texpts_NPxC

def randsample_higherr(errmap:Tensor, thresh:int=0.1, topK:int=500):
    """
    errmap: HxW
    thresh: For efficiency, only consider points with value above thresh
    topK: Return (atmost) topK subpixels with highest errmap

    Sample points with value > thresh
    """
    h,w = errmap.shape
    gridintpts_Px2 = (errmap>thresh).nonzero()
    gridsubpts_Px2 = gridintpts_Px2 + torch.rand_like(gridintpts_Px2.float())
    gridsubpts_Px2 = 2 * gridsubpts_Px2 / torch.tensor([h,w], device=errmap.device) - 1
    gridsubpts_Px2 = gridsubpts_Px2[:,[1,0]]
    gridsubval_Px1 = sample_points(errmap[None], gridsubpts_Px2)
    gridsubval_P = gridsubval_Px1.squeeze(1)
    topK = min(gridsubval_P.shape[0], topK)
    gridsubval_K, gridsubidx_K = torch.topk(gridsubval_P, topK)
    gridsubpts_Kx2 = gridsubpts_Px2[gridsubidx_K, :]
    return gridsubpts_Kx2

def convert_uv_to_3d_coordinates(uv, rad=1):
    '''
    Takes a uv coordinate between [-1,1] and returns a 3d point on the sphere.
    uv -- > [......, 2] shape

    U: Azimuth: Angle with +X [-pi,pi]
    V: Inclination: Angle with +Z [0,pi]
    '''
    phi = np.pi*(uv[...,0])
    theta = np.pi*(uv[...,1]+1)/2

    if type(uv) == torch.Tensor:
        x = torch.sin(theta)*torch.cos(phi)
        y = torch.sin(theta)*torch.sin(phi)
        z = torch.cos(theta)
        points3d = torch.stack([x,y,z], dim=-1)
    else:
        x = np.sin(theta)*np.cos(phi)
        y = np.sin(theta)*np.sin(phi)
        z = np.cos(theta)
        points3d = np.stack([x,y,z], axis=-1)
    return points3d*rad


def convert_3d_to_uv_coordinates(X):
    """
    X : N,3
    Returns UV: N,2 normalized to [-1, 1]
    U: Azimuth: Angle with +X [-pi,pi]
    V: Inclination: Angle with +Z [0,pi]
    """
    if type(X) == torch.Tensor:
        eps=1e-4
        rad = torch.norm(X, dim=-1).clamp(min=eps)
        theta = torch.acos( (X[..., 2] / rad).clamp(min=-1+eps,max=1-eps) )    # Inclination: Angle with +Z [0,pi]
        phi = torch.atan2(X[..., 1], X[..., 0])  # Azimuth: Angle with +X [-pi,pi]
        vv = (theta / np.pi) * 2 - 1
        uu = ((phi + np.pi) / (2*np.pi)) * 2 - 1
        uv = torch.stack([uu, vv],dim=-1)
    else:
        rad = np.linalg.norm(X, axis=-1)
        rad = np.clip(rad, 1e-12, None)
        theta = np.arccos(X[..., 2] / rad)      # Inclination: Angle with +Z [0,pi]
        phi = np.arctan2(X[..., 1], X[..., 0])  # Azimuth: Angle with +X [-pi,pi]
        vv = (theta / np.pi) * 2 - 1
        uu = ((phi + np.pi) / (2*np.pi)) * 2 - 1
        uv = np.stack([uu, vv],-1)
    return uv


def sample_textures(texture_flow, images):
    """
    texture_flow: B x ... x 2
    (In normalized coordinate [-1, 1])
    images: B x 3 x N x N

    output: B x ... x 3
    """
    b = texture_flow.shape[0]
    assert(texture_flow.shape[-1]==2)

    # Reshape into B x 1 x . x 2
    flow_grid_bx1xdx2 = texture_flow.view(b, 1, -1, 2)

    # B x 3 x 1 x .
    # print('images', images.device, images.shape)
    # print('flow_grid', flow_grid.device, flow_grid.shape)
    # print('flow_grid.range', flow_grid.min().item(), flow_grid.max().item())
    samples_bx3x1xd = F.grid_sample(images, flow_grid_bx1xdx2, align_corners=False)
    # print('samples start')
    # print('samples', samples.view(-1))
    # _ = samples.detach().cpu().numpy()
    # print('samples finish')
    # B x 3 x F x T x T
    samples_bx1xdx3 = samples_bx3x1xd.permute(0,2,3,1)
    samples_bxdddx3 = samples_bx1xdx3.view((b,)+texture_flow.shape[1:-1]+(3,))
    return samples_bxdddx3

def orthographic_proj(X, cam):
    """
    X: B x N x 3
    cam: B x 7: [sc, tx, ty, quaternions]
    """
    quat = cam[:, -4:]
    X_rot = quat_rotate(X, quat)

    scale = cam[:, 0].contiguous().view(-1, 1, 1)
    trans = cam[:, 1:3].contiguous().view(cam.size(0), 1, -1)

    return scale * X_rot[:, :, :2] + trans

def orthographic_proj_withz(X, cam, offset_z=0.):
    """
    X: B x ... x 3
    cam: B x 7: [sc, tx, ty, quaternions]
    Orth preserving the z.


    Oparation:
        X_ndc = scale * R * X + [tx,ty,offset_z]'
        R = rotation from quat
        T = [tx , ty , offset_z] / scale
        P = [
                [scale,        0,         0,  0],
                [0,        scale,         0,  0],
                [0,            0,     scale,  0],
                [0,            0,         0,  1],
        ]
    """
    assert(X.shape[0] == cam.shape[0])
    X_flat = X.view(X.shape[0], -1, 3)
    quat = cam[:, -4:]
    X_rot = quat_rotate(X_flat, quat)

    scale = cam[:, 0].contiguous().view(-1, 1, 1)
    trans = cam[:, 1:3].contiguous().view(cam.size(0), 1, -1)

    proj = scale * X_rot

    proj_xy = proj[:, :, :2] + trans
    proj_z = proj[:, :, 2, None] + offset_z

    proj_xyz = torch.cat((proj_xy, proj_z), 2)
    return proj_xyz.view(X.shape)


def perspective_proj(X, cam, cam_f):
    """
    X: B x ... x 3
    cam: B x 7: [sc, tx, ty, quaternions]
    cam_f: B: focal length of cam

    Oparation:
        X_ndc = scale * R * X + [tx,ty,f]'
        x = (RX + tx/scale) * f / (f/scale + RZ)
        R = rotation from quat
        T = [tx , ty , f] / scale
        P = [
                [f,        0,         0,  0],
                [0,        f,         0,  0],
                [0,        0,         0,  1],
                [0,        0,         1,  0],
        ]
    """
    assert(X.shape[0] == cam.shape[0])
    X_flat = X.view(X.shape[0], -1, 3)
    cam_f = cam_f.contiguous().view(-1, 1, 1)

    proj_ortho = orthographic_proj_withz(X_flat, cam, offset_z=0.)

    proj_xy = proj_ortho[:, :, :2]
    proj_z = proj_ortho[:, :, 2, None]
    proj_xy = proj_xy * cam_f / (cam_f + proj_z)

    proj_xyz = torch.cat((proj_xy, proj_z), 2)
    return proj_xyz.view(X.shape)


def orthographic_proj_withz_inverse(X, cam, offset_z=0):
    """
    X: B x N x 3
    cam: B x 7: [sc, tx, ty, quaternions]
    Inverse otho projection
    """
    quat = cam[:, -4:]
    scale = cam[:, 0].contiguous().view(-1, 1, 1)
    trans = cam[:, 1:3].contiguous().view(cam.size(0), 1, -1)

    X_xy = X[:,:,:2] - trans
    X_z  = X[:,:,2:] - offset_z
    X = torch.cat((X_xy, X_z), 2)

    X = X / scale
    X = quat_rotate(X, quat_inverse(quat))

    return X


def cross_product(qa, qb):
    """Cross product of va by vb.

    Args:
        qa: B X N X 3 vectors
        qb: B X N X 3 vectors
    Returns:
        q_mult: B X N X 3 vectors
    """
    qa_0 = qa[..., 0]
    qa_1 = qa[..., 1]
    qa_2 = qa[..., 2]

    qb_0 = qb[..., 0]
    qb_1 = qb[..., 1]
    qb_2 = qb[..., 2]

    # See https://en.wikipedia.org/wiki/Cross_product
    q_mult_0 = qa_1*qb_2 - qa_2*qb_1
    q_mult_1 = qa_2*qb_0 - qa_0*qb_2
    q_mult_2 = qa_0*qb_1 - qa_1*qb_0

    return torch.stack([q_mult_0, q_mult_1, q_mult_2], dim=-1)


def hamilton_product(qa, qb):
    """Multiply qa by qb.

    Args:
        qa: B X N X 4 quaternions
        qb: B X N X 4 quaternions
    Returns:
        q_mult: B X N X 4
    """
    qa_0 = qa[..., 0]
    qa_1 = qa[..., 1]
    qa_2 = qa[..., 2]
    qa_3 = qa[..., 3]

    qb_0 = qb[..., 0]
    qb_1 = qb[..., 1]
    qb_2 = qb[..., 2]
    qb_3 = qb[..., 3]

    # See https://en.wikipedia.org/wiki/Quaternion#Hamilton_product
    q_mult_0 = qa_0*qb_0 - qa_1*qb_1 - qa_2*qb_2 - qa_3*qb_3
    q_mult_1 = qa_0*qb_1 + qa_1*qb_0 + qa_2*qb_3 - qa_3*qb_2
    q_mult_2 = qa_0*qb_2 - qa_1*qb_3 + qa_2*qb_0 + qa_3*qb_1
    q_mult_3 = qa_0*qb_3 + qa_1*qb_2 - qa_2*qb_1 + qa_3*qb_0

    return torch.stack([q_mult_0, q_mult_1, q_mult_2, q_mult_3], dim=-1)


def quat_rotate(X, quat):
    """Rotate points by quaternions.

    Args:
        X: B X N X 3 points
        quat: B X 4 quaternions

    Returns:
        X_rot: B X N X 3 (rotated points)
    """
    # repeat q along 2nd dim
    # import ipdb; ipdb.set_trace()
    # ones_x = X[[0], :, :][:, :, [0]]*0 + 1
    # q = torch.unsqueeze(q, 1)*ones_x
    quat = quat[:,None,:].expand(-1,X.shape[1],-1)

    quat_conj = torch.cat([ quat[:, :, 0:1] , -1*quat[:, :, 1:4] ], dim=-1)
    X = torch.cat([ X[:, :, 0:1]*0, X ], dim=-1)

    X_rot = hamilton_product(quat, hamilton_product(X, quat_conj))
    return X_rot[:, :, 1:4]

def quat_inverse(quat):
    """
    quat: B x 4: [quaternions]
    returns inverted quaternions
    """
    flip = torch.tensor([1,-1,-1,-1],dtype=quat.dtype,device=quat.device)
    quat_inv = quat * flip.view((1,)*(quat.dim()-1)+(4,))
    return quat_inv

def quat2axisangle(quat):
    """
    quat: B x 4: [quaternions]
    returns quaternion axis, angle
    """
    cos = quat[..., 0]
    sin = quat[..., 1:].norm(dim=-1)
    axis = quat[..., 1:]/sin[..., None]
    angle = 2*cos.clamp(-1+1e-6,1-1e-6).acos()
    return axis, angle

def axisangle2quat(axis, angle):
    """
    axis: B x 3: [axis]
    angle: B: [angle]
    returns quaternion: B x 4
    """
    axis = F.normalize(axis,dim=-1)
    angle = angle.unsqueeze(-1)/2
    quat = torch.cat([angle.cos(), angle.sin()*axis], dim=-1)
    return quat

def get_base_quaternions(num_pose_az=8, num_pose_el=1, initial_quat_bias_deg=45., elevation_bias=0):
    _axis = torch.eye(3).float()

    # Quaternion base bias
    xxx_base = [1.,0.,0.]
    aaa_base = initial_quat_bias_deg
    axis_base = torch.tensor(xxx_base).float()
    angle_base = torch.tensor(aaa_base).float() / 180. * np.pi
    qq_base = axisangle2quat(axis_base, angle_base) # 4

    # Quaternion multipose bias
    azz = torch.as_tensor(np.linspace(0,2*np.pi,num=num_pose_az,endpoint=False)).float()
    ell = torch.as_tensor(np.linspace(-np.pi/2,np.pi/2,num=(num_pose_el+1),endpoint=False)[1:]).float() + elevation_bias * np.pi/180
    quat_azz = axisangle2quat(_axis[1], azz) # num_pose_az,4
    quat_ell = axisangle2quat(_axis[0], ell) # num_pose_el,4
    quat_el_az = hamilton_product(quat_ell[None,:,:], quat_azz[:,None,:]) # num_pose_az,num_pose_el,4
    quat_el_az = quat_el_az.view(-1,4)                  # num_pose_az*num_pose_el,4
    _quat = hamilton_product(quat_el_az, qq_base[None,...]).float()

    return _quat

def azElRot_to_quat(azElRot):
    """
    azElRot: ...,az el ro
    """
    _axis = torch.eye(3, dtype=azElRot.dtype, device=azElRot.device)
    num_dims = azElRot.dim()-1
    _axis = _axis.view((1,)*num_dims+(3,3))
    azz = azElRot[..., 0]
    ell = azElRot[..., 1]
    rot = azElRot[..., 2]
    quat_azz = axisangle2quat(_axis[...,1], azz) # ...,4
    quat_ell = axisangle2quat(_axis[...,0], ell) # ...,4
    quat_rot = axisangle2quat(_axis[...,2], rot) # ...,4

    quat = hamilton_product(quat_ell, quat_azz)
    quat = hamilton_product(quat_rot, quat)

    return quat

def reflect_cam_pose(cam_pose):
    batch_dims = cam_pose.dim()-1
    cam_pose = cam_pose * torch.tensor([1, -1, 1, 1, 1, -1, -1],
                                        dtype=cam_pose.dtype,
                                        device=cam_pose.device).view((1,)*batch_dims + (-1,))
    return cam_pose

def reflect_cam_pose_z(cam_pose):
    batch_dims = cam_pose.dim()-1
    axis = torch.tensor([[0, 1, 0]], dtype=cam_pose.dtype, device=cam_pose.device)
    angle = torch.tensor([np.pi], dtype=cam_pose.dtype, device=cam_pose.device)
    rot180 = axisangle2quat(axis, angle).view((1,)*batch_dims + (-1,))
    quat = hamilton_product(rot180, cam_pose[..., 3:7])
    quat = quat * torch.tensor([1, 1, -1, -1],
                                    dtype=quat.dtype,
                                    device=quat.device).view((1,)*batch_dims + (-1,))
    cam_pose = torch.cat((cam_pose[...,:3], quat), dim=-1)
    return cam_pose

def random_point_on_sphere(shape, dtype=torch.float32, device='cpu'):
    zero = torch.zeros(shape, dtype=dtype, device=device)
    one = torch.ones(shape, dtype=dtype, device=device)
    U01 = torch.distributions.Uniform(zero, one)
    theta = 2 * np.pi * U01.sample()
    phi = torch.acos(torch.clamp(1 - 2 * U01.sample(), min=-1+1e-6, max=1-1e-6))
    x = torch.sin(phi) * torch.cos(theta)
    y = torch.sin(phi) * torch.sin(theta)
    z = torch.cos(phi)
    return torch.stack([x,y,z], dim=-1)

def quat_to_azElCr(quat):
    """
        Quat: ...,4
        Assuming camera is right-hand coordinate system with x towards right, y vertically down, z +ve depth.
    """
    assert(quat.shape[-1]==4)
    quat_flat = quat.view(-1,4)

    _axis = torch.eye(3)
    xyz_w = torch.eye(3)
    xyz_c = quat_rotate(xyz_w[None,:,:], quat_flat)

    # Undo cyclo-rot by rotating xyz_c about z-axis s.t. xyz_c[1,:] aligns with y [0,1,0] (upto a z-projection)
    y_c_proj = xyz_c[:,1,0:2]
    angle_cr = torch.atan2(-y_c_proj[:,0], y_c_proj[:,1])
    quat_cr = axisangle2quat(_axis[None,:,2], angle_cr) # ...,4
    xyz_c = quat_rotate(xyz_c, quat_inverse(quat_cr))


    # Undo elev by rotating xyz_c about x-axis s.t. xyz_c[1,:] aligns with y [0,1,0] (exactly)
    y_c_proj = xyz_c[:,1,1:3]
    angle_el = torch.atan2(y_c_proj[:,1], y_c_proj[:,0])
    quat_el = axisangle2quat(_axis[None,:,0], angle_el) # ...,4
    xyz_c = quat_rotate(xyz_c, quat_inverse(quat_el))


    # Undo azim by rotating xyz_c about y-axis s.t. xyz_c[0,:] aligns with x [1,0,0] (exactly)
    x_c = xyz_c[:,0,:]
    angle_az = torch.atan2(-x_c[:,2], x_c[:,0])
    quat_az = axisangle2quat(_axis[None,:,1], angle_az) # ...,4
    xyz_c = quat_rotate(xyz_c, quat_inverse(quat_az))

    angle_azelcr = torch.stack((angle_az,angle_el,angle_cr), dim=-1)
    return angle_azelcr.view(quat.shape[:-1]+(3,))


def invert_poses(poses):
    """
    poses: Nx4x4
    """
    assert((poses[:,3,:3]==0).all())
    assert((poses[:,3,3]==1).all())
    R = poses[:,:3,:3]
    assert((torch.det(R)-1).abs().max() < 1e-4)
    t = poses[:,:3,3:4]
    R = R.transpose(1,2)
    t = -R @ t
    res = torch.zeros_like(poses)
    res[:,:3,:3] = R
    res[:,:3,3:4] = t
    res[:,3,3] = 1
    return res


if __name__ == "__main__":
    azElCr = torch.tensor([
        [np.pi/6,np.pi/5,np.pi/4],
        [-np.pi/6,-np.pi/5,-np.pi/4],
        [-3*np.pi/4,np.pi/5,-2*np.pi/3],
        [0.1,0.2,0.3]
    ])
    quat = azElRot_to_quat(azElCr)
    azElCr1 = quat_to_azElCr(quat)
    print(azElCr)
    print(azElCr1)
    print((azElCr1-azElCr).norm())
