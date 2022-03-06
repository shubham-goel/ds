import logging
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from pytorch3d.renderer import PerspectiveCameras
from pytorch3d.renderer.cameras import CamerasBase
from pytorch3d.transforms import axis_angle_to_matrix, quaternion_to_matrix


class dollyParamCameras(torch.nn.Module):
    def __init__(self, poses, hfovs, centre=torch.zeros(3),
            param_R_axis_angle = False,
            optimize_cam=False, optimize_first=True,
            optimize_R=True, optimize_T=True, optimize_Fov=True,
            hfov_range=10, img_size=(-1,-1), device=None,
            trans_scale=0.1,
        ):
        super().__init__()
        self.param_R_axis_angle = param_R_axis_angle
        self.optimize_first = optimize_first
        self.hfov_range = hfov_range
        self.device = device if device is not None else poses.device
        self.img_size = img_size
        self.N, _, _ = poses.shape

        # Allow float/int hfovs input
        if isinstance(hfovs, int) or isinstance(hfovs, float):
            hfovs = torch.full_like(poses[:,0,0], hfovs)
        elif isinstance(hfovs, torch.Tensor):
            if hfovs.dim()==0:
                hfovs = hfovs[None].expand(self.N)
        else:
            raise TypeError
        assert(hfovs.shape == (self.N,))

        self.register_buffer('trans_scale', torch.tensor(trans_scale))
        self.register_buffer('poses', poses)
        self.register_buffer('hfovs', hfovs)
        self.register_buffer('centre', centre)

        # See self.create_cameras for explanation on camera parameterization
        if param_R_axis_angle:
            self.zero_quat = [0,0,0]
            rel_quats = torch.zeros((self.N - int(not optimize_first),3)) + torch.tensor(self.zero_quat)
        else:
            self.zero_quat = [1,0,0,0]
            rel_quats = torch.zeros((self.N - int(not optimize_first),4)) + torch.tensor(self.zero_quat)
        rel_trans = torch.zeros((self.N - int(not optimize_first),3))
        rel_hfovs = torch.zeros((self.N - int(not optimize_first),))
        if optimize_cam and optimize_R:
            logging.debug('Optimizing rel_R, set rel_R parameters.')
            self.register_parameter('rel_quats', torch.nn.Parameter(rel_quats))
        else:
            logging.debug('Not optimizing rel_R, set rel_R buffers.')
            self.register_buffer('rel_quats', rel_quats)
        if optimize_cam and optimize_T:
            logging.debug('Optimizing rel_T, set rel_T parameters.')
            self.register_parameter('rel_trans', torch.nn.Parameter(rel_trans))
        else:
            logging.debug('Not optimizing rel_T, set rel_T buffers.')
            self.register_buffer('rel_trans', rel_trans)
        if optimize_cam and optimize_Fov:
            logging.debug('Optimizing rel_Fov, set rel_Fov parameters.')
            self.register_parameter('rel_hfovs', torch.nn.Parameter(rel_hfovs))
        else:
            logging.debug('Not optimizing rel_Fov, set rel_Fov buffers.')
            self.register_buffer('rel_hfovs', rel_hfovs)
        logging.info(f'Initialized dollyParamCam (optRTF ' +
                        f'{int(optimize_cam and optimize_R)}' +
                        f'{int(optimize_cam and optimize_T)}' +
                        f'{int(optimize_cam and optimize_Fov)}' +
                        ')')

        self.to(self.device)

    def __len__(self) -> int:
        return self.N

    def create_cameras_list(self, **kwargs) -> List[CamerasBase]:
        return [self.create_cameras(id=i, **kwargs) for i in range(len(self))]

    def get_RTfovF(self, id=slice(None), **kwargs):
        eps = 1e-10

        if isinstance(id, int):
            id = slice(id,id+1)
        poses = self.poses[id]
        hfovs = self.hfovs[id]
        rel_quats = self.rel_quats
        rel_trans = self.rel_trans
        rel_hfovs = self.rel_hfovs
        if not self.optimize_first:
            rel_quats = torch.cat((rel_quats.new_tensor([self.zero_quat]), rel_quats), dim=0)
            rel_trans = torch.cat((rel_trans.new_zeros((1,3)), rel_trans), dim=0)
            rel_hfovs = torch.cat((rel_hfovs.new_zeros((1,)), rel_hfovs), dim=0)
        rel_quats = rel_quats[id]
        rel_trans = rel_trans[id]
        rel_hfovs = rel_hfovs[id]

        # R,T about self.centre
        R = poses[:,:3,:3]
        T = poses[:,:3,3] + R @ self.centre
        Tx, Ty, Tz = T.unbind(-1)

        if not self.param_R_axis_angle:
            rel_quats = F.normalize(rel_quats, dim=-1)
        rel_trans = rel_trans * self.trans_scale
        rel_hfovs = self.hfov_range*torch.tanh(rel_hfovs/self.hfov_range) * np.pi/180      # Additive (upto +-self.hfov_range degrees) change to HFOV

        # incorporate rel_hfovs into Tz, hfovs and new_focal
        new_hfovs = (hfovs + rel_hfovs).clamp(min=eps)
        new_focal = 1 / torch.tan(new_hfovs)

        # new translation
        Tz = Tz * torch.tan(hfovs) / torch.tan(new_hfovs)
        new_T = torch.stack([Tx,Ty,Tz], dim=-1) + rel_trans

        # new rotation
        if self.param_R_axis_angle:
            rel_R = axis_angle_to_matrix(rel_quats)
        else:
            rel_R = quaternion_to_matrix(rel_quats)
        new_R = rel_R @ R

        # Update R,T to be about (0,0,0)
        new_T = new_T - new_R @ self.centre

        return new_R, new_T, new_hfovs, new_focal

    def create_cameras(self, id=slice(None), space='ndc', **kwargs) -> CamerasBase:
        """
        Returns a batch of PerspectiveCameras
        """
        # Build R,T,focal
        new_R, new_T, new_hfov, new_focal = self.get_RTfovF(id=id, **kwargs)

        # points are row-vectors
        new_R = new_R.transpose(-1,-2)

        # PerspectiveCameras(
        #     focal_length=((0.17875, 0.11718),),  # fx = fx_screen / half_imwidth,
        #                                         # fy = fy_screen / half_imheight
        #     principal_point=((-0.5, 0),),  # px = - (px_screen - half_imwidth) / half_imwidth,
        #                                    # py = - (py_screen - half_imheight) / half_imheight
        # )
        if space=='ndc':
            cam_kwargs = {
                'focal_length':new_focal
            }
        elif space=='screen':
            H,W = kwargs.get('img_size', self.img_size)
            assert(W>0)
            cam_kwargs = {
                'focal_length':new_focal * min(H,W)/2,
                'principal_point': ((H/2, W/2),),
                'image_size': ((H, W),),
            }
        else:
            raise ValueError

        device = kwargs.get('device', self.device)
        cameras = PerspectiveCameras(
            device=device,
            R=new_R, T=new_T,
            **cam_kwargs
        )
        assert(cameras.T.isfinite().all())
        assert(cameras.R.isfinite().all())
        return cameras
