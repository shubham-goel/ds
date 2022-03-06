import json
import logging
import math
import os

import imageio
import numpy as np
import pytorch3d.io
import torch

trans_t = lambda t : torch.tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1],
], dtype=torch.float32)

rot_phi = lambda phi : torch.tensor([
    [1,0,0,0],
    [0,math.cos(phi),-math.sin(phi),0],
    [0,math.sin(phi), math.cos(phi),0],
    [0,0,0,1],
], dtype=torch.float32)

rot_theta = lambda th : torch.tensor([
    [math.cos(th),0,-math.sin(th),0],
    [0,1,0,0],
    [math.sin(th),0, math.cos(th),0],
    [0,0,0,1],
], dtype=torch.float32)


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.tensor([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]]).float() @ c2w
    return c2w


def load_blender_data(basedir, half_res=False, testskip=1):
    splits = ['train', 'val', 'test']
    metas = {}
    for s in splits:
        with open(os.path.join(basedir, 'transforms_{}.json'.format(s)), 'r') as fp:
            metas[s] = json.load(fp)

    all_imgs = []
    all_poses = []
    counts = [0]
    for s in splits:
        meta = metas[s]
        imgs = []
        poses = []
        if s=='train' or testskip==0:
            skip = 1
        else:
            skip = testskip

        for frame in meta['frames'][::skip]:
            fname = os.path.join(basedir, frame['file_path'] + '.png')
            imgs.append(imageio.imread(fname))
            poses.append(torch.tensor(frame['transform_matrix']))
        imgs = torch.as_tensor(np.array(imgs)).float().permute(0,3,1,2) / 255 # keep all 4 channels (RGBA)
        poses = torch.stack(poses, dim=0)
        counts.append(counts[-1] + imgs.shape[0])
        all_imgs.append(imgs)
        all_poses.append(poses)

    i_split = [torch.arange(counts[i], counts[i+1]) for i in range(3)]

    imgs = torch.cat(all_imgs, dim=0)
    poses = torch.cat(all_poses, dim=0)

    H, W = imgs[0].shape[:2]
    camera_angle_x = float(meta['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)
    hfov = .5 * camera_angle_x

    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]],0)

    if half_res:
        imgs = tf.image.resize_area(imgs, [400, 400]).numpy()
        H = H//2
        W = W//2
        focal = focal/2.

    # Try to load gt mesh if available
    try:
        obj_path = os.path.join(basedir,'mesh.obj')
        gtmesh = pytorch3d.io.load_objs_as_meshes([obj_path])
        verts = gtmesh.verts_packed()
        print('verts', verts.shape)
    except:
        gtmesh = None
        print('verts NA')

    return imgs, poses, hfov, render_poses, i_split, gtmesh


def load_all_blender_data(datasetdir, metadata_path, max_views=-1, load_mesh=True):
    """
    Loads all data from transforms.json
    """
    with open(metadata_path, 'r') as fp:
        meta = json.load(fp)

    imgs = []
    poses = []
    hfovs = []
    skip = 1

    for i,frame in enumerate(meta['frames'][::skip]):
        # Break early to speed up loading time
        if max_views>0 and i >= max_views:
            break
        fname = f'{datasetdir}/{frame["file_path"]}.png'
        imgs.append(imageio.imread(fname))
        poses.append(torch.tensor(frame['transform_matrix']))
        hfovs.append(float(frame['camera']['angle_x'])/2)

    imgs =  torch.as_tensor(np.array(imgs)).float().permute(0,3,1,2) / 255 # keep all 4 channels (RGBA)
    poses = torch.stack(poses).float()
    hfovs = torch.tensor(hfovs).float()


    if load_mesh:
        # Load GT shape
        obj_path = f'{datasetdir}/{meta["obj_path"]}'
        gtmesh = pytorch3d.io.load_objs_as_meshes([obj_path])

        verts = gtmesh.verts_packed()
        logging.info(f'loaded GT mesh with #verts {verts.shape}')
        vcen = (verts.max(0)[0]+verts.min(0)[0])/2
    else:
        gtmesh = None
        vcen = torch.zeros_like(poses[0,:3,3])

    radius = float((poses[0,:3,3] - vcen).norm())
    render_poses = torch.stack([pose_spherical(angle, -20.0, radius) for angle in np.linspace(-180,180,40+1)[:-1]],0)
    render_poses[:,:3,3] += vcen

    return imgs, poses, hfovs, gtmesh, render_poses, float(hfovs[0])
