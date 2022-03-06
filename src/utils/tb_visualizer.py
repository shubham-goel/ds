import os
import os.path as osp
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter


class TBVisualizer():
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.stats_dir = f'{self.log_dir}/stats/'
        self.mesh_dir = f'{self.log_dir}/mesh/'
        if not osp.exists(self.log_dir):
            os.makedirs(self.log_dir)
        if not osp.exists(self.stats_dir):
            os.makedirs(self.stats_dir)
        if not osp.exists(self.mesh_dir):
            os.makedirs(self.mesh_dir)

        print("Logging to {}".format(log_dir))
        self.viz = SummaryWriter(f'{self.log_dir}')

    def __del__(self):
        # self.viz.close()  # Uncommenting this causes code to hang on interrupts
        self.flush()

    def flush(self):
        self.viz.flush()

    def plot_images(self, images, global_step):
        for label, image in images.items():
            assert(np.isfinite(image).all()), label
            if(len(image.shape) == 2):
                dataformats = 'HW'
                self.viz.add_image(label,image, global_step, dataformats=dataformats)
            elif(len(image.shape) == 3):
                dataformats = 'HWC' if (image.shape[2]<=4) else 'CHW'
                self.viz.add_image(label,image, global_step, dataformats=dataformats)
            elif(len(image.shape) == 4):
                dataformats = 'NHWC' if (image.shape[3]<=4) else 'NCHW'
                self.viz.add_images(label,image, global_step, dataformats=dataformats)
            else:
                raise NotImplementedError


    def plot_videos(self, videos, global_step, fps=4):
        for label, video in videos.items():
            assert(np.isfinite(video).all()), label
            if(len(video.shape) == 4): # t,C,H,W
                assert video.shape[1]==3, f'Invalid video shape:{video.shape}'
                self.viz.add_video(label, video[None], global_step, fps=fps)
            elif(len(video.shape) == 5):
                assert video.shape[2]==3, f'Invalid video shape:{video.shape}'
                self.viz.add_video(label, video, global_step, fps=fps)
            else:
                raise NotImplementedError

    def plot_meshes(self, meshes, global_step):
        for label, mesh in meshes.items():
            vert = mesh['v']
            assert(torch.isfinite(vert).all()), label
            face = mesh['f'] if 'f' in mesh else None
            color = mesh['c'] if 'c' in mesh else None
            config = mesh['cfg'] if 'cfg' in mesh else {}
            self.viz.add_mesh(label,vert,colors=color,faces=face,config_dict=config,global_step=global_step)

    def save_meshes(self, meshes, global_step):
        for label, mesh in meshes.items():
            vert = mesh['v']
            assert(torch.isfinite(vert).all()), label
            face = mesh['f']

            vert = vert[0] if len(vert.shape)==3 else vert
            face = face[0] if len(face.shape)==3 else face
            outdir = f'{self.mesh_dir}/{label}'
            if not osp.exists(outdir):
                os.makedirs(outdir)
            # import pymesh
            # pmesh = pymesh.form_mesh(vert.numpy(), face.numpy())
            # pymesh.save_mesh(f'{outdir}/{global_step}.obj', pmesh)
            import pytorch3d
            import pytorch3d.io
            pytorch3d.io.save_obj(Path(f'{outdir}/{global_step}.obj'), vert, face)

    def plot_embeddings(self, embeddings, global_step):
        for label, embed in embeddings.items():
            if isinstance(embed,dict):
                mat = embed['mat']
                metadata = embed['metadata'] if 'metadata' in embed else None
                metadata_header = embed['metadata_header'] if 'metadata_header' in embed else None
                label_img = embed['label_img'] if 'label_img' in embed else None
                self.viz.add_embedding(mat,tag=label,global_step=global_step,metadata=metadata, label_img=label_img, metadata_header=metadata_header)
            else:
                assert(torch.isfinite(embed).all()), label
                self.viz.add_embedding(embed,tag=label,global_step=global_step)

    def plot_histograms(self, histograms, global_step):
        for label, hist in histograms.items():
            if isinstance(hist,dict):
                values = hist['values']
                bins = hist['bins'] if 'bins' in hist else 'tensorflow'
                max_bins = hist['max_bins'] if 'max_bins' in hist else None
                self.viz.add_histogram(label,values,global_step=global_step,bins=bins, max_bins=max_bins)
            else:
                assert(torch.isfinite(hist).all()), label
                self.viz.add_histogram(label,hist,global_step=global_step)

    def plot_texts(self, texts, global_step):
        for label, text in texts.items():
            self.viz.add_text(label,text,global_step=global_step)

    # |visuals|: dictionary of images to display or save
    def display_current_results(self, visuals, global_step, save_meshes=False):
        if 'img' in visuals:
            self.plot_images(visuals['img'], global_step)
        if 'image' in visuals:
            self.plot_images(visuals['image'], global_step)

        if 'video' in visuals:
            fps = visuals['video_fps'] if 'video_fps' in visuals else 4
            self.plot_videos(visuals['video'], global_step, fps=fps)

        if 'mesh' in visuals:
            self.plot_meshes(visuals['mesh'], global_step)
            if save_meshes:
                self.save_meshes(visuals['mesh'], global_step)

        if 'embed' in visuals:
            self.plot_embeddings(visuals['embed'], global_step)

        if 'hist' in visuals:
            self.plot_histograms(visuals['hist'], global_step)

        if 'text' in visuals:
            self.plot_texts(visuals['text'], global_step)

        if 'scalar' in visuals:
            self.plot_current_scalars(visuals['scalar'], global_step)

    def save_raw_stats(self, stats, name, epoch):
        if epoch is None:
            path = f'{self.stats_dir}/{name}'
        else:
            path = f'{self.stats_dir}/{name}_{epoch}'
        Path(path).parents[0].mkdir(exist_ok=True)
        np.savez(path, **stats)

    def plot_current_scalars(self, scalars, global_step):
        for key, value in scalars.items():
            if isinstance(value, dict):
                self.viz.add_scalars(key, value, global_step)
            else:
                self.viz.add_scalar(key, value, global_step)

    # scatter plots
    def plot_current_points(self, points, disp_offset=10):
        idx = disp_offset
        for label, pts in points.items():
            #image_numpy = np.flipud(image_numpy)
            self.vis.scatter(
                pts, opts=dict(title=label, markersize=1), win=self.display_id + idx)
            idx += 1

    # scalars: same format as |scalars| of plot_current_scalars
    def print_current_scalars(self, epoch, i, scalars, start_time=None):
        if start_time is None:
            message = '(time : %.3f, epoch: %d, iters: %d) ' % (epoch, i)
        else:
            time_diff = (time.time() - start_time)
            message = '(time : %.2f, epoch: %d, iters: %d) ' % (time_diff, epoch, i)
        for k, v in scalars.items():
            message += '%s: %.3f ' % (k, v)

        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)
