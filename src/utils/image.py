import logging
import math
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from ..utils.misc import (concat, is_array, is_tensor, permute, repeat, split,
                          squeeze, zeros)

IMAGE_TYPE = Union[torch.Tensor, np.ndarray]


def square_expand(img, bg=1):
    """
    Expand img (h,w,c) to a square, and fill paddings with bg value.
    Returns (square_image, top_left_position)
    """
    h,w = img.shape[0], img.shape[1]
    if h!=w:
        mm = max(h,w)
        if len(img.shape)==3:
            shape = (mm,mm,img.shape[2])
        else:
            shape = (mm,mm)
        new_img = np.zeros(shape, dtype=img.dtype) + bg

        h_start = mm//2 - h//2
        w_start = mm//2 - w//2
        h_end = h_start + h
        w_end = w_start + w

        new_img[h_start:h_end, w_start:w_end, ...] = img
        return new_img, (h_start, w_start)
    else:
        return img, (0, 0)

def get_img_format(img:IMAGE_TYPE) -> str:
    """ Find format of img as NCHW, NHWC, CHW, HWC or HW"""
    dim = len(img.shape)
    if dim==4:
        # Assume first dim is batchsize
        return "N" + get_img_format(img[0])
    elif dim==3:
        # Assume channel is the dim with size <=4 (allowing for rgba at most)
        if img.shape[0] <=4:
            return "CHW"
        elif img.shape[2] <=4:
            return "HWC"
        else:
            raise ValueError(f'Unknown format for tensor with shape {img.shape}')
    elif dim==2:
        return "HW"
    else:
        raise ValueError(f'Unknown format for tensor with shape {img.shape}')

def change_img_format(
        img:IMAGE_TYPE, out_format: str, inp_format: Optional[str] = None,
    ) -> IMAGE_TYPE:
    """ Permutes input img to match output format (HWC,CHW,NCWH etc).
        Squeezes/Unsqueezes dims if necessary.
    """
    out_format = out_format.upper()
    if inp_format is None:
        inp_format = get_img_format(img).upper()
    if inp_format==out_format:
        return img
    logging.debug(f'change_img_format: input (shape {tuple(img.shape)}, format {inp_format})')
    logging.debug(f'change_img_format: output format {out_format}')

    # inp_format may have extra dimensions. Ensure these are of size 1.
    # input_dim_map_count = [perumte_seq.count(in_dim) for in_dim in inp_format]
    # non_singlular = [c==0 for i, c in enumerate(input_dim_map_count)]
    to_squeeze = []
    for idx, in_dim in enumerate(inp_format):
        num_occur = out_format.count(in_dim)
        if num_occur==0:
            if img.shape[idx] > 1:
                raise ValueError(f'Input dimension {idx} of input '
                    + f'(shape: {tuple(img.shape)}, format: {inp_format}) is not '
                    + f'singular, yet absent in out_format ({out_format})'
                )
            to_squeeze.append(idx)
        elif num_occur>1:
            raise ValueError(f'Input dimension {idx} of input '
                + f'(shape: {tuple(img.shape)}, format: {inp_format}) occurs '
                + f'more than once in out_format ({out_format})'
            )
    for i in reversed(to_squeeze):
        img = squeeze(img, i)
        inp_format = inp_format[:i] + inp_format[i+1:]
    logging.debug(f'change_img_format: squeezing {to_squeeze}')
    logging.debug(f'change_img_format: input (shape {tuple(img.shape)}, format {inp_format})')

    # out_format may have extra dimensions
    to_unsqueeze = []
    for idx, out_dim in enumerate(out_format):
        num_occur = inp_format.count(out_dim)
        if num_occur==0:
            to_unsqueeze.append(idx)
        elif num_occur>1:
            raise ValueError(f'out_format ({out_format}) dimension {idx} '
                + f'occurs more than once in input (shape: {tuple(img.shape)}, '
                + f'format: {inp_format}).'
            )
    for i in reversed(to_unsqueeze):
        img = img[..., None]
        inp_format = inp_format + out_format[i]
    logging.debug(f'change_img_format: unsqueezing {to_unsqueeze}')
    logging.debug(f'change_img_format: input (shape {tuple(img.shape)}, format {inp_format})')

    # Map dimensions in output to dims in input
    assert(sorted(inp_format) == sorted(out_format))
    perumte_seq = [inp_format.find(out_dim) for out_dim in out_format]
    img = permute(img, tuple(perumte_seq))
    logging.debug(f'change_img_format: permuting {perumte_seq}')
    logging.debug(f'change_img_format: input (shape {tuple(img.shape)}, format {inp_format})')

    return img

def pad(
        img: IMAGE_TYPE, H: int, W: int, out_format: str = 'HWC',
        mode: str = 'constant', value: float = 1.
    ) -> IMAGE_TYPE:
    # Check if format is valid
    out_format = out_format.upper()
    if out_format not in ['HW', 'HWC', 'CHW', 'NCHW', 'NHWC']:
        raise ValueError(f'Unkown img format {out_format}')

    # Format image
    img = change_img_format(img, out_format)

    H_dim = out_format.find('H')
    W_dim = out_format.find('W')

    if is_array(img):
        pad_size = ((0, H-img.shape[H_dim]), (0, W-img.shape[W_dim]))
        if out_format=='HWC':
            pad_size = pad_size + ((0,0),)
        elif out_format=='CHW':
            pad_size = ((0,0),) + pad_size
        elif out_format=='NHWC':
            pad_size = ((0,0),) + pad_size + ((0,0),)
        elif out_format=='NCHW':
            pad_size = ((0,0),) + ((0,0),) + pad_size
        elif out_format=='HW':
            pass
        img = np.pad(img, pad_size, mode, constant_values=value)
    elif is_tensor(img):
        # See https://pytorch.org/docs/stable/nn.functional.html#pad
        # pad_size is 1D tensor containing (pad_start, pad_end) of dimensions starting from the end
        pad_size = (0, W-img.shape[W_dim], 0, H-img.shape[H_dim],)
        if out_format=='HWC':
            pad_size = (0,0) + pad_size
        elif out_format=='CHW':
            pad_size = pad_size + (0,0)
        elif out_format=='NHWC':
            pad_size = (0,0) + pad_size + (0,0)
        elif out_format=='NCHW':
            pad_size = pad_size + (0,0) + (0,0)
        elif out_format=='HW':
            pass
        img = F.pad(img, pad_size, mode, value)
    else:
        raise TypeError(f'Unknown input type {type(img)}')

    return img

def concatenate_images1d(img_list, hstack=True, out_format=None, last_alpha=None):
    """
    img_list is a list of np images.
    concatenate images horizontally if hstack is true, else vertically.
    Pads images with 1 if needed
    """
    # Either all arras or tensors
    are_arrays = [is_array(i) for i in img_list]
    are_tensors = [is_tensor(i) for i in img_list]
    if not (all(are_arrays) or all(are_tensors)):
        raise TypeError(f'Input img_list ({[type(i) for i in img_list]}) contains a mixture of np arrays and tensors')

    # Set format to input images
    if out_format is None:
        out_format = get_img_format(img_list[0])
    img_list = [change_img_format(img, out_format) for img in img_list]

    # Find max h/w/c
    H_dim = out_format.upper().find('H')
    W_dim = out_format.upper().find('W')
    C_dim = out_format.upper().find('C')

    max_h = max([img.shape[H_dim] for img in img_list])
    max_w = max([img.shape[W_dim] for img in img_list])
    max_c = max([img.shape[C_dim] for img in img_list])

    if last_alpha is None: last_alpha = (max_c==2) or (max_c >= 4)

    imgs = []
    for img in img_list:
        if hstack:
            # TODO: If last_alpha, pad last channel with 0-alphas
            img = pad(img, max_h, img.shape[W_dim], out_format=out_format, mode='constant', value=1)
        else:
            img = pad(img, img.shape[H_dim], max_w, out_format=out_format, mode='constant', value=1)
        if img.shape[C_dim] != max_c:
            if last_alpha:
                if img.shape[C_dim]==1: # grayscale image, set alpha = 1
                    alpha = zeros(img, img.shape) + 1
                elif img.shape[C_dim]==2: # grayscale+alpha image, extract alpha
                    img, alpha, _ = split_alpha(img, Cdim=C_dim)
                else:
                    raise ValueError
            assert(img.shape[C_dim]==1)
            num_rep = max_c-1 if last_alpha else max_c
            img = repeat(img, tuple([num_rep if i==C_dim else 1 for i in range(len(img.shape))]))
            if last_alpha:
                img = concat([img,alpha], dim=C_dim)
        imgs.append(img)
    if hstack:
        return concat(imgs, W_dim)
    else:
        return concat(imgs, H_dim)

def concatenate_images2d(img_list, out_format=None, last_alpha=None):
    """
    img_list is a possible 2d list of images.
    Concatenate as 2d grid of images
    """
    h_imgs = []
    for h_list in img_list:
        img = concatenate_images1d(h_list, hstack=True, out_format=out_format, last_alpha=last_alpha)
        h_imgs.append(img)
    img = concatenate_images1d(h_imgs, hstack=False, out_format=out_format, last_alpha=last_alpha)
    return img

def concat_images_grid(img_list: List[torch.Tensor], **kwargs):
    nrows = kwargs.get('nrows', round(math.sqrt(len(img_list))))
    list2d = []
    for i in range(0,len(img_list),nrows):
        list2d.append(img_list[i:i+nrows])
    return concatenate_images2d(list2d)

def split_alpha(img, Cdim=None) -> Tuple[torch.Tensor, torch.Tensor, int]:
    if Cdim is None: Cdim = get_img_format(img).find('C')
    if Cdim==-1: raise ValueError
    numC = img.shape[Cdim]
    if numC<=1: raise ValueError
    img, alpha = split(img, [numC-1, 1], Cdim)
    return img, alpha, Cdim

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    img = torch.zeros(1,3,128,256)
    change_img_format(img, 'HWC')
    print(get_img_format(torch.zeros(1,3,128,256)))
    print(get_img_format(torch.zeros(3,128,256)))
    print(get_img_format(torch.zeros(128,256,3)))
    print(get_img_format(torch.zeros(128,256)))
    import ipdb; ipdb.set_trace()
