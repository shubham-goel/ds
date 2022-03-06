import logging
import os
import os.path as osp
import traceback
from typing import Callable, List, NamedTuple, Optional, Tuple, Union

import numpy as np
import torch
from dotmap import DotMap
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from omegaconf.errors import InterpolationKeyError
from torch import is_tensor

is_dict = lambda x: isinstance(x, dict)
is_array = lambda x: isinstance(x, np.ndarray)
identity = lambda x: x

def apply_dict(dd: dict, fk: Callable=identity, fv: Callable=identity) -> dict:
    """ Apply fk to keys and fv to values """
    return dd.__class__({fk(k):fv(v) for k,v in dd.items()})

def apply_dict_rec(dd: dict, fk: Callable=identity, fv: Callable=identity, **kwargs) -> dict:
    """ Recursively apply fk to keys and fv to leaves """
    return dd.__class__({
        fk(k): apply_dict_rec(v, fk, fv, **kwargs) if is_dict(v) else fv(v)
        for k,v in dd.items()
    }, **kwargs)

def add_prefix_to_keys(dd: dict, prefix: str = '') -> dict:
    return apply_dict(dd, fk=lambda k:f'{prefix}{k}')

def try_move_device(x, device:torch.device):
    """ Try to move torch object (tensor/Mesh etc.) to device """
    try:
        x = x.to(device)
    except AttributeError:
        pass
    return x

def flatten_dict(dd: dict, delimiter: str = '/') -> dict:
    res = {}
    for k,v in dd.items():
        if isinstance(v, dict):
            res.update(add_prefix_to_keys(flatten_dict(v, delimiter), f'{k}{delimiter}'))
        else:
            res.update({f'{k}':v})
    return res

def symlink_submitit(dest = 'submitit_dir') -> None:
    """ Symlink hydra.launcher.submitit_folder to dest. """
    # Fetch job id/num
    hydra_cfg = HydraConfig.get()
    job_id, job_num = get_jobid_jobnum_hack(hydra_cfg)
    logging.info(f'job_id  {job_id}')
    logging.info(f'job_num {job_num}')

    # Find and symlink submitit directory (contains stdout/stderr)
    try:
        submitit_dir = hydra_cfg.launcher.get('submitit_folder', None)
    except InterpolationKeyError:
        logging.warning(f'Could not find submitit_folder in hydra config. Skipping symlink.')
        return

    logging.info(f'submitit_dir {submitit_dir}')
    if None not in [submitit_dir,job_id]:
        submitit_dir = submitit_dir.replace('%j', job_id, 1)
        # if os.path.exists(dest):
        logging.debug(f'Unlinking existing symlink')
        try:
            os.unlink(dest)
        except:
            pass
        logging.info(f'Symlinking {submitit_dir} to "{dest}"')
        os.symlink(submitit_dir, dest)
        if not os.path.exists(submitit_dir):
            logging.warn(f'Symlink source does not exist: {submitit_dir}')
    else:
        logging.debug(f'Skipping symlink')

def represents_int(s: str) -> bool:
    """ Returns True if string represents a valid integer. """
    try:
        int(s)
        return True
    except ValueError:
        return False

def _is_valid_jobid(jobid: str) -> bool:
    """ jobid is either integer or integer_integer. """
    jobids = jobid.split('_')
    return (
        (len(jobids)==1 and represents_int(jobids[0]))
        or (len(jobids)==2 and all([represents_int(i) for i in jobids]))
    )

def get_jobid_jobnum_hack(hydra_cfg: DictConfig) -> \
                    Tuple[Optional[str], Optional[str]]:
    """ Try fetching job.id/job.num from (in order):
        1. hydra_cfg
        2. slurm environment variables
        3. [hack] cwd (hydra.sweep.subdir) which is assumed
            to be .../${hydra.job.id}/${hydra.job.num}.
    """
    jobid = hydra_cfg.job.get('id', None)
    jobnum = hydra_cfg.job.get('num', None)
    if jobid is None:
        try:
            jobid = f'{os.environ["SLURM_ARRAY_JOB_ID"]}_{os.environ["SLURM_ARRAY_TASK_ID"]}'
            logging.warn(f'Set jobid to {jobid} from SLURM_ARRAY env variables ')
        except KeyError:
            try:
                jobid = os.environ["SLURM_JOB_ID"]
                logging.warn(f'Set jobid to {jobid} from SLURM_JOB_ID env variable')
            except KeyError:
                pass
    if jobid==None and jobnum==None:
        logging.warn('Trying to set jobid,jobnum from cwd')
        jobnum = osp.basename(os.getcwd())
        jobid = osp.basename(osp.dirname(os.getcwd()))
        if not _is_valid_jobid(jobid):
            logging.warn('jobid  failed validity check, setting to None')
            jobid = None
        if not represents_int(jobnum):
            logging.warn('jobnum failed validity check, setting to None')
            jobnum = None
    return jobid, jobnum

def to_tensor(dd: DotMap, float_type=torch.float, int_type=torch.long, **kwargs):
    def f(v):
        if is_array(v) or is_tensor(v):
            v = torch.as_tensor(v)
            if v.dtype in {torch.float, torch.float64}:
                v = v.to(float_type)
            elif v.dtype in {torch.int, torch.long}:
                v = v.to(int_type)
        return v
    return apply_dict_rec(dd, fv = f, **kwargs)

def batchify_func(batch, f:Callable):
    return torch.stack([f(b) for b in batch])

DATA_TYPE = Union[torch.Tensor, np.ndarray]
def permute(t:DATA_TYPE, args:Tuple):
    if is_tensor(t):
        return t.permute(*args)
    elif is_array(t):
        return t.transpose(*args)

def repeat(t:DATA_TYPE, args:Tuple):
    if is_tensor(t):
        return t.repeat(*args)
    elif is_array(t):
        return np.tile(t, args)

def reshape(t:DATA_TYPE, args:Tuple):
    if is_tensor(t):
        return t.reshape(*args)
    elif is_array(t):
        return np.reshape(t, args)

def squeeze(t:DATA_TYPE, dim:int):
    if is_tensor(t):
        return t.squeeze(dim)
    elif is_array(t):
        return np.squeeze(t, dim)

def concat(ts:List[DATA_TYPE], dim:int = 0):
    if len(ts)==0:
        raise ValueError(f'concat got input of length 0: {ts}')
    if is_tensor(ts[0]):
        return torch.cat(ts, dim=dim)
    elif is_array(ts[0]):
        return np.concatenate(ts, axis=dim)

def stack(ts:List[DATA_TYPE], dim:int = 0):
    if len(ts)==0:
        raise ValueError(f'stack got input of length 0: {ts}')
    if is_tensor(ts[0]):
        return torch.stack(ts, dim=dim)
    elif is_array(ts[0]):
        return np.stack(ts, axis=dim)

def split(t:DATA_TYPE, split_sizes: List[int], dim:int):
    if is_tensor(t):
        return t.split(split_sizes, dim=dim)
    elif is_array(t):
        return np.split(t, split_sizes, axis=dim)

def zeros(t:DATA_TYPE, shape: List[int]):
    if is_tensor(t):
        return torch.zeros(shape, dtype=t.dtype, device=t.device)
    elif is_array(t):
        return np.zeros(shape, dtype=t.dtype)

def add_suffix_to_path(path: str, suffix: str) -> str:
    _basefname, _ext = os.path.splitext(path)
    return f'{_basefname}{suffix}{_ext}'

def read_file_as_list(fname):
    with open(fname, 'r') as f:
        ll = f.readlines()
    return [l.strip() for l in ll if len(l.strip())>0]

class EmptyContext(object):
    def __init__(self):
        pass
    def __enter__(self):
        pass
    def __exit__(self, exc_type, exc_value, tb):
        if exc_type is not None:
            traceback.print_exception(exc_type, exc_value, tb)
            return False # uncomment to pass exception through
        return True

def split_into_equally_sized_chunks(total_size, number_of_baskets):
    basket_sizes = [0 for _ in range(number_of_baskets)]
    chunk_size =  int(total_size/number_of_baskets)
    for i in range(number_of_baskets):
        basket_sizes[i] += chunk_size
    num_remaining = total_size - chunk_size*number_of_baskets
    assert(num_remaining <= number_of_baskets)
    for i in range(num_remaining):
        basket_sizes[i] += 1
    assert(sum(basket_sizes)==total_size)
    return basket_sizes

def chunk_items_into_baskets(items, number_of_baskets):
    basket_sizes = split_into_equally_sized_chunks(len(items), number_of_baskets)
    cumsize = 0
    baskets = [[] for _ in range(number_of_baskets)]
    for i,bs in enumerate(basket_sizes):
        baskets[i].extend(items[cumsize:cumsize+bs])
        cumsize += bs
    return baskets

class Metrics(NamedTuple):
    shape_metrics: Optional[dict]
    camera_metrics: Optional[dict]
    image_metrics: Optional[dict]
    finished_iters: int
