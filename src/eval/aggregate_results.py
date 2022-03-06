import logging
import os
from bdb import BdbQuit
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import hydra
import torch
from hydra.core.config_store import ConfigStore
from hydra.utils import to_absolute_path

from ..eval.evaluate_ds import eval_overfit_single
from ..eval.evaluate_nerf import eval_nerf
from ..utils.dict_merge import dict_merge
from ..utils.metrics import camera_metric_keys
from ..utils.misc import Metrics, apply_dict_rec, flatten_dict, try_move_device

logging.basicConfig(level=logging.INFO, filename='agg_res.out')
logger = logging.getLogger(__name__)

SHAPE_METRIC_KEYS = {'Chamfer-L2', 'Chamfer-L2-p2g', 'Chamfer-L2-g2p', 'NormalConsistency', 'AbsNormalConsistency',
                    'Precision@0.100000', 'Recall@0.100000', 'F1@0.100000',
                    'Precision@0.200000', 'Recall@0.200000', 'F1@0.200000',
                    'Precision@0.300000', 'Recall@0.300000', 'F1@0.300000',
                    'Precision@0.400000', 'Recall@0.400000', 'F1@0.400000',
                    'Precision@0.500000', 'Recall@0.500000', 'F1@0.500000'}
BAD_SHAPE_METRIC = {k: 0 for k in SHAPE_METRIC_KEYS}
BAD_SHAPE_METRIC['Chamfer-L2'] = float('inf')
BAD_SHAPE_METRIC['Chamfer-L2-p2g'] = float('inf')
BAD_SHAPE_METRIC['Chamfer-L2-g2p'] = float('inf')
BAD_SHAPE_METRIC['NormalConsistency'] = float('-inf')
CAM_METRIC_KEYS = camera_metric_keys(per_camera=True, N=20)
BAD_CAM_METRIC = {k:float('inf') for k in CAM_METRIC_KEYS}
BAD_METRICS = Metrics(
    shape_metrics=BAD_SHAPE_METRIC,
    camera_metrics=BAD_CAM_METRIC,
    image_metrics=None,
    finished_iters=1e6
)

EXPROOT = Path(f'{os.environ["SHARED_HOME"]}') / 'dump_facebook'
OURS_EXPDIRS = tuple([EXPROOT / 'google/__tex_transfer_March9__/'])
NERF_EXPDIRS = tuple([EXPROOT / 'nerf/__nerf_wmask__/'])
NERFOPT_EXPDIRS = tuple([EXPROOT / 'nerf/__nerf_wmask_camOpt__/'])

def fetch_evaluator(dirpath: Path, iter='latest', exptype='ours'):
    dirpath = str(dirpath.absolute())
    if exptype=='ours':
        evaluator = eval_overfit_single(dirpath, iter=iter, best=False)
    elif exptype=='nerf':
        evaluator = eval_nerf(dirpath, iter=iter)
    elif exptype=='colmap':
        raise NotImplementedError
    return evaluator

def gather_results(dirpath: Path, iter='latest', exptype='ours', save_if_iter_exceeds=None, recompute=False, align_scale_view0=False, align_fine='icp'):
    align_kwargs = dict(align_scale_view0=align_scale_view0)
    if align_fine.startswith('icp'):
        save_metric_name = f'___aggregation_shapecam_metrics_{align_fine}'
        align_kwargs.update(use_icp = True)
        align_kwargs.update(icp_type = align_fine[len('icp_'):])
    elif align_fine=='none':
        save_metric_name = '___aggregation_shapecam_metrics_noicp'
        align_kwargs.update(use_icp = False)
    else:
        raise ValueError(f'align_fine={align_fine}')
    save_metric_path = dirpath/f'{save_metric_name}_i{iter}.pth'

    if not recompute and save_metric_path.is_file():
        logging.info(f'Loading metrics from {save_metric_path}')
        metrics = torch.load(str(save_metric_path))
    else:
        # while exptype=='nerf':
        # logger.info(torch.cuda.memory_summary())
        logger.info(('Gathering results from ', dirpath))
        evaluator = fetch_evaluator(dirpath, iter=iter, exptype=exptype)
        shape_metrics = evaluator.compute_shape_metrics(uni_chamfer=True, **align_kwargs)
        camera_metrics = evaluator.compute_camera_metrics(**align_kwargs)
        finished_iters = evaluator.finished_iters()
        metrics = Metrics(
            shape_metrics=apply_dict_rec(shape_metrics, fv = lambda x: try_move_device(x, 'cpu')),
            camera_metrics=apply_dict_rec(camera_metrics, fv = lambda x: try_move_device(x, 'cpu')),
            finished_iters=finished_iters,
            image_metrics=None,
        )

        if finished_iters >= save_if_iter_exceeds:
            logging.info(f'Saving metrics to {save_metric_path}')
            torch.save(metrics, str(save_metric_path))
            if iter=='latest':
                torch.save(metrics, dirpath/f'{save_metric_name}_i{finished_iters-1}.pth')
    return metrics

def gather_results_recursive(dirpath: Path, exclude_dirs=[], include_dirs=[], **kwargs):
    if (dirpath/'.hydra'/'config.yaml').is_file():
        return gather_results(dirpath, **kwargs)

    logger.info(('Gathering results from ', dirpath))

    results = dict()
    for subd in dirpath.iterdir():
        if subd.name in exclude_dirs:
            logger.info(f'Skipping1 {subd}')
            continue
        if len(include_dirs)>0 and subd.name not in include_dirs:
            logger.info(f'Skipping2 {subd}')
            continue

        if subd.is_dir():
            try:
                results[subd.name] = gather_results_recursive(subd, exclude_dirs=exclude_dirs, include_dirs=include_dirs, **kwargs)
            except KeyboardInterrupt as e:
                exit(0)
            except BdbQuit as e:
                exit(0)
            except RuntimeError as e:
                logger.error((subd, kwargs.get('iter', 'latest'), e), exc_info=True)
                breakpoint()
                x = 0
            except Exception as e:
                logger.error((subd, kwargs.get('iter', 'latest'), e))
                results[subd.name] = e
    return results

def gather_recursive_and_merge(dirpath_list: List[Path], **kwargs):
    results = [gather_results_recursive(d, **kwargs) for d in dirpath_list]
    final_dict = results[0]
    for d in results[1:]:
        dict_merge(final_dict, d)
    return final_dict

@dataclass
class MyConfig:
    exclude_dirs: Tuple[str] = ('r40t0h0',)
    include_dirs: Tuple[str] = tuple([])
    r_noises: Tuple[int] = (10,20,30)
    all_num_views: Tuple[int] = (4,6,8,12)
    reaggregate: bool = False
    recompute_ours: bool = False
    recompute_nerf: bool = False
    make_plots: Tuple[str] = ('vs_views', 'grid', 'grid_compare')
    # align_fine: str = 'icp_p2g_noscale_centered'    # Used in CVPR camera_release with estimate_scale=False
    # align_fine: str = 'icp_g2p_noscale_centered'    # Used in CVPR camera_release with estimate_scale=False
    align_fine: str = 'none'                          # Used in CVPR camera_release with estimate_scale=False

cs = ConfigStore.instance()
# Registering the Config class with the name 'config'.
cs.store(name="config", node=MyConfig)

@hydra.main(config_name="config")
def main(cfg: MyConfig):
    logger.info('#' * 30)
    logger.info('#' * 30)
    logger.info('#' * 30)
    logger.info('Starting script')
    logger.info('#' * 30)

    addregated_dir = Path(to_absolute_path('results/aggregated/'))
    if cfg.align_fine.startswith('icp'):
        addregated_dir = addregated_dir.parent / (addregated_dir.name + f'_align{cfg.align_fine}')
    elif cfg.align_fine == 'none':
        addregated_dir = addregated_dir.parent / (addregated_dir.name + '_alignNone')
    else:
        raise ValueError(f'align_fine={cfg.align_fine} must be one of [icp.<type>, none]')
    addregated_dir.mkdir(exist_ok=True, parents=True)

    # Each <method>_results should have structure:
    #   [r20t0h0][v8][Sonny]: Metrics
    our_kwargs = dict(exptype='ours', save_if_iter_exceeds=49_999, )
    nerf_kwargs = dict(exptype='nerf', save_if_iter_exceeds=99_999, )
    common_kwargs = dict(align_scale_view0=True, align_fine=cfg.align_fine, recompute=cfg.recompute_nerf, exclude_dirs=cfg.exclude_dirs, include_dirs=cfg.include_dirs)
    if cfg.reaggregate:
        ours_results = gather_recursive_and_merge(OURS_EXPDIRS, iter='latest', **our_kwargs, **common_kwargs)
        torch.save(ours_results, addregated_dir / 'ours.pth')
        nerf_results = gather_recursive_and_merge(NERF_EXPDIRS, iter='latest', **nerf_kwargs, **common_kwargs)
        torch.save(nerf_results, addregated_dir / 'nerf.pth')
        nerf_results = gather_recursive_and_merge(NERF_EXPDIRS, iter=100000, **nerf_kwargs, **common_kwargs)
        torch.save(nerf_results, addregated_dir / 'nerf-100k.pth')
        nerfopt_results = gather_recursive_and_merge(NERFOPT_EXPDIRS, iter='latest', **nerf_kwargs, **common_kwargs)
        torch.save(nerfopt_results, addregated_dir / 'nerfopt.pth')
        nerfopt_results = gather_recursive_and_merge(NERFOPT_EXPDIRS, iter=100000, **nerf_kwargs, **common_kwargs)
        torch.save(nerfopt_results, addregated_dir / 'nerfopt-100k.pth')

    ours_results = torch.load(addregated_dir / 'ours.pth')
    lnerf_results = torch.load(addregated_dir / 'nerf.pth')
    nerf_results = torch.load(addregated_dir / 'nerf-100k.pth')
    lnerfopt_results = torch.load(addregated_dir / 'nerfopt.pth')
    nerfopt_results = torch.load(addregated_dir / 'nerfopt-100k.pth')
    logger.info(('ours', min([v.finished_iters for v in flatten_dict(ours_results).values() if isinstance(v, Metrics)])))
    logger.info(('nerf-latest', min([v.finished_iters for v in flatten_dict(lnerf_results).values() if isinstance(v, Metrics)])))
    logger.info(('nerf', min([v.finished_iters for v in flatten_dict(nerf_results).values() if isinstance(v, Metrics)])))
    logger.info(('nerfopt-latest', min([v.finished_iters for v in flatten_dict(lnerfopt_results).values() if isinstance(v, Metrics)])))
    logger.info(('nerfopt', min([v.finished_iters for v in flatten_dict(nerfopt_results).values() if isinstance(v, Metrics)])))

    # Print what data we got
    def print_stats(label, m_results):
        for k in m_results:
            for k2 in m_results[k]:
                logger.info((f'{label} {k:8s} {k2:3s}', len([x for x in m_results[k][k2].values() if isinstance(x, Metrics)])))
    print_stats('our_results', ours_results)
    print_stats('nerf_results', nerf_results)
    print_stats('nerfopt_results', nerfopt_results)
    print_stats('lnerf_results', lnerf_results)
    print_stats('lnerfopt_results', lnerfopt_results)

if __name__ == '__main__':
    main()
