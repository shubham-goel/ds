import logging
from itertools import chain
from pathlib import Path

import hydra
from omegaconf.dictconfig import DictConfig

from aggregate_results import (NERF_EXPDIRS, NERFOPT_EXPDIRS, OURS_EXPDIRS,
                               gather_results)

CONFIG_DIR='../../configs'
@hydra.main(config_path=CONFIG_DIR, config_name="eval_tex_transfer")
def main(cfg: DictConfig):

    common_kwargs = dict(align_scale_view0=True, align_fine=cfg.align_fine, )

    # Evaluate metrics on all experiments for this asin
    for r in cfg.r_noises:
        for v in cfg.all_num_views:
            suffix = f'r{r}t0h0/v{v}/{cfg.asin}'
            for expdir in OURS_EXPDIRS:
                try:
                    gather_results(Path(expdir)/suffix, iter='latest', exptype='ours', save_if_iter_exceeds=49999, recompute=True, **common_kwargs)
                except KeyboardInterrupt as e:
                    exit(0)
                except RuntimeError as e:
                    logging.error((Path(expdir)/suffix, 'latest', e), exc_info=True)
                    breakpoint()
                    x = 0
                except Exception as e:
                    logging.error((Path(expdir)/suffix, 'latest', e))
            for expdir in chain(NERF_EXPDIRS, NERFOPT_EXPDIRS):
                for it in ['latest', 100000]:
                    try:
                        gather_results(Path(expdir)/suffix, iter=it, exptype='nerf', save_if_iter_exceeds=99999, recompute=True, **common_kwargs)
                    except KeyboardInterrupt as e:
                        exit(0)
                    except RuntimeError as e:
                        logging.error((Path(expdir)/suffix, it, e), exc_info=True)
                        breakpoint()
                        x = 0
                    except Exception as e:
                        logging.error((Path(expdir)/suffix, 'latest', e))

if __name__ == '__main__':
    main()
