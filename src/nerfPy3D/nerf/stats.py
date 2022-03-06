from collections import defaultdict
import logging
import time
import warnings
from itertools import cycle
from typing import Dict, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors
from visdom import Visdom


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.history = []
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.min = float('inf')

    def update(self, val, n=1, epoch=0):
        # make sure the history is of the same len as epoch
        while len(self.history) <= epoch:
            self.history.append([])
        self.history[epoch].append(val / n)
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.min = min(self.min, val)

    def get_epoch_averages(self):
        """
        Returns:
            averages: A list of average values of the metric for each epoch
                in the history buffer.
        """
        if len(self.history) == 0:
            return None
        return [
            (i, float(np.array(h).mean())) for i,h in enumerate(self.history)
            if len(h) > 0
        ]

class SimpleStats(object):
    def __init__(self):
        self.meter_dict = defaultdict(AverageMeter)

    def update(self, scalar_dict: Dict[str, float], iter: int):
        scalar_dict.update(_iter=iter)
        for k,v in scalar_dict.items():
            self.meter_dict[k].update(v)

class Stats(object):
    """
    stats logging object useful for gathering statistics of training a deep net in pytorch

    Example:
        ```
        # Init stats structure that logs statistics 'objective' and 'top1e'.
        stats = Stats( ('objective','top1e') )

        network = init_net()  # init a pytorch module (=nueral network)
        dataloader = init_dataloader()  # init a dataloader

        for epoch in range(10):

            # start of epoch -> call new_epoch
            stats.new_epoch()

            # Iterate over batches.
            for batch in dataloader:
                # Run a model and save into a dict of output variables "output"
                output = network(batch)

                # stats.update() automatically parses the 'objective' and 'top1e'
                # from the "output" dict and stores this into the db.
                stats.update(output)
                stats.print() # prints the averages over given epoch

            # Stores the training plots into '/tmp/epoch_stats.pdf'
            # and plots into a visdom server running at localhost (if running).
            stats.plot_stats(plot_file='/tmp/epoch_stats.pdf')
        ```
    """

    def __init__(
        self,
        log_vars,
        verbose=False,
        epoch=-1,
        plot_file=None,
    ):
        self.verbose = verbose
        self.log_vars = log_vars
        self.plot_file = plot_file
        self.hard_reset(epoch=epoch)

    def reset(self):  # to be called after each epoch
        stat_sets = list(self.stats.keys())
        logging.debug("stats: epoch %d - reset" % self.epoch)
        self.it = {k: -1 for k in stat_sets}
        for stat_set in stat_sets:
            for stat in self.stats[stat_set]:
                self.stats[stat_set][stat].reset()

    def hard_reset(self, epoch=-1):  # to be called during object __init__
        self.epoch = epoch
        logging.debug("stats: epoch %d - hard reset" % self.epoch)
        self.stats = {}
        # reset
        self.reset()

    def new_epoch(self):
        logging.debug("stats: new epoch %d" % (self.epoch + 1))
        self.epoch += 1
        self.reset()  # zero the stats + increase epoch counter

    def _gather_value(self, val):
        if type(val) == float:
            pass
        else:
            val = val.data.cpu().numpy()
            val = float(val.sum())
        return val

    def update(self, preds, time_start=None, stat_set="train"):

        if self.epoch == -1:  # uninitialized
            warnings.warn(
                "self.epoch==-1 means uninitialized stats structure"
                " -> new_epoch() called"
            )
            self.new_epoch()

        if stat_set not in self.stats:
            self.stats[stat_set] = {}
            self.it[stat_set] = -1

        self.it[stat_set] += 1

        epoch = self.epoch
        it = self.it[stat_set]

        for stat in self.log_vars:

            if stat not in self.stats[stat_set]:
                self.stats[stat_set][stat] = AverageMeter()

            if stat == "sec/it":  # compute speed
                if time_start is None:
                    elapsed = 0.0
                else:
                    elapsed = time.time() - time_start
                time_per_it = float(elapsed) / float(it + 1)
                val = time_per_it
                # self.stats[stat_set]['sec/it'].update(time_per_it,epoch=epoch,n=1)
            else:
                if stat in preds:
                    val = self._gather_value(preds[stat])
                else:
                    val = None

            if val is not None:
                self.stats[stat_set][stat].update(val, epoch=epoch, n=1)

    def print(self, max_it=None, stat_set="train", vars_print=None, get_str=False):

        epoch = self.epoch
        stats = self.stats

        str_out = ""

        it = self.it[stat_set]
        stat_str = ""
        stats_print = sorted(stats[stat_set].keys())
        for stat in stats_print:
            if stats[stat_set][stat].count == 0:
                continue
            stat_str += " {0:.12}: {1:1.3f} |".format(stat, stats[stat_set][stat].avg)

        head_str = f"[{stat_set}] | epoch {epoch} | it {it}"
        if max_it:
            head_str += f"/ {max_it}"

        str_out = f"{head_str} | {stat_str}"

        logging.info(str_out)

    def plot_stats(
        self,
        viz : Optional[Visdom] = None,
        visdom_env : Optional[str] = None,
        plot_file : Optional[str] = None,
    ):

        stat_sets = list(self.stats.keys())

        if viz is None:
            novisdom = True
        elif not viz.check_connection():
            warnings.warn("Cannot connect to the visdom server! Skipping visdom plots.")
            novisdom = True
        else:
            novisdom = False

        lines = []

        for stat in self.log_vars:
            xs = []
            vals = []
            stat_sets_now = []
            for stat_set in stat_sets:
                x_val = self.stats[stat_set][stat].get_epoch_averages()
                if x_val is None:
                    continue
                else:
                    val = np.array([v for x,v in x_val]).reshape(-1)
                    x = np.array([x for x,v in x_val]).reshape(-1)
                    stat_sets_now.append(stat_set)

                vals.append(val)
                xs.append(x)

            if len(vals) == 0:
                continue

            def expand_max(arrays):
                max_len = max([len(a) for a in arrays])
                return [np.concatenate((a, np.full(max_len-len(a), np.nan))) for a in arrays]

            vals = np.stack(expand_max(vals), axis=1)
            x = np.stack(expand_max(xs), axis=1)

            lines.append((stat_sets_now, stat, x, vals))

        if not novisdom:
            for tmodes, stat, x, vals in lines:
                if vals.shape[1] == 1:  # prevent visdom crash
                    vals = vals[:, 0]
                if x.shape[1] == 1:  # prevent visdom crash
                    x = x[:, 0]
                title = "%s" % stat
                opts = {"title": title, "legend": list(tmodes)}
                viz.line(
                    Y=vals,
                    X=x,
                    env=visdom_env,
                    opts=opts,
                    win=f"stat_plot_{title}",
                )

        if plot_file is not None:
            print("Exporting stats to %s" % plot_file)
            ncol = 3
            nrow = int(np.ceil(float(len(lines)) / ncol))
            matplotlib.rcParams.update({"font.size": 5})
            color = cycle(plt.cm.tab10(np.linspace(0, 1, 10)))
            fig = plt.figure(1)
            plt.clf()
            for idx, (tmodes, stat, x, vals) in enumerate(lines):
                c = next(color)
                plt.subplot(nrow, ncol, idx + 1)
                for vali, vals_ in enumerate(vals.T):
                    c_ = c * (1.0 - float(vali) * 0.3)
                    plt.plot(x, vals_, c=c_, linewidth=1)
                plt.ylabel(stat)
                plt.xlabel("epoch")
                plt.gca().yaxis.label.set_color(c[0:3] * 0.75)
                plt.legend(tmodes)
                gcolor = np.array(mcolors.to_rgba("lightgray"))
                plt.grid(
                    b=True, which="major", color=gcolor, linestyle="-", linewidth=0.4
                )
                plt.grid(
                    b=True, which="minor", color=gcolor, linestyle="--", linewidth=0.2
                )
                plt.minorticks_on()

            plt.tight_layout()
            plt.show()
            fig.savefig(plot_file)

if __name__ == '__main__':
    stats = Stats(['v0', 'v1'])
    dummy_data1 = {
            'v0': 1.0,
            'v1': 1.1,
        }
    dummy_data2 = {
            'v0': 2.0,
            'v1': 2.1,
        }
    stats.update(dummy_data1, stat_set='train')
    stats.update(dummy_data2, stat_set='val')
    stats.new_epoch()
    stats.update(dummy_data1, stat_set='train')
    stats.new_epoch()
    stats.update(dummy_data1, stat_set='train')
    stats.update(dummy_data2, stat_set='val')

    from visdom import Visdom
    viz = Visdom(
        server='http://100.96.161.120',
        port=6009,
    )
    stats.plot_stats(
        viz=viz,
        visdom_env='dummy'
    )
