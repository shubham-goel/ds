# %%
from pathlib import Path
import itertools
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from hydra.utils import to_absolute_path

from .aggregate_results import BAD_SHAPE_METRIC, BAD_CAM_METRIC

# %% Functions for loading aggregated results from `./results/aggregated*`
def load_results(align_fine = 'none'):
    addregated_dir = Path(to_absolute_path('results/aggregated/'))
    figure_dir =  Path(to_absolute_path('figures/'))
    if align_fine.startswith('icp'):
        addregated_dir = addregated_dir.parent / (addregated_dir.name + f'_align{align_fine}')
        figure_dir = figure_dir.parent / (figure_dir.name + f'_align{align_fine}')
    elif align_fine == 'none':
        addregated_dir = addregated_dir.parent / (addregated_dir.name + '_alignNone')
        figure_dir = figure_dir.parent / (figure_dir.name + '_alignNone')
    else:
        raise ValueError(f'align_fine={align_fine} must be one of [icp.<type>, none]')

    # Load results
    ours_results = torch.load(addregated_dir / 'ours.pth')
    lnerfopt_results = torch.load(addregated_dir / 'nerfopt.pth')
    return ours_results, lnerfopt_results

def flatten_results(ours_results):
    ours_results.keys()
    keys1 = list(ours_results.keys())
    keys2 = list(ours_results[keys1[0]].keys())
    keys3 = list(ours_results[keys1[0]][keys2[0]].keys())
    shape_keys4 = list(BAD_SHAPE_METRIC.keys())
    cam_keys4 = list(BAD_CAM_METRIC.keys())

    flat_results = {(k1,k2,k3): {**{k4: getattr(ours_results[k1][k2][k3], 'shape_metrics', BAD_SHAPE_METRIC).get(k4, None) for k4 in shape_keys4},
                                **{k4: getattr(ours_results[k1][k2][k3], 'camera_metrics', BAD_CAM_METRIC).get(k4, None) for k4 in cam_keys4}
                                }
                    for k1,k2,k3 in itertools.product(keys1, keys2, keys3)}
    return flat_results

def load_results_pd(align_fine):
    ours_results, lnerfopt_results = load_results(align_fine)
    ours_pd = pd.DataFrame.from_dict(flatten_results(ours_results), orient='index')
    nerf_pd = pd.DataFrame.from_dict(flatten_results(lnerfopt_results), orient='index')
    ours_pd.index.set_names(['noise', 'views', 'asin'], inplace=True)
    nerf_pd.index.set_names(['noise', 'views', 'asin'], inplace=True)
    return ours_pd, nerf_pd

# %% Combining results across alignments
def combine_results_from_alignments(pd_list, pd_cam, keys=None):
    if keys is None:
        keys = range(len(pd_list))
    assert(len(keys) == len(pd_list))
    pd_all = pd.concat(pd_list, keys=keys, names=['alignment'])
    pd_best_min = pd_all.groupby(['noise', 'views', 'asin']).min()
    pd_best_max = pd_all.groupby(['noise', 'views', 'asin']).max()

    def startswith_any(s:str, l:list):
        return any([s.startswith(i) for i in l])
    max_columns = [c for c in pd_best_max.columns if startswith_any(c, ('F1', 'Precision', 'Recall')) or ('Normal' in c)]
    min_columns = [c for c in pd_best_max.columns if c not in max_columns]
    min_columns = ['Chamfer-L2', 'Chamfer-L2-g2p', 'Chamfer-L2-p2g']
    # print(min_columns)
    # print(max_columns)
    pd_best = pd.concat([pd_best_min[min_columns], pd_best_max[max_columns], pd_cam[['R/avg']]], axis=1)
    pd_best.sort_index(axis=1, ascending=True, inplace=True)

    ## Ensure computation is correct
    assert (pd_best['R/avg'].sort_index(axis=0) == pd_cam['R/avg'].sort_index(axis=0)).all()
    # for metric in ['Chamfer-L2', 'Chamfer-L2-g2p', 'Chamfer-L2-p2g']:
    for metric in ['Chamfer-L2']:
        assert all((pd_best[metric].sort_index(axis=0) <= pd_i[metric].sort_index(axis=0)).all() for pd_i in pd_list)
    for metric in ['NormalConsistency', 'AbsNormalConsistency',
                    'Precision@0.100000', 'Recall@0.100000', 'F1@0.100000',
                    'Precision@0.200000', 'Recall@0.200000', 'F1@0.200000',
                    'Precision@0.300000', 'Recall@0.300000', 'F1@0.300000',
                    'Precision@0.400000', 'Recall@0.400000', 'F1@0.400000',
                    'Precision@0.500000', 'Recall@0.500000', 'F1@0.500000']:
        assert all((pd_best[metric].sort_index(axis=0) >= pd_i[metric].sort_index(axis=0)).all() for pd_i in pd_list)
    return pd_best

# %% Plotting functions
def plot_graphs_vs_views(ours_q, nerf_q, r_noises=[10,20,30], views=[4,6,8,12], figure_dir=Path('figures/')):
    # collated['R']['r10t0h0']['ours-v8'] = np.array([...])

    titles = {
        'R': 'Rotation Error (degrees) ↓',
        'Chamfer-L2': 'Chamfer-L2 ↓',
        'NormalConsistency': 'NormalConsistency ↑',
        'AbsNormalConsistency': 'AbsNormalConsistency ↑',
        'F1@0.100000': 'F1@0.1 ↑',
        'F1@0.200000': 'F1@0.2 ↑',
        'F1@0.300000': 'F1@0.3 ↑',
        # 'F1@0.400000': 'F1@0.4 ↑',
        # 'F1@0.500000': 'F1@0.5 ↑',
        'Precision@0.100000': 'Precision@0.1 ↑',
        'Precision@0.200000': 'Precision@0.2 ↑',
        'Precision@0.300000': 'Precision@0.3 ↑',
        # 'Precision@0.400000': 'Precision@0.4 ↑',
        # 'Precision@0.500000': 'Precision@0.5 ↑',
        'Recall@0.100000': 'Recall@0.1 ↑',
        'Recall@0.200000': 'Recall@0.2 ↑',
        'Recall@0.300000': 'Recall@0.3 ↑',
        # 'Recall@0.400000': 'Recall@0.4 ↑',
        # 'Recall@0.500000': 'Recall@0.5 ↑',
    }
    k0s = list(titles.keys())
    k1s_plot = [f'r{n}' for n in r_noises]
    k2s_plot = [f'v{v}' for v in views]
    cmap = matplotlib.cm.get_cmap('Set2')
    k1s_colors = [cmap(i) for i in range(len(k1s_plot))]

    for i,k0 in enumerate(k0s):
        fig, ax = plt.subplots(figsize=(6,3))
        for j, (k1, color) in enumerate(zip(k1s_plot,k1s_colors)):
            ours_Ys = [ours_q.at[(f'{k1}t0h0', k2), k0] for k2 in k2s_plot]
            nerf_Ys = [nerf_q.at[(f'{k1}t0h0', k2), k0] for k2 in k2s_plot]
            plt.plot(k2s_plot, ours_Ys, color=color, label=f'DS {k1}', linestyle='-')
            plt.plot(k2s_plot, nerf_Ys, color=color, label=f'nerf-opt {k1}', linestyle='--')

        # sort both labels and handles by labels
        handles, labels = ax.get_legend_handles_labels()
        labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
        ax.legend(handles, labels)

        fig.tight_layout(pad=0, h_pad=0, w_pad=0)
        fig.suptitle(titles[k0], fontsize=16)

        figure_dir.mkdir(exist_ok=True, parents=True)
        plt.savefig(figure_dir / f'{k0}.pdf', bbox_inches='tight', pad_inches=0)
        plt.close('all')

def plot_graphs_grid(ours_q, nerf_q, r_noises=[10,20,30], views=[4,6,8,12], figure_dir=Path('figures/'), percentile=50):
    # collated['R']['r10t0h0']['ours-v8'] = np.array([...])
    titles = {
        'R': None,
        'Chamfer-L2': None,
        'NormalConsistency': None,
        'AbsNormalConsistency': None,
        'F1@0.100000': None,
        'F1@0.200000': None,
        'F1@0.300000': None,
        # 'F1@0.400000': None,
        # 'F1@0.500000': None,
        'Precision@0.100000': None,
        'Precision@0.200000': None,
        'Precision@0.300000': None,
        # 'Precision@0.400000': None,
        # 'Precision@0.500000': None,
        'Recall@0.100000': None,
        'Recall@0.200000': None,
        'Recall@0.300000': None,
        # 'Recall@0.400000': None,
        # 'Recall@0.500000': None,
    }
    valfmts = {
        'R': "{x:.2f}",
        'Chamfer-L2': "{x:.3f}",
        'NormalConsistency': "{x:.2f}",
        'AbsNormalConsistency': "{x:.2f}",
        'F1@0.100000': "{x:.1f}",
        'F1@0.200000': "{x:.1f}",
        'F1@0.300000': "{x:.1f}",
        # 'F1@0.400000': "{x:.1f}",
        # 'F1@0.500000': "{x:.1f}",
        'Precision@0.100000': "{x:.1f}",
        'Precision@0.200000': "{x:.1f}",
        'Precision@0.300000': "{x:.1f}",
        # 'Precision@0.400000': "{x:.1f}",
        # 'Precision@0.500000': "{x:.1f}",
        'Recall@0.100000': "{x:.1f}",
        'Recall@0.200000': "{x:.1f}",
        'Recall@0.300000': "{x:.1f}",
        # 'Recall@0.400000': "{x:.1f}",
        # 'Recall@0.500000': "{x:.1f}",
    }
    k0s = list(titles.keys())
    k1s_plot = [f'{n}' for n in r_noises]
    k2s_plot = [f'ours v{v}' for v in views] + [f'nerf v{v}' for v in views]

    # perc = collated_to_grid(ours_results_collated, k0s, k1s, k2s, reduce_func = lambda x: np.percentile(x, p))
    for i,k0 in enumerate(k0s):
        # fig, ax = plt.subplots()
        fig = plt.figure()
        ax = None
        if k0 in ['Chamfer-L2', 'R']:
            textcolors=("white", "black")
            cmap = 'YlGn_r'
        else:
            textcolors=("black", "white")
            cmap = 'YlGn'

        # Create data grid
        ours_Ys = np.array([[ours_q.at[(f'r{rn}t0h0', f'v{v}'), k0] for v in views] for rn in r_noises])
        nerf_Ys = np.array([[nerf_q.at[(f'r{rn}t0h0', f'v{v}'), k0] for v in views] for rn in r_noises])
        data = np.concatenate([ours_Ys, nerf_Ys], axis=1)

        # im, cbar = heatmap(perc[i], k1s, k2s, ax=ax, cmap=cmap, cbarlabel=k0)
        im, cbar = heatmap(data, k1s_plot, k2s_plot, ax=ax, cmap=cmap, show_colorbar=False)
        texts = annotate_heatmap(im, valfmt=valfmts[k0], textcolors=textcolors, size=14)
        plt.title(k0)

        fig.tight_layout()
        figure_dir.mkdir(exist_ok=True, parents=True)
        plt.savefig(figure_dir / f'{k0}_{percentile}.pdf', bbox_inches='tight', pad_inches=0)
        plt.close('all')


def plot_graphs_grid_compare(ours_q, nerf_q, r_noises=[10,20,30], views=[4,6,8,12], figure_dir=Path('figures/'), percentile=50):
    # collated['R']['r10t0h0']['ours-v8'] = np.array([...])

    titles = {
        'R': 'Rotation Error (degrees) ↓',
        'Chamfer-L2': 'Chamfer-L2 ↓',
        'NormalConsistency': 'NormalConsistency ↑',
        'AbsNormalConsistency': 'AbsNormalConsistency ↑',
        'F1@0.100000': 'F1@0.1 ↑',
        'F1@0.200000': 'F1@0.2 ↑',
        'F1@0.300000': 'F1@0.3 ↑',
        # 'F1@0.400000': 'F1@0.4 ↑',
        # 'F1@0.500000': 'F1@0.5 ↑',
        'Precision@0.100000': 'Precision@0.1 ↑',
        'Precision@0.200000': 'Precision@0.2 ↑',
        'Precision@0.300000': 'Precision@0.3 ↑',
        # 'Precision@0.400000': 'Precision@0.4 ↑',
        # 'Precision@0.500000': 'Precision@0.5 ↑',
        'Recall@0.100000': 'Recall@0.1 ↑',
        'Recall@0.200000': 'Recall@0.2 ↑',
        'Recall@0.300000': 'Recall@0.3 ↑',
        # 'Recall@0.400000': 'Recall@0.4 ↑',
        # 'Recall@0.500000': 'Recall@0.5 ↑',
    }
    titles = {k:(k if v is None else v) for k,v in titles.items()}
    valfmts = {
        'R': "{x:.2f}",
        'Chamfer-L2': "{x:.3f}",
        'NormalConsistency': "{x:.2f}",
        'AbsNormalConsistency': "{x:.2f}",
        'F1@0.100000': "{x:.1f}",
        'F1@0.200000': "{x:.1f}",
        'F1@0.300000': "{x:.1f}",
        # 'F1@0.400000': "{x:.1f}",
        # 'F1@0.500000': "{x:.1f}",
        'Precision@0.100000': "{x:.1f}",
        'Precision@0.200000': "{x:.1f}",
        'Precision@0.300000': "{x:.1f}",
        # 'Precision@0.400000': "{x:.1f}",
        # 'Precision@0.500000': "{x:.1f}",
        'Recall@0.100000': "{x:.1f}",
        'Recall@0.200000': "{x:.1f}",
        'Recall@0.300000': "{x:.1f}",
        # 'Recall@0.400000': "{x:.1f}",
        # 'Recall@0.500000': "{x:.1f}",
    }
    k0s = list(titles.keys())
    k2s_plot = [f'v{v}' for v in views]
    k1s_plot = [f'{n}' for n in r_noises]

    # perc_ours = collated_to_grid(ours_results_collated, k0s, k1s, k2s_ours, reduce_func = lambda x: np.percentile(x, p))
    # perc_nerfopt = collated_to_grid(ours_results_collated, k0s, k1s, k2s_nerfopt, reduce_func = lambda x: np.percentile(x, p))
    for i,k0 in enumerate(k0s):
        if k0 in ['Chamfer-L2', 'R']:
            textcolors=("white", "black")
            textcolors2=("black", "black")
            cmap = 'YlGn_r'
            cmap2 = 'PiYG_r'
        else:
            textcolors=("black", "white")
            textcolors2=("black", "black")
            cmap = 'YlGn'
            cmap2 = 'PiYG'

        fig, (ax1,ax2) = plt.subplots(1,2)

        # Create data grid
        ours_Ys = np.array([[ours_q.at[(f'r{rn}t0h0', f'v{v}'), k0] for v in views] for rn in r_noises])
        nerf_Ys = np.array([[nerf_q.at[(f'r{rn}t0h0', f'v{v}'), k0] for v in views] for rn in r_noises])

        # im, cbar = heatmap(perc[i], k1s, k2s, ax=ax, cmap=cmap, cbarlabel=k0)
        im, cbar = heatmap(ours_Ys, k1s_plot, k2s_plot, ax=ax1, cmap=cmap,
                            show_colorbar=False, slant_labels=False, xticksize=13, yticksize=13)
        texts = annotate_heatmap(im, valfmt=valfmts[k0], textcolors=textcolors, size=14)
        ax1.set_title('ours', fontsize=15)
        # plt.yticks(fontsize=13)
        # plt.xticks(fontsize=13)

        # # Difference
        # vvv = np.abs((perc_nerfopt[i] - perc_ours[i])).max()
        # im, cbar = heatmap(perc_nerfopt[i] - perc_ours[i], k1s_plot, k2s_plot, ax=ax2, cmap=cmap2,
        #                     show_colorbar=False, slant_labels=False, xticksize=13, yticksize=13,
        #                     vmin=-vvv, vmax=vvv)
        # texts = annotate_heatmap(im, valfmt=valfmts2[k0], textcolors=textcolors2, size=14)
        # ax2.set_title('nerf-opt', fontsize=15)

        # Percentage difference
        percent_diff = (nerf_Ys - ours_Ys)/(ours_Ys) * 100
        vvv = np.abs(percent_diff).max()
        im, cbar = heatmap(percent_diff, k1s_plot, k2s_plot, ax=ax2, cmap=cmap2,
                            show_colorbar=False, slant_labels=False, xticksize=13, yticksize=13,
                            vmin=-vvv, vmax=vvv)
        texts = annotate_heatmap(im, valfmt="{x:+.0f}%", textcolors=textcolors2, size=12)
        ax2.set_title('nerf-opt', fontsize=15)

        ax2.get_yaxis().set_ticks([])
        # plt.tick_params(
        #     axis='x',          # changes apply to the x-axis
        #     which='both',      # both major and minor ticks are affected
        #     bottom=False,      # ticks along the bottom edge are off
        #     top=False,         # ticks along the top edge are off
        #     labelbottom=False) # labels along the bottom edge are off

        # plt.yticks(fontsize=13)
        # plt.xticks(fontsize=13)

        # breakpoint()

        # for tick in chain(ax1.xaxis.get_major_ticks(), ax2.xaxis.get_major_ticks()):
        #     tick.label.set_fontsize(16)
        # for tick in chain(ax1.yaxis.get_major_ticks(), ax2.yaxis.get_major_ticks()):
        #     tick.label.set_fontsize(16)

        fig.tight_layout(pad=0, h_pad=0, w_pad=0)
        fig.suptitle(titles[k0], y=0.9, fontsize=16)
        # plt.tight_layout(pad=0, h_pad=0, w_pad=0)
        # plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

        figure_dir.mkdir(exist_ok=True, parents=True)
        plt.savefig(figure_dir / f'{k0}_{percentile}.pdf', bbox_inches='tight', pad_inches=0)
        plt.close('all')


def heatmap(data, row_labels, col_labels, ax=None,
            show_colorbar=True, slant_labels=True,
            xticksize=10, yticksize=10,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    if show_colorbar:
        # Create colorbar
        cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
        cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
    else:
        cbar = None

    # We want to show all ticks...
    # ... and label them with the respective list entries.
    if xticksize>0:
        ax.set_xticks(np.arange(data.shape[1]))
        ax.set_xticklabels(col_labels, fontsize=xticksize)
    if yticksize>0:
        ax.set_yticks(np.arange(data.shape[0]))
        ax.set_yticklabels(row_labels, fontsize=yticksize)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    if slant_labels:
        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
                rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    if xticksize>0: ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    if yticksize>0: ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


# %% Main function
if __name__ == '__main__':
    ours_g2p, nerf_g2p = load_results_pd('icp_g2p_noscale_centered')
    ours_p2g, nerf_p2g = load_results_pd('icp_p2g_noscale_centered')
    ours_noicp, nerf_noicp = load_results_pd('none')

    # Check to make sure Chamfer-L2-p2g and Chamfer-L2-g2p behave as expected
    assert (ours_g2p['Chamfer-L2-g2p'] <= (ours_noicp['Chamfer-L2-g2p'])).mean() >= 0.98
    assert (ours_p2g['Chamfer-L2-p2g'] <= (ours_noicp['Chamfer-L2-p2g'])).mean() >= 0.98
    assert (nerf_g2p['Chamfer-L2-g2p'] <= (nerf_noicp['Chamfer-L2-g2p'])).mean() >= 0.98
    assert (nerf_p2g['Chamfer-L2-p2g'] <= (nerf_noicp['Chamfer-L2-p2g'])).mean() >= 0.98

    ours_best = combine_results_from_alignments([ours_g2p, ours_p2g, ours_noicp], ours_noicp)
    nerf_best = combine_results_from_alignments([nerf_g2p, nerf_p2g, nerf_noicp], nerf_noicp)

    figure_dir = Path('figures/')
    ours_q50 = ours_best.groupby(['noise', 'views']).quantile().sort_index(axis=1).rename(columns={'R/avg': 'R'})
    nerf_q50 = nerf_best.groupby(['noise', 'views']).quantile().sort_index(axis=1).rename(columns={'R/avg': 'R'})
    plot_graphs_vs_views(ours_q50, nerf_q50, figure_dir = figure_dir / 'vs_views_together')
    plot_graphs_grid(ours_q50, nerf_q50, figure_dir = figure_dir / 'grid' / 'ours_nerfopt')
    plot_graphs_grid_compare(ours_q50, nerf_q50, figure_dir = figure_dir / 'grid_compare' / 'ours_nerfopt')
