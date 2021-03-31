"""Main plotting module."""

import math
import locale
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def init_params(german_labels=True, font_size=20, font_family='Carlito',
                pdf_padding=0.1, pdf_bbox='tight', pdf_fonttype=42,
                deact_warnings=True):
    """Initialize RC parameters for matplotlib plots."""
    if german_labels:
        locale.setlocale(locale.LC_TIME, 'de_DE.UTF-8')
    mpl.rcParams['font.size'] = font_size
    mpl.rcParams['font.family'] = font_family
    mpl.rcParams['savefig.pad_inches'] = pdf_padding
    mpl.rcParams['savefig.bbox'] = pdf_bbox
    plt.rcParams['pdf.fonttype'] = pdf_fonttype
    mpl.rcParams['hatch.linewidth'] = 2

    if deact_warnings:
        mpl.rcParams.update({'figure.max_open_warning': 0})


def znes_colors(n=None):
    """Return dict with ZNES colors.

    Examples
    --------
    >>> znes_colors().keys()  # doctest: +ELLIPSIS
    dict_keys(['darkblue', 'red', 'lightblue', 'orange', 'grey',...

    Original author: @ckaldemeyer
    """
    colors = {
        'darkblue': '#00395B',
        'red': '#B54036',
        'lightblue': '#74ADC0',
        'orange': '#EC6707',
        'grey': '#BFBFBF',
        'dimgrey': 'dimgrey',
        'lightgrey': 'lightgrey',
        'slategrey': 'slategrey',
        'darkgrey': '#A9A9A9'
    }

    # allow for a dict of n colors
    if n is not None:
        if n > len(colors):
            raise IndexError('Number of requested colors is too big.')
        else:
            return {k: colors[k] for k in list(colors)[:n]}
    else:
        return colors


def znes_colors_hatched(n, diff_colors=4):
    """Return list of dicts with ZNES colors with hatches."""
    colors = list(znes_colors().values())
    hatches = ['//', '\\\\', '////', '\\\\\\\\']

    return_list = list()

    for i in range(n):
        if i < diff_colors:
            return_list += [{'color': colors[i % diff_colors],
                             'edgecolor': 'w'}]
        else:
            return_list += [{'color': colors[i % diff_colors],
                             'hatch': hatches[((math.floor(i/diff_colors) - 1)
                                               % 4)],
                             'edgecolor': 'w'}]

    return return_list


def get_colors(nr_cols, **kwargs):
    """Get color parameters list of dictionaries."""
    color_params = list()
    if 'colors' in kwargs:
        for color in kwargs['colors']:
            color_params += [{'color': color}]
    elif 'hatches' in kwargs:
        if 'diff_colors' in kwargs:
            color_params = znes_colors_hatched(
                nr_cols, diff_colors=kwargs['diff_colors'])
        else:
            color_params = znes_colors_hatched(nr_cols)
    else:
        colors = list(znes_colors(nr_cols).values())
        for color in colors:
            color_params += [{'color': color}]

    return color_params


def create_multipage_pdf(file_name='plots.pdf', figs=None, dpi=300,
                         mute=False):
    """Save all open matplotlib figures into a multipage pdf-file.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>>
    >>> df1 = pd.DataFrame(np.random.randn(24, 2))
    >>> ax1 = df1.plot(kind='line')
    >>>
    >>> df2 = pd.DataFrame(np.random.randn(24, 2))
    >>> ax2 = df2.plot(kind='scatter', x=0, y=1)
    >>>
    >>> # mute is set to true to surpress writing a pdf file
    >>> create_multipage_pdf(file_name='plots.pdf', dpi=300, mute=True)
    False

    Original author: @ckaldemeyer
    """
    if mute is True:
        # set return flag to false if no output is written
        flag = False
    else:
        pp = PdfPages(file_name)
        if figs is None:
            figs = [plt.figure(n) for n in plt.get_fignums()]
        for fig in figs:
            fig.savefig(pp, format='pdf')
        pp.close()

        # close all existing figures
        for fig in figs:
            plt.close(fig)

        # set return flag
        flag = True

    return flag


def monthlyBar(data, figsize=[12, 5.5], legend_loc='best', return_objs=False,
               **kwargs):
    """Create bar chart of sum of monthly unit commitment."""
    monSum = data.resample('M').sum()/1e3
    monSum.rename(index=lambda x: x.strftime('%b'), inplace=True)

    nr_cols = len(monSum.columns)

    if 'color_params' in kwargs:
        color_params = kwargs['color_params']
    else:
        color_params = get_colors(nr_cols, **kwargs)

    fig, ax = plt.subplots(figsize=figsize)

    pos_bottom = 0
    neg_bottom = 0

    for col, color_param in zip(monSum.columns, color_params):
        mean_val = monSum[col].mean()
        if mean_val >= 0:
            ax.bar(monSum.index, monSum[col],
                   bottom=pos_bottom, **color_param)
            pos_bottom += monSum[col]
        elif mean_val < 0:
            ax.bar(monSum.index, monSum[col],
                   bottom=neg_bottom, **color_param)
            neg_bottom += monSum[col]

    if 'demand' in kwargs:
        monDemand = kwargs['demand'].resample('M').sum()/1e3
        monDemand.rename(index=lambda x: x.strftime('%b'), inplace=True)
        ax.bar(monSum.index, monDemand,
               width=0.25, color=znes_colors()['lightgrey'], alpha=0.75,
               linewidth=0)

    ax.grid(linestyle='--', which='major', axis='y')

    if 'ylabel' in kwargs:
        ax.set_ylabel(kwargs['ylabel'])
    else:
        ax.set_ylabel('Gesamtwärmemenge in GWh')

    if 'xlabel' in kwargs:
        ax.set_xlabel(kwargs['xlabel'])

    if 'title' in kwargs:
        ax.set_title(kwargs['title'])

    if 'suptitle' in kwargs:
        fig.suptitle(kwargs['suptitle'])

    if 'labels' in kwargs:
        labels = kwargs['labels']
    else:
        labels = monSum.columns.to_list()
    if legend_loc[:7] == 'outside':
        if legend_loc[8:] == 'right':
            ax.legend(labels=labels, loc='upper right',
                      bbox_to_anchor=(1.27, 1),
                      ncol=1)
        elif legend_loc[8:] == 'bottom':
            ax.legend(labels=labels, loc='lower left',
                      bbox_to_anchor=(0, -0.265),
                      ncol=nr_cols)
    else:
        ax.legend(labels=labels, loc=legend_loc)

    if return_objs:
        return fig, ax


def load_curve(data, figsize=[8, 5], linewidth=2.5, legend_loc='best', return_objs=False,
               **kwargs):
    """Plot the sorted (annual) load curves of units."""
    data = data.apply(lambda x: x.sort_values(ascending=False).values)
    data.reset_index(drop=True, inplace=True)

    nr_cols = len(data.columns)

    color_params = get_colors(nr_cols, **kwargs)

    fig, ax = plt.subplots(figsize=figsize)

    for col, color_param in zip(data.columns, color_params):
        ax.plot(data[col], linewidth=linewidth, **color_param)

    ax.grid(linestyle='--')

    if 'ylabel' in kwargs:
        ax.set_ylabel(kwargs['ylabel'])
    else:
        ax.set_ylabel(r'Wärmestrom $\dot{Q}$ in MW')

    if 'xlabel' in kwargs:
        ax.set_xlabel(kwargs['xlabel'])
    else:
        ax.set_xlabel('Stunden')

    if 'title' in kwargs:
        ax.set_title(kwargs['title'])

    if 'suptitle' in kwargs:
        fig.suptitle(kwargs['suptitle'])

    if 'labels' in kwargs:
        labels = kwargs['labels']
    else:
        labels = data.columns.to_list()
    if legend_loc[:7] == 'outside':
        if legend_loc[8:] == 'right':
            ax.legend(labels=labels, loc='upper right',
                      bbox_to_anchor=(1.33, 1),
                      ncol=1)
        elif legend_loc[8:] == 'bottom':
            anchor = (0, -0.35)
            if nr_cols > 4:
                nr_cols = round(nr_cols/2)
                anchor = (0, -0.45)
            ax.legend(labels=labels, loc='lower left',
                      bbox_to_anchor=anchor,
                      ncol=nr_cols)
    else:
        ax.legend(labels=labels, loc=legend_loc)

    if return_objs:
        return fig, ax
