"""Main plotting module."""

import locale
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def init_params(font_size=20, font_family='Carlito', pdf_padding=0.1,
                pdf_bbox='tight', pdf_fonttype=42,
                deact_warnings=True):
    """Initialize RC parameters for matplotlib plots."""
    locale.setlocale(locale.LC_TIME, 'de_DE.UTF-8')
    mpl.rcParams['font.size'] = font_size
    mpl.rcParams['font.family'] = font_family
    mpl.rcParams['savefig.pad_inches'] = pdf_padding
    mpl.rcParams['savefig.bbox'] = pdf_bbox
    plt.rcParams['pdf.fonttype'] = pdf_fonttype

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


def monthlyBar(data, figsize=[12, 5.5], return_ax=False, **kwargs):
    """Create bar chart of sum of monthly unit commitment."""
    monSum = data.resample('M').sum()/1e3
    monSum.rename(index=lambda x: x.strftime('%b'), inplace=True)

    nr_cols = len(monSum.columns)

    if 'colors' in kwargs:
        colors = kwargs['colors']
    else:
        colors = list(znes_colors(nr_cols).values())

    fig, ax = plt.subplots(figsize=figsize)

    for i, col in enumerate(monSum.columns):
        i += 1
        bottom = 0
        if i < nr_cols:
            bottom_cols = monSum.columns[i:]
            for bottom_col in bottom_cols:
                bottom += monSum[bottom_col]

        ax.bar(monSum.index, monSum[col], bottom=bottom, color=colors[-i])

    ax.grid(linestyle='--', which='major', axis='y')

    if 'ylabel' in kwargs:
        ax.set_ylabel(kwargs['ylabel'])
    else:
        ax.set_ylabel('Gesamtwärmemenge in GWh')

    if 'title' in kwargs:
        ax.set_title(kwargs['title'])

    if 'suptitle' in kwargs:
        fig.suptitle(kwargs['suptitle'])

    if 'labels' in kwargs:
        labels = kwargs['labels']
    else:
        labels = monSum.columns.to_list()
    ax.legend(labels=labels)

    if return_ax:
        return ax
