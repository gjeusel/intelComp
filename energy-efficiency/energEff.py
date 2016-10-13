#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import subprocess

import matplotlib
import matplotlib.gridspec as gridspec
matplotlib.use("Qt4Agg") # enable plt.show() to display
import matplotlib.pyplot as plt
from matplotlib.externals import six

import pandas as pd
import seaborn as sns
import numpy as np

def normalize_df(df): #{{{
    df_norm = (df - df.mean()) / (df.max() - df.min())
    return df_norm
#}}}

def render_mpl_table(data, col_width=3.0, row_height=0.625, font_size=14, #{{{
                     header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w',
                     bbox=[0, 0, 1, 1], header_columns=0,
                     ax=None, **kwargs):
    if ax is None:
        size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
        fig, ax = plt.subplots(figsize=size)
        ax.axis('off')

    mpl_table = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns, **kwargs)

    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    for k, cell in six.iteritems(mpl_table._cells):
        cell.set_edgecolor(edge_color)
        if k[0] == 0 or k[1] < header_columns:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0]%len(row_colors) ])
    return ax
#}}}

# write_df_table_desc, write_df_histogram, write_df_box : {{{
def write_df_table_desc(_dataset, _fout='out.png', _header_color='#40466e',
                    _row_colors=['#f1f1f2', 'w'], _edge_color='w'):

    table_describe = _dataset.describe()
    ax = render_mpl_table(table_describe, header_color=_header_color,
                          row_colors=_row_colors, edge_color=_edge_color)
    plt.savefig(_fout)
    plt.close()

def write_df_histogram(_dataset, _fout, _color='#40466e', _fig_size=(20,10)):
    _dataset.hist(color=_color, alpha=0.8, bins=40, figsize=_fig_size)
    plt.savefig(_fout)
    plt.close()

def write_df_box(_dataset, _fout, _fig_size=(20,10)):
    fig = _dataset.plot.box(figsize = _fig_size)
    plt.savefig(_fout)
    plt.close()
#}}}

def write_df_statistics(dataset): #{{{
    # Describe
    write_df_table_desc(dataset.iloc[:, 0:8], "Xvar_desc.png", _header_color='#40466e')
    write_df_table_desc(dataset.iloc[:, 8:10], "Yvar_desc.png", _header_color='#7b241b')

    # Histograms :
    dataset_normed = normalize_df(dataset)
    write_df_histogram(dataset_normed.iloc[:, 0:8], "Xvar_histograms.png", _color='#40466e', _fig_size=(20,10))
    write_df_histogram(dataset_normed.iloc[:, 8:10], "Yvar_histograms.png", _color='#7b241b', _fig_size=(10,5))

    # Boxplot :
    dataset_normed = normalize_df(dataset)
    write_df_box(dataset_normed.iloc[:, 0:8], "Xvar_boxplot.png", _fig_size=(20,10))
    write_df_box(dataset_normed.iloc[:, 8:10], "Yvar_boxplot.png", _fig_size=(10,5))
#}}}

def main():
    """Main program : energeff"""
    parser = argparse.ArgumentParser(description='Energetic Study of Building according to their shapes')

    #ArgumentParser.add_argument(name or flags...[, action][, nargs][, const][, default][, type][, choices][, required][, help][, metavar][, dest])

    parser.add_argument('dataset', type=str, action='store',
            metavar='<dataset>', help='dataset path')

    args = parser.parse_args()  # de type <class 'argparse.Namespace'>

    print 'Reading ' + args.dataset + ' ...'
    dataset = pd.read_excel(args.dataset)
    dataset.columns = [ "X1 Rel. Compactness",          \
                        "X2 Surface Area",              \
                        "X3 Wall Area",                 \
                        "X4 Roof Area",                 \
                        "X5 Overall Height",            \
                        "X6 Orientation",               \
                        "X7 Glazing Area",              \
                        "X8 Glazing Area Distr.",       \
                        "y1 Heating Load",              \
                        "y2 Cooling Load",              \
                      ]

    write_df_statistics(dataset)

    # cm = sns.light_palette("green", as_cmap=True)
    # table_describe.style.background_gradient(cmap=cm)

    # df.style.set_caption('Colormaps, with a caption.')\
    #         .background_gradient(cmap=cm)

    # plt.show() # show all figures


if __name__ == '__main__':
    main()
