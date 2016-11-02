#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import subprocess

import matplotlib
matplotlib.use("Qt4Agg") # enable plt.show() to display
import matplotlib.pyplot as plt

import seaborn as sns
import numpy as np

import pandas as pd

def fout_to_title(_fout):
    import re
    title = re.sub(r"_", " ", _fout)
    title = re.sub(r"\.png", "", title)
    return title

# Norms functions : {{{
def normalize_min_max(df):
    result = df.copy()
    for feature_name in df.columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result

def normalize_Z_score(df):
    result = df.copy()
    for feature_name in df.columns:
        mean_value = df[feature_name].mean()
        standard_deviation = df[feature_name].std()
        result[feature_name] = (df[feature_name] - mean_value) / standard_deviation
    return result
#}}}


def render_mpl_table(data, col_width=3.0, row_height=0.625, font_size=14, #{{{
                     header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w',
                     bbox=[0, 0, 1, 1], header_columns=0,
                     ax=None, **kwargs):

    from matplotlib.externals import six

    if ax is None:
        size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
        fig, ax = plt.subplots(figsize=size)
        ax.axis('off')

    mpl_table = ax.table(cellText=data.values, bbox=bbox, rowLabels=data.index, colLabels=data.columns, **kwargs)

    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    for k, cell in six.iteritems(mpl_table._cells):
        cell.set_edgecolor(edge_color)
        if k[0] == 0 or k[1] < header_columns:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0]%len(row_colors) ])
    return fig, ax
#}}}

def write_df_table_desc(_dataset, _save_fig, _fout, _header_color='#40466e', #{{{
                    _row_colors=['#f1f1f2', 'w'], _edge_color='w'):

    table_describe = _dataset.describe()
    fig, ax = render_mpl_table(table_describe, header_color=_header_color,
                          row_colors=_row_colors, edge_color=_edge_color)
    plt.title(fout_to_title(_fout), fontsize=20)
    if(_save_fig):
        plt.savefig(_fout, bbox_inches='tight')
#}}}

def write_df_histogram(_dataset, _save_fig, _fout, _color='#40466e', _fig_size=(20,10)):#{{{
    _dataset.hist(color=_color, alpha=0.8, bins=20, figsize=_fig_size)
    plt.title(fout_to_title(_fout), fontsize=20)
    if(_save_fig):
        plt.savefig(_fout, bbox_inches='tight')
#}}}

def write_df_box(_dataset, _save_fig, _fout, _fig_size=(20,10)):#{{{
    fig = _dataset.plot.box(figsize = _fig_size, showfliers=True)
    plt.title(fout_to_title(_fout), fontsize=20)
    if(_save_fig):
        plt.savefig(_fout, bbox_inches='tight')
#}}}


def write_scatter_matrix(df, _save_fig, _fout, _fig_size=(25,15)): #{{{
    from pandas.tools.plotting import scatter_matrix # grafico de projecçao
    axs = scatter_matrix(df, alpha=0.5, figsize=_fig_size)

    for ax in axs[:,0]: # the left boundary
        # ax.grid('off', axis='both')
        ax.set_ylabel(ax.get_ylabel(), rotation=0, verticalalignment='center', labelpad=55)
        ax.set_yticks([])

    plt.title(fout_to_title(_fout), fontsize=20)

    if(_save_fig):
        plt.savefig(_fout, bbox_inches='tight')
#}}}

def write_correlation_mat(df, _save_fig, _fout, _fig_size=(15,15)): #{{{
    # Compute correlation matrix
    corrmat = df.corr()

    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=_fig_size)

    # Set ax & colormap with seaborn. But do not touch at axes
    ax = sns.heatmap(corrmat, vmax=1, square=True, linewidths=1, xticklabels=False, yticklabels=False)

    ax.set_xticklabels(df.columns, minor=False, rotation='vertical')
    ax.set_yticklabels(df.columns[df.shape[1]::-1], minor=False, rotation='horizontal')

    ax.xaxis.tick_top()
    ax.yaxis.tick_left()

    figure_title = fout_to_title(_fout)
    plt.text(0.5, -0.05, figure_title,
             horizontalalignment='center',
             verticalalignment='bottom',
             fontsize=20,
             transform = ax.transAxes)

    # plt.subplots_adjust(left=0.1, right=1, top=0.85)

    if(_save_fig):
        plt.savefig(_fout, bbox_inches='tight')

#}}}

def write_distance_matrix(df, _metric, _save_fig, _fout, _fig_size=(15,15)):#{{{
    from scipy.spatial.distance import pdist, squareform
    DistMatrix = squareform(pdist(df, metric=_metric))

    fig, ax = plt.subplots(figsize=_fig_size)
    plt.colorbar(ax.matshow(DistMatrix, alpha=0.8, cmap="jet")) #matshow with colormap legend

    plt.title(fout_to_title(_fout), fontsize=20)
    if(_save_fig):
        plt.savefig(_fout, bbox_inches='tight')
    # return DistMatrix
#}}}


def df_descr_statistics(dataset, savefig): #{{{
    # Normalize
    dataset_Z_normed = normalize_Z_score(dataset)

    # Describe
    print "Describe ..."
    write_df_table_desc(dataset.iloc[:, 0:8], _save_fig=savefig, _fout="Xvar_desc.png", _header_color='#40466e')
    write_df_table_desc(dataset.iloc[:, 8:10], _save_fig=savefig, _fout="Yvar_desc.png", _header_color='#7b241b')

    # Histograms :
    print "Histograms ..."
    write_df_histogram(dataset.iloc[:, 0:8], _save_fig=savefig, _fout="Xvar_histograms.png", _color='#40466e', _fig_size=(20,10))
    write_df_histogram(dataset.iloc[:, 8:10], _save_fig=savefig, _fout="Yvar_histograms.png", _color='#7b241b', _fig_size=(10,7))
    write_df_histogram(dataset_Z_normed.iloc[:, 0:8], _save_fig=savefig, _fout="Xvar_Znormalized_histograms.png", _color='#40466e', _fig_size=(20,10))
    write_df_histogram(dataset_Z_normed.iloc[:, 8:10], _save_fig=savefig, _fout="Yvar_Znormalized_histograms.png", _color='#7b241b', _fig_size=(10,7))

    # Boxplot :
    print "Boxplot ..."
    write_df_box(dataset_Z_normed.iloc[:, 0:8], _save_fig=savefig, _fout="Xvar_Znormalized_boxplot.png", _fig_size=(20,10))
    write_df_box(dataset_Z_normed.iloc[:, 8:10], _save_fig=savefig, _fout="Yvar_Znormalized_boxplot.png", _fig_size=(10,5))
#}}}

def df_descr_matrices(dataset, savefig): #{{{
    # Normalize
    dataset_Z_normed = normalize_Z_score(dataset)

    # Scatter matrix :
    print "Scatter matrix ..."
    write_scatter_matrix(dataset, _save_fig=savefig, _fout="Scatter_matrix.png", _fig_size=(25,15))

    # Correlation matrix :
    print "Correlation matrix ..."
    write_correlation_mat(dataset, _save_fig=savefig, _fout="correlation_matrix.png")

    # Distance matrix :
    print "Distance matrix ..."
    write_distance_matrix(dataset_Z_normed, _metric="euclidean", _save_fig=savefig, _fout="distance_matrix_euclidean_Znormed.png")

#}}}


def ordinary_least_square(data): #{{{
    from sklearn.linear_model import LinearRegression
    lr = LinearRegression()
    lr.fit(data[:, 0:8], data[:, 8:10])
    return lr
#}}}


def main():
    """Main program : energeff"""
    parser = argparse.ArgumentParser(description='Energetic Study of Building according to their shapes')

    #ArgumentParser.add_argument(name or flags...[, action][, nargs][, const][, default][, type][, choices][, required][, help][, metavar][, dest])

    parser.add_argument('dataset', type=str, action='store',
            metavar='<dataset_path>', help='dataset path')

    parser.add_argument('--descr', action='store_true', default=False, dest='descr_datas',
            help='whether to right data infos or not')

    parser.add_argument('--save_fig', action='store_true', default=False, dest='save_fig',
            help='whether to save figures generated by --descr in png')

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


    if (args.descr_datas is True):
        df_descr_statistics(dataset, savefig=args.save_fig)
        df_descr_matrices(dataset, savefig=args.save_fig)

        # figs = list(map(plt.figure, plt.get_fignums()))
        # axes = plt.gca()    # Get current axes
        # axes.lines.remove(figs) # Removes the (first and only) line created in ax2
        # plt.draw()          # Updates the graph (in interactive mode)

        # plt.show()


    # ###################  Linear regression: ###################### {{{
    # dataset_X = dataset.iloc[:, 0:8]
    # dataset_Y = dataset.iloc[:, 8:10]

    # data = np.asarray(dataset) #convert datafram to numpy Narray
    # # Normalize
    # dataset_Z_normed = normalize_Z_score(dataset)
    # data_Z_normed = np.asarray(dataset_Z_normed)

    # # # res = ols(y=dataset_Y, x=dataset_X)
    # # res = ols(y=dataset['y1 Heating Load'], x=dataset_X)
    # # print res

    # lr_ols = ordinary_least_square(data_Z_normed)
    # print "lr_ols.coef_ = ", lr_ols.coef_
    # print "lr_ols.residues_ = ", lr_ols.residues_

    # #}}}


if __name__ == '__main__':
    main()
