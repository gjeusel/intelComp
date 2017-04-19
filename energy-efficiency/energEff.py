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

from sklearn.cross_validation import train_test_split, cross_val_score, cross_val_predict

# Classifications :
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, confusion_matrix, classification_report

# Regression :
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

# Global variables :
_save_fig = False;
_with_X6 = False;

_blue_color = '#40466e';
_red_color = '#7b241b';
_beige_color = '#ebcb92';
_brown_color = '#905000';


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


def render_mpl_table(data, col_width=3.0, row_height=0.625, font_size=12, #{{{
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


# Data analysis : description, histogram, box-plot functions : {{{
def write_df_table_desc(_dataset, _save_fig, _fout, _header_color='#40466e', #{{{
                    _row_colors=['#f1f1f2', 'w'], _edge_color='w'):

    table_describe = _dataset.describe()
    fig, ax = render_mpl_table(table_describe, header_color=_header_color,
                          row_colors=_row_colors, edge_color=_edge_color)
    if(_save_fig):
        _fout = "images/" + _fout
        plt.savefig(_fout, bbox_inches='tight')
#}}}

def write_df_histogram(_dataset, _save_fig, _fout, _color='#40466e', _fig_size=(20,10)):#{{{
    _dataset.hist(color=_color, alpha=0.8, bins=20, figsize=_fig_size)
    if(_save_fig):
        _fout = "images/" + _fout
        plt.savefig(_fout, bbox_inches='tight')
#}}}

def write_df_box(_dataset, _save_fig, _fout, _fig_size=(20,10)):#{{{
    fig = _dataset.plot.box(figsize = _fig_size, showfliers=True)
    if(_save_fig):
        _fout = "images/" + _fout
        plt.savefig(_fout, bbox_inches='tight')
#}}}
#}}}

# Data analysis : scatter, correlation, distance matrices functions : {{{
def write_scatter_matrix(df, _save_fig, _fout, _fig_size=(25,15)): #{{{
    from pandas.tools.plotting import scatter_matrix # grafico de projecçao
    axs = scatter_matrix(df, alpha=0.5, figsize=_fig_size)

    for ax in axs[:,0]: # the left boundary
        # ax.grid('off', axis='both')
        ax.set_ylabel(ax.get_ylabel(), rotation=0, verticalalignment='center', labelpad=55)
        ax.set_yticks([])

    if(_save_fig):
        _fout = "images/" + _fout
        plt.savefig(_fout, bbox_inches='tight')
#}}}

def write_correlation_mat(df, _save_fig, _fout, _fig_size=(15,15)): #{{{
    # Compute correlation matrix
    corrmat = df.corr()

    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=_fig_size)

    # Set ax & colormap with seaborn.
    ax = sns.heatmap(corrmat, vmin=-1, vmax=1, center=0, square=True, linewidths=1, xticklabels=True, yticklabels=True)

    ax.set_xticklabels(df.columns, minor=False, rotation='vertical')
    ax.set_yticklabels(df.columns[df.shape[1]::-1], minor=False, rotation='horizontal')

    if(_save_fig):
        _fout = "images/" + _fout
        plt.savefig(_fout, bbox_inches='tight')

#}}}

def compute_distance_matrix(df, _metric): #{{{
    from scipy.spatial.distance import pdist, squareform
    dist_mat = pdist(df, metric=_metric)
    dist_mat = squareform(dist_mat) #translates this flattened form into a full matrix
    return dist_mat
#}}}

def write_graph_mean_dist(df, _metric, _save_fig, _fout, _fig_size=(8,5)):#{{{
    from scipy.spatial.distance import pdist, squareform
    dist_array = pdist(df.values, metric=_metric)
    dist_mat = squareform(dist_array)

    abscisse = np.arange(dist_mat.shape[1])
    dist_mean_array = dist_mat.mean(1)
    dist_mean_array = np.sort(dist_mean_array)

    fig, ax = plt.subplots(figsize=_fig_size)
    plt.plot(abscisse, dist_mean_array, "o", markersize=2)

    if(_save_fig):
        _fout = "images/" + _fout
        plt.savefig(_fout, bbox_inches='tight')
#}}}

def write_distance_matrix(_dist_mat, _save_fig, _fout, _fig_size=(15,15)):#{{{

    fig, ax = plt.subplots(figsize=_fig_size)
    plt.colorbar(ax.matshow(_dist_mat, alpha=0.8, cmap="jet")) #matshow with colormap legend

    if(_save_fig):
        _fout = "images/" + _fout
        plt.savefig(_fout, bbox_inches='tight')
#}}}
#}}}

#{{{ Data analysis : Higher level functions call
def df_descr_statistics(dataset, savefig): #{{{
    # Normalize
    dataset_Z_normed = normalize_Z_score(dataset)

    # Describe
    print "Describe ..."
    write_df_table_desc(dataset.iloc[:, 0:4], _save_fig=savefig, _fout="Xvar_desc_part1.png", _header_color='#40466e')
    write_df_table_desc(dataset.iloc[:, 4:8], _save_fig=savefig, _fout="Xvar_desc_part2.png", _header_color='#40466e')
    write_df_table_desc(dataset.iloc[:, 8:10], _save_fig=savefig, _fout="Yvar_desc.png", _header_color='#7b241b')

    # Histograms :
    print "Histograms ..."
    write_df_histogram(dataset.iloc[:, 0:4], _save_fig=savefig, _fout="Xvar_histograms_part1.png", _color='#40466e', _fig_size=(20,10))
    write_df_histogram(dataset.iloc[:, 4:8], _save_fig=savefig, _fout="Xvar_histograms_part2.png", _color='#40466e', _fig_size=(20,10))
    write_df_histogram(dataset.iloc[:, 8:10], _save_fig=savefig, _fout="Yvar_histograms.png", _color='#7b241b', _fig_size=(10,7))
    write_df_histogram(dataset_Z_normed.iloc[:, 0:4], _save_fig=savefig, _fout="Xvar_Znormalized_histograms_part1.png", _color='#40466e', _fig_size=(20,10))
    write_df_histogram(dataset_Z_normed.iloc[:, 4:8], _save_fig=savefig, _fout="Xvar_Znormalized_histograms_part2.png", _color='#40466e', _fig_size=(20,10))
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
    # write_scatter_matrix(dataset, _save_fig=savefig, _fout="Scatter_matrix.png", _fig_size=(25,15))

    # Correlation matrix :
    print "Correlation matrix ..."
    write_correlation_mat(dataset, _save_fig=savefig, _fout="correlation_matrix.png")

    # Distance graph :
    print "Distance graph ..."
    dataset_X = dataset.iloc[:, 0:8]
    dataset_X_Z_normed = normalize_Z_score(dataset_X)

    write_graph_mean_dist(dataset_X_Z_normed, _metric="euclidean", _save_fig=savefig, _fout="graph_mean_euclidean_dist_Znormed.png")
    write_graph_mean_dist(dataset_X, _metric="mahalanobis", _save_fig=savefig, _fout="graph_mean_mahalanobis_dist.png")

    # Distance matrix :
    print "Distance matrix ..."
    dist_mat_euclidean = compute_distance_matrix(dataset_X_Z_normed, _metric="euclidean")
    write_distance_matrix(_dist_mat=dist_mat_euclidean, _save_fig=savefig, _fout="distance_matrix_euclidean_Z_normed.png")

    dist_mat_mahalanobis = compute_distance_matrix(dataset_X, _metric="mahalanobis")
    write_distance_matrix(_dist_mat=dist_mat_mahalanobis, _save_fig=savefig, _fout="distance_matrix_mahalanobis.png")

#}}}
#}}}


#{{{ Classification functions :
def df_reg_to_clf_problem(dataset, _save_fig): #{{{
    y_sum_real = dataset.iloc[:,8:9].as_matrix().ravel() + dataset.iloc[:,9:10].as_matrix().ravel()

    y_num_class = np.zeros((y_sum_real.size,), dtype = np.int)
    for i in range(0, y_sum_real.size):
        if y_sum_real[i] <= 50:
            y_num_class[i] = 1
        if y_sum_real[i] > 50 and y_sum_real[i] <= 90:
            y_num_class[i] = 2
        if y_sum_real[i] > 90:
            y_num_class[i] = 3

    df_class_pb = dataset.iloc[:,0:8]
    df_class_pb["Y Sum HL and CL"] = pd.Series(y_num_class, index = df_class_pb.index)

    df_tmp = pd.DataFrame(columns=['y1 Heating Load', 'y2 Cooling Load', 'y1 + y2', 'y'])
    df_tmp['y1 Heating Load'] = dataset['y1 Heating Load']
    df_tmp['y2 Cooling Load'] = dataset['y2 Cooling Load']
    df_tmp['y1 + y2'] = dataset['y1 Heating Load'] + dataset['y2 Cooling Load']
    df_tmp['y'] = y_num_class

    table_head = df_tmp.head(3)
    table_mid = df_tmp.iloc[300:303,:]
    table_tail = df_tmp.tail(3)
    table = pd.concat([table_head, table_mid, table_tail])

    fig, ax = render_mpl_table(table, header_color='#7b241b')
    if(_save_fig):
        _fout = "images/" + "datafram_classification_form"
        plt.savefig(_fout, bbox_inches='tight')

    return df_class_pb
#}}}

def write_confusion_mat(y_true, y_predicted, _scores, _save_fig, _fout, _fig_size=(6,5)): #{{{
    # Compute confusion matrix
    conf_mat = confusion_matrix(y_true, y_predicted)

    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=_fig_size)

    ax = sns.heatmap(conf_mat, annot=True, fmt="d", annot_kws={"size": 25}, cmap="Blues")

    ax.set_xticklabels(["True C1", "True C2"], minor=False, rotation='horizontal')
    ax.set_yticklabels(["Predicted C2", "Predicted C1"], minor=False, rotation='horizontal')

    ax.xaxis.tick_top()
    ax.yaxis.tick_left()

    label_acc = ("Accuracy: %0.2f (+/- %0.2f)" % (_scores.mean(), _scores.std() * 2))
    plt.figtext(0.18, -0.1, label_acc, fontsize=20)

    if(_save_fig):
        _fout = "images/" + _fout
        plt.savefig(_fout, bbox_inches='tight')

#}}}

def write_table_scores(df, _save_fig, _fout, _fig_size=(10,10)): #{{{
    fig, ax = render_mpl_table(df, header_color=_brown_color, row_colors=['#f1f1f2', 'w'], edge_color='w')

    if(_save_fig):
        _fout = "images/" + _fout
        plt.savefig(_fout, bbox_inches='tight')
#}}}

def classify(df_classification, _save_fig, k = 10): #{{{
    X = df_classification.iloc[:,0:8].as_matrix()
    y = df_classification.iloc[:,8:9].as_matrix().ravel()

    clfs = {
        "Naive_Bayes": {"clf_func_name": GaussianNB(), "y": None},
        "k-neighbors_k_1": {"clf_func_name": KNeighborsClassifier(n_neighbors=1), "y": None},
        "k-neighbors_5": {"clf_func_name": KNeighborsClassifier(n_neighbors=5), "y": None},
        "Decision_Tree_max_depth_none": {"clf_func_name": DecisionTreeClassifier(max_depth=None), "Y": None},
        "Decision_Tree_max_depth_5": {"clf_func_name": DecisionTreeClassifier(max_depth=5), "Y": None},
        "Random_Forest_10_trees_max_depth_none": {"clf_func_name": RandomForestClassifier(max_depth=None, n_estimators=10, max_features=1), "y": None},
        "Random_Forest_10_trees_max_depth_5": {"clf_func_name": RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1), "y": None},
        # "neural_network": {"clf_func_name": MLPClassifier(alpha=1), "Y": None}
        #"SVC": {"clf_func_name": SVC(gamma=2, C=1), "Y":None},
        #"gaussian_process": {"clf_func_name": GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True), "Y": None},
        }

    list_name = []
    for name,clf in clfs.iteritems():
        list_name.append(name)
    df_scores_results = pd.DataFrame(index = list_name, columns=['ACC', 'AUC'])

    for name,clf in clfs.iteritems():
        print "Cross validating with ", name
        y_predicted = cross_val_predict(clf["clf_func_name"], X, y, cv=k)
        scores = cross_val_score(clf["clf_func_name"], X, y, cv=k)

        scores_mean = "Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)

        name_fig_mat = "Confusion matrix of " + name + " classifier"
        write_confusion_mat(y, y_predicted, _scores=scores, _save_fig = _save_fig, _fout = name_fig_mat)

        ACC = accuracy_score(y, y_predicted)*100
        AUC = roc_auc_score(y-1, y_predicted)
        df_scores_results.loc[name,:] = [ACC, AUC]

    name_fig_table_scores = "Table of scores per classifier"
    write_table_scores(df_scores_results, _save_fig = _save_fig, _fout = name_fig_table_scores)
#}}}
#}}}

#{{{ Regression functions :
def MAPE_score(y_true, y_predicted): #{{{
    n = y_true.size
    error_MAPE = 0
    for i in range(0, n):
        error_MAPE += abs((y_true[i] - y_predicted[i])/y_true[i])

    error_MAPE = 1./n*error_MAPE*100
    return error_MAPE
#}}}

def short_study_df_reg(df_reg_pb): #{{{

    # Short description :
    df_reg_pb.iloc[:,7:8].hist(color=_red_color, alpha=0.8, bins=20, figsize=(8,6))
    if(_save_fig):
        _fout = "images/" + "Yvar_summed_histograms.png"
        plt.savefig(_fout, bbox_inches='tight')

    # Correlation matrice :
    write_correlation_mat(df_reg_pb, _save_fig=_save_fig, _fout="correlation_matrix_reg_prob", _fig_size=(15,15))
#}}}

def df_to_df_reg_problem(dataset): #{{{
    print "Formatting dataset for regression problem ..."
    y_sum = dataset.iloc[:,8:9].as_matrix().ravel() + dataset.iloc[:,9:10].as_matrix().ravel()

    df_reg_pb = dataset.iloc[:,0:8]
    if(_with_X6 is False):
        df_reg_pb = df_reg_pb.drop('X6 Orientation', 1)
    df_reg_pb["y (HL + CL)"] = pd.Series(y_sum, index = df_reg_pb.index)

    short_study_df_reg(df_reg_pb)

    return df_reg_pb
#}}}

def write_graph_cv_predict(y_true, y_predicted, _fout, _fig_size=(10,10)): #{{{
    fig, ax = plt.subplots(figsize=_fig_size)

    ax.scatter(y_true, y_predicted)
    ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--', lw=4)
    ax.set_xlabel('HL + CL Measured')
    ax.set_ylabel('HL + CL Predicted')

    score = r2_score(y_true, y_predicted)
    title_tmp = "R2 score : " + str(score)
    plt.title(title_tmp)

    if(_save_fig):
        _fout = "images/" + _fout
        plt.savefig(_fout, bbox_inches='tight')
#}}}


def compute_regression(df_regression, k = 10): #{{{
    n_registers = df_regression.shape[0]
    if (_with_X6):
        X = df_regression.iloc[:,0:8].as_matrix()
        y = df_regression.iloc[:,8:9].as_matrix().ravel()
    else:
        X = df_regression.iloc[:,0:7].as_matrix()
        y = df_regression.iloc[:,7:8].as_matrix().ravel()

    model_pol3 = Pipeline([('poly', PolynomialFeatures(degree=3)), ('linear', linear_model.LinearRegression(fit_intercept=False))])

    regressors_model = {
        "Linear": {"reg_model": linear_model.LinearRegression(copy_X = True)},
        # "Linear normalized": {"reg_model": linear_model.LinearRegression(normalize = True, copy_X = True)},
        "Linear + SVD regularization alpha=0.001": {"reg_model": linear_model.Ridge(alpha = 0.001, copy_X = True)},
        "Polynomial deg 3": {"reg_model": model_pol3},
        # "k-neighbors k = 5": {"reg_model": KNeighborsRegressor(n_neighbors=5)},
        "Random Forest 10 trees": {"reg_model": RandomForestRegressor(n_estimators=10)},
        }

    list_name = []
    for name,clf in regressors_model.iteritems():
        list_name.append(name)

    df_validation_results = pd.DataFrame(index = list_name, columns=['R2', 'RMS', 'MAPE'])

    for name, reg_mod in regressors_model.iteritems():
        print "Cross validating with ", name
        y_predicted = cross_val_predict(reg_mod["reg_model"], X, y, cv=k)

        # scores_R2 = cross_val_score(reg_mod["reg_model"], X, y, cv=k)
        # mean_score_R2 = np.sum(scores_R2)/scores_R2.size

        score_R2 = r2_score(y, y_predicted)
        score_RMS = mean_squared_error(y, y_predicted)
        score_MAPE = MAPE_score(y, y_predicted)

        df_validation_results.loc[name, 'R2'] = score_R2
        df_validation_results.loc[name, 'RMS'] = score_RMS
        df_validation_results.loc[name, 'MAPE'] = score_MAPE

        graph_name = "graph_cv_predict_normalized_datas" + name + ".png"
        write_graph_cv_predict(y, y_predicted, _fout=graph_name)


    if (_with_X6):
        name_fig_table_validation = "Table of validation metrics per Regressor model with normalized Datas"
    else:
        name_fig_table_validation = "Table of validation metrics per Regressor model with normalized Datas without X6"

    write_table_scores(df_validation_results, _save_fig = _save_fig, _fout = name_fig_table_validation)

#}}}


def write_graph(x, x_label, df_validation_results, _fout, _markersize=200, _fig_size=(15,6)): #{{{

    fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=_fig_size)

    # R2 score
    y = df_validation_results.loc[:,'R2']
    ax1.scatter(x, y, color=_brown_color, s=_markersize)
    ax1.plot(x, y, 'k', color=_brown_color, alpha=.4)
    ax1.set_title("R2 score", fontsize=14, fontweight='bold', color=_brown_color, rotation='horizontal')
    # ax1.set_ylim((0.8,1))

    # RMS score
    y = df_validation_results.loc[:,'RMS']
    ax2.scatter(x, y, color=_brown_color, s=_markersize)
    ax2.plot(x, y, 'k', color=_brown_color, alpha=.4)
    ax2.set_title("RMS score", fontsize=14, fontweight='bold', color=_brown_color, rotation='horizontal')
    # ax2.set_ylim((0,100))

    # MAPE score
    y = df_validation_results.loc[:,'MAPE']
    ax3.scatter(x, y, color=_brown_color, s=_markersize)
    ax3.plot(x, y, 'k', color=_brown_color, alpha=.4)
    ax3.set_title("MAPE score", fontsize=14, fontweight='bold', color=_brown_color, rotation='horizontal')
    # ax3.set_ylim((0,15))

    for ax in fig.axes:
        ax.set_xlabel(x_label)
        # ax.set_xlim(xmin=0)
        # ax.set_xticks(np.arange(min(x), max(x)+1, 1.0))

    if(_save_fig):
        _fout = "images/" + _fout
        plt.savefig(_fout, bbox_inches='tight')
#}}}

def polynomial_model_study(df_regression, _deg_max=11, k=10): #{{{
    n_registers = df_regression.shape[0]
    if (_with_X6):
        X = df_regression.iloc[:,0:8].as_matrix()
        y = df_regression.iloc[:,8:9].as_matrix().ravel()
    else:
        X = df_regression.iloc[:,0:7].as_matrix()
        y = df_regression.iloc[:,7:8].as_matrix().ravel()

    deg_pol = np.arange(1, _deg_max+1, 1)

    list_name = []
    for i in range(0, deg_pol.size):
        model_name = "Polynomial deg " + str(deg_pol[i])
        list_name.append(model_name)

    df_validation_results = pd.DataFrame(index = list_name, columns=['R2', 'RMS', 'MAPE'])

    for i in range(0, deg_pol.size):
        print "Processing deg = ", deg_pol[i]
        model_pol_deg = Pipeline([('poly', PolynomialFeatures(degree=deg_pol[i])), ('linear', linear_model.LinearRegression(fit_intercept=False))])

        # scores_R2 = cross_val_score(model_pol_deg, X, y, cv=k)
        # mean_score_R2 = np.sum(scores_R2)/scores_R2.size
        # df_validation_results.loc[list_name[i], 'MAPE'] = mean_score_MAPE

        y_predicted = cross_val_predict(model_pol_deg, X, y, cv=k)

        score_R2 = r2_score(y, y_predicted)
        score_RMS = mean_squared_error(y, y_predicted)
        score_MAPE = MAPE_score(y, y_predicted)

        df_validation_results.loc[list_name[i], 'R2'] = score_R2
        df_validation_results.loc[list_name[i], 'RMS'] = score_RMS
        df_validation_results.loc[list_name[i], 'MAPE'] = score_MAPE


    name_fig_table_validation = "Table of scores per Polynomial Regressor model"
    write_table_scores(df_validation_results, _save_fig = _save_fig, _fout = name_fig_table_validation)

    name_graph = "Polynomial regressor scores vs deg"
    write_graph(deg_pol, "Polynomial deg", df_validation_results, _fout=name_graph)
#}}}

def randomForest_model_study(df_regression, _n_tree_max=50, k=10): #{{{
    n_registers = df_regression.shape[0]
    if (_with_X6):
        X = df_regression.iloc[:,0:8].as_matrix()
        y = df_regression.iloc[:,8:9].as_matrix().ravel()
    else:
        X = df_regression.iloc[:,0:7].as_matrix()
        y = df_regression.iloc[:,7:8].as_matrix().ravel()

    n_trees = np.arange(2, _n_tree_max+1, 1)

    list_name = []
    for i in range(0, n_trees.size):
        model_name = "Random Forest " + str(n_trees[i]) + " trees"
        list_name.append(model_name)

    df_validation_results = pd.DataFrame(index = list_name, columns=['R2', 'RMS', 'MAPE'])

    for i in range(0, n_trees.size):
        print "Processing Random Forest with ", n_trees[i], " trees"
        model = RandomForestRegressor(n_estimators=n_trees[i])

        y_predicted = cross_val_predict(model, X, y, cv=k)

        score_R2 = r2_score(y, y_predicted)
        score_RMS = mean_squared_error(y, y_predicted)
        score_MAPE = MAPE_score(y, y_predicted)

        df_validation_results.loc[list_name[i], 'R2'] = score_R2
        df_validation_results.loc[list_name[i], 'RMS'] = score_RMS
        df_validation_results.loc[list_name[i], 'MAPE'] = score_MAPE


    name_fig_table_validation = "Table of scores per Random Forest Regressor model - num trees"
    write_table_scores(df_validation_results, _save_fig = _save_fig, _fout = name_fig_table_validation)

    name_graph = "Random Forest regressor scores vs num trees"
    write_graph(n_trees, "number of trees", df_validation_results, _fout=name_graph, _markersize=50)
#}}}

def randomForest_prof_model_study(df_regression, n_trees=10, max_depth=50, k=10): #{{{
    n_registers = df_regression.shape[0]
    if (_with_X6):
        X = df_regression.iloc[:,0:8].as_matrix()
        y = df_regression.iloc[:,8:9].as_matrix().ravel()
    else:
        X = df_regression.iloc[:,0:7].as_matrix()
        y = df_regression.iloc[:,7:8].as_matrix().ravel()

    v_max_depth = np.arange(1, max_depth+1, 1)

    list_name = []
    for i in range(0, v_max_depth.size):
        model_name = "Random Forest max_depth = " + str(v_max_depth[i])
        list_name.append(model_name)

    df_validation_results = pd.DataFrame(index = list_name, columns=['R2', 'RMS', 'MAPE'])

    for i in range(0, v_max_depth.size):
        print "Processing Random Forest with max_depth = ", v_max_depth[i]
        model = RandomForestRegressor(n_estimators=n_trees, max_depth = v_max_depth[i])

        y_predicted = cross_val_predict(model, X, y, cv=k)

        score_R2 = r2_score(y, y_predicted)
        score_RMS = mean_squared_error(y, y_predicted)
        score_MAPE = MAPE_score(y, y_predicted)

        df_validation_results.loc[list_name[i], 'R2'] = score_R2
        df_validation_results.loc[list_name[i], 'RMS'] = score_RMS
        df_validation_results.loc[list_name[i], 'MAPE'] = score_MAPE


    name_fig_table_validation = "Table of scores per Random Forest Regressor model - max_depth"
    write_table_scores(df_validation_results, _save_fig = _save_fig, _fout = name_fig_table_validation)

    name_graph = "Random Forest regressor scores vs max_depth"
    write_graph(v_max_depth, "max_depth", df_validation_results, _fout=name_graph, _markersize=50)
#}}}

def Ridge_model_study(df_regression, _alpha_min=10**-4, k=10): #{{{
    n_registers = df_regression.shape[0]
    if (_with_X6):
        X = df_regression.iloc[:,0:8].as_matrix()
        y = df_regression.iloc[:,8:9].as_matrix().ravel()
    else:
        X = df_regression.iloc[:,0:7].as_matrix()
        y = df_regression.iloc[:,7:8].as_matrix().ravel()

    v_alpha = np.arange(1, 0, -0.01)

    list_name = []
    for i in range(0, v_alpha.size):
        model_name = "Linear + SVD regularization alpha = " + str(v_alpha[i])
        list_name.append(model_name)

    df_validation_results = pd.DataFrame(index = list_name, columns=['R2', 'RMS', 'MAPE'])

    for i in range(0, v_alpha.size):
        print "Processing Ridge (=Linear + SVD) with alpha = ", v_alpha[i]
        model = linear_model.Ridge(alpha=v_alpha[i], copy_X = True)

        y_predicted = cross_val_predict(model, X, y, cv=k)

        score_R2 = r2_score(y, y_predicted)
        score_RMS = mean_squared_error(y, y_predicted)
        score_MAPE = MAPE_score(y, y_predicted)

        df_validation_results.loc[list_name[i], 'R2'] = score_R2
        df_validation_results.loc[list_name[i], 'RMS'] = score_RMS
        df_validation_results.loc[list_name[i], 'MAPE'] = score_MAPE

    name_fig_table_validation = "Table of scores per Ridge Regressor model"
    write_table_scores(df_validation_results, _save_fig = _save_fig, _fout = name_fig_table_validation)

    name_graph = "Ridge regressor scores vs alpha"
    write_graph(v_alpha, "alpha", df_validation_results, _fout=name_graph, _markersize=50)
#}}}

#}}}


def main():

#{{{ Argument Parsing :
    """Main program : energeff"""
    parser = argparse.ArgumentParser(description='Energetic Study of Building according to their shapes')

    #ArgumentParser.add_argument(name or flags...[, action][, nargs][, const][, default][, type][, choices][, required][, help][, metavar][, dest])

    parser.add_argument('dataset', type=str, action='store',
            metavar='<dataset_path>', help='dataset path')

    parser.add_argument('--descr', action='store_true', default=False, dest='descr_datas',
            help='whether to process data analysis or not')

    parser.add_argument('--class', action='store_true', default=False, dest='classification',
            help='whether to run classification algorithm or not')

    parser.add_argument('--reg', action='store_true', default=False, dest='regression',
            help='whether to run regression algorithm or not')

    parser.add_argument('--save_fig', action='store_true', default=False, dest='save_fig',
            help='whether to save figures generated by --descr in png')
#}}}
    args = parser.parse_args()  # de type <class 'argparse.Namespace'>

    global _save_fig
    _save_fig = args.save_fig

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
        # df_descr_statistics(dataset, savefig=args.save_fig)
        df_descr_matrices(dataset, savefig=args.save_fig)

    if (args.classification is True):
        print 'Classification study : '
        df_class_pb = df_reg_to_clf_problem(dataset, _save_fig = args.save_fig)
        classify(df_class_pb, args.save_fig)

    if (args.regression is True):
        print 'Regression study : '
        df_reg_pb = df_to_df_reg_problem(dataset)
        df_reg_pb = normalize_Z_score(df_reg_pb)
        compute_regression(df_reg_pb, k=10)

        # print 'Additional studies :\n'

        # print 'Polynomial ...'
        # polynomial_model_study(df_reg_pb, k=10)

        # print 'Random Forest ...'
        # randomForest_model_study(df_reg_pb, k=10)
        # randomForest_prof_model_study(df_reg_pb, k=10)

        # print 'Ridge ...'
        # Ridge_model_study(df_reg_pb, k=10)


if __name__ == '__main__':
    main()
