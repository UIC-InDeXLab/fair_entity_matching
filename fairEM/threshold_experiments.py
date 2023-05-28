import math
import os

import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tabulate import tabulate
import warnings
warnings.filterwarnings("ignore")

plt.rcParams["font.family"] = "serif"


def heatmap(data, row_labels, col_labels, ax=None, cbar_kw={}, cbarlabel="", **kwargs):
    if not ax:
        ax = plt.gca()
    im = ax.imshow(data, **kwargs)
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw, shrink=0.75, ticks=np.linspace(0,8,9))
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center", rotation_mode="anchor")
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - .5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)
    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.3f}", **textkw):
    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


# thresholds = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
# path = os.listdir("../data/probs/cameras")
# model = ""
# for threshold in thresholds:
#     for file in path:
#         df = pd.read_csv("../data/probs/cameras/" + file)
#         predictions = [True if df.iloc[idx]["match_score"] > threshold else False for idx in
#                        range(len(df))]
#         if file == "preds_deepmatcher.csv":
#             model = "DeepMatcher"
#         elif file == "preds_ditto.csv":
#             model = "Ditto"
#         elif file == "preds_dt.csv":
#             model = "DecisionTree"
#         elif file == "preds_gnem.csv":
#             model = "GNEM"
#         elif file == "preds_hiermatcher.csv":
#             model = "HierMatcher"
#         elif file == "preds_mcan.csv":
#             model = "MCAN"
#         elif file == "preds_lg.csv":
#             model = "LogisticRegression"
#         elif file == "preds_ln.csv":
#             model = "LinearRegression"
#         elif file == "preds_rf.csv":
#             model = "RandomForest"
#         elif file == "preds_nb.csv":
#             model = "NaiveBayes"
#         elif file == "preds_svm.csv":
#             model = "SVM"
#
#         with open("../data/FairEM/" + model + "/cameras/preds_" + str(threshold) + ".csv", "w") as f:
#             f.write("preds\n")
#             for prediction in predictions:
#                 f.write("1\n" if prediction else "0\n")

datasets = ['iTunes-Amazon', 'Cameras', 'DBLP-ACM', 'DBLP-Scholar']
res_files = ['threhsold_data/res_itunes_amazon', 'threhsold_data/res_cameras', 'threhsold_data/res_dblp_acm', 'threhsold_data/res_dblp_scholar']
l2_tprp = []
l2_ppvp = []
for idx in range(len(datasets)):
    arrs = []
    with open(res_files[idx]) as f:
        arr = []
        for line in f:
            if '-------------------------------------------------' in line:
                arrs.append(arr)
                arr = []
                continue
            else:
                arr.append(line.split("\n")[0])

    df1 = pd.DataFrame(arrs, columns=['Model', 'Threshold', 'F1', 'TPR','PPV']).sort_values(by=['Model', 'Threshold']).reset_index()
    df1['Threshold'] = df1['Threshold'].astype(float)
    df1['F1'] = df1['F1'].astype(float)
    df1['TPR'] = df1['TPR'].astype(float)
    df1['PPV'] = df1['PPV'].astype(float)


    thresholds = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
    path = "../experiments/" + datasets[idx] + "/results/preds_"

    tprp_arr = []
    ppvp_arr = []
    for t in thresholds:
        for file in os.listdir(path + str(t)):
            model = file.split("_")[0]
            df = pd.read_csv(path + str(t) + "/" + file)
            tprp = \
                df.loc[
                    (df["measure"] == "true_positive_rate_parity") & (df["is_fair"] == False) & (
                            df["counts"] >= 10)].shape[
                    0]
            ppvp = df.loc[(df["measure"] == "positive_predictive_value_parity") & (df["is_fair"] == False) & (
                    df["counts"] >= 10)].shape[0]
            tprp_arr.append([model, t, tprp])
            ppvp_arr.append([model, t, ppvp])
    df = pd.DataFrame(tprp_arr, columns=['Model', 'Threshold', 'Count']).sort_values(
        by=['Model', 'Threshold']).reset_index()

    df = df.merge(df1, on=['Model', 'Threshold'])
    x_axis = np.unique(df.Threshold.values)
    y_axis = np.unique(df.Model.values)
    colors = df.Count.values
    tpr = df.TPR.values
    colors = np.reshape(colors, (len(y_axis), len(x_axis)))

    squared_sum = []
    for row in colors:
        sum = 0
        for i in range(len(row) - 1):
            sum += (row[i] - row[i + 1]) ** 2
        squared_sum.append(math.sqrt(sum))
    squared_sum.insert(0,datasets[idx])
    l2_tprp.append(squared_sum)


    tpr = np.reshape(tpr, (len(y_axis), len(x_axis)))
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 8))
    im, cbar = heatmap(colors, y_axis, x_axis, ax=ax, cmap="RdYlGn_r",
                       cbarlabel="Number of discriminated subgroups")
    y_tick_labels = ['DTMatcher', 'DeepMatcher', 'Ditto', 'GNEM', 'HierMatcher', 'LinRegMatcher', 'LogRegMatcher',
                     'MCAN', 'NBMatcher', 'RFMatcher', 'SVMMatcher']
    ax.set_yticklabels(y_tick_labels)
    text = annotate_heatmap(im, data=tpr, valfmt="{x:.2f}")
    plt.tight_layout()
    plt.savefig('TPRP_' + datasets[idx] + '.png')

    df = pd.DataFrame(ppvp_arr, columns=['Model', 'Threshold', 'Count']).sort_values(
        by=['Model', 'Threshold']).reset_index()
    df = df.merge(df1, on=['Model', 'Threshold'])
    y_axis = np.unique(df.Model.values)
    x_axis = np.unique(df.Threshold.values)
    colors = df.Count.values
    ppv = df.PPV.values
    colors = np.reshape(colors, (len(y_axis), len(x_axis)))

    squared_sum = []
    for row in colors:
        sum = 0
        for i in range(len(row) - 1):
            sum += (row[i] - row[i + 1]) ** 2
        squared_sum.append(math.sqrt(sum))
    squared_sum.insert(0,datasets[idx])
    l2_ppvp.append(squared_sum)

    ppv = np.reshape(ppv, (len(y_axis), len(x_axis)))
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 8))
    im, cbar = heatmap(colors, y_axis, x_axis, ax=ax, cmap="RdYlGn_r",
                       cbarlabel="Number of discriminated subgroups")
    y_tick_labels=['DTMatcher','DeepMatcher','Ditto', 'GNEM','HierMatcher','LinRegMatcher','LogRegMatcher','MCAN','NBMatcher','RFMatcher','SVMMatcher']
    ax.set_yticklabels(y_tick_labels)
    text = annotate_heatmap(im, data=ppv, valfmt="{x:.2f}")
    plt.tight_layout()
    plt.savefig('PPVP_' + datasets[idx] + '.png')

print("TPRP:")
print(tabulate(l2_tprp, headers=["-", "DecisionTree", "DeepMatcher", "Ditto", "GNEM", "HierMatcher", "LinearRegression",
                                 "LogisticRegression", "MCAN", "NaiveBayes", "RandomForest", "SVM"],tablefmt="fancy_grid"))
print()
print("PPVP:")
print(tabulate(l2_ppvp, headers=["-", "DecisionTree", "DeepMatcher", "Ditto", "GNEM", "HierMatcher", "LinearRegression",
                                 "LogisticRegression", "MCAN", "NaiveBayes", "RandomForest", "SVM"],tablefmt="fancy_grid"))
