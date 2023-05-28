import os

import pandas as pd

import FairEM as fem
import workloads as wl
from pathlib import Path



def save_pandas_csv_if_not_exists(dataframe, outname, outdir):
    Path(outdir).mkdir(parents=True, exist_ok=True)
    fullname = os.path.join(outdir, outname)
    dataframe.to_csv(fullname, index=False)


def run_one_workload(model, dataset, left_sens_attribute, right_sens_attribute, test_file,
                     single_fairness=True,
                     k_combinations=1, delimiter=','):
    predictions = pd.read_csv("data/" + model + "/" + dataset + "/preds.csv").values.tolist()
    test_file = "data/DeepMatcher/" + dataset + "/" + test_file

    workload = wl.Workload(pd.read_csv(test_file),
                           left_sens_attribute, right_sens_attribute,
                           predictions, label_column="label",
                           multiple_sens_attr=True, delimiter=delimiter,
                           single_fairness=single_fairness, k_combinations=k_combinations)
    return [workload]


def single_fairness(model, dataset, left_sens_attribute, right_sens_attribute, threshold,
                    single_fairness=True,
                    test_file="test.csv"):
    workloads = run_one_workload(model, dataset, left_sens_attribute, right_sens_attribute, test_file,
                                 single_fairness=single_fairness)

    fairEM = fem.FairEM(model, workloads, alpha=0.05, full_workload_test=test_file, threshold=threshold,
                        single_fairness=single_fairness)

    binary_fairness = []
    measures = ["accuracy_parity", "statistical_parity",
                "true_positive_rate_parity", "false_positive_rate_parity",
                "negative_predictive_value_parity", "positive_predictive_value_parity"]

    attribute_names = []
    for k_comb in workloads[0].k_combs_to_attr_names:
        curr_attr_name = workloads[0].k_combs_to_attr_names[k_comb]
        attribute_names.append(curr_attr_name)
    df = pd.DataFrame(columns=["measure", "sens_attr", "is_fair"])

    aggregate = "distribution"
    for measure in measures:
        print("measure", measure)
        temp_df = pd.DataFrame(columns=["measure", "sens_attr", "is_fair"])
        is_fair, counts = fairEM.is_fair(measure, aggregate)
        binary_fairness.append(is_fair)
        temp_df["measure"] = [measure] * len(is_fair)
        temp_df["sens_attr"] = attribute_names
        temp_df["is_fair"] = is_fair
        temp_df["counts"] = counts
        df = df.append(temp_df, ignore_index=True)

    save_pandas_csv_if_not_exists(dataframe=df, outname=model + "_results_single_fairness.csv",
                                  outdir="experiments/" + dataset + "/")


# Pairwise Fairness
def pairwise_fairness(model, dataset, left_sens_attribute, right_sens_attribute, test_file, threshold,
                      single_fairness=False):
    workloads = run_one_workload(model, dataset, left_sens_attribute, right_sens_attribute, test_file=test_file,
                                 single_fairness=single_fairness)
    fairEM = fem.FairEM(model, workloads, alpha=0.05, full_workload_test=test_file, threshold=threshold,
                        single_fairness=single_fairness)

    binary_fairness = []
    measures = ["accuracy_parity", "statistical_parity",
                "true_positive_rate_parity", "false_positive_rate_parity",
                "negative_predictive_value_parity", "positive_predictive_value_parity"]

    attribute_names = []
    for k_comb in workloads[0].k_combs_to_attr_names:
        curr_attr_name = workloads[0].k_combs_to_attr_names[k_comb]
        attribute_names.append(curr_attr_name)
        df = pd.DataFrame(columns=["measure", "sens_attr", "is_fair"])

    aggregate = "distribution"
    for measure in measures:
        temp_df = pd.DataFrame(columns=["measure", "sens_attr", "is_fair"])
        is_fair, counts = fairEM.is_fair(measure, aggregate)
        binary_fairness.append(is_fair)
        temp_df["measure"] = [measure] * len(is_fair)
        temp_df["sens_attr"] = attribute_names
        temp_df["is_fair"] = is_fair
        temp_df["counts"] = counts
        df = df.append(temp_df, ignore_index=True)

    save_pandas_csv_if_not_exists(dataframe=df, outname=model + "_results_pairwise_fairness.csv",
                                  outdir="/experiments/" + dataset + "/")


def dataset_experiments(dataset, sens_att, test_file, threshold=0.1):  # Threshold specifies the fairness threshold
    models = ["DeepMatcher", "Ditto", "GNEM", "HierMatcher", "MCAN", "SVM", "RuleBasedMatcher", "RandomForest",
              "NaiveBayes", "LogisticRegression", "LinearRegression", "DecisionTree"]
    for mod in models:
        single_fairness(model=mod, dataset=dataset,
                        left_sens_attribute="left_" + sens_att,
                        right_sens_attribute="right_" + sens_att,
                        single_fairness=True,
                        threshold=threshold,
                        test_file=test_file)

        pairwise_fairness(model=mod, dataset=dataset,
                          left_sens_attribute="left_" + sens_att,
                          right_sens_attribute="right_" + sens_att,
                          single_fairness=False,
                          threshold=threshold, test_file=test_file)


dataset_experiments("Compas", "Ethnic_Code_Text", test_file="test.csv")
