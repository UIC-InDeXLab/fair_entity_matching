from itertools import combinations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

import measures


class Workload:
    # encoding is a list of vectors. len(encoding) = # of entities pairs
    # each vector is an encoding for that particular entity pair
    def __init__(self, df, sens_att_left, sens_att_right, prediction, label_column="label",
                 multiple_sens_attr=False, delimiter=",", single_fairness=True, k_combinations=1):
        self.df = df
        self.label_column = label_column
        self.prediction = prediction
        self.sens_att_left = sens_att_left
        self.sens_att_right = sens_att_right
        self.multiple_sens_attr = multiple_sens_attr
        self.delimiter = delimiter
        self.single_fairness = single_fairness
        self.sens_attr_vals = self.find_all_sens_attr()
        self.sens_att_to_index = self.create_sens_att_to_index()
        self.name_to_encode = {}
        self.encoding = self.encode()

        self.TP = 0
        self.FP = 1
        self.TN = 2
        self.FN = 3

        self.workload_conf_matrix = self.calculate_workload_conf_matrix()
        self.entitites_to_count = self.create_entities_to_count()
        self.k_combs = self.create_k_combs(k_combinations)
        self.k_combs_to_attr_names = self.k_combs_to_attribute_names()

    def find_all_sens_attr(self):
        sens_att = set()
        for it_index, row in self.df.iterrows():
            left = row[self.sens_att_left]
            right = row[self.sens_att_right]
            if self.multiple_sens_attr:
                for item in left.split(self.delimiter):
                    sens_att.add(item.strip())
                for item in right.split(self.delimiter):
                    sens_att.add(item.strip())
            else:
                sens_att.add(left.strip())
                sens_att.add(right.strip())

        sens_attr_vals = list(sens_att)
        sens_attr_vals.sort()
        return sens_attr_vals

    def encode(self):
        encoding = []
        for it_index, row in self.df.iterrows():
            left_att = row[self.sens_att_left]
            right_att = row[self.sens_att_right]
            left_vector = np.zeros(len(self.sens_attr_vals))
            right_vector = np.zeros(len(self.sens_attr_vals))
            if self.multiple_sens_attr:
                for item in left_att.split(self.delimiter):
                    curr_item = item.strip()
                    idx = self.sens_attr_vals.index(curr_item)
                    left_vector[idx] = 1
                for item in right_att.split(self.delimiter):
                    curr_item = item.strip()
                    idx = self.sens_attr_vals.index(curr_item)
                    right_vector[idx] = 1
            else:
                curr_item = left_att.strip()
                idx = self.sens_attr_vals.index(curr_item)
                left_vector[idx] = 1
                curr_item = right_att.strip()
                idx = self.sens_attr_vals.index(curr_item)
                right_vector[idx] = 1
            encoding.append(np.concatenate([left_vector, right_vector]))
        return encoding

    def create_sens_att_to_index(self):
        sens_att_to_index = {}
        for i in range(len(self.sens_attr_vals)):
            att = self.sens_attr_vals[i]
            sens_att_to_index[att] = i
        return sens_att_to_index

    def create_hashmap_key(self, row):
        left_att = row[self.sens_att_left]
        right_att = row[self.sens_att_right]

        key_left = []
        key_right = []

        if self.multiple_sens_attr:
            for item in left_att.split(self.delimiter):
                curr_item = item.strip()
                key_left.append(self.sens_att_to_index[curr_item])
            for item in right_att.split(self.delimiter):
                curr_item = item.strip()
                key_right.append(self.sens_att_to_index[curr_item])
            key_left.sort()
            key_right.sort()
        else:
            curr_item = left_att.strip()
            key_left.append(self.sens_att_to_index[curr_item])
            curr_item = right_att.strip()
            key_right.append(self.sens_att_to_index[curr_item])

        # -1 added as a delimiter between the left and right keys
        res = key_left + [-1] + key_right if key_left[0] <= key_right[0] else key_right + [-1] + key_left
        return tuple(res)

    def create_entities_to_count(self):
        entitites_to_count = {}
        for ind, row in self.df.iterrows():
            key = self.create_hashmap_key(row)
            entitites_to_count[key] = [0, 0, 0, 0]
        for ind, row in self.df.iterrows():
            key = self.create_hashmap_key(row)
            self.fill_entities_to_count(entitites_to_count, key, ind, self.prediction[ind], row[self.label_column])
        return entitites_to_count

    def fill_entities_to_count(self, entitites_to_count, key, ind, pred, ground_truth):
        if pred[0]:
            if ground_truth:
                entitites_to_count[key][self.TP] += 1
            else:
                entitites_to_count[key][self.FP] += 1
        else:
            if ground_truth:
                entitites_to_count[key][self.FN] += 1
            else:
                entitites_to_count[key][self.TN] += 1

    def find_border_in_key(self, key):
        return key.index(-1)

    def create_k_combs(self, k):
        k_combs = {}
        if self.single_fairness:
            for entity in self.entitites_to_count:
                number_of_pairs_with_entity = sum(self.entitites_to_count[entity]) + 1  # because of -1 for the border
                border = self.find_border_in_key(entity)
                for comb in combinations(entity[:border], k):
                    if comb not in k_combs:
                        k_combs[comb] = number_of_pairs_with_entity
                    else:
                        k_combs[comb] = k_combs[comb] + number_of_pairs_with_entity
                for comb in combinations(entity[border + 1:], k):
                    if comb not in k_combs:
                        k_combs[comb] = number_of_pairs_with_entity
                    else:
                        k_combs[comb] = k_combs[comb] + number_of_pairs_with_entity

        else:
            for entity in self.entitites_to_count:
                number_of_pairs_with_entity = sum(self.entitites_to_count[entity]) + 1  # because of -1 for the border
                border = self.find_border_in_key(entity)
                for comb1 in combinations(entity[:border], k):
                    for comb2 in combinations(entity[border + 1:], k):
                        k_comb = comb1 + comb2
                        if k_comb not in k_combs:
                            k_combs[k_comb] = number_of_pairs_with_entity
                        else:
                            k_combs[k_comb] = k_combs[k_comb] + number_of_pairs_with_entity
        return k_combs

    def k_combs_to_attribute_names(self):
        comb_to_attribute_names = {}
        if self.single_fairness:
            for comb in self.k_combs:
                names = []
                for elem in comb:
                    names.append(elem)
                names.sort()  # sort by index
                name = ""
                for elem in names:
                    name += self.sens_attr_vals[elem] + "~"
                name = name[:-1]  # delete the last ~
                comb_to_attribute_names[comb] = name
        else:
            for comb in self.k_combs:
                comb_list = list(comb)
                names_left = []
                names_right = []
                for i in range(int(len(comb_list) / 2)):
                    names_left.append(comb_list[i])
                    names_right.append(comb_list[i + int(len(comb_list) / 2)])
                name_left = ""
                name_right = ""
                for elem in names_left:
                    name_left += self.sens_attr_vals[elem] + "~"
                name_left = name_left[:-1]
                for elem in names_right:
                    name_right += self.sens_attr_vals[elem] + "~"
                name_right = name_right[:-1]

                comb_to_attribute_names[comb] = name_left + "|" + name_right
        return comb_to_attribute_names

    def create_subgroup_encoding_from_subgroup_single(self, subgroup):
        subgroup_encoding = [0] * len(self.sens_attr_vals)
        for group in subgroup:
            subgroup_encoding[group] = 1
        return subgroup_encoding

    def create_subgroup_encodings_from_subgroup_pairwise(self, subgroup):
        subgroup_encoding = [0] * 2 * len(self.sens_attr_vals)
        subgroup = list(subgroup)
        border = int(len(subgroup) / 2)

        left = subgroup[:border]
        right = subgroup[border:]
        left_encoding = [0] * len(self.sens_attr_vals)
        right_encoding = [0] * len(self.sens_attr_vals)
        for k in left:
            left_encoding[k] = 1
        for k in right:
            right_encoding[k] = 1

        return left_encoding + right_encoding, right_encoding + left_encoding

    def calculate_fairness_single(self, subgroup, measure):
        if measure == "accuracy_parity":
            return measures.accuracy_parity_single(self, subgroup)
        elif measure == "statistical_parity":
            return measures.statistical_parity_single(self, subgroup)
        elif measure == "true_positive_rate_parity":
            return measures.true_positive_rate_parity_single(self, subgroup)
        elif measure == "false_positive_rate_parity":
            return measures.false_positive_rate_parity_single(self, subgroup)
        elif measure == "negative_predictive_value_parity":
            return measures.negative_predictive_value_parity_single(self, subgroup)
        elif measure == "positive_predictive_value_parity":
            return measures.positive_predictive_value_parity_single(self, subgroup)

    def calculate_fairness_pairwise(self, subgroup, measure):
        if measure == "accuracy_parity":
            return measures.accuracy_parity_pairwise(self, subgroup)
        elif measure == "statistical_parity":
            return measures.statistical_parity_pairwise(self, subgroup)
        elif measure == "true_positive_rate_parity":
            return measures.true_positive_rate_parity_pairwise(self, subgroup)
        elif measure == "false_positive_rate_parity":
            return measures.false_positive_rate_parity_pairwise(self, subgroup)
        elif measure == "negative_predictive_value_parity":
            return measures.negative_predictive_value_parity_pairwise(self, subgroup)
        elif measure == "positive_predictive_value_parity":
            return measures.positive_predictive_value_parity_pairwise(self, subgroup)

    def calculate_workload_conf_matrix(self):
        conf_matr = [0] * 4

        print('accuracy', accuracy_score(self.df.label, [i for [i] in self.prediction]))
        print('f1', f1_score(self.df.label, [i for [i] in self.prediction]))
        print('precision', precision_score(self.df.label, [i for [i] in self.prediction]), ", ", 'recall',
              recall_score(self.df.label, [i for [i] in self.prediction]))

        for ind, row in self.df.iterrows():
            pred = self.prediction[ind][0]
            ground_truth = row[self.label_column]
            if pred:
                if ground_truth:
                    conf_matr[self.TP] += 1
                else:
                    conf_matr[self.FP] += 1
            else:
                if ground_truth:
                    conf_matr[self.FN] += 1
                else:
                    conf_matr[self.TN] += 1
        print("workload TP FP TN FN", tuple(conf_matr))
        print()
        return tuple(conf_matr)

    def calculate_workload_fairness(self, measure):
        TP, FP, TN, FN = self.workload_conf_matrix
        if measure == "accuracy_parity":
            return measures.AP(TP, FP, TN, FN)
        elif measure == "statistical_parity":
            return measures.SP(TP, FP, TN, FN)
        elif measure == "true_positive_rate_parity":
            return measures.TPR(TP, FP, TN, FN)
        elif measure == "false_positive_rate_parity":
            return measures.FPR(TP, FP, TN, FN)
        elif measure == "negative_predictive_value_parity":
            return measures.NPV(TP, FP, TN, FN)
        elif measure == "positive_predictive_value_parity":
            return measures.PPV(TP, FP, TN, FN)

    def fairness(self, subgroups, measure, aggregate="distribution"):
        counts = []
        if self.single_fairness:
            values = [self.calculate_fairness_single(subgroup, measure) for subgroup in subgroups]
            for subgroup in subgroups:
                print("subgroup", self.sens_attr_vals[subgroup[0]])
                print("TP, FP, TN, FN", measures.get_confusion_matrix_single(self, subgroup))
                print("No. of tuples", str(measures.get_confusion_matrix_single(self, subgroup)[0] +
                                           measures.get_confusion_matrix_single(self, subgroup)[1] +
                                           measures.get_confusion_matrix_single(self, subgroup)[2] +
                                           measures.get_confusion_matrix_single(self, subgroup)[3]))
                counts.append(measures.get_confusion_matrix_single(self, subgroup)[0] +
                              measures.get_confusion_matrix_single(self, subgroup)[1] +
                              measures.get_confusion_matrix_single(self, subgroup)[2] +
                              measures.get_confusion_matrix_single(self, subgroup)[3])
                print()
        else:
            values = [self.calculate_fairness_pairwise(subgroup, measure) for subgroup in subgroups]
            for subgroup in subgroups:
                print("subgroup", self.sens_attr_vals[subgroup[0]] + "|" + self.sens_attr_vals[subgroup[1]])
                print("TP, FP, TN, FN", measures.get_confusion_matrix_pairwise(self, subgroup))
                print("No. of tuples", str(measures.get_confusion_matrix_pairwise(self, subgroup)[0] +
                                           measures.get_confusion_matrix_pairwise(self, subgroup)[1] +
                                           measures.get_confusion_matrix_pairwise(self, subgroup)[2] +
                                           measures.get_confusion_matrix_pairwise(self, subgroup)[3]))
                counts.append(measures.get_confusion_matrix_pairwise(self, subgroup)[0] +
                              measures.get_confusion_matrix_pairwise(self, subgroup)[1] +
                              measures.get_confusion_matrix_pairwise(self, subgroup)[2] +
                              measures.get_confusion_matrix_pairwise(self, subgroup)[3])
                print()

        # make the measure a parity by subtracting the model performance
        print(measure," Values:", values)
        workload_fairness = self.calculate_workload_fairness(measure)
        values = [x - workload_fairness for x in values]
        # values = [values[0]-values[1], values[1]-values[0]]
        print("Disparity Values:", values)
        print("---------------------")

        if aggregate == "max":
            return max(values)
        elif aggregate == "min":
            return min(values)
        elif aggregate == "max_minus_min":
            return max(values) - min(values)
        elif aggregate == "average":
            return np.mean(values)
        return values, counts
