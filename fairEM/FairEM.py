from statsmodels.stats.weightstats import ztest

class FairEM:
    # the input is a list of objects of class Workload
    # alpha is used for the Z-Test 
    def __init__(self, model, workloads, alpha, full_workload_test, threshold, single_fairness=True):
        self.model = model
        self.workloads = workloads
        self.alpha = alpha
        self.threshold = threshold
        self.single_fairness = single_fairness
        self.distances_unfaired = {}
        self.distances_all = {}

        self.TP = 0
        self.FP = 1
        self.TN = 2
        self.FN = 3

    # creates a two dimensional matrix, subgroups x workload fairness value
    # used only for distribution
    def separate_distributions_from_workloads(self, subgroups, workloads_fairness):
        num_of_subgroups = len(subgroups)
        subgroup_precisions = []
        for i in range(num_of_subgroups):
            subgroup_precisions.append([])
        for i in range(num_of_subgroups):
            for workload_distr in workloads_fairness:
                subgroup_precisions[i].append(workload_distr[i])
        return subgroup_precisions

    # true would mean something is good, i.e. is fair
    # so for accuracy if x0 - avg(x) > -threshold, this is good
    # if we want a measure to be as low as possible, 
    # then x0 - avg(x) < threshold
    def is_fair_measure_specific(self, measure, workload_fairness):
        if measure == "accuracy_parity" or \
                measure == "statistical_parity" or \
                measure == "true_positive_rate_parity" or \
                measure == "true_negative_rate_parity" or \
                measure == "positive_predictive_value_parity" or \
                measure == "negative_predictive_value_parity":
            return workload_fairness >= -self.threshold
        if measure == "false_positive_rate_parity" or \
                measure == "false_negative_rate_parity" or \
                measure == "false_discovery_rate_parity" or \
                measure == "false_omission_rate_parity":
            return workload_fairness <= self.threshold

    def is_fair(self, measure, aggregate, real_distr=False):
        if len(self.workloads) == 1:
            workload_fairness, counts = self.workloads[0].fairness(self.workloads[0].k_combs, measure, aggregate)
            if aggregate is not "distribution":
                return self.is_fair_measure_specific(measure, workload_fairness)
            else:
                if real_distr:
                    return workload_fairness
                else:
                    return [self.is_fair_measure_specific(measure, subgroup_fairness) \
                            for subgroup_fairness in workload_fairness], counts
        else:
            workloads_fairness = []
            for workload in self.workloads:
                workloads_fairness.append(workload.fairness(workload.k_combs, measure, aggregate))

            # general idea of how the entity matching is performed
            if aggregate is not "distribution":
                p_value = ztest(workloads_fairness, value=self.threshold)[1]
                return p_value <= self.alpha
            # specific for each measure
            else:
                subgroup_to_list_of_fairneses = {}
                for i in range(len(self.workloads)):
                    workload = self.workloads[i]
                    fairnesses = workloads_fairness[i]

                    k_combs_list = [x for x in workload.k_combs_to_attr_names]

                    for j in range(len(fairnesses)):
                        subgroup_index = k_combs_list[j]
                        subgroup_name = workload.k_combs_to_attr_names[subgroup_index]
                        if subgroup_name not in subgroup_to_list_of_fairneses:
                            subgroup_to_list_of_fairneses[subgroup_name] = []
                        subgroup_to_list_of_fairneses[subgroup_name].append(fairnesses[j])

                subroups_is_fair = {}
                for subgroup in subgroup_to_list_of_fairneses:
                    if len(subgroup_to_list_of_fairneses[subgroup]) >= 30:  # limit for a valid z-test
                        p_value = ztest(subgroup_to_list_of_fairneses[subgroup], value=self.threshold)[1]
                        subroups_is_fair[subgroup] = (p_value <= self.alpha)

                return subroups_is_fair
