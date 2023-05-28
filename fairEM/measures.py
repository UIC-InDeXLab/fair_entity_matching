from utils import clauses_satisfied


def get_confusion_matrix_single(workload, subgroup):
    subgroup_encoding = workload.create_subgroup_encoding_from_subgroup_single(subgroup)
    match_TP = match_FP = match_TN = match_FN = 0
    for entity_to_count in workload.entitites_to_count:
        left_entity = list(entity_to_count)[:workload.find_border_in_key(entity_to_count)]
        right_entity = list(entity_to_count)[workload.find_border_in_key(entity_to_count) + 1:]
        left_entity_encoding = workload.create_subgroup_encoding_from_subgroup_single(left_entity)
        right_entity_encoding = workload.create_subgroup_encoding_from_subgroup_single(right_entity)
        if clauses_satisfied(subgroup_encoding, left_entity_encoding) or clauses_satisfied(subgroup_encoding,
                                                                                           right_entity_encoding):
            match_TP += workload.entitites_to_count[entity_to_count][workload.TP]
            match_TN += workload.entitites_to_count[entity_to_count][workload.TN]
            match_FP += workload.entitites_to_count[entity_to_count][workload.FP]
            match_FN += workload.entitites_to_count[entity_to_count][workload.FN]

    return match_TP, match_FP, match_TN, match_FN


def get_confusion_matrix_pairwise(workload, subgroup):
    encoding1, encoding2 = workload.create_subgroup_encodings_from_subgroup_pairwise(subgroup)
    match_TP = match_FP = match_TN = match_FN = 0
    for entity_to_count in workload.entitites_to_count:
        left_entity = list(entity_to_count)[:workload.find_border_in_key(entity_to_count)]
        right_entity = list(entity_to_count)[workload.find_border_in_key(entity_to_count) + 1:]
        left_entity_encoding = workload.create_subgroup_encoding_from_subgroup_single(left_entity)
        right_entity_encoding = workload.create_subgroup_encoding_from_subgroup_single(right_entity)
        entity_encoding = left_entity_encoding + right_entity_encoding
        if clauses_satisfied(encoding1, entity_encoding) or clauses_satisfied(encoding2, entity_encoding):
            match_TP += workload.entitites_to_count[entity_to_count][workload.TP]
            match_TN += workload.entitites_to_count[entity_to_count][workload.TN]
            match_FP += workload.entitites_to_count[entity_to_count][workload.FP]
            match_FN += workload.entitites_to_count[entity_to_count][workload.FN]
    return match_TP, match_FP, match_TN, match_FN


def AP(TP, FP, TN, FN):
    if (TP + TN + FP + FN) == 0:  # denominator
        return 1
    else:
        return (TP + TN) / (TP + TN + FP + FN)


def SP(TP, FP, TN, FN):
    if (TP + FP + TN + FN) == 0:  # denominator
        return 1
    else:
        return TP / (TP + FP + TN + FN)


def TPR(TP, FP, TN, FN):
    if (TP + FN) == 0:  # denominator
        return 1
    else:
        return TP / (TP + FN)


def FPR(TP, FP, TN, FN):
    if (FP + TN) == 0:  # denominator
        return 1
    else:
        return FP / (FP + TN)


def FNR(TP, FP, TN, FN):
    if (FN + TP) == 0:  # denominator
        return 1
    else:
        return FN / (FN + TP)


def TNR(TP, FP, TN, FN):
    if (TN + FP) == 0:  # denominator
        return 1
    else:
        return TN / (TN + FP)


def PPV(TP, FP, TN, FN):
    if (TP + FP) == 0:  # denominator
        return 1
    else:
        return TP / (TP + FP)


def NPV(TP, FP, TN, FN):
    if (TN + FN) == 0:  # denominator
        return 1
    else:
        return TN / (TN + FN)


def FDR(TP, FP, TN, FN):
    if (TP + FP) == 0:  # denominator
        return 1
    else:
        return FP / (TP + FP)


def FOR(TP, FP, TN, FN):
    if (TN + FN) == 0:  # denominator
        return 1
    else:
        return FN / (TN + FN)


def accuracy_parity_single(workload, subgroup):
    (match_TP, match_FP, match_TN, match_FN) = get_confusion_matrix_single(workload, subgroup)
    return AP(match_TP, match_FP, match_TN, match_FN)


def accuracy_parity_pairwise(workload, subgroup):
    (match_TP, match_FP, match_TN, match_FN) = get_confusion_matrix_pairwise(workload, subgroup)
    return AP(match_TP, match_FP, match_TN, match_FN)


def statistical_parity_single(workload, subgroup):
    (match_TP, match_FP, match_TN, match_FN) = get_confusion_matrix_single(workload, subgroup)
    return SP(match_TP, match_FP, match_TN, match_FN)


def statistical_parity_pairwise(workload, subgroup):
    (match_TP, match_FP, match_TN, match_FN) = get_confusion_matrix_pairwise(workload, subgroup)
    return SP(match_TP, match_FP, match_TN, match_FN)


def true_positive_rate_parity_single(workload, subgroup):
    (match_TP, match_FP, match_TN, match_FN) = get_confusion_matrix_single(workload, subgroup)
    return TPR(match_TP, match_FP, match_TN, match_FN)


def true_positive_rate_parity_pairwise(workload, subgroup):
    (match_TP, match_FP, match_TN, match_FN) = get_confusion_matrix_pairwise(workload, subgroup)
    return TPR(match_TP, match_FP, match_TN, match_FN)


def false_positive_rate_parity_single(workload, subgroup):
    (match_TP, match_FP, match_TN, match_FN) = get_confusion_matrix_single(workload, subgroup)
    return FPR(match_TP, match_FP, match_TN, match_FN)


def false_positive_rate_parity_pairwise(workload, subgroup):
    (match_TP, match_FP, match_TN, match_FN) = get_confusion_matrix_pairwise(workload, subgroup)
    return FPR(match_TP, match_FP, match_TN, match_FN)


def false_negative_rate_parity_single(workload, subgroup):
    (match_TP, match_FP, match_TN, match_FN) = get_confusion_matrix_single(workload, subgroup)
    return FNR(match_TP, match_FP, match_TN, match_FN)


def false_negative_rate_parity_pairwise(workload, subgroup):
    (match_TP, match_FP, match_TN, match_FN) = get_confusion_matrix_pairwise(workload, subgroup)
    return FNR(match_TP, match_FP, match_TN, match_FN)


def true_negative_rate_parity_single(workload, subgroup):
    (match_TP, match_FP, match_TN, match_FN) = get_confusion_matrix_single(workload, subgroup)
    return TNR(match_TP, match_FP, match_TN, match_FN)


def true_negative_rate_parity_pairwise(workload, subgroup):
    (match_TP, match_FP, match_TN, match_FN) = get_confusion_matrix_pairwise(workload, subgroup)
    return TNR(match_TP, match_FP, match_TN, match_FN)


def positive_predictive_value_parity_single(workload, subgroup):
    (match_TP, match_FP, match_TN, match_FN) = get_confusion_matrix_single(workload, subgroup)
    return PPV(match_TP, match_FP, match_TN, match_FN)


def negative_predictive_value_parity_single(workload, subgroup):
    (match_TP, match_FP, match_TN, match_FN) = get_confusion_matrix_single(workload, subgroup)
    return NPV(match_TP, match_FP, match_TN, match_FN)


def positive_predictive_value_parity_pairwise(workload, subgroup):
    (match_TP, match_FP, match_TN, match_FN) = get_confusion_matrix_pairwise(workload, subgroup)
    return PPV(match_TP, match_FP, match_TN, match_FN)


def negative_predictive_value_parity_pairwise(workload, subgroup):
    (match_TP, match_FP, match_TN, match_FN) = get_confusion_matrix_pairwise(workload, subgroup)
    return NPV(match_TP, match_FP, match_TN, match_FN)


def false_discovery_rate_parity_single(workload, subgroup):
    (match_TP, match_FP, match_TN, match_FN) = get_confusion_matrix_single(workload, subgroup)
    return FDR(match_TP, match_FP, match_TN, match_FN)


def false_omission_rate_parity_single(workload, subgroup):
    (match_TP, match_FP, match_TN, match_FN) = get_confusion_matrix_single(workload, subgroup)
    return FOR(match_TP, match_FP, match_TN, match_FN)


def false_discovery_rate_parity_pairwise(workload, subgroup):
    (match_TP, match_FP, match_TN, match_FN) = get_confusion_matrix_pairwise(workload, subgroup)
    return FDR(match_TP, match_FP, match_TN, match_FN)


def false_omission_rate_parity_pairwise(workload, subgroup):
    (match_TP, match_FP, match_TN, match_FN) = get_confusion_matrix_pairwise(workload, subgroup)
    return FOR(match_TP, match_FP, match_TN, match_FN)
