import numpy as np
# returns true if all attributes in subgroup1 are present in subgroup2
def clauses_satisfied(subgroup1, subgroup2):
    for i in range(len(subgroup1)):
        if subgroup1[i] == 1 and subgroup2[i] == 0:
            return False
    return True

# converts a k-combination (2-comb (45, 55, 13, 46)) to a subgroup (encoding)
# with 1's at those indices
def comb_to_encoding(combination, full_encoding_len):
    # by definition, 2|len(combination) and 2|full_encoding_len
    boundary = int(len(combination) / 2)
    encoding = np.zeros(full_encoding_len)

    for ind in range(len(combination)):
        if ind < boundary:
            encoding[combination[ind]] = 1
        else:
            encoding[combination[ind] + int(full_encoding_len / 2)] = 1

    return encoding