import csv

count_id = 0


def to_deepmatcher_helper(left_vals, right_vals, table, new_schema, res_filename):
    global count_id
    res_file = open(res_filename, "w")
    curr_line = "id,"
    for sch_attr in new_schema:
        curr_line += sch_attr + ","
    curr_line = curr_line[:-1]
    curr_line += "\n"
    res_file.write(curr_line)
    for line in table:
        left_id = line[0]
        right_id = line[1]
        label = line[2]
        new_line = [label] + left_vals[left_id] + right_vals[right_id]

        res_file.write(str(count_id) + ",")
        count_id += 1
        for i in range(len(new_line)):
            attr_val = new_line[i]
            if "," in attr_val:
                res_file.write("\"")
                res_file.write(attr_val)
                res_file.write("\"")
            else:
                res_file.write(attr_val)
            if i == len(new_line) - 1:
                res_file.write("\n")
            else:
                res_file.write(",")


def to_deepmatcher_input(dataset_name, tableA, tableB, test, train, valid):
    left_table = open(tableA, "r")

    schema = left_table.readlines()[0].split(",")[0:]  # remove the id
    left_schema = ["left_" + sch.strip() for sch in schema]
    right_schema = ["right_" + sch.strip() for sch in schema]
    new_schema = ["label"] + left_schema + right_schema

    left_table = list(csv.reader(open(tableA)))[1:]
    right_table = list(csv.reader(open(tableB)))[1:]

    left_vals = {}
    right_vals = {}

    for left_line in left_table:
        left_id = str(left_line[0])
        left_line = left_line[0:]  # remove the id
        left_vals[left_id] = left_line
    for right_line in right_table:
        right_id = str(right_line[0])
        right_line = right_line[0:]  # remove the id
        right_vals[right_id] = right_line

    train_table = list(csv.reader(open(train)))[1:]
    test_table = list(csv.reader(open(test)))[1:]
    valid_table = list(csv.reader(open(valid)))[1:]

    to_deepmatcher_helper(left_vals, right_vals, train_table, new_schema,
                          "data/" + dataset_name + "/train.csv")
    to_deepmatcher_helper(left_vals, right_vals, valid_table, new_schema,
                          "data/" + dataset_name + "/valid.csv")
    to_deepmatcher_helper(left_vals, right_vals, test_table, new_schema,
                          "data/" + dataset_name + "/test.csv")


# This code can be used to convert data from Magellan Data Repository to the format accepted by Deepmatcher.
# Link to the Magellan Data Repository:
# https://sites.google.com/site/anhaidgroup/useful-stuff/the-magellan-data-repository?authuser=0

to_deepmatcher_input("DBLP-ACM",
                     "data/magellan/DBLP-ACM/tableA.csv",
                     "data/magellan/DBLP-ACM/tableB.csv",
                     "data/magellan/DBLP-ACM/test.csv",
                     "data/magellan/DBLP-ACM/train.csv",
                     "data/magellan/DBLP-ACM/valid.csv")
