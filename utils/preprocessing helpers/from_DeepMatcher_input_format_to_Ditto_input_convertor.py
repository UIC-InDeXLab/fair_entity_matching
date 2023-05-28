import pandas as pd
from pathlib import Path

def from_deepmatcher_input_to_ditto_input_helper(path_in, path_out, data_in, data_out):
    df = pd.read_csv(path_in + data_in, index_col="id")
    schema = list(df.columns)[0:]
    ditto_schema = [x.replace("left_", "").replace("right_", "") for x in schema]

    Path(path_out).mkdir(parents=True, exist_ok=True)
    res_file = open(path_out + data_out, "w")
    for idx, row in df.iterrows():
        label = row["label"]
        ditto_row = ""
        for i in range(len(schema)-1):
            ditto_row += "COL " + ditto_schema[i] + " "
            ditto_row += "VAL " + str(row[schema[i]]) + " "
            if "left_" in schema[i] and "right_" in schema[i + 1]:
                ditto_row += "\t"
        ditto_row += "\t" + str(label)
        res_file.write(ditto_row)
        res_file.write("\n")


def from_deepmatcher_input_to_ditto_input(path_in, path_out, test, train, valid):
    from_deepmatcher_input_to_ditto_input_helper(path_in, path_out, test, "test.txt")
    from_deepmatcher_input_to_ditto_input_helper(path_in, path_out, train, "train.txt")
    from_deepmatcher_input_to_ditto_input_helper(path_in, path_out, valid, "valid.txt")


from_deepmatcher_input_to_ditto_input("data/Compas/",
                                      "data/Ditto/Compas/",
                                      "test.csv",
                                      "train.csv",
                                      "valid.csv")



