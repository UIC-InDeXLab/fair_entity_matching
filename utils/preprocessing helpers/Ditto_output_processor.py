import json

def jsonl_to_predictions(path, dataset_name):
    predictions = []
    with open(path, 'r') as json_file:
        json_list = list(json_file)

    for json_line in json_list:
        line = json.loads(json_line)
        predictions.append(line["match"])

    with open("data/Ditto/" + dataset_name + "/preds.csv", "w") as f:
        for prediction in predictions:
            f.write(str(prediction) + '\n')

datasets = ['Compas']
paths = ['/Ditto/output_compas.jsonl']

for i in range(len(paths)):
    jsonl_to_predictions(paths[i], datasets[i])

