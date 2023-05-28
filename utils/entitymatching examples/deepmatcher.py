import deepmatcher as dm


def run_deepmatcher(directory, train, validation, test, epochs=10, prediction_threshold=0.5):
    train, validation, test = dm.data.process(path=directory, train=train, validation=validation, test=test)

    dm_model = dm.MatchingModel()
    dm_model.run_train(train, validation, best_save_path='best_model.pth', epochs=epochs)
    dm_scores = dm_model.run_prediction(test)
    prediction = [True if dm_scores.iloc[idx]["match_score"] > prediction_threshold else False for idx in
                  range(len(dm_scores))]
    return prediction


predictions = run_deepmatcher("data/Compas/",
                              train="train.csv",
                              validation="valid.csv",
                              test="test.csv",
                              epochs=10,
                              prediction_threshold=0.5)

with open("data/Compas/DeepMatcher/preds.csv", "w") as f:
    f.write("preds\n")
    for prediction in predictions:
        f.write("1\n" if prediction else "0\n")
