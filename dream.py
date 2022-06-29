from vectors import tokenize
import json
import numpy as np
from collections import OrderedDict
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm
import os

# TODO: hyperparameter optimization (including different kmer sizes)
# TODO: validation, cross validation
# TODO: incorporate reverse complement awareness
# TODO: try boosting, in particular CatBoost
# TODO: script currently supports evaluation on test only if submission flag is active
# TODO: consider augmenting the frequency matrix instead of padding

# directory definitions
RF_DIR_TRAIN = "./data/rf/results/train/"
RF_DIR_TEST = "./data/rf/results/test/"
RF_DIR_VALIDATION = "./data/rf/results/validation/"

def dump_predictions(prediction_dict, prediction_file):
    with open(prediction_file, 'w') as f:
        json.dump(prediction_dict, f)

def submit(Y_pred=np.zeros((71103,), dtype=np.float16)):
    with open('./sample_submission.json', 'r') as f:
        ground = json.load(f)
    indices = np.array([int(indice) for indice in list(ground.keys())])
    PRED_DATA = OrderedDict()
    for i in indices:
        PRED_DATA[str(i)] = float(Y_pred[i])
    dump_predictions(PRED_DATA, 'pred.json')

def main():
    # train baseline model, save results to training directory
    size = 4
    database, exprs, is_submission = tokenize(size=size, is_submission=False)
    sequences, vectors = list(database.index), database.values   
    rf = RandomForestRegressor(n_estimators=100, max_depth=None, min_samples_split=2, bootstrap=False, verbose=2)
    rf.fit(vectors, exprs)
    scores = rf.predict(vectors)
    if not os.path.exists(RF_DIR_TRAIN):
        os.makedirs(RF_DIR_TRAIN, exist_ok=True)
        with open(RF_DIR_TRAIN + "train.txt", "w+") as f:
            for _, pred in enumerate(tqdm(zip(sequences, scores))):
                f.write(pred[0] + "\t" + str(pred[1]) + "\n")
    # evaluate results on the test data
    # submission flag has to be active in this case
    if not os.path.exists(RF_DIR_TEST): 
        os.makedirs(RF_DIR_TEST, exist_ok=True)
        # pad if necessary
        padding = vectors.shape[1]
        database, exprs, is_submission = tokenize(size=size, is_submission=True)
        sequences, vectors = list(database.index), database.values
        # padding
        while vectors.shape[1] < padding:
            vectors = np.append(vectors, np.zeros((len(sequences),1)), axis=1)
        scores = rf.predict(vectors)
        with open(RF_DIR_TEST + "test.txt", "w+") as f:
            for _, pred in enumerate(tqdm(zip(sequences, scores))):
                f.write(pred[0] + "\t" + str(pred[1]) + "\n")
        assert is_submission == True
    # create submission .json file
    submit()

if __name__ == "__main__":
    main()