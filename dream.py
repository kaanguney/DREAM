from vectors import tokenize
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm

def main():
    size = 4
    database, exprs = tokenize(size=size)
    sequences, vectors = list(database.index), database.values
    rf = RandomForestRegressor(n_estimators=100, max_depth=None, min_samples_split=2, bootstrap=False, verbose=2)
    rf.fit(vectors, exprs)
    scores = rf.predict(vectors)
    # save model predictions
    with open("rf_predictions.txt", "w+") as f:
       for _, pred in enumerate(tqdm(zip(sequences, scores))):
            f.write(pred[0] + "\t" + str(pred[1]) + "\n")

if __name__ == "__main__":
    main()