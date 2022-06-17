import os
import pandas as pd
import shutil
import numpy as np
from more_itertools import sliced

# specify path to data
PATH = os.getcwd()
path = os.path.join(PATH, 'data')
print(f"I/O path: {path}")

# process sequence, expression level
def load(filename, path):
    """
    Parse and load passed files into numpy arrays for Pandas API.
    :param:
           str filename: filename for the sequences.
           str path: path to the data folder that stores the sequences.
    :return:
            np.ndarray sequences: sequence and expression pairs wrapped into numpy rows.
    """
    with open(os.path.join(path, filename)) as f:
        sequences = f.readlines()
        sequences = [sequence.split('\t') for sequence in sequences]
        sequences = [[sequence_list[0], sequence_list[1].replace('\n', '')] for sequence_list in sequences]
    return np.array(sequences)

# load sequence, expression couples to pandas
training = pd.DataFrame(load('train_sequences.txt', path))
testing = pd.DataFrame(load('test_sequences.txt', path))
training.columns, testing.columns = ['sequence', 'expression'], ['sequence', 'expression']

# cast accordingly
training.sequence = training.sequence.apply(lambda x: str(x))
testing.sequence = testing.sequence.apply(lambda x: str(x))

training.expression = training.expression.apply(lambda x: float(x))
testing.expression = testing.expression.apply(lambda x: float(x))

# delete data folder, create a fresh directory for batch files
shutil.rmtree('data')
os.mkdir('training')
os.mkdir('testing')

# batch training
slices = sliced(range(len(training)), 500000)
filecount = 1
for slice in slices:
  batch = training.iloc[slice]
  batch.to_csv(os.path.join(os.getcwd(), 'training', f'train_000{filecount}.csv'), index=False)
  filecount += 1

# no need to batch testing
testing.to_csv(os.path.join(os.getcwd(), 'testing', 'test.csv'), index=False)
