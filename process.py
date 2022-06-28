import sys
import os
import pandas as pd
import shutil
import numpy as np
from more_itertools import sliced
from tqdm import tqdm

# select processing mode
PROCESSING_MODES = ['raw', 'onehot']
while True:
  processing_mode = input('Please enter the processing mode: ')
  if processing_mode in PROCESSING_MODES: break
  else: print("Valid processing modes are 'raw' or 'onehot'.")

# if processing mode 'onehot', export subset with .npz extension
MEMORY_MODES = [5, 10, 15]
if processing_mode == 'onehot':
  while True:
    try:
      memory_mode = int(input('Please enter the percentage of allocated memory: '))
    except ValueError:
      print("Valid memory modes are 5, 10 or 15.")
      continue
    if memory_mode in MEMORY_MODES: break
    else: print("Valid memory modes are 5, 10 or 15.")

# helper function for proessing mode 'onehot'
def remove_prefix(text: str, prefix: str) -> str:
    if text.startswith(prefix):
        return text[len(prefix):]
    raise ValueError(prefix)

# helper function for proessing mode 'onehot'
def remove_suffix(text: str, suffix: str) -> str:
    if text.endswith(suffix):
        return text[len(suffix):]
    raise ValueError(suffix)

# global variables for processing mode 'onehot'
TOTAL_SEQS = 6739258
PREFIX = 'TGCATTTTTTTCACATC'
SUFFIX = 'GGTTACGGCTGTT'
MAX_SEQ_LEN = 142 - len(PREFIX) - len(SUFFIX)  # 112
nuc_map = {k: i for i, k in enumerate(['A', 'C', 'T', 'G', 'N'])}

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

if processing_mode == 'raw':
  # specify path to data
  PATH = os.getcwd()
  path = os.path.join(PATH, 'data')
  print(f"I/O path: {path}")

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
  print('Processing, done.')
  sys.exit()

# execute 'onehot' snippet, @vinnik-dmitry07
else:
  x = np.empty((TOTAL_SEQS, MAX_SEQ_LEN * 4), np.float16)
  y = np.empty(TOTAL_SEQS, np.float16)

  with open('data/train_sequences.txt', 'r') as f:
      for i, line in enumerate(tqdm(f.readlines())):
        if not (TOTAL_SEQS // (100 / memory_mode)) == i:
          seq, expr = line.split('\t')
          seq = remove_suffix(remove_prefix(seq, PREFIX), SUFFIX)
          nuc_idx = np.array([nuc_map[s] for s in seq], dtype=int)
          pos_idx = np.arange(nuc_idx.size)[nuc_idx != 4]
          nuc_idx = nuc_idx[nuc_idx != 4]
          one_hot = np.zeros((MAX_SEQ_LEN, 4))
          one_hot[pos_idx, nuc_idx] = 1 / MAX_SEQ_LEN
          x[i] = one_hot.ravel()
          y[i] = float(expr)
        else:
          np.savez('data/onehot', x=x, y=y)
          print('Processing, done.')
          sys.exit()