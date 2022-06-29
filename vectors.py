from tqdm import tqdm
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

# kmerization might be an expensive process
MEMORY_MODES = [15, 20, 25]
while True:
    try:
        memory_mode = int(input('Please enter the percentage of allocated memory: '))
    except ValueError:
        print("Valid memory modes are 15, 20 or 25.")
        continue
    if memory_mode in MEMORY_MODES: break
    else: print("Valid memory modes are 15, 20 or 25.")

# helper function
def remove_prefix(text: str, prefix: str) -> str:
    if text.startswith(prefix):
        return text[len(prefix):]
    raise ValueError(prefix)

# helper function
def remove_suffix(text: str, suffix: str) -> str:
    if text.endswith(suffix):
        return text[len(suffix):]
    raise ValueError(suffix)

# global variables
TOTAL_SEQS = 6739258
PREFIX = 'TGCATTTTTTTCACATC'
SUFFIX = 'GGTTACGGCTGTT'
MAX_SEQ_LEN = 142 - len(PREFIX) - len(SUFFIX)  # 112
nuc_map = {k: i for i, k in enumerate(['A', 'C', 'T', 'G', 'N'])}

# replace cutoff with commented statement on GPU
cutoff = 1000
# cutoff = int(TOTAL_SEQS // (100 / memory_mode))

def kmerize(filename="./data/train_sequences.txt", stride=1, size=4):
    """
    Generate kmers for specified subset of sequences.
    :param:
           str filename: file with sequences.
           int stride: slide of the window for kmer generation.
           int size: size of the kmer.
    :return:
            pd.DataFrame: Pandas object with row pair as sequence and the corresponding kmer.
            list exprs: expressions of promoter sequences.
    """
    sequences, kmers, exprs = [], [], []
    with open(filename) as f:
        for i, line in enumerate(tqdm(f.readlines())):
            if i == cutoff:
                print("Processing, done.")
                break
            seq, expr = line.split('\t')
            exprs.append(float(expr.replace("\n", "")))
            seq = remove_suffix(remove_prefix(seq, PREFIX), SUFFIX)
            try:
                # suboptimal solution to padding
                # consider alternative(s)
                kmer = [seq[i:(i+size)] for i in range(0, stride, len(seq)) if len(seq[i:(i+size)]) == size and len(seq) == 80]
                kmer = ",".join(kmer)
                sequences.append(seq)
                kmers.append(kmer)
            except IndexError:
                pass
    database = pd.DataFrame({'sequence': sequences, "kmers": kmers})
    exprs = np.array(exprs, dtype=np.float16)
    return database, exprs

def tokenize(size):
    """
    Calculate frequency for each kmer given sequence.
    :param:
           int size: length of each kmer.
    :return:
            pd.DataFrame: Pandas object consisting of kmer frequencies indexed by sequences.
            list exprs: expressions of promoter sequences.
    """
    database, exprs = kmerize(size=size)
    vectorizer = CountVectorizer()
    vectorizer.fit(database["kmers"])
    data = vectorizer.transform(database["kmers"])
    vectors = pd.DataFrame(data.toarray(), database["sequence"].values, vectorizer.get_feature_names())
    return vectors, exprs