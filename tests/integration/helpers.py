import os
import numpy as np
import pickle

# get the absolute file path of this file
FILE_PATH = os.path.dirname(os.path.abspath(__file__))

def fiqa_test_data() -> tuple[np.ndarray, list, np.ndarray, np.ndarray]:

    # Get the vectors
    with open(FILE_PATH + '/../data/fiqa_vectors.pickle', 'rb') as handle:
        vectors = pickle.load(handle)
    
    # Get the query data
    with open(FILE_PATH + '/../data/fiqa_queries.pickle', 'rb') as handle:
        queries = pickle.load(handle)
    
    # Get the text data
    with open(FILE_PATH + '/../data/fiqa_text.pickle', 'rb') as handle:
        text = pickle.load(handle)
    
    # Get the ground truths
    with open(FILE_PATH + '/../data/fiqa_ground_truths.pickle', 'rb') as handle:
        ground_truths = pickle.load(handle)

    return vectors, text, queries, ground_truths
