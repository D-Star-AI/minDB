""" Test for a small minDB """
import numpy as np
import unittest
import os
import sys

import helpers

FILE_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(FILE_PATH, '../../'))

from mindb.mindb import minDB


def evaluate(db, queries: np.ndarray, ground_truths: np.ndarray, query_k: int, gt_k: int) -> tuple[float, list]:

    all_unique_ids = []
    all_cosine_similarity = []
    total_sum = 0
    for i in range(queries.shape[0]):
        results = db.query(queries[i], query_k, gt_k)
        reranked_I = results["ids"]
        cosine_similarity = results["cosine_similarity"]

        all_cosine_similarity.append(cosine_similarity)
        # compute recall
        total_sum += sum([1 for x in reranked_I[:gt_k] if x in ground_truths[i, :gt_k]]) / gt_k
        unique_ids = np.unique(reranked_I)
        all_unique_ids.append(unique_ids)

    recall = total_sum / ground_truths.shape[0]

    return recall, all_unique_ids, all_cosine_similarity


class TestSmallSpdbEvaluation(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.db_name = "small_spdb_test"
        self.query_k = 500
        self.gt_k = 50
        self.db = minDB(self.db_name, LMDB_MAP_SIZE=1*1024*1024*1024)
        self.vectors, self.text, self.queries, self.ground_truths = helpers.fiqa_test_data()
    
    def test__001_small_db_eval(self):
        vectors = self.vectors[0:2500]
        text = self.text[0:2500]
        data = []
        for i, vector in enumerate(vectors):
            data.append((vector, {"text": text[i]}))
        # Add a subset of the vectors
        self.db.add(data)
        # Train the index
        self.db.train(False)
        # Make sure the vectors are in the index
        self.assertTrue(self.db.faiss_index.ntotal, 2500)

        vectors = self.vectors[2500:]
        text = self.text[2500:]
        data = []
        for i, vector in enumerate(vectors):
            data.append((vector, {"text": text[i]}))
        # Add the rest of the vectors
        self.db.add(data)

        recall, all_unique_ids, all_cosine_similarity = evaluate(
            self.db, self.queries, self.ground_truths, self.query_k, self.gt_k
        )

        # Make sure cosine similarity is between 0 and 1
        # all_cosine_similarity is a list of lists, so we need to flatten it
        all_cosine_similarity = [item for sublist in all_cosine_similarity for item in sublist]
        self.assertTrue(all([x >= 0 and x <= 1 for x in all_cosine_similarity]))

        # Recall should be 1.0
        self.assertGreaterEqual(recall, 0.999)
        self.assertLessEqual(recall, 1.001)

        # Make sure the unique ids are the same length as the gt_k
        self.assertTrue(all([len(x) == self.gt_k for x in all_unique_ids]))

    @classmethod
    def tearDownClass(self):
        self.db.delete()


if __name__ == "__main__":
    unittest.main()