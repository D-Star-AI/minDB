""" Full end to end evaluation of the minDB using the fiqa Beir dataset """
import os
import sys
import numpy as np
import time
import unittest

import helpers

FILE_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(FILE_PATH, '../../'))

from mindb.mindb import minDB, load_db


def evaluate(db, queries: np.ndarray, ground_truths: np.ndarray, query_k: int, gt_k: int) -> tuple[float, float, list]:

    start_time = time.time()
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

    end_time = time.time()
    recall = total_sum / ground_truths.shape[0]
    latency = (end_time - start_time) * 1000 / queries.shape[0]

    return recall, latency, all_unique_ids, all_cosine_similarity


class TestFullSpdbEvaluation(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.db_name = "full_eval_test"
        self.pca_dimension = 256
        self.opq_dimension = 128
        self.compressed_vector_bytes = 32
        self.omit_opq = True # This speeds up the test with a very small performance hit
        self.query_k = 500
        self.gt_k = 50
        self.db = minDB(self.db_name)
        self.vectors, self.text, self.queries, self.ground_truths = helpers.fiqa_test_data()
        data = [(self.vectors[i], {"text": self.text[i]}) for i in range(len(self.vectors))]
        self.db.add(data)


    def test__001_full_eval(self):

        # Load the index ()
        db = load_db(self.db_name)

        # Train the index
        db.train(True, self.pca_dimension, self.opq_dimension, self.compressed_vector_bytes, self.omit_opq)

        # Evaluate the index
        recall, latency, all_unique_ids, all_cosine_similarity = evaluate(
            db, self.queries, self.ground_truths, self.query_k, self.gt_k
        )

        # Make sure cosine similarity is between 0 and 1
        # all_cosine_similarity is a list of lists
        all_cosine_similarity = [item for sublist in all_cosine_similarity for item in sublist]
        self.assertTrue(all([x >= 0 and x <= 1 for x in all_cosine_similarity]))

        # Set the recall cutoff at 0.97 and less than 1
        # If recall is above 1, something went wrong
        self.assertGreater(recall, 0.97)
        self.assertLessEqual(recall, 1)

        # Make sure latency is less than 30ms (this includes the re-ranking from disk step)
        self.assertLess(latency, 30)

        # Make sure the length of each unique ID list is equal to the gt_k
        self.assertTrue(all([len(x) == self.gt_k for x in all_unique_ids]))


    def test__002_full_eval_no_two_level(self):

        # Load the index ()
        db = load_db(self.db_name)

        # Train the index, but without two level clustering
        db.train(False)

        # Evaluate the index
        recall, latency, all_unique_ids, all_cosine_similarity = evaluate(
            db, self.queries, self.ground_truths, self.query_k, self.gt_k
        )

        print ("recall", recall)

        # Make sure cosine similarity is between 0 and 1
        # all_cosine_similarity is a list of lists
        all_cosine_similarity = [item for sublist in all_cosine_similarity for item in sublist]
        self.assertTrue(all([x >= 0 and x <= 1 for x in all_cosine_similarity]))

        # Set the recall cutoff at 0.97 and less than 1
        # If recall is above 1, something went wrong
        self.assertGreater(recall, 0.97)
        self.assertLessEqual(recall, 1)

        # Make sure latency is less than 40ms (this includes the re-ranking from disk step)
        self.assertLess(latency, 40)

        # Make sure the length of each unique ID list is equal to the gt_k
        self.assertTrue(all([len(x) == self.gt_k for x in all_unique_ids]))

    @classmethod
    def tearDownClass(self):
        db = load_db(self.db_name)
        db.delete()

if __name__ == "__main__":
    unittest.main()