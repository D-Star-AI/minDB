import unittest
import os
import sys

FILE_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(FILE_PATH, '../../'))

import helpers
from mindb.mindb import minDB


class TestIndexCoverageRatio(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.db_name = "index_coverage_ratio_eval_test"
        self.pca_dimension = 256
        self.opq_dimension = 128
        self.compressed_vector_bytes = 32
        self.omit_opq = True
        self.vectors, self.text, self.queries, self.ground_truths = helpers.fiqa_test_data()
        self.db = minDB(self.db_name)
    
    def test__001_index_coverage_ratio(self):

        # Test that the index coverage ratio is 0 before adding any vectors
        coverage_ratio = self.db.trained_index_coverage_ratio
        self.assertEqual(coverage_ratio, 0)

        # Test that the index coverage ratio is still 0 after adding vectors
        add_data = [(self.vectors[i], {"text": self.text[i]}) for i in range(len(self.vectors))]
        self.db.add(add_data)
        coverage_ratio = self.db.trained_index_coverage_ratio
        self.assertEqual(coverage_ratio, 0)

        # Test that the index coverage ratio is 1 after training the index
        self.db.train(True, self.pca_dimension, self.opq_dimension, self.compressed_vector_bytes, self.omit_opq)
        coverage_ratio = self.db.trained_index_coverage_ratio
        self.assertEqual(coverage_ratio, 1)

        # Test that the index coverage ratio is 0.5 after adding another set of vectors
        self.db.add(add_data)
        coverage_ratio = self.db.trained_index_coverage_ratio
        self.assertEqual(coverage_ratio, 0.5)

        num_new_vectors = self.db.num_new_vectors
        self.assertEqual(num_new_vectors, len(self.vectors))

        # Remove the first 30000 vectors (which is the length of the fiqa test data)
        ids = [i for i in range(30000)]
        self.db.remove(ids)
        # The index coverage ratio should be 0 now
        coverage_ratio = self.db.trained_index_coverage_ratio
        self.assertEqual(coverage_ratio, 0)

        num_trained_vectors_removed = self.db.num_trained_vectors_removed
        self.assertEqual(num_trained_vectors_removed, 30000)

    @classmethod
    def tearDownClass(self):
        self.db.delete()

if __name__ == "__main__":
    unittest.main()