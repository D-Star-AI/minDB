import unittest
import shutil
import numpy as np
import os
import sys

FILE_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(FILE_PATH, '../../'))

from mindb.utils import lmdb_utils
import helpers


class TestLmdbUtils(unittest.TestCase):

    @classmethod
    def setUpClass(self):

        self.name = "lmdb_test"
        self.lmdb_path = os.path.join(os.path.expanduser("~"), '.mindb', self.name)
        self.lmdb_uncompressed_vectors_path = os.path.join(self.lmdb_path, 'uncompressed_vectors')
        self.lmdb_metadata_path = os.path.join(self.lmdb_path, 'metadata')

        self.vectors, self.text, _, _ = helpers.fiqa_test_data()
        self.ids = [i for i in range(len(self.vectors))]
        self.metadata = [{"text": t} for t in self.text]

    def test_001__create_lmdb(self):
        vectors_db_path = lmdb_utils.create_lmdb(self.lmdb_path, 'uncompressed_vectors')
        metadata_db_path = lmdb_utils.create_lmdb(self.lmdb_path, 'metadata')
        self.assertEqual(vectors_db_path, self.lmdb_uncompressed_vectors_path)
        self.assertEqual(metadata_db_path, self.lmdb_metadata_path)

    def test_002__add_items_to_lmdb__vectors(self):
        lmdb_utils.add_items_to_lmdb(
            self.lmdb_uncompressed_vectors_path, self.vectors, self.ids, encode_fn=np.ndarray.tobytes, LMDB_MAP_SIZE=1*1024*1024*1024
        )

        # Make sure there are 30,000 items in the lmdb
        db_count = lmdb_utils.get_db_count(self.lmdb_uncompressed_vectors_path)
        self.assertEqual(db_count, 30_000)
    
    def test_003__add_items_to_lmdb__metadata(self):
        lmdb_utils.add_items_to_lmdb(
            self.lmdb_metadata_path, self.metadata, self.ids, encode_fn=str.encode, LMDB_MAP_SIZE=1*1024*1024*1024
        )

        # Make sure there are 30,000 items in the lmdb
        db_count = lmdb_utils.get_db_count(self.lmdb_metadata_path)
        self.assertEqual(db_count, 30_000)
    
    def test_004__get_lmdb_vectors_by_ids(self):
        ids = [0]
        vectors = lmdb_utils.get_lmdb_vectors_by_ids(self.lmdb_uncompressed_vectors_path, ids)
        # Make sure the vectors are the same
        self.assertTrue(type(vectors), np.ndarray)
        self.assertTrue(np.array_equal(vectors[0], self.vectors[0]))
    
    def test_005__get_lmdb_metadata_by_ids(self):
        ids = [0]
        metadata = lmdb_utils.get_lmdb_metadata_by_ids(self.lmdb_metadata_path, ids)
        # Make sure the metadata is the same
        self.assertTrue(type(metadata), dict)
        self.assertEqual(metadata[0], self.metadata[0])

    def test_006__remove_from_lmdb(self):
        ids = [0, 1, 2]
        lmdb_utils.remove_from_lmdb(self.lmdb_uncompressed_vectors_path, ids, LMDB_MAP_SIZE=1*1024*1024*1024)
        lmdb_utils.remove_from_lmdb(self.lmdb_metadata_path, ids, LMDB_MAP_SIZE=1*1024*1024*1024)

        db_count = lmdb_utils.get_db_count(self.lmdb_uncompressed_vectors_path)
        self.assertEqual(db_count, 29_997)

        db_count = lmdb_utils.get_db_count(self.lmdb_metadata_path)
        self.assertEqual(db_count, 29_997)

    @classmethod
    def tearDownClass(self):
        shutil.rmtree(self.lmdb_path)


if __name__ == '__main__':
    unittest.main()