import sys
import os
import unittest
import faiss
import numpy as np
import random
import string

FILE_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(FILE_PATH, '../../'))

from mindb.cache.cache import LRUCache
from mindb.mindb import minDB, load_db


def generate_random_vectors_with_text(N, D):
    random_vectors = np.random.rand(N, D).astype(np.float32) 
    random_text = [''.join(random.choices(string.ascii_lowercase, k=D)) for _ in range(N)]
    return random_vectors, random_text


class TestLRUCache(unittest.TestCase):

    flat_faiss_index = faiss.IndexFlat(768)
    flat_faiss_index = faiss.IndexIDMap(flat_faiss_index)

    def setUp(self):
        self.cache = LRUCache(100 * 1024 * 1024) # 200 MB
        self.training_params = {
            'pca_dimension': 256,
            'opq_dimension': 128,
            'compressed_vector_bytes': 32,
            'omit_opq': True,
            'num_clusters': 1000
        }
        self.operations = {
            "cache_test_db": "in progress",
            "cache_test_db_2": "complete",
            "cache_test_db_3": "untrained"
        }
        self.random_vectors, self.random_text = generate_random_vectors_with_text(15000, 768)
        
    
    def test_001__add_to_cache(self):

        db_name = "cache_test_db"
        db = minDB(name=db_name, vector_dimension=768,
            max_memory_usage=1000 * 1024 * 1024, LMDB_MAP_SIZE=int(0.25*1024*1024*1024))
        
        # Add an item to the cache
        self.cache.put(db_name, db)

        # Make sure the item exists in the cache now
        self.assertTrue(db_name in self.cache.cache)
    

    def test_002__add_vectors(self):
        
        db_name = "cache_test_db"
        db = load_db(db_name)
        self.cache.put(db_name, db)
        
        num_vectors = 15000
        vector_dimension = 768
        random_vectors = self.random_vectors.tolist()
        
        batch_size = 1000
        for i in range(0, len(random_vectors), batch_size):
            data = []
            for j in range(i, i+batch_size):
                data.append((random_vectors[j], {"text": self.random_text[j]}))
            db.add(data)

        # Update the cache (this gets called in the add function in FastAPI)
        #self.cache.put(db_name, db)
        self.cache.update_memory_usage()

        # Get the memory usage of the cache now
        memory_usage = self.cache.current_memory_usage
        print ("memory_usage", memory_usage)
        self.assertEqual(memory_usage, (num_vectors * vector_dimension * 4) + 240090)
    

    def test_003__auto_remove_from_cache(self):

        # Add the first db to the cache. setUp gets called before each test, so we need to add the db again
        db_name = "cache_test_db"
        db = load_db(db_name)
        self.cache.put(db_name, db)

        memory_usage = self.cache.current_memory_usage
        print ("memory_usage 2", memory_usage)


        # Create a couple more DBs and add 30,000 vectors to each in order to get above 200MB
        db = minDB(name="cache_test_db_2", vector_dimension=768,
            max_memory_usage=1000 * 1024 * 1024, LMDB_MAP_SIZE=int(0.25*1024*1024*1024))
        
        # Add the db to the cache
        self.cache.put("cache_test_db_2", db)
        print ("cache after 2nd db", self.cache.cache)

        random_vectors = self.random_vectors.tolist()
        batch_size = 1000
        for i in range(0, len(random_vectors), batch_size):
            data = []
            for j in range(i, i+batch_size):
                data.append((random_vectors[j], {"text": self.random_text[j]}))
            db.add(data)


        print (db.max_id)
        print (db.faiss_index)
        print (db.faiss_index.ntotal)

        # Update the cache
        self.cache.update_memory_usage()
        memory_usage = self.cache.current_memory_usage
        print (self.cache.cache)
        print ("memory_usage 3", memory_usage)

        # View the cache keys. There should be 2 keys in the cache now
        cache_keys = self.cache.cache.keys()
        self.assertEqual(len(cache_keys), 2)


        # Create another DB
        db = minDB(name="cache_test_db_3", vector_dimension=768,
            max_memory_usage=1000 * 1024 * 1024, LMDB_MAP_SIZE=int(0.25*1024*1024*1024))
        
        # Add the db to the cache
        self.cache.put("cache_test_db_3", db)

        random_vectors = self.random_vectors.tolist()

        # Run this in batches to simulate how it is run in FastAPI
        batch_size = 1000
        for i in range(0, len(random_vectors), batch_size):
            # Get the db from the cache each time (this matches what happens in FastAPI when adding vectors)
            db = self.cache.get("cache_test_db_3", check_memory_usage=True, operations=self.operations)
            data = []
            for j in range(i, i+batch_size):
                data.append((random_vectors[j], {"text": self.random_text[j]}))
            db.add(data)

        # Update the cache
        self.cache.update_memory_usage()
        print ("cache memory usage after 3rd db", self.cache.current_memory_usage)

        # View the cache keys. There should still be 2 cache keys, and cache_test_db_2 should not be in it.
        # Even though cache_test_db_2 was more recently used than cache_test_db, it should be removed since
        # the operation status of cache_test_db is "in progress"
        cache_keys = self.cache.cache.keys()
        print ("cache_keys", cache_keys)
        self.assertEqual(len(cache_keys), 2)
        self.assertTrue("cache_test_db_2" not in cache_keys)
    

    def test_004__update_max_cache_memory_usage(self):
        
        # Update the max memory usage of the cache
        self.cache.update_max_memory_usage(300 * 1024 * 1024)
        self.assertEqual(self.cache.max_memory_usage, 300 * 1024 * 1024)


    def test_009__tear_down(self):
        db = load_db("cache_test_db")
        db.delete()

        db2 = load_db("cache_test_db_2")
        db2.delete()

        db3 = load_db("cache_test_db_3")
        db3.delete()


if __name__ == '__main__':
    unittest.main()