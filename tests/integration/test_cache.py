from fastapi.testclient import TestClient # requires httpx
import time
import unittest
import os
import sys
import random
import string
import numpy as np
import json

FILE_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(FILE_PATH, '../../'))

from helpers import fiqa_test_data

from mindb.api.fastapi import app


def generate_random_vectors_with_text(N, D):
    random_vectors = np.random.rand(N, D).astype(np.float32) 
    random_text = [''.join(random.choices(string.ascii_lowercase, k=D)) for _ in range(N)]
    return random_vectors, random_text


class TestAutoTrain(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.client = TestClient(app)
        self.db_names = ["fiqa_test_1", "fiqa_test_2"]
        self.pca_dimension = 256
        self.opq_dimension = 128
        self.compressed_vector_bytes = 32
        self.omit_opq = False

        vectors, text, _, _ = fiqa_test_data()
        self.vectors = vectors.tolist()
        self.text = text
    

    def test__001_setup_dbs(self):
        # Create a few databases and add vectors to them

        vectors = self.vectors
        text = self.text

        for db_name in self.db_names:
            response = self.client.post("/db/create", json={"name": db_name})
            self.assertTrue(response.status_code == 200)

            # Add vectors to the database (below the training cutoff so the db is a flat index)
            batch_size = 1000
            for i in range(0, 20000, batch_size):
                data = []
                for j in range(i, i+batch_size):
                    data.append((vectors[j], {"text": text[j]}))
                response = self.client.post(f"/db/{db_name}/add", json={"add_data": data})
            self.assertTrue(response.status_code == 200)

        # View the cache
        response = self.client.get("/db/view_cache")
        print (response.json())
        cache_keys = response.json()["cache_keys"]
        self.assertTrue(len(cache_keys) == 2)

        response = self.client.get("/db/view_cache")
        current_memory_usage = response.json()["current_memory_usage"]
        print ("current_memory_usage", current_memory_usage)

        db_name = self.db_names[0]

        # This is the memory usage needed for the tests to work properly
        new_max_memory_usage = 150 * 1024 * 1024
        response = self.client.post("/db/update_max_memory_usage", json={"max_memory_usage": new_max_memory_usage})
        print ("response", response.json())
    

    def test__002_train(self):

        # Train the first db, then check the info of the db
        db_name = self.db_names[0]
        response = self.client.post(f"/db/{db_name}/train")

        # Wait for the training to finish
        for i in range(20):
            response = self.client.get(f"/db/{db_name}/train")
            status = response.json()["status"]
            if status == "complete":
                break
            time.sleep(10)
        
        response = self.client.get(f"/db/{db_name}/info")
        db_info  = response.json()["db_info"]
        # Convert the db info from a string to a dictionary
        db_info = json.loads(db_info)
        n_total = db_info["n_total"]
        self.assertTrue(n_total == 20000)

        # View the cache
        response = self.client.get("/db/view_cache")
        current_memory_usage = response.json()["current_memory_usage"]
        print ("current_memory_usage", current_memory_usage)

        # Make sure the memory usage is less than 80MB
        # If it's higher, then the cache memory wouldn't have updated the after training
        self.assertTrue(current_memory_usage < (80 * 1024 * 1024))
    

    def test__003_auto_remove_cache(self):

        # Create 2 more DBs and add 20,000 vectors to each in order to get above 150MB
        new_db_names = ["fiqa_test_3", "fiqa_test_4"]
        for db_name in new_db_names:
            response = self.client.post("/db/create", json={"name": db_name})
            self.assertTrue(response.status_code == 200)

            # Add vectors to the database
            batch_size = 1000
            for i in range(0, 20000, batch_size):
                data = []
                for j in range(i, i+batch_size):
                    data.append((self.vectors[j], {"text": self.text[j]}))
                response = self.client.post(f"/db/{db_name}/add", json={"add_data": data})
            self.assertTrue(response.status_code == 200)
        
        # View the cache
        response = self.client.get("/db/view_cache")
        # The fiqa_test_1 db should have been removed from the cache
        cache_keys = response.json()["cache_keys"]
        self.assertTrue(len(cache_keys) == 3)
        # Make sure the fiqa_test_2 db is not in the cache
        self.assertTrue("fiqa_test_2" not in cache_keys)
    

    def test__004_remove_from_cache(self):
        # Just testing removing a database from the cache
        
        db_name = "fiqa_test_1"
        response = self.client.post(f"/db/{db_name}/remove_from_cache")

        # View the cache
        response = self.client.get("/db/view_cache")
        cache_keys = response.json()["cache_keys"]
        self.assertTrue(len(cache_keys) == 2)
        # Make sure the fiqa_test_1 db is not in the cache
        self.assertTrue("fiqa_test_1" not in cache_keys)
    

    def test__005_remove_from_cache_while_training(self):
        # Make sure the DB is not removed from the cache while training

        # Initialize the training operation for the fiqa_test_3 db
        db_name = "fiqa_test_3"
        response = self.client.post(f"/db/{db_name}/train")

        # Sleep for enough time for the training to begin
        time.sleep(5)

        # Make sure fiqa_test_3 is in the cache now
        response = self.client.get("/db/view_cache")
        cache_keys = response.json()["cache_keys"]
        self.assertTrue("fiqa_test_3" in cache_keys)

        # Add 1000 vectors to the 4th db. It was not previously in the cache, but it should be added now
        # We need to add the vectors to this db since it's still a flat index, which will increase memory usage
        # enough to trigger a db to be removed from the cache
        db_name = "fiqa_test_4"
        batch_size = 1000
        data = []
        for j in range(0, batch_size):
            data.append((self.vectors[j], {"text": self.text[j]}))
        response = self.client.post(f"/db/{db_name}/add", json={"add_data": data})
        self.assertTrue(response.status_code == 200)
        
        # View the cache. fiqa_test_4 and fiqa_test_3 should be in the cache
        response = self.client.get("/db/view_cache")
        cache_keys = response.json()["cache_keys"]
        self.assertTrue("fiqa_test_4" in cache_keys)
        self.assertTrue("fiqa_test_3" in cache_keys)


        # Add 1000 vectors to the 2nd db. It was not previously in the cache, but it should be added now
        db_name = "fiqa_test_2"
        batch_size = 1000
        data = []
        for j in range(0, batch_size):
            data.append((self.vectors[j], {"text": self.text[j]}))
        response = self.client.post(f"/db/{db_name}/add", json={"add_data": data})
        self.assertTrue(response.status_code == 200)
        
        # View the cache. fiqa_test_4 should not be in the cache, even though it was more recently used
        # than fiqa_test_3. This is because fiqa_test_3 is currently training
        response = self.client.get("/db/view_cache")
        cache_keys = response.json()["cache_keys"]
        self.assertTrue("fiqa_test_2" in cache_keys)
        self.assertTrue("fiqa_test_4" not in cache_keys)


        # Wait for the training to finish
        db_name = "fiqa_test_3"
        for i in range(20):
            response = self.client.get(f"/db/{db_name}/train")
            status = response.json()["status"]
            if status == "complete":
                break
            time.sleep(10)
        
        print ("training status", status)

    
    def test__006_update_max_memory_usage(self):
        # Update the max memory usage of the cache

        new_max_memory_usage = 400 * 1024 * 1024
        response = self.client.post("/db/update_max_memory_usage", json={"max_memory_usage": new_max_memory_usage})
        self.assertTrue(response.status_code == 200)

        # View the cache
        response = self.client.get("/db/view_cache")
        current_memory_usage = response.json()["current_memory_usage"]
        self.assertTrue(current_memory_usage < new_max_memory_usage)

        # Add more vectors to the 4th db. This will get the memory usage above the original 200MB
        # We need to make sure the cache keys stay the same
        db_name = "fiqa_test_4"
        batch_size = 1000
        for i in range(0, 15000, batch_size):
            data = []
            for j in range(i, i+batch_size):
                data.append((self.vectors[j], {"text": self.text[j]}))
            response = self.client.post(f"/db/{db_name}/add", json={"add_data": data})
        self.assertTrue(response.status_code == 200)

        # View the cache
        response = self.client.get("/db/view_cache")
        current_memory_usage = response.json()["current_memory_usage"]
        print ("current_memory_usage", current_memory_usage)
        cache_keys = response.json()["cache_keys"]
        print ("cache_keys test 006", cache_keys)
        self.assertTrue(len(cache_keys) == 3)

        # Set the memory usage back to 150MB
        new_max_memory_usage = 150 * 1024 * 1024
        response = self.client.post("/db/update_max_memory_usage", json={"max_memory_usage": new_max_memory_usage})
        print ("response", response.json())

        # View the cache. A db should be removed since the memory usage is above the max memory usage
        response = self.client.get("/db/view_cache")
        current_memory_usage = response.json()["current_memory_usage"]
        print ("current_memory_usage", current_memory_usage)
        cache_keys = response.json()["cache_keys"]
        print ("cache_keys test 006", cache_keys)
        self.assertTrue(len(cache_keys) == 2)
        self.assertTrue("fiqa_test_2" not in cache_keys)
    
    @classmethod
    def tearDownClass(self):
        for db_name in self.db_names:
            response = self.client.post(f"/db/{db_name}/delete")
        
        for db_name in ["fiqa_test_3", "fiqa_test_4"]:
            response = self.client.post(f"/db/{db_name}/delete")

if __name__ == "__main__":
    unittest.main()