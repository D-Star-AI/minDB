from fastapi.testclient import TestClient # requires httpx
import numpy as np
import time
import unittest
import json
import os
import sys

FILE_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(FILE_PATH, '../../'))

from helpers import fiqa_test_data

from mindb.api.fastapi import app


class TestFastAPI(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.client = TestClient(app)
        self.db_name = "fiqa_test"
        self.pca_dimension = 256
        self.opq_dimension = 128
        self.compressed_vector_bytes = 32
        self.omit_opq = True
        self.query_k = 500
        self.gt_k = 50

        vectors, text, _, _ = fiqa_test_data()
        self.vectors = vectors.tolist()
        self.text = text
    
    def test__001_create(self):
        # Create a new database
        response = self.client.post("/db/create", json={"name": self.db_name})
        print (response.text)
        self.assertTrue(response.status_code == 200)

    def test__002_add(self):
        # Add vectors to the index
        batch_size = 1000
        for i in range(0, 20000, batch_size):
            print (i)
            data = []
            for j in range(i, i+batch_size):
                data.append((
                    self.vectors[j],
                    {"text": self.text[j]}
                ))
            response = self.client.post(f"/db/{self.db_name}/add", json={"add_data": data})
        self.assertTrue(response.status_code == 200)
    
    def test__003__delete_db_while_training(self):
        # Begin a training operation, then delete the DB. We need to make sure there is no complete failure
        # when this happens
        response = self.client.post(f"/db/{self.db_name}/train")
        print (response.status_code)
        self.assertEqual(response.status_code, 200)
        time.sleep(10)

        response = self.client.post(f"/db/{self.db_name}/delete")

        # Wait for the training to complete
        response = self.client.get(f"/db/{self.db_name}/train")
        status = response.json()["status"]
        print (status)

if __name__ == "__main__":
    unittest.main()