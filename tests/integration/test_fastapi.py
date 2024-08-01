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


def evaluate(client, db_name: str, queries: np.ndarray, ground_truths: np.ndarray, query_k: int, gt_k: int):
    start_time = time.time()
    all_unique_ids = []
    total_sum = 0
    for i in range(queries.shape[0]):
        response = client.post(f"/db/{db_name}/query", json={"query_vector": queries[i].tolist(), "preliminary_top_k": query_k, "final_top_k": gt_k})
        reranked_I = np.array(response.json()['ids'])
        # compute recall
        total_sum += sum([1 for x in reranked_I[:gt_k] if x in ground_truths[i, :gt_k]]) / gt_k
        unique_ids = np.unique(reranked_I)
        all_unique_ids.append(unique_ids)

    end_time = time.time()
    recall = total_sum / ground_truths.shape[0]
    latency = (end_time - start_time) * 1000 / queries.shape[0]

    return recall, latency, all_unique_ids


class TestFastAPI(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.client = TestClient(app)
        self.db_name = "fast_api_test"
        self.pca_dimension = 256
        self.opq_dimension = 128
        self.compressed_vector_bytes = 32
        self.omit_opq = True
        self.query_k = 500
        self.gt_k = 50

        vectors, text, queries, ground_truths = fiqa_test_data()
        self.vectors = vectors.tolist()
        self.text = text
        self.queries = queries
        self.ground_truths = ground_truths


    def test__001_create(self):
        # Create a new database
        response = self.client.post("/db/create", json={"name": self.db_name})
        print (response.text)
        self.assertTrue(response.status_code == 200)

    def test__002_add(self):
        # Add vectors to the index
        batch_size = 1000
        for i in range(0, len(self.vectors), batch_size):
            print (i)
            data = []
            for j in range(i, i+batch_size):
                data.append((
                    self.vectors[j],
                    {"text": self.text[j]}
                ))
            response = self.client.post(f"/db/{self.db_name}/add", json={"add_data": data})
        self.assertTrue(response.status_code == 200)

    def test__003_train(self):

        # Try to train the index. This should fail since it was auto-trained when the vectors were added
        response = self.client.post(f"/db/{self.db_name}/train", json={
            "use_two_level_clustering": True,
            "pca_dimension": self.pca_dimension,
            "opq_dimension": self.opq_dimension,
            "compressed_vector_bytes": self.compressed_vector_bytes,
            "omit_opq": True
        })
        print ("test__003_train: ", response.status_code)
        #self.assertTrue(response.status_code == 400)

        tries = 0
        while tries < 50:
            response = self.client.get(f"/db/{self.db_name}/train")
            status = response.json()["status"]
            if status == "complete":
                break
            else:
                tries += 1
                time.sleep(20)

        self.assertEqual(status, "complete")
    

    def test__004_add_while_training(self):
        response = self.client.post(f"/db/{self.db_name}/train", json={
            "use_two_level_clustering": True,
            "pca_dimension": self.pca_dimension,
            "opq_dimension": self.opq_dimension,
            "compressed_vector_bytes": self.compressed_vector_bytes,
            "omit_opq": True
        })
        self.assertTrue(response.status_code == 200)
        time.sleep(5)

        batch_size = 100
        for i in range(0, 2000, batch_size):
            print (i)
            data = []
            for j in range(i, i+batch_size):
                data.append((
                    self.vectors[j],
                    {"text": self.text[j]}
                ))
            response = self.client.post(f"/db/{self.db_name}/add", json={"add_data": data})
        self.assertTrue(response.status_code == 200)

        # Wait for the training to complete
        tries = 0
        while tries < 50:
            response = self.client.get(f"/db/{self.db_name}/train")
            status = response.json()["status"]
            if status == "complete":
                break
            else:
                tries += 1
                time.sleep(40)

        self.assertEqual(status, "complete")

        response = self.client.get(f"/db/{self.db_name}/info")
        self.assertEqual(response.status_code, 200)

        db_info = json.loads(response.json()["db_info"])
        print ("db_info", db_info)
        num_vectors = db_info["num_vectors"]
        n_total = db_info["n_total"]
        num_new_vectors = db_info["num_new_vectors"]
        trained_index_coverage_ratio = db_info["trained_index_coverage_ratio"]
        print ("trained_index_coverage_ratio", trained_index_coverage_ratio)

        self.assertEqual(num_vectors, 32000)
        self.assertEqual(n_total, 32000)
        self.assertEqual(num_new_vectors, 2000)
        self.assertEqual(trained_index_coverage_ratio, 0.9375) # 30,000 / 32,000
    

    def test__005_remove(self):

        # Remove the extra 2000 vectors that were added, since they are just copies of the first 2000 vectors
        batch_size = 1000
        # The ids are 30000 to 31999
        for i in range(30000, 32000, batch_size):
            print (i)
            ids = list(range(i, i+batch_size))
            response = self.client.post(f"/db/{self.db_name}/remove", json={"ids": ids})
        self.assertTrue(response.status_code == 200)

        # Make sure there are no new vectors left after the removal
        response = self.client.get(f"/db/{self.db_name}/info")
        db_info = json.loads(response.json()["db_info"])
        num_new_vectors = db_info["num_new_vectors"]
        self.assertEqual(num_new_vectors, 0)

        # The trained index coverage ratio should be 1 now
        trained_index_coverage_ratio = db_info["trained_index_coverage_ratio"]
        self.assertEqual(trained_index_coverage_ratio, 1.0)


    def test__006_query(self):
        # Test a single query
        response = self.client.post(f"/db/{self.db_name}/query", json={"query_vector": self.queries[0].tolist()})
        self.assertTrue(response.status_code == 200)

    def test__007_full_eval(self):
        # Run a full evaluation. This will tell us if everything is working properly
        recall, latency, all_unique_ids = evaluate(
            self.client, self.db_name, self.queries, self.ground_truths, self.query_k, self.gt_k
        )

        # Set the recall cutoff at above 0.97 and less than 1
        # If recall is above 1, something went wrong
        self.assertGreater(recall, 0.97)
        self.assertLessEqual(recall, 1)

        # Make sure latency is less than 65ms (higher cutoff than the other test since there's an http request)
        self.assertLess(latency, 65)

        # Make sure the length of each unique ID list is equal to the gt_k
        self.assertTrue(all([len(x) == self.gt_k for x in all_unique_ids]))
    

    def test__008_remove_trained_vectors(self):

        # Remove the first 15000 vectors (Done after the full eval so it doesn't mess with recall)
        batch_size = 1000
        for i in range(0, 15000, batch_size):
            print (i)
            ids = list(range(i, i+batch_size))
            response = self.client.post(f"/db/{self.db_name}/remove", json={"ids": ids})
        self.assertTrue(response.status_code == 200)

        # Make sure there are no new vectors left after the removal
        response = self.client.get(f"/db/{self.db_name}/info")
        db_info = json.loads(response.json()["db_info"])
        num_trained_vectors_removed = db_info["num_trained_vectors_removed"]
        self.assertEqual(num_trained_vectors_removed, 15000)

        # The trained index coverage ratio should be 0.5 now
        trained_index_coverage_ratio = db_info["trained_index_coverage_ratio"]
        self.assertEqual(trained_index_coverage_ratio, 0.5)


    """def test__009_tear_down(self):
        response = self.client.post(f"/db/{self.db_name}/delete")
        self.assertTrue(response.status_code == 200)"""
    
    @classmethod
    def tearDownClass(self):
        response = self.client.post(f"/db/{self.db_name}/delete")

if __name__ == "__main__":
    unittest.main()