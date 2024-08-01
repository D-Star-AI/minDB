from fastapi.testclient import TestClient # requires httpx
import sys
import os
import numpy as np
import time
import unittest
import json

from helpers import fiqa_test_data

FILE_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(FILE_PATH, '../../'))

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
        self.db_name = "fiqa_test"
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
    

    def test__001_add_while_training(self):

        vectors = self.vectors
        text = self.text

        # 2x the vectors list (just to have a larger dataset)
        #vectors.extend(vectors)
        #text.extend(text)

        # Using different sleep times to test different scenarios
        # One for adding all vectors while training is running, but before it is complete
        # One for adding vectors both during training, and after training is complete
        # One for adding vectors only after training is complete
        for sleep_time in [5, 25, 45]:

            ### Create a new database ###
            response = self.client.post("/db/create", json={"name": self.db_name})
            print (response.text)
            self.assertTrue(response.status_code == 200)


            ### Add the data ###
            
            batch_size = 1000
            for i in range(0, len(vectors), batch_size):
                data = []
                for j in range(i, i+batch_size):
                    data.append((vectors[j], {"text": text[j]}))
                response = self.client.post(f"/db/{self.db_name}/add", json={"add_data": data})
            self.assertTrue(response.status_code == 200)


            ### Train the index ###
            response = self.client.post(f"/db/{self.db_name}/train", json={
                "use_two_level_clustering": True,
                "pca_dimension": self.pca_dimension,
                "opq_dimension": self.opq_dimension,
                "compressed_vector_bytes": self.compressed_vector_bytes,
                "omit_opq": True
            })
            print ("response.status_code", response.status_code)
            #self.assertTrue(response.status_code == 200)

            time.sleep(sleep_time) # Sleep long enough for the training to be mostly complete
            # This way we are adding vectors when the training process completes

            batch_size = 500
            for i in range(0, 10000, batch_size):
                data = []
                for j in range(i, i+batch_size):
                    data.append((vectors[j], {"text": text[j]}))
                response = self.client.post(f"/db/{self.db_name}/add", json={"add_data": data})
            self.assertTrue(response.status_code == 200)

            # Wait for the training to complete
            tries = 0
            while tries < 30:
                response = self.client.get(f"/db/{self.db_name}/train")
                status = response.json()["status"]
                if status == "complete":
                    break
                else:
                    tries += 1
                    time.sleep(10)

            self.assertEqual(status, "complete")

            time.sleep(5)

            response = self.client.get(f"/db/{self.db_name}/info")
            self.assertEqual(response.status_code, 200)

            # Make sure there are 40000 vectors in the database, both for num_vectors and n_total (which is the faiss index)
            db_info = json.loads(response.json()["db_info"])
            self.assertEqual(db_info["num_vectors"], 40000)
            self.assertEqual(db_info["n_total"], 40000)



            ### Remove ###
            # This is necessary in order to run the full eval. Otherwise the duplicate data will cause issues

            # Remove the extra vectors that were added, since they are just copies of the first vectors
            batch_size = 1000
            # The ids are 30,000 to 69,999
            for i in range(30000, 40000, batch_size):
                print (i)
                ids = list(range(i, i+batch_size))
                response = self.client.post(f"/db/{self.db_name}/remove", json={"ids": ids})
            self.assertTrue(response.status_code == 200)



            ### Full eval ###
            # Run a full evaluation. This will tell us if everything is working properly
            recall, latency, all_unique_ids = evaluate(
                self.client, self.db_name, self.queries, self.ground_truths, self.query_k, self.gt_k
            )

            # Set the recall cutoff at above 0.97 and less than 1
            # If recall is above 1, something went wrong
            self.assertGreater(recall, 0.97)
            self.assertLessEqual(recall, 1)

            # Make sure latency is less than 25ms (higher cutoff than the other test since there's an http request)
            self.assertLess(latency, 25)

            # Make sure the length of each unique ID list is equal to the gt_k
            self.assertTrue(all([len(x) == self.gt_k for x in all_unique_ids]))


            ### Delete the DB
            response = self.client.post(f"/db/{self.db_name}/delete")
            self.assertTrue(response.status_code == 200)
    


    def test__002_remove_while_training(self):

        vectors = self.vectors
        text = self.text

        new_db_name = "fiqa_test_remove"

        ### Create a new database ###
        response = self.client.post("/db/create", json={"name": new_db_name})
        self.assertTrue(response.status_code == 200)


        ### Add the data (just below the auto train cutoff) ###
        batch_size = 1000
        for i in range(0, 24000, batch_size):
            data = []
            for j in range(i, i+batch_size):
                data.append((vectors[j], {"text": text[j]}))
            response = self.client.post(f"/db/{new_db_name}/add", json={"add_data": data})
        self.assertTrue(response.status_code == 200)


        ### Train the index ###
        response = self.client.post(f"/db/{new_db_name}/train", json={
            "use_two_level_clustering": True,
            "pca_dimension": self.pca_dimension,
            "opq_dimension": self.opq_dimension,
            "compressed_vector_bytes": self.compressed_vector_bytes,
            "omit_opq": True
        })
        self.assertTrue(response.status_code == 200)

        time.sleep(5)

        ### Remove vectors while training ###
        batch_size = 1000
        for i in range(0, 4000, batch_size):
            print (i)
            ids = list(range(i, i+batch_size))
            response = self.client.post(f"/db/{new_db_name}/remove", json={"ids": ids})
        self.assertTrue(response.status_code == 200)

        # There should be 24000 vectors total, but only 20000 in the faiss index
        # This is because we remove the vectors from faiss, but not from LMDB when a training operation is happening
        response = self.client.get(f"/db/{new_db_name}/info")
        db_info = json.loads(response.json()["db_info"])

        n_total = db_info["n_total"]
        num_vectors = db_info["num_vectors"]
        self.assertEqual(n_total, 20000)
        self.assertEqual(num_vectors, 24000)


        # Wait for the training to complete
        tries = 0
        while tries < 30:
            response = self.client.get(f"/db/{new_db_name}/train")
            status = response.json()["status"]
            if status == "complete":
                break
            else:
                tries += 1
                time.sleep(10)

        self.assertEqual(status, "complete")

        # Once the training is complete, wait a little so the vectors can be removed
        time.sleep(5)

        # Get the db info again
        response = self.client.get(f"/db/{new_db_name}/info")
        self.assertEqual(response.status_code, 200)

        # There should be 20000 vectors total, and 20000 in the faiss index
        db_info = json.loads(response.json()["db_info"])
        print ("db_info", db_info)

        n_total = db_info["n_total"]
        num_vectors = db_info["num_vectors"]
        self.assertEqual(n_total, 20000)
        self.assertEqual(num_vectors, 20000)

    
    @classmethod
    def tearDownClass(self):
        db_name = "fiqa_test_remove"
        response = self.client.post(f"/db/{db_name}/delete")
        # This can fail since the DB should have already deleted, so we can't assert a status
        # But it's fine if it fails, since we just want to make sure the DB is deleted

if __name__ == "__main__":
    unittest.main()