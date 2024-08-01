import unittest
import faiss

from mindb.utils import training_utils, faiss_utils

class TestGetNumClusters(unittest.TestCase):

    test_cases = [
        (10000, 200),
        (1000000, 6324),
        (100000000, 200000),
    ]
    def test__get_num_clusters(self):
        for num_vectors, expected_num_clusters in self.test_cases:
            with self.subTest(num_vectors=num_vectors, expected_num_clusters=expected_num_clusters):
                num_clusters = training_utils.get_num_clusters(num_vectors)
                self.assertEqual(num_clusters, expected_num_clusters)


class TestGetNProbe(unittest.TestCase):

    test_cases = [
        (200, 100),
        (1000, 250),
        (6350, 444),
        (200000, 6000),
    ]
    def test__get_n_probe(self):
        for num_clusters, expected_n_probe in self.test_cases:
            with self.subTest(num_clusters=num_clusters, expected_n_probe=expected_n_probe):
                n_probe = training_utils.get_n_probe(num_clusters)
                self.assertEqual(n_probe, expected_n_probe)


class TestGetTrainingMemoryUsage(unittest.TestCase):

    ### Test get_training_memory_usage ###
    def test__get_training_memory_usage(self):
        memory_usage = training_utils.get_training_memory_usage(vector_dimension = 768, num_vectors = 100000)
        self.assertEqual(memory_usage, 921600000)


class TestGetNumBatches(unittest.TestCase):

    ### Test get_num_batches ###
    def test__get_num_batches(self):
        num_batches = training_utils.get_num_batches(num_vectors = 1000000, vector_dimension = 768, max_memory_usage = 4*1024*1024*1024)
        self.assertEqual(num_batches, 3)


class TestDetermineOptimalTrainingMethod(unittest.TestCase):

    ### Test determine_optimal_training_method ###
    def test__is_two_level_clustering_optimal__clustering(self):
        # 5M vectors
        use_two_level_clustering = training_utils.is_two_level_clustering_optimal(max_memory_usage = 4*1024*1024*1024, vector_dimension = 768, num_vectors = 5000000)
        self.assertEqual(use_two_level_clustering, True)
    
    def test__is_two_level_clustering_optimal__subsampling(self):
        # 1M vectors
        use_two_level_clustering = training_utils.is_two_level_clustering_optimal(max_memory_usage = 4*1024*1024*1024, vector_dimension = 768, num_vectors = 1000000)
        self.assertEqual(use_two_level_clustering, False)


class TestCalculateTrainedIndexCoverageRatio(unittest.TestCase):

    ### Full coverage ###
    def test__full_coverage(self):

        num_vectors_trained_on = 100_000
        num_new_vectors = 0
        num_trained_vectors_removed = 0

        coverage_ratio = training_utils.calculate_trained_index_coverage_ratio(num_vectors_trained_on, num_new_vectors, num_trained_vectors_removed)

        self.assertEqual(coverage_ratio, 1.0)

    ### Partial coverage ###
    def test__partial_coverage(self):

        num_vectors_trained_on = 100_000
        num_new_vectors = 100_000
        num_trained_vectors_removed = 0

        coverage_ratio = training_utils.calculate_trained_index_coverage_ratio(num_vectors_trained_on, num_new_vectors, num_trained_vectors_removed)

        self.assertEqual(coverage_ratio, 0.5)
    
    ### No coverage ###
    def test__no_coverage(self):

        num_vectors_trained_on = 0
        num_new_vectors = 100_000
        num_trained_vectors_removed = 0

        coverage_ratio = training_utils.calculate_trained_index_coverage_ratio(num_vectors_trained_on, num_new_vectors, num_trained_vectors_removed)

        self.assertEqual(coverage_ratio, 0)
    
    ### Partial coverage with vectors removed ###
    def test__partial_coverage_vectors_removed(self):

        num_vectors_trained_on = 100_000
        num_new_vectors = 0
        num_trained_vectors_removed = 50_000

        coverage_ratio = training_utils.calculate_trained_index_coverage_ratio(num_vectors_trained_on, num_new_vectors, num_trained_vectors_removed)

        self.assertEqual(coverage_ratio, 0.5)
    
    ### Partial coverage with vectors added and removed ###
    def test__partial_coverage_vectors_added_and_removed(self):

        num_vectors_trained_on = 100_000
        num_new_vectors = 60_000
        num_trained_vectors_removed = 20_000

        coverage_ratio = training_utils.calculate_trained_index_coverage_ratio(num_vectors_trained_on, num_new_vectors, num_trained_vectors_removed)

        self.assertEqual(coverage_ratio, 0.5)


class TestCheckIsFlatIndex(unittest.TestCase):
    
    def test__check_is_flat_index__True(self):
        faiss_index = faiss.IndexFlat(768)
        faiss_index = faiss.IndexIDMap(faiss_index)
        is_index_flat = faiss_utils.check_is_flat_index(faiss_index)
        self.assertTrue(is_index_flat)
    
    def test__check_is_flat_index__False(self):
        faiss_index = faiss.index_factory(768, "PCA256,IVF4096,PQ32")
        is_index_flat = faiss_utils.check_is_flat_index(faiss_index)
        self.assertFalse(is_index_flat)
