import unittest
import numpy as np

import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../../")

from mindb.utils import input_validation

class TestNameInputParameters(unittest.TestCase):

    valid_database_names = [
        "test_db",
        "test-db-2"
        "Test db"
    ]

    invalid_database_names = [
        "test.db",
        "#test_db"
        "test/db"
    ]

    ### Test valid database names ###
    def test__validate_database_name__valid_names(self):
        for database_name in self.valid_database_names:
            is_valid, _ = input_validation.validate_database_name(database_name)
            self.assertTrue(is_valid)
    
    ### Test invalid database names ###
    def test__validate_database_name__invalid_names(self):
        for database_name in self.invalid_database_names:
            is_valid, _ = input_validation.validate_database_name(database_name)
            self.assertFalse(is_valid)


class TestTrainInputParameters(unittest.TestCase):

    # vector_dimension, pca, opq, pq
    valid_train_parameters = [
        (768, 256, 128, 32),
        (1024, 256, 256, 64),
        (512, 512, 256, 64),
        (768, None, None, None),
    ]

    invalid_train_parameters = [
        (None, 128, 100, 32, "No vectors have been added to the database"),
        (768, None, 128, None, "compressed_vector_bytes must be set if opq_dimension is set"),
        (768, 128.3, 100, 32, "pca_dimension is not the correct type. Expected type: int. Actual type"),
        (768, '128', 100, 32, "pca_dimension is not the correct type. Expected type: int. Actual type"),
        (768, 1024, 128, 32, "pca_dimension is larger than the number of columns in the data. Number of columns in data"),
        (768, 128, 256, 32, "opq_dimension is larger than pca_dimension"),
        (768, 128, 100, 32, "opq_dimension is not divisible by compressed_vector_bytes")
    ]

    def test__validate_train__valid_parameters(self):
        for vector_dimension, pca_dimension, opq_dimension, compressed_vector_bytes in self.valid_train_parameters:
            is_valid, _ = input_validation.validate_train(vector_dimension, pca_dimension, opq_dimension, compressed_vector_bytes)
            self.assertTrue(is_valid)

    def test__validate_train__invalid_parameters(self):
        for vector_dimension, pca_dimension, opq_dimension, compressed_vector_bytes, expected_reason in self.invalid_train_parameters:
            is_valid, reason = input_validation.validate_train(vector_dimension, pca_dimension, opq_dimension, compressed_vector_bytes)
            self.assertFalse(is_valid)
            self.assertTrue(expected_reason in reason)


class TestAddInputParameters(unittest.TestCase):

    # Create a valid numpy array, an invalid one, and a text list
    num_vectors = 10
    vector_dimension = 768
    vector_array = np.random.rand(1, vector_dimension)
    vector_array_inverted_dimensions = np.random.rand(vector_dimension, 1)
    invalid_array = np.random.rand(1, 512)
    # Normalize these vectors
    norm_vector_array = vector_array / np.linalg.norm(vector_array, axis=1)[:, None]
    norm_invalid_array = invalid_array / np.linalg.norm(invalid_array, axis=1)[:, None]
    # Reshape a vector from (1, 768) to (768, 1)
    norm_vector_array_inverted_dimensions = norm_vector_array.reshape(vector_dimension, 1)
    max_memory = 4 * 1024 * 1024 * 1024
    metadata = [{"text": "test"}]

    valid_data = [(norm_vector_array, metadata)] * num_vectors
    valid_data_list = [(norm_vector_array.tolist(), metadata)] * num_vectors
    invalid_data = [(norm_invalid_array, metadata)] * num_vectors
    invalid_data_multiple_arrays = [(np.random.rand(2, vector_dimension), metadata)] * num_vectors
    non_normalized_data = [(invalid_array, metadata)] * num_vectors

    valid_add_parameters = [
        (valid_data, vector_dimension, num_vectors, max_memory, False),
        (valid_data, None, num_vectors, max_memory, False),
        (valid_data_list, vector_dimension, num_vectors, max_memory, False),
        (valid_data_list, None, num_vectors, max_memory, False),
        ([(norm_vector_array_inverted_dimensions, metadata)*num_vectors], vector_dimension, num_vectors, max_memory, False),
        (non_normalized_data, None, num_vectors, max_memory, False),
    ]

    invalid_add_parameters = [
        (invalid_data, vector_dimension, num_vectors, max_memory, False, "Vector is not the correct size. Expected size"),
        (invalid_data_multiple_arrays, vector_dimension, num_vectors, max_memory, False, "Each vector should be a single array. Actual size"),
        (valid_data, vector_dimension, num_vectors, 1, True, "Adding these vectors will exceed the max memory usage. Max memory usage"),
    ]

    def test__validate_add__valid_parameters(self):
        for data, vector_dimension, num_vectors, max_memory, is_flat_index in self.valid_add_parameters:
            vectors, _, is_valid, reason = input_validation.validate_add(data, vector_dimension, num_vectors, max_memory, is_flat_index)
            self.assertTrue(is_valid)
            self.assertTrue(type(vectors[0][0]) == np.float32)

    def test__validate_add__invalid_parameters(self):
        for data, vector_dimension, num_vectors, max_memory, is_flat_index, expected_reason in self.invalid_add_parameters:
            _, _, is_valid, reason = input_validation.validate_add(data, vector_dimension, num_vectors, max_memory, is_flat_index)
            self.assertFalse(is_valid)
            self.assertTrue(expected_reason in reason)


class TestRemoveInputParameters(unittest.TestCase):

    valid_remove_parameters = [
        np.random.randint(0, 100, 10)
    ]

    invalid_remove_parameters = [
        (np.array([1.2, 2.3, 3.4, 4.5, 5.6]), "IDs are not integers"),
        (np.array([-1, -2, 0, 1, 2]), "Negative IDs found. All IDs must be positive"),
        (np.random.randint(0, 100, (10, 768)), "IDs are not 1D.")
    ]

    def test__validate_remove__valid_parameters(self):
        for ids in self.valid_remove_parameters:
            is_valid, _ = input_validation.validate_remove(ids)
            self.assertTrue(is_valid)

    def test__validate_remove__invalid_parameters(self):
        for ids, expected_reason in self.invalid_remove_parameters:
            is_valid, reason = input_validation.validate_remove(ids)
            self.assertFalse(is_valid)
            self.assertTrue(expected_reason in reason)

if __name__ == '__main__':
    unittest.main()