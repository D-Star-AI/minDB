import numpy as np
import re
from mindb.utils import training_utils


def validate_database_name(name: str) -> tuple[bool, str]:
    # Make sure the DB name is valid. It must be valid for a file name
    name_regex = r'^[a-zA-Z0-9_ -]+$'
    if not re.match(name_regex, name):
        return False, "The name is invalid. It must only contain alphanumeric characters, spaces, underscores, and hyphens."
    else:
        return True, ""


def validate_train(vector_dimension: int, pca_dimension: int, opq_dimension: int, compressed_vector_bytes: int) -> tuple[bool, str]:

    # If the vector dimension is not set, that means there are no vectors in the database
    if vector_dimension == None:
        return False, "No vectors have been added to the database"
    
    if compressed_vector_bytes is None and opq_dimension is not None:
        return False, "compressed_vector_bytes must be set if opq_dimension is set"

    # Make sure pca, pq_bytes, and opq_dimension are integers and are all positive
    if pca_dimension is not None and not isinstance(pca_dimension, int):
        return False, "pca_dimension is not the correct type. Expected type: int. Actual type: " + str(type(pca_dimension))
    if opq_dimension is not None and not isinstance(opq_dimension, int):
        return False, "opq_dimension is not the correct type. Expected type: int. Actual type: " + str(type(opq_dimension))
    if compressed_vector_bytes is not None and not isinstance(compressed_vector_bytes, int):
        return False, "compressed_vector_bytes is not the correct type. Expected type: int. Actual type: " + str(type(compressed_vector_bytes))
    
    if pca_dimension is not None and pca_dimension < 1:
        return False, "pca_dimension is not positive. pca_dimension: " + str(pca_dimension)
    if opq_dimension is not None and opq_dimension < 1:
        return False, "opq_dimension is not positive. opq_dimension: " + str(opq_dimension)
    if compressed_vector_bytes is not None and compressed_vector_bytes < 1:
        return False, "compressed_vector_bytes is not positive. compressed_vector_bytes: " + str(compressed_vector_bytes)

    # Make sure PCA is less than the number of columns in the data
    if pca_dimension is not None and pca_dimension > vector_dimension:
        return False, "pca_dimension is larger than the number of columns in the data. Number of columns in data: " + str(vector_dimension) + " pca_dimension: " + str(pca_dimension)
    
    # OPQ has to be less than or equal to PCA
    if (opq_dimension is not None and pca_dimension is not None) and opq_dimension > pca_dimension:
        return False, "opq_dimension is larger than pca_dimension. pca_dimension: " + str(pca_dimension) + " opq_dimension: " + str(opq_dimension)
    
    # opq_dimension has to be divisible by compressed_vector_bytes
    if opq_dimension is not None and opq_dimension % compressed_vector_bytes != 0:
        return False, "opq_dimension is not divisible by compressed_vector_bytes. opq_dimension: " + str(opq_dimension) + " compressed_vector_bytes: " + str(compressed_vector_bytes)
    
    return True, "Success"


def validate_add(data, vector_dimension: int, num_vectors: int, max_memory_usage: int, is_flat_index: bool) -> tuple[np.ndarray, list, bool, str]:

    # Make sure the data is a list
    if not isinstance(data, list):
        return [], [], False, "Data is not the correct type. Expected type: list. Actual type: " + str(type(data))
    
    # Make sure each element in the data is a tuple
    for item in data:
        if not isinstance(item, tuple):
            return [], [], False, "List item is not the correct type. Expected type: tuple. Actual type: " + str(type(item))
    
    vectors = [item[0] for item in data]
    metadata = [item[1] for item in data]

    if len(vectors) < 1:
        return [], [], False, "There are no vectors in the data"

    # Make sure the vector dimension is greater than 0
    if vector_dimension == None:
        if len(vectors[0]) == 0:
            return [], [], False, "Vector dimension cannot be 0"

    # Double check that the vector is the right type
    for i,vector in enumerate(vectors):
        # Make sure the vector is a numpy array or list. If it's a list, convert it to a numpy array
        if isinstance(vector, list):
            vector = np.array(vector, dtype=np.float32)
            vectors[i] = vector
        if not isinstance(vector, np.ndarray):
            return [], [], False, "Vector is not the correct type. Expected type: numpy array or list. Actual type: " + str(type(vector))
        
        # Make sure each vector is the right size
        if len(vector.shape) != 1:
            # If the vector is a 2D array, make sure it's a single row (so (1, 768) is ok, but (2, 768) is not)
            if vector.shape[0] != 1 and vector.shape[1] != 1:
                return [], [], False, "Each vector should be a single array. Actual size: " + str(vector.shape)
            vector = np.squeeze(vector)
            vectors[i] = vector
        
        if vector_dimension != None and vector.shape[0] != vector_dimension:
            return [], [], False, "Vector is not the correct size. Expected size: " + str(vector_dimension) + " Actual size: " + str(vector.shape[0])
    
    # Normalize all of the vectors (no need to check first, since we are just dividing by 1 if the vector is already normalized)
    for i,vector in enumerate(vectors):
        vector = vector / np.linalg.norm(vector)
        vectors[i] = vector
            
    if is_flat_index:
        # Make sure adding the vectors won't exceed the max memory usage
        new_memory_usage = training_utils.get_training_memory_usage(vectors[0].shape[0], num_vectors + len(vectors))
        if (max_memory_usage is not None and new_memory_usage > max_memory_usage):
            return [], [], False, "Adding these vectors will exceed the max memory usage. Max memory usage: " + str(max_memory_usage) + " New memory usage: " + str(new_memory_usage)
    
    # Convert the vectors from a list to a numpy array
    vectors = np.array(vectors, dtype=np.float32)

    return vectors, metadata, True, "Success"


def validate_remove(ids: np.ndarray) -> tuple[bool, str]:

    # Make sure the data is the correct type (numpy array)
    if not isinstance(ids, np.ndarray):
        return False, "IDs are not the correct type. Expected type: numpy array. Actual type: " + str(type(ids))
    
    # Make sure the IDs are integers
    if not np.issubdtype(ids.dtype, np.integer):
        return False, "IDs are not integers. IDs: " + str(ids.dtype)
    
    # Make sure the IDs are positive
    if np.any(ids < 0):
        # This won't actually cause an error, but it should never happen so we want to warn the user
        return False, "Negative IDs found. All IDs must be positive"
    
    # Make sure the data is a 1D array
    if len(ids.shape) != 1:
        return False, "IDs are not 1D. IDs: " + str(ids.shape)
    
    return True, "Success"


def validate_query(query_vector: np.ndarray, vector_dimension: int) -> tuple[bool, str]:

    # Make sure the data is the correct type (numpy array)
    if not isinstance(query_vector, np.ndarray):
        return False, "Query vectors are not the correct type. Expected type: numpy array. Actual type: " + str(type(query_vector))
    
    if len(query_vector.shape) == 1:
        query_vector = query_vector.reshape((-1, vector_dimension))
    
    # Make sure the query vector is the correct size. The 
    if vector_dimension != None and query_vector.shape[1] != vector_dimension:
        return False, "Query vector is not the correct size. Expected size: " + str(vector_dimension) + " Actual size: " + str(query_vector.shape[1])

    return True, "Success"
