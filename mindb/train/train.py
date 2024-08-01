import faiss
import logging
import numpy as np

from mindb.train import two_level_clustering
from mindb.utils import lmdb_utils, training_utils


logger = logging.getLogger(__name__)


def train_with_two_level_clustering(uncompressed_vectors_lmdb_path: str, vector_dimension: int, pca_dimension: int, opq_dimension: int, compressed_vector_bytes: int, max_memory_usage: int, omit_opq: bool, lmdb_lock, num_clusters: int = None) -> faiss.IndexPreTransform:

    # Load the vector ids from the LMDB
    with lmdb_lock:
        vector_ids = lmdb_utils.get_lmdb_index_ids(uncompressed_vectors_lmdb_path)
    num_vectors = len(vector_ids)

    # Get the parameters for training the index
    if num_clusters is None:
        num_clusters = training_utils.get_num_clusters(num_vectors)
    index_factory_parameter_string = training_utils.create_index_factory_parameter_string(pca_dimension, opq_dimension, compressed_vector_bytes, num_clusters, vector_dimension, omit_opq)
    logger.info(f'index_factory_parameter_string: {index_factory_parameter_string}')

    # create the index
    faiss_index = faiss.index_factory(
        vector_dimension, index_factory_parameter_string)

    # Train the index
    index = two_level_clustering.train_ivf_index_with_two_level_clustering(
        faiss_index, num_clusters, max_memory_usage, vector_dimension, uncompressed_vectors_lmdb_path, lmdb_lock)
    logger.info(f'index.is_trained: {index.is_trained}')

    index = add_vectors_to_faiss(
        uncompressed_vectors_lmdb_path, index, vector_ids, vector_dimension, max_memory_usage, lmdb_lock)
    logger.info(f'added {index.ntotal} vectors to index')

    # Set the n_probe parameter
    n_probe = training_utils.get_n_probe(num_clusters)
    faiss.ParameterSpace().set_index_parameter(index, "nprobe", n_probe)

    return index, vector_ids


def train_with_subsampling(uncompressed_vectors_lmdb_path: str, vector_dimension: int, pca_dimension: int, opq_dimension: int, compressed_vector_bytes: int, max_memory_usage: int, omit_opq: bool, lmdb_lock, num_clusters: int = None) -> faiss.IndexPreTransform:

    # Load the vector ids from the LMDB
    with lmdb_lock:
        vector_ids = lmdb_utils.get_lmdb_index_ids(uncompressed_vectors_lmdb_path)
    num_vectors = len(vector_ids)

    # Get the parameters for training the index
    if num_clusters is None:
        num_clusters = training_utils.get_num_clusters(num_vectors)
    index_factory_parameter_string = training_utils.create_index_factory_parameter_string(pca_dimension, opq_dimension, compressed_vector_bytes, num_clusters, vector_dimension, omit_opq)
    logger.info(f'index_factory_parameter_string: {index_factory_parameter_string}')

    # Get a subset of the vectors
    memory_usage = training_utils.get_training_memory_usage(
        vector_dimension, num_vectors)
    logger.info(f'memory_usage: {memory_usage}')
    # Define the percentage to train on based off the max memory usage and memory usage
    percentage_to_train_on = min(1, max_memory_usage / memory_usage)
    num_vectors_to_train = int(num_vectors * percentage_to_train_on)
    logger.info(f'num_vectors_to_train: {num_vectors_to_train}')

    # Get a random subset of the vectors
    random_indices = np.random.choice(
        vector_ids, num_vectors_to_train, replace=False)
    with lmdb_lock:
        vectors = lmdb_utils.get_lmdb_vectors_by_ids(
            uncompressed_vectors_lmdb_path, random_indices)
    
    if vectors is None:
        logger.debug("returning None, None")
        return None, None

    # create the index
    index = faiss.index_factory(
        vector_dimension, index_factory_parameter_string)
    index.train(vectors)

    index = add_vectors_to_faiss(
        uncompressed_vectors_lmdb_path, index, vector_ids, vector_dimension, max_memory_usage, lmdb_lock)
    if index is not None:
        logger.info(f'added {index.ntotal} vectors to index')

    if index is None:
        logger.debug("returning None, None")
        return None, None

    # Set the n_probe parameter (I think it makes sense here since n_probe is dependent on num_clusters)
    n_probe = training_utils.get_n_probe(num_clusters)
    faiss.ParameterSpace().set_index_parameter(index, "nprobe", n_probe)

    return index, vector_ids


def train_small_index(uncompressed_vectors_lmdb_path: str, vector_dimension: int, max_memory_usage: int, lmdb_lock) -> faiss.IndexPreTransform:
    # This won't actually train the index, it will create a flat index and add the vectors to it

    with lmdb_lock:
        vector_ids = lmdb_utils.get_lmdb_index_ids(uncompressed_vectors_lmdb_path)

    index = faiss.IndexFlat(vector_dimension)
    faiss_index = faiss.IndexIDMap(index)
    faiss_index = add_vectors_to_faiss(uncompressed_vectors_lmdb_path, faiss_index, vector_ids, vector_dimension, max_memory_usage, lmdb_lock)

    return faiss_index


def add_vectors_to_faiss(uncompressed_vectors_lmdb_path: str, index: faiss.IndexPreTransform, vector_ids: list, vector_dimension: int, max_memory_usage: int, lmdb_lock) -> faiss.IndexPreTransform:
    
    num_vectors = len(vector_ids)
    # Add all of the vectors to the index. We need to know the number of batches to do this in
    num_batches = training_utils.get_num_batches(num_vectors, vector_dimension, max_memory_usage)
    # Calculate the number of vectors per batch
    num_per_batch = np.ceil(num_vectors / num_batches).astype(int)
    for i in range(num_batches):
        # Get the batch ids (based off the number of batches and the current i)
        batch_ids = vector_ids[i *num_per_batch: (min((i + 1) * num_per_batch, num_vectors))]
        with lmdb_lock:
            vectors = lmdb_utils.get_lmdb_vectors_by_ids(
                uncompressed_vectors_lmdb_path, batch_ids)
        # Make sure the vectors are not None. This happens if the lmdb folder is not found
        if vectors is None:
            logger.debug("returning None")
            return None
        
        # Add the vectors to the index         
        index.add_with_ids(vectors, batch_ids)

    return index