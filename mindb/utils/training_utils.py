import numpy as np
from mindb.utils.faiss_utils import check_is_flat_index
from mindb.train import training_params

def get_num_clusters(num_vectors: int) -> int:
    # Get the number of clusters to use for the IVF index, based on the number of vectors
    scaling_factor = 0.2
    num_clusters = int((num_vectors**0.75) * scaling_factor)
    return num_clusters

def get_n_probe(num_clusters: int) -> int:
    # Get the number of probes to use for the IVF index, based on the number of clusters
    # This is a piecewise linear function, based on the log of the number of clusters

    log_num_clusters = np.log(num_clusters)

    # Define the piecewise linear function breakpoints and y values
    x_breakpoints = [np.log(200), np.log(1000), np.log(6350), np.log(200000)]
    y_values = [0.5, 0.25, 0.07, 0.03]

    if log_num_clusters <= x_breakpoints[0]:
        n_probe_factor = y_values[0]
    elif log_num_clusters <= x_breakpoints[1]:
        n_probe_factor = np.interp(log_num_clusters, [x_breakpoints[0], x_breakpoints[1]], [y_values[0], y_values[1]])
    elif log_num_clusters <= x_breakpoints[2]:
        n_probe_factor = np.interp(log_num_clusters, [x_breakpoints[1], x_breakpoints[2]], [y_values[1], y_values[2]])
    elif log_num_clusters <= x_breakpoints[3]:
        n_probe_factor = np.interp(log_num_clusters, [x_breakpoints[2], x_breakpoints[3]], [y_values[2], y_values[3]])
    else:
        n_probe_factor = y_values[3]

    return int(n_probe_factor * num_clusters)

def create_index_factory_parameter_string(pca_dimension: int, opq_dimension: int, compressed_vector_bytes: int, num_clusters: int, vector_dimension: int, omit_opq: bool) -> str:

    index_factory_parameters = []
    
    if pca_dimension is not None:
        index_factory_parameters.append(f'PCA{pca_dimension}')
    else:
        # If there is no PCA or OPQ, then the Faiss index is no longer an IndexPreTransform object, which would cause failures
        index_factory_parameters.append(f'PCA{vector_dimension}')
    
    # If omit_opq is True, then we don't want to include OPQ in the index factory parameter string
    if not omit_opq and compressed_vector_bytes is not None:
        index_factory_parameters.append(f'OPQ{compressed_vector_bytes}_{opq_dimension}')
    
    index_factory_parameters.append(f'IVF{num_clusters}')

    if compressed_vector_bytes is not None:
        index_factory_parameters.append(f'PQ{compressed_vector_bytes}')
    else:
        index_factory_parameters.append('Flat')

    index_factory_parameter_string = ','.join(index_factory_parameters)
    return index_factory_parameter_string

def get_training_memory_usage(vector_dimension: int, num_vectors: int) -> int:
    # 1M 768 dimension vectors uses ~10GB of memory
    memory_usage = int(num_vectors * vector_dimension * 4 * 3) # 4 bytes per float, with a 3x multiplier for overhead
    return memory_usage

def get_num_batches(num_vectors: int, vector_dimension: int, max_memory_usage: int) -> int:
    memory_usage = num_vectors * vector_dimension * 4
    # We don't really need to push memory requirements here, so we'll just use 1/4 of the max memory usage
    num_batches = int(np.ceil(memory_usage / (max_memory_usage / 4)))
    return num_batches

def get_num_vectors_per_batch(max_memory_usage: int, vector_dimension: int) -> int:
    # This does basically the same thing as get_num_batches, but returns the number of vectors per batch
    num_vectors = int(max_memory_usage / (vector_dimension * 4 * 3)) # 4 bytes per float, plus 3x for overhead
    # We don't really need to push memory requirements here, so we'll just use 1/4 of the max memory usage
    return int(num_vectors/4)

def is_two_level_clustering_optimal(max_memory_usage: int, vector_dimension: int, num_vectors: int) -> bool:

    memory_usage = get_training_memory_usage(vector_dimension, num_vectors)
    max_num_vectors = int((max_memory_usage / memory_usage) * num_vectors)
    num_clusters = get_num_clusters(num_vectors)
    num_vectors_per_cluster = int(max_num_vectors / num_clusters)

    # faiss recommends a minimum of 39 vectors per cluster
    if num_vectors_per_cluster < 39:
        # We need to use the clustering method
        return True
    else:
        # We can use the subsampling method
        return False

def get_default_faiss_params(vector_dimension: int) -> dict:
    if vector_dimension < 150:
        return {
            "pca_dimension": max(64, vector_dimension),
            "opq_dimension": max(64, vector_dimension),
            "compressed_vector_bytes": 16,
        }
    elif vector_dimension < 300:
        return {
            "pca_dimension": 128,
            "opq_dimension": 64,
            "compressed_vector_bytes": 16,
        }
    # e5-small, miniLM
    elif vector_dimension < 600:
        return {
            "pca_dimension": 256,
            "opq_dimension": 128,
            "compressed_vector_bytes": 32,
        }
    # e5-base, Cohere multilingual
    elif vector_dimension < 1000:
        return {
            "pca_dimension": 256,
            "opq_dimension": 128,
            "compressed_vector_bytes": 32,
        }
    # e5-large, OpenAI Ada
    elif vector_dimension < 2000:
        return {
            "pca_dimension": 512,
            "opq_dimension": 256,
            "compressed_vector_bytes": 32,
        }
    else:
        return {
            "pca_dimension": 1024,
            "opq_dimension": 512,
            "compressed_vector_bytes": 128,
        }

def calculate_trained_index_coverage_ratio(num_vectors_trained_on: int, num_new_vectors: int, num_trained_vectors_removed: int) -> float:

    if num_vectors_trained_on == 0:
        return 0

    # Number of vectors trained that are still in the index
    num_trained_vectors_left = num_vectors_trained_on - num_trained_vectors_removed

    # Total number of vectors in the index
    num_total_vectors = num_vectors_trained_on + num_new_vectors

    # Calculate the coverage ratio
    coverage_ratio = num_trained_vectors_left / num_total_vectors

    return coverage_ratio


def check_needs_initial_training(db_name: str, num_vectors: int, faiss_index, operations: dict) -> bool:
    
    # If there are fewer than 50k vectors, we don't need to train
    if num_vectors < training_params.num_vector_training_cutoff:
        #print ("num vectors is less than 50k", num_vectors)
        return False
    
    # If there is already an index, we don't need to train
    is_flat_index = check_is_flat_index(faiss_index)
    if not is_flat_index:
        print ("index already trained", type(faiss_index))
        return False

    # Check if there is a training operation already in progress for this database
    if db_name in operations:
        if operations[db_name] == "in progress" or operations[db_name] == "training":
            print ("training already in progress", operations[db_name])
            return False
    
    return True



def check_needs_training(db_name: str, num_vectors: int, operations: dict, index_coverage_ratio: float) -> bool:
    
    # If there are fewer than 50k vectors, we don't need to train
    if num_vectors < training_params.num_vector_training_cutoff:
        print ("num vectors is less than 50k", num_vectors)
        return False
    
    if index_coverage_ratio > training_params.trained_index_coverage_ratio_cutoff:
        print ("index coverage ratio is greater than 0.5", index_coverage_ratio)
        return False

    # Check if there is a training operation already in progress for this database
    if db_name in operations:
        if operations[db_name] == "in progress" or operations[db_name] == "training":
            print ("training already in progress", operations[db_name])
            return False
    
    return True


def get_training_params(max_memory_usage: int, vector_dimension: int, num_vectors: int) -> dict:
    
    use_two_level_clustering = is_two_level_clustering_optimal(max_memory_usage, vector_dimension, num_vectors)
    pca_dimension = training_params.pca_dimension
    omit_opq = training_params.omit_opq
    opq_dimension = training_params.opq_dimension
    compressed_vector_bytes = training_params.compressed_vector_bytes

    return {
        "use_two_level_clustering": use_two_level_clustering,
        "pca_dimension": pca_dimension,
        "omit_opq": omit_opq,
        "opq_dimension": opq_dimension,
        "compressed_vector_bytes": compressed_vector_bytes,
    }
