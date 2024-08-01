import faiss
import logging
import numpy as np
import json
import os
import shutil
import threading
from faiss.contrib.exhaustive_search import knn

from mindb.utils import lmdb_utils, training_utils, faiss_utils, query_utils, input_validation
from mindb.train import train


def configure_logging(level=logging.INFO):
    logging.basicConfig(
        level=level,
        format='%(asctime)s %(name)s:%(lineno)d in %(funcName)s() %(levelname)-8s %(message)s',
        datefmt='%m-%d %H:%M:%S'
    )

configure_logging()
logger = logging.getLogger(__name__)


def get_mindb_path(name: str, save_path: str):
    """
    Get the path to the minDB directory.

    :param name: The name of the database.
    :param save_path: The path where the database files will be saved. Defaults to ~/.mindb/{name}. If this directory does not exist, it will be created.
    """
    # Set the save path to the current directory if it is not specified
    if save_path is None:
        save_path = os.path.join(os.path.expanduser("~"), '.mindb', name)
    return save_path


class minDB:
    """
    A class representing a searchable database using Faiss and LMDB for efficient storage and retrieval of text and vectors.
    """
    def __init__(self, name: str, save_path: str = None, vector_dimension: int = None, max_memory_usage: int = 4*1024*1024*1024, LMDB_MAP_SIZE: int = 100*1024*1024*1024, logging_level: int = logging.INFO, create_or_load: str ='create'):
        """
        Initialize the minDB object.

        :param name: The name of the database.
        :param save_path: The path where the database files will be saved. Defaults to ~/.mindb/{name}. If this directory does not exist, it will be created.
        :param vector_dimension: The dimension of the vectors to be stored in the database. 
        :param max_memory_usage: The maximum memory usage allowed for the construction and querying of the database, in bytes. Defaults to 4 GB.
        """
        self.name = name
        self._faiss_lock = threading.Lock()
        self._lmdb_lock = threading.Lock()
        self._save_path = get_mindb_path(name, save_path)
        self.LMDB_MAP_SIZE = LMDB_MAP_SIZE

        if create_or_load == 'create':
            self._initialize_new_db(name, vector_dimension, max_memory_usage)
        else:
            self._initialize_from_config()

    def _initialize_new_db(self, name: str, vector_dimension: int, max_memory_usage: int) -> None:

        logger.info('Initializing minDB')
        self._vector_dimension = vector_dimension
        self.max_id = -1
        self.max_memory_usage = max_memory_usage
        self.faiss_index = None
        self.training_params = None

        self.max_trained_id = 0
        self.num_vectors_trained_on = 0
        self.num_new_vectors = 0
        self.num_trained_vectors_removed = 0

        # Create the save directory if it doesn't exist and return an exception if it already does exist
        if os.path.exists(self.save_path):
            raise Exception(f"Database with name {self.name} already exists. Please choose a different name.")

        os.makedirs(self.save_path)
        logger.info(f'Initializing minDB with name {name} at {self.save_path}')

        # set the lmdb path
        self._lmdb_path = os.path.join(self.save_path, 'lmdb')

        # create the lmdb databases
        self._lmdb_uncompressed_vectors_path = lmdb_utils.create_lmdb(self.lmdb_path, 'uncompressed_vectors')
        logger.info(f'lmdb uncompressed vectors database location: {self.lmdb_uncompressed_vectors_path}')
        self._lmdb_metadata_path = lmdb_utils.create_lmdb(self.lmdb_path, 'metadata')
        logger.info(f'lmdb text database location: {self.lmdb_metadata_path}')

        self.save()

    def _initialize_from_config(self) -> None:
        
        config_params = self.read_config_params()
        self._vector_dimension = config_params["vector_dimension"]
        self.max_id = config_params["max_id"]
        self.max_memory_usage = config_params["max_memory_usage"]
        self.LMDB_MAP_SIZE=config_params["LMDB_MAP_SIZE"]

        self.max_trained_id = config_params["max_trained_id"]
        self.num_vectors_trained_on = config_params["num_vectors_trained_on"]
        self.num_new_vectors = config_params["num_new_vectors"]
        self.num_trained_vectors_removed = config_params["num_trained_vectors_removed"]
        self.training_params = config_params["training_params"]

        # set the lmdb path
        self._lmdb_path = os.path.join(self.save_path, 'lmdb')
        self._lmdb_uncompressed_vectors_path = os.path.join(self.lmdb_path, 'uncompressed_vectors')
        self._lmdb_metadata_path = os.path.join(self.lmdb_path, 'metadata')

        # load faiss index from save path
        try:
            self.faiss_index = faiss.read_index(os.path.join(self.save_path, 'faiss_index.index'))
        except:
            self.faiss_index = None

    @property
    def vector_dimension(self):
        """
        The dimension of vectors in the database.
        """
        return self._vector_dimension

    @property
    def save_path(self):
        """
        The path where the associated database files will be saved on disk.
        """
        return self._save_path

    @property
    def lmdb_path(self):
        return self._lmdb_path

    @property
    def lmdb_uncompressed_vectors_path(self):
        return self._lmdb_uncompressed_vectors_path

    @property
    def lmdb_metadata_path(self):
        return self._lmdb_metadata_path
    
    @property
    def num_vectors(self):
        """
        Get the number of vectors in the database.
        """
        with self._lmdb_lock:
            return lmdb_utils.get_db_count(self.lmdb_uncompressed_vectors_path)
    
    @property
    def trained_index_coverage_ratio(self):
        """
        Get the coverage ratio of the trained index.
        """

        return training_utils.calculate_trained_index_coverage_ratio(
            self.num_vectors_trained_on, self.num_new_vectors, self.num_trained_vectors_removed)

    def add(self, data: list[tuple[np.ndarray, dict]], add_to_new_faiss_index: bool = False) -> list:
        """
        Add vectors and their corresponding text to the database.

        :param data: A list of tuples containing a vector and the associated metadata.
        
        NOTE: The metadata must be a dictionary
        """

        # Check if the index is flat or not
        is_flat_index = faiss_utils.check_is_flat_index(self.faiss_index)

        # Validate the inputs
        vectors, metadata, is_valid, reason = input_validation.validate_add(
            data, self.vector_dimension, self.num_vectors, self.max_memory_usage, is_flat_index)
        if not is_valid:
            raise ValueError(reason)
                
        if is_flat_index:
            # Check the number of vectors in the index
            if self.faiss_index.ntotal + vectors.shape[0] >= 50000:
                # Show a warning message
                logger.warning('The number of vectors in the index is greater than 50k. Please train your index for faster performance.')

        # Check if the index is still None (This should only happen on the first add operation)
        if self.faiss_index is None:
            index = faiss.IndexFlat(vectors.shape[1])
            faiss_index = faiss.IndexIDMap(index)
            with self._faiss_lock:
                self.faiss_index = faiss_index
        ids = faiss_utils.create_faiss_index_ids(self.max_id, vectors.shape[0])
        self.max_id = ids[-1]

        with self._lmdb_lock:
            lmdb_utils.add_items_to_lmdb(
                db_path=self.lmdb_uncompressed_vectors_path,
                items=vectors,
                ids=ids,
                encode_fn=np.ndarray.tobytes,
                LMDB_MAP_SIZE=self.LMDB_MAP_SIZE
            )
        
        with self._lmdb_lock:
            lmdb_utils.add_items_to_lmdb(
                db_path=self.lmdb_metadata_path,
                items=metadata,
                ids=ids,
                encode_fn=str.encode,
                LMDB_MAP_SIZE=self.LMDB_MAP_SIZE
            )

        # If the index is not trained, don't add the vectors to the index
        if self.faiss_index is not None:
            # TODO: transform vectors if necessary
            with self._faiss_lock:
                self.faiss_index.add_with_ids(vectors, ids)
        
        # Add the vectors to the new faiss index if necessary
        if add_to_new_faiss_index:
            self.new_faiss_index.add_with_ids(vectors, ids)
        
        #logger.info(f'Added vectors and text to LMDB and faiss index')

        self._vector_dimension = vectors.shape[1]
        self.update_training_data_stats(ids_added=ids)
        self.save()

        return ids

    def train(self, use_two_level_clustering: bool = None, pca_dimension: int = None, opq_dimension: int = None, compressed_vector_bytes: int = None, omit_opq: bool = False, num_clusters: int = None, perform_clean_up: bool = False) -> None:
        """
        Train the Faiss index for efficient vector search.

        :param use_two_level_clustering: Whether to use two-level clustering for training. If None, the optimal method will be determined based on memory usage and number of vectors.
        :param pca_dimension: The target dimension for PCA dimensionality reduction. If None, a default value will be used.
        :param opq_dimension: The target dimension for OPQ dimensionality reduction. If None, a default value will be used.
        :param compressed_vector_bytes: The number of bytes to use for compressed vectors. If None, a default value will be used.
        :param omit_opq: Whether to omit the OPQ step during training. This reduces training time with a slight drop in accuracy. Defaults to False.
        :param num_clusters: The number of clusters to use for training. If None, a default value will be calculated.
        """

        logger.info('Training the Faiss index')

        # get default parameters
        default_params = training_utils.get_default_faiss_params(self.vector_dimension)
        if pca_dimension is None:
            pca_dimension = default_params['pca_dimension']
        if opq_dimension is None:
            opq_dimension = default_params['opq_dimension']
        if compressed_vector_bytes is None:
            compressed_vector_bytes = default_params['compressed_vector_bytes']
        
        self.training_params = {
            'pca_dimension': pca_dimension,
            'opq_dimension': opq_dimension,
            'compressed_vector_bytes': compressed_vector_bytes,
            'omit_opq': omit_opq,
            'num_clusters': num_clusters
        }

        # log the training parameters individually
        logger.info(f'pca_dimension: {pca_dimension}')
        logger.info(f'opq_dimension: {opq_dimension}')
        logger.info(f'compressed_vector_bytes: {compressed_vector_bytes}')
        logger.info(f'omit_opq: {omit_opq}')

        # Validate the inputs
        is_valid, reason = input_validation.validate_train(
            self.vector_dimension, pca_dimension, opq_dimension, compressed_vector_bytes)
        if not is_valid:
            logger.error(reason)
            raise ValueError(reason)
        
        # If there are fewer than 5k vectors, don't train the index. We will just create a flat index.
        if (self.num_vectors < 5000):
            logger.info('Skipping training because there are fewer than 5k vectors')
            index = train.train_small_index(
                uncompressed_vectors_lmdb_path=self.lmdb_uncompressed_vectors_path,
                vector_dimension=self.vector_dimension,
                max_memory_usage=self.max_memory_usage,
                lmdb_lock=self._lmdb_lock,
            )
            with self._faiss_lock:
                self.faiss_index = index
            self.save()
            return

        if use_two_level_clustering is None:
            # Figure out which training method is optimal based off the max memory usage and number of vectors
            use_two_level_clustering = training_utils.is_two_level_clustering_optimal(
                max_memory_usage=self.max_memory_usage,
                vector_dimension=self.vector_dimension,
                num_vectors=self.num_vectors
            )
        
        if use_two_level_clustering:
            logger.info('Training with two-level clustering')
            new_faiss_index, lmdb_index_ids = train.train_with_two_level_clustering(
                uncompressed_vectors_lmdb_path=self.lmdb_uncompressed_vectors_path,
                vector_dimension=self.vector_dimension,
                pca_dimension=pca_dimension,
                opq_dimension=opq_dimension,
                compressed_vector_bytes=compressed_vector_bytes,
                max_memory_usage=self.max_memory_usage,
                omit_opq=omit_opq,
                lmdb_lock=self._lmdb_lock,
                num_clusters=num_clusters
            )
        else:
            logger.info('Training with subsampling')
            new_faiss_index, lmdb_index_ids = train.train_with_subsampling(
                uncompressed_vectors_lmdb_path=self.lmdb_uncompressed_vectors_path,
                vector_dimension=self.vector_dimension,
                pca_dimension=pca_dimension,
                opq_dimension=opq_dimension,
                compressed_vector_bytes=compressed_vector_bytes,
                max_memory_usage=self.max_memory_usage,
                omit_opq=omit_opq,
                lmdb_lock=self._lmdb_lock,
                num_clusters=num_clusters
            )

        if new_faiss_index is None:
            logger.error('Training failed. Index was likely deleted during the training operation.')
            self.new_faiss_index = None
            return
        
        # Save the max id used when training the index
        # Convert each value of lmdb_index_ids to an int
        lmdb_index_ids = [int(i) for i in lmdb_index_ids]
        self.max_trained_id = max(lmdb_index_ids)
        self.num_vectors_trained_on = len(lmdb_index_ids)

        # Reset the number of new vectors and number of trained vectors removed to 0
        self.num_new_vectors = 0
        self.num_trained_vectors_removed = 0

        logger.info('Setting the new faiss index')
        self.new_faiss_index = new_faiss_index

        self.save()
    
    def add_unassigned_vectors(self, vector_ids: list) -> None:
        # Add the unassigned vectors to the new faiss index

        # First, get the vectors from the LMDB for the unassigned vector IDs
        with self._lmdb_lock:
            unassigned_vectors = lmdb_utils.get_lmdb_vectors_by_ids(self.lmdb_uncompressed_vectors_path, vector_ids)

        try:
            self.new_faiss_index.add_with_ids(unassigned_vectors, vector_ids)
        except Exception as e:
            logger.info("Exception in add_unassigned_vectors " + e)
            logger.info("No new faiss index")
            return False
        
        # These vectors won't be part of the training dataset, so they are new vectors
        self.num_new_vectors += len(vector_ids)

        logger.info(f'Added {len(vector_ids)} unassigned vectors to the new Faiss index')
        
        return True


    def query(self, query_vector: np.ndarray, preliminary_top_k: int = 500, final_top_k: int = 100) -> list:
        """
        Query the database to find the most similar text to the given query vector.

        :param query_vector: A 1D numpy array representing the query vector.
        :param preliminary_top_k: The number of preliminary results to retrieve from the compressed Faiss index. Should be 5-10x higher than final_top_k. Defaults to 500.
        :param final_top_k: The number of final results to return after reranking. Defaults to 100.

        :return: two lists containing the reranked text and their corresponding IDs, respectively.
        """

        final_top_k = min(final_top_k, self.num_vectors)

        # Check if the query vector is a list, and if so, convert it to a numpy array
        if isinstance(query_vector, list):
            query_vector = np.array(query_vector, dtype=np.float32)

        # query_vector needs to be a 1D array
        is_valid, reason = input_validation.validate_query(query_vector, self.vector_dimension)
        if not is_valid:
            raise ValueError(reason)

        # Check if we need to reshape the query vector
        if len(query_vector.shape) == 1:
            query_vector = query_vector.reshape((-1, self.vector_dimension))

        # query faiss index
        with self._faiss_lock:
            # For a flat index, there is no need for a preliminary top k
            is_flat_index = faiss_utils.check_is_flat_index(self.faiss_index)
            if is_flat_index:
                # Check the number of vectors in the index
                if self.faiss_index.ntotal >= 50000:
                    # Show a warning message
                    logger.warning('The number of vectors in the index is greater than 50k. Please train your index for faster performance.')
                _, I = self.faiss_index.search(query_vector, final_top_k)
                with self._lmdb_lock:
                    corpus_vectors, _ = lmdb_utils.get_ranked_vectors(
                        self.lmdb_uncompressed_vectors_path, I)
                with self._lmdb_lock:
                    metadata = lmdb_utils.get_lmdb_metadata_by_ids(self.lmdb_metadata_path, I.tolist()[0])
                cosine_similarity = query_utils.calculate_cosine_similarity(query_vector, corpus_vectors)
                
                return {
                    "ids": I[0].tolist(),
                    'metadata': metadata,
                    'cosine_similarity': cosine_similarity
                }
            else:
                _, I = self.faiss_index.search(query_vector, preliminary_top_k)

        with self._lmdb_lock:
            corpus_vectors, position_to_id_map = lmdb_utils.get_ranked_vectors(
                self.lmdb_uncompressed_vectors_path, I)

        # brute force search full vectors to find true top_k
        _, reranked_I = knn(query_vector, corpus_vectors, final_top_k)

        # Get the final vectors. reranked_I is a list of indices in the corpus_vectors array
        final_vectors = corpus_vectors[reranked_I[0]]
        cosine_similarity = query_utils.calculate_cosine_similarity(query_vector, final_vectors)

        # Get the ids of the reranked vectors
        reranked_ids = [position_to_id_map[position] for position in reranked_I[0]]

        with self._lmdb_lock:
            reranked_metadata = lmdb_utils.get_lmdb_metadata_by_ids(
                self.lmdb_metadata_path, reranked_ids
            )

        return {
            "ids": reranked_ids,
            'metadata': reranked_metadata,
            'cosine_similarity': cosine_similarity
        }
    
    def remove(self, vector_ids: np.ndarray, remove_from_lmdb: bool = True) -> None:
        """
        Remove vectors and their corresponding text from the database.

        :param vector_ids: A numpy array or list of vector IDs to be removed.
        """

        if isinstance(vector_ids, list):
            vector_ids = np.array(vector_ids)

        # Validate the inputs
        is_valid, reason = input_validation.validate_remove(vector_ids)
        if not is_valid:
            raise ValueError(reason)

        # Remove the vectors from the faiss index (has to be done first)
        with self._faiss_lock:
            self.faiss_index.remove_ids(vector_ids)
        # Save here in case something fails in the LMDB removal.
        # We can't have ids in the faiss index that don't exist in the LMDB
        self.save()

        # If there is a training operation going on, remove_from_lmdb will be False
        # We can't remove from the LMDB while it is being trained, since we could 
        # remove a vector that is part of the training data, which would throw and error
        if not remove_from_lmdb:
            logger.info(f'Finished removing vectors from faiss index')
            return

        # remove vectors from LMDB
        with self._lmdb_lock:
            ids_deleted = lmdb_utils.remove_from_lmdb(
                db_path=self.lmdb_uncompressed_vectors_path,
                ids=vector_ids,
                LMDB_MAP_SIZE=self.LMDB_MAP_SIZE
            )
        # Update the number of ids removed that were part of the training data, and the number of new ids
        self.update_training_data_stats(ids_deleted=ids_deleted)

        # remove text from LMDB
        with self._lmdb_lock:
            lmdb_utils.remove_from_lmdb(
                db_path=self.lmdb_metadata_path,
                ids=vector_ids,
                LMDB_MAP_SIZE=self.LMDB_MAP_SIZE
            )
        logger.info(f'Finished removing vectors and text from LMDB and faiss index')

    def save(self) -> None:
        """
        Save the minDB object and its associated Faiss index to disk.
        """
        with self._faiss_lock:
            if self.faiss_index is not None:
                faiss_index_path = os.path.join(self.save_path, f'faiss_index.index')
                #logger.info(f'Saving faiss index to disk at {faiss_index_path}')

                faiss.write_index(self.faiss_index, faiss_index_path)
                #logger.info(f'faiss index saved to disk at {faiss_index_path}')

            self.save_config_params()

    def read_config_params(self):
        config_file_path = os.path.join(self.save_path, 'config.json')
        # Read in the json
        with open(config_file_path, 'r') as f:
            config_params = json.load(f)
        return config_params

    def save_config_params(self):
        config_params = {
            "max_id": self.max_id,
            "vector_dimension": self.vector_dimension,
            "max_memory_usage": self.max_memory_usage,
            "LMDB_MAP_SIZE": self.LMDB_MAP_SIZE,
            "max_trained_id": self.max_trained_id,
            "num_vectors_trained_on": self.num_vectors_trained_on,
            "num_trained_vectors_removed": self.num_trained_vectors_removed,
            "num_new_vectors": self.num_new_vectors,
            "training_params": self.training_params
        }
        config_file_path = os.path.join(self.save_path, 'config.json')
        with open(config_file_path, 'w') as f:
            json.dump(config_params, f)
    
    def update_training_data_stats(self, ids_deleted: list = [], ids_added: list = []):

        # Handle the ids deleted
        count_removed = 0
        for id in ids_deleted:
            if id > self.max_trained_id:
                # This id was not part of the training data
                self.num_new_vectors -= 1
            else:
                # This id was part of the training data
                self.num_trained_vectors_removed += 1
                count_removed += 1
        
        # These will always be new vectors
        count_added = 0
        for id in ids_added:
            self.num_new_vectors += 1
            count_added += 1


    def delete(self):
        """Remove the mindb object and its associated files from disk."""
        shutil.rmtree(self.save_path)


def load_db(name: str, save_path: str = None) -> minDB:
    """
    Load an existing minDB object and its associated Faiss index from disk.

    :param name: The name of the database to load.
    :param save_path: The path where the database files are saved. Defaults to the .mindb folder in the the current directory.

    :return: An minDB object.
    """
    # use default save path if none is provided
    save_path = get_mindb_path(name, save_path)

    # make sure the database exists
    if not os.path.exists(save_path):
        raise ValueError(f'No database named {name} exists in {save_path}')

    db = minDB(name=name, save_path=save_path, create_or_load="load")

    return db
