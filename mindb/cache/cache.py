from collections import OrderedDict
from mindb.mindb import load_db
from mindb.utils import faiss_utils

class LRUCache:
    def __init__(self, max_memory_usage):
        self.max_memory_usage = max_memory_usage
        self.cache = OrderedDict()
        self.current_memory_usage = 0
    
    def update_max_memory_usage(self, max_memory_usage, operations={}):
        self.max_memory_usage = max_memory_usage
        self.update_memory_usage()
        while (self.current_memory_usage > self.max_memory_usage) and (len(self.cache) > 0):

            # Loop through the cache and find the first key that isn't in the operations list
            first_key = None
            keys = list(self.cache.keys())
            for item in keys:
                print (item)
                if (item not in operations) or (item in operations and (operations[item] == "complete" or operations[item] == "untrained")):
                    first_key = item
                    break
            if first_key is None:
                break

            evicted_value = self.cache.pop(first_key)
            self.current_memory_usage -= estimate_memory_usage(evicted_value)

    def update_memory_usage(self):
        memory_usage = 0
        for _, value in self.cache.items():
            memory_usage += estimate_memory_usage(value)
        self.current_memory_usage = memory_usage

    def get(self, key, check_memory_usage=False, operations={}):
        """
        Method to get the value of a key in the cache. If the key is not in the cache, it will raise a KeyError.
        - If check_memory_usage is True, it will check if the current memory usage is greater than the max memory usage and evict the least recently used key until the memory usage is less than the max memory usage.
        - We only want to check memory usage and modify the cache for operations that are not latency sensitive. Probably just for adding and removing vectors.
        """
        if key not in self.cache:
            try:
                db = load_db(key)
            except ValueError:
                return None
            self.cache[key] = db
        
        self.cache.move_to_end(key)
        
        if check_memory_usage:
            self.update_memory_usage()
            while (self.current_memory_usage > self.max_memory_usage) and (len(self.cache) > 0):

                # Loop through the cache and find the first key that isn't in the operations list
                first_key = None
                keys = list(self.cache.keys())
                for item in keys:
                    if (item not in operations) or (item in operations and (operations[item] == "complete" or operations[item] == "untrained")):
                        first_key = item
                        break
                if first_key is None:
                    break

                evicted_value = self.cache.pop(first_key)
                self.current_memory_usage -= estimate_memory_usage(evicted_value)
        return self.cache[key]

    def put(self, key, value, operations={}):
        """
        Only needed when we add a new database.
        """
        if key in self.cache:
            self.cache.move_to_end(key)
        else:

            self.update_memory_usage() # put only called in non-latency sensitive operations, so we can always check memory usage here
            index_memory_usage = estimate_memory_usage(value)
            while (self.current_memory_usage + index_memory_usage > self.max_memory_usage) and (len(self.cache) > 0):
                # Loop through the cache and find the first key that isn't in the operations list
                first_key = None
                keys = list(self.cache.keys())
                for key in keys:
                    if (key not in operations) or (key in operations and (operations[key] == "complete" or operations[key] == "untrained")):
                        first_key = key
                        break
                if first_key is None:
                    break

                evicted_value = self.cache.pop(first_key)
                self.current_memory_usage -= estimate_memory_usage(evicted_value)

            self.cache[key] = value
            self.current_memory_usage += index_memory_usage

    def remove(self, key):
        """
        Only needed when we remove a database.
        """
        if key in self.cache:
            self.cache.pop(key)
            self.update_memory_usage()


def estimate_memory_usage(db):
    # Get an estimate of the memory usage based off the number of vectors, the type of index,
    # the vector dimension, and the params used to train the index (if trained)

    if db.faiss_index is None:
        return 48 # This is the memory usage of an empty database

    vector_dimension = db._vector_dimension
    n_total = db.faiss_index.ntotal

    # Case where the index is a flat index
    is_flat_index = faiss_utils.check_is_flat_index(db.faiss_index)
    if is_flat_index:
        memory_usage = n_total * vector_dimension * 4 + 240090 # There is an extra 240,090 bytes of overhead for the index
        return memory_usage

    # Case where the index is a trained index
    compressed_vector_bytes = db.training_params["compressed_vector_bytes"]

    ## TODO: refine this a bit more. These are very good estimates, but it would be better to 
    # have an equation that can predict the memory usage more accurately
    if vector_dimension == 256:
        constant_addition = 1479168
    elif vector_dimension == 512:
        constant_addition = 2529792
    elif vector_dimension == 768:
        constant_addition = 4104704 # There is an extra 4,104,704 bytes of overhead for the index
    elif vector_dimension == 1024:
        constant_addition = 6203904
    else:
        constant_addition = vector_dimension * 4092 # Roughly 4KB
    
    memory_usage = n_total * (compressed_vector_bytes + 8) + constant_addition

    return memory_usage