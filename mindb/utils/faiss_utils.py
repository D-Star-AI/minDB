
def check_is_flat_index(index) -> bool:

    if (str(type(index)) == "<class 'faiss.swigfaiss_avx2.IndexIDMap'>" or str(type(index)) == "<class 'faiss.swigfaiss.IndexIDMap'>"):
        return True
    else:
        return False
    
def create_faiss_index_ids(max_id: int, num_new_vectors: int) -> list:
    # Create a sequential list of IDs for the new vectors
    # The IDs start at max_id + 1 and go up to max_id + num_new_vectors
    new_ids = range(max_id + 1, max_id + num_new_vectors + 1)
    return new_ids