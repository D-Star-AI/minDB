from mindb.mindb import minDB, load_db
from mindb.utils import training_utils, lmdb_utils
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Tuple
import threading
import json
import time
from mindb.cache.cache import LRUCache

FILE_PATH = os.path.dirname(os.path.abspath(__file__))


app = FastAPI()

# Load all databases located in the ~/.mindb folder into a dictionary
db_path = os.path.join(os.path.expanduser("~"), ".mindb")

# Initialize the Cache class
databases = LRUCache(max_memory_usage=1 * 1024 * 1024 * 1024) # 1 GB

operations = {}
unassigned_vectors = {}
training_queue = []
initial_training_queue = []
initial_training_queue_lock = threading.Lock()
vectors_to_remove = {}

# Define request and response models

class AddInput(BaseModel):
    add_data: List[Tuple]

class RemoveInput(BaseModel):
    ids: List[int]

class QueryInput(BaseModel):
    query_vector: List[float]
    preliminary_top_k: Optional[int] = 500
    final_top_k: Optional[int] = 100


class QueryOutput(BaseModel):
    metadata: List[dict]
    ids: List[int]
    cosine_similarity: List[float]


class CreateDBInput(BaseModel):
    name: str
    vector_dimension: Optional[int] = None
    max_memory_usage: Optional[int] = 4 * 1024 * 1024 * 1024


class TrainDBInput(BaseModel):
    use_two_level_clustering: Optional[bool] = None
    pca_dimension: Optional[int] = None
    opq_dimension: Optional[int] = None
    compressed_vector_bytes: Optional[int] = None
    omit_opq: Optional[bool] = False

class MaxMemoryInput(BaseModel):
    max_memory_usage: int

# API routes
@app.get("/health")
def read_root():
    return {"status": "healthy"}

@app.get("/test")
def read_root():
    return {"status": "healthy"}

@app.get("/db/{db_name}/info")
def get_info(db_name: str):
    # Return a json object with the class attributes
    db = databases.get(db_name, operations=operations)
    if db == None:
        raise HTTPException(status_code=404, detail="Database not found")

    if db.faiss_index == None:
        n_total = 0
    else:
        n_total = db.faiss_index.ntotal
    db_info = {
        "name": db.name,
        "vector_dimension": db.vector_dimension,
        "num_vectors": db.num_vectors,
        "trained_index_coverage_ratio": db.trained_index_coverage_ratio,
        "max_memory_usage": db.max_memory_usage,
        "n_total": n_total,
        "max_id": db.max_id,
        "lmdb_uncompressed_vectors_path": db.lmdb_uncompressed_vectors_path,
        "max_trained_id": db.max_trained_id,
        "num_vectors_trained_on": db.num_vectors_trained_on,
        "num_new_vectors": db.num_new_vectors,
        "num_trained_vectors_removed": db.num_trained_vectors_removed,
        "training_params": db.training_params,
    }

    # Turn the object into a string so it can be returned
    db_info = json.dumps(db_info)

    return {"db_info": db_info}


@app.post("/db/create")
def create_db(create_db_input: CreateDBInput):
    if create_db_input.name in databases.cache:
        raise HTTPException(
            status_code=400, detail="Database with this name already exists")
    db = minDB(name=create_db_input.name, vector_dimension=create_db_input.vector_dimension,
              max_memory_usage=create_db_input.max_memory_usage)

    # Add this to the cache
    databases.put(create_db_input.name, db)

    return {"message": "Database created successfully"}


def check_for_initial_training(db_name: str):
    # Check to see if we need to train the database. This will be called during the add function
    db = databases.get(db_name, operations=operations)
    if db == None:
        # Not sure what needs to happen here. This really should never happen
        pass
    needs_training = training_utils.check_needs_initial_training(db_name, db.num_vectors, db.faiss_index, operations)
    if needs_training:
        train_db(db_name)


def train_initial_indexes():
    # This function will be called in a separate thread to train the indexes that need to be trained
    while len(initial_training_queue) > 0:
        db_name = initial_training_queue[0]
        
        if db_name not in operations:
            try:
                # We can't have this throw an error, so we wrap it in a try/except
                train_db(db_name)
            except:
                # TODO: Alert us that something went wrong
                pass
            with initial_training_queue_lock:
                initial_training_queue.remove(db_name)
        else:
            initial_training_queue.remove(db_name)


@app.post("/db/{db_name}/add")
def add_vectors(db_name: str, data: AddInput):
    db = databases.get(db_name, check_memory_usage=True, operations=operations)
    if db == None:
        raise HTTPException(status_code=404, detail="Database not found")

    training_status = "untrained"
    if (db_name in operations):
        training_status = operations[db_name]

    # If the status of the training operation is 'trained', add the vectors to the new faiss index as well
    if training_status == "trained":
        ids = db.add(data=data.add_data, add_to_new_faiss_index=True)
    else:
        ids = db.add(data=data.add_data)

    # If there is a training process happening, add the vectors to a list of unassigned vectors
    if training_status == "in progress":
        if db_name not in unassigned_vectors:
            unassigned_vectors[db_name] = []
        unassigned_vectors[db_name].extend(ids)
    
    needs_training = training_utils.check_needs_initial_training(db_name, db.num_vectors, db.faiss_index, operations)
    if needs_training:
        with initial_training_queue_lock:
            # get the length of the queue before adding the db_name
            len_training_queue = len(initial_training_queue)
            # Make sure the db_name is not already in the queue
            if (db_name not in initial_training_queue):
                initial_training_queue.append(db_name)
        
        # Do this outside the thread lock to avoid locking the training queue for too long
        if (len_training_queue == 0):
            # If there are no indexes already in the queue, start training the indexes in a new thread
            thread = threading.Thread(target=train_initial_indexes)
            thread.start()
    
    return {"message": "Vectors and text added successfully"}


@app.post("/db/{db_name}/remove")
def remove_vectors_by_id(db_name: str, ids: RemoveInput):

    training_status = "untrained"
    if (db_name in operations):
        training_status = operations[db_name]
    
    remove_from_lmdb = True
    if training_status == "trained" or training_status == "in progress":
        # We can't remove vectors from LMDB if the index is being trained
        remove_from_lmdb = False
        # Add these vector ids to a list of vectors to be removed from LMDB once training is complete
        if db_name not in vectors_to_remove:
            vectors_to_remove[db_name] = []
        vectors_to_remove[db_name].extend(ids.ids)
    
    db = databases.get(db_name, operations=operations)
    if db == None:
        raise HTTPException(status_code=404, detail="Database not found")
    db.remove(vector_ids=ids.ids, remove_from_lmdb=remove_from_lmdb)

    return {"message": f"{len(ids.ids)} vectors removed successfully"}


def cleanup_training(db_name: str):

    # One last check to make sure there are no unassigned vectors
    # Make sure the db name is in the list of databases (this is a safeguard in case the db was deleted)
    db = databases.get(db_name, operations=operations)
    if db == None:
        return
    
    if (db_name in unassigned_vectors):

        # Add the vectors to the new faiss index in batches of 10,000
        batch_size = 10000
        while len(unassigned_vectors[db_name]) > 0:
            batch_ids = unassigned_vectors[db_name][:batch_size]
            
            with db._lmdb_lock:
                vectors = lmdb_utils.get_lmdb_vectors_by_ids(db.lmdb_uncompressed_vectors_path, batch_ids)
            with db._faiss_lock:
                db.faiss_index.add_with_ids(vectors, batch_ids)
                unassigned_vectors[db_name] = unassigned_vectors[db_name][batch_size:]
    
    # Handle any vectors that need to be removed from LMDB
    if (db_name in vectors_to_remove):
        batch_size = 500
        while len(vectors_to_remove[db_name]) > 0:
            batch_ids = vectors_to_remove[db_name][:batch_size]
            db.remove(vector_ids=batch_ids, remove_from_lmdb=True)
            vectors_to_remove[db_name] = vectors_to_remove[db_name][batch_size:]
            

def train_db(db_name: str):
    db = databases.get(db_name, operations=operations)
    if db == None:
        raise HTTPException(status_code=404, detail="Database not found")
    operations[db_name] = "in progress"

    training_params = training_utils.get_training_params(db.max_memory_usage, db.vector_dimension, db.num_vectors)

    db.train(
        use_two_level_clustering=training_params["use_two_level_clustering"],
        pca_dimension=training_params["pca_dimension"],
        opq_dimension=training_params["opq_dimension"],
        compressed_vector_bytes=training_params["compressed_vector_bytes"],
        omit_opq=training_params["omit_opq"]
    )
    operations[db_name] = "trained"

    # Sleep for 5 seconds. This in unnecessarily long, but it makes sure we don't start adding the unassigned vectors
    # to the faiss index in the middle of an add operation. 
    time.sleep(5)
    
    # Perform the cleanup operation to make sure all of the vectors have been added to the faiss index
    if (db_name in unassigned_vectors):

        # Add the vectors to the new faiss index in batches of 1,000
        batch_size = 1000
        expected_num_batches = int(len(unassigned_vectors[db_name]) / batch_size)
        num_iters = 0
        while len(unassigned_vectors[db_name]) > 0:
            batch_ids = unassigned_vectors[db_name][:batch_size]
            
            success = db.add_unassigned_vectors(vector_ids=batch_ids)

            if not success:
                # TODO: Alert us that something went wrong
                pass
            unassigned_vectors[db_name] = unassigned_vectors[db_name][batch_size:]
            num_iters += 1

            if (num_iters > expected_num_batches*2):
                # This is a safeguard to make sure we don't get stuck in an infinite loop
                # TODO: Alert us that something went wrong
                break
    
    # When the operation is complete, update the operation status to 'complete', set the faiss index
    # to the new faiss index, and remove the new faiss index
    with db._faiss_lock:
        if db.new_faiss_index is not None:
            db.faiss_index = db.new_faiss_index
            db.new_faiss_index = None
        else:
            print ("No new faiss index found")
            operations[db_name] = "failed"
            return
    
    # This has to be done outside of the faiss lock since the save operation will acquire the lock
    print ("Saving the new faiss index")
    db.save()
    print ("Done saving the new faiss index")
    operations[db_name] = "complete"

    # Update the memory usage after training
    databases.update_memory_usage()

    time.sleep(5)
    # Perform the cleanup operation to make sure all of the vectors have been added to the faiss index
    cleanup_training(db_name)


@app.post("/db/{db_name}/train")
def start_train_db(db_name: str):
    db = databases.get(db_name, operations=operations)
    if db == None:
        raise HTTPException(status_code=404, detail="Database not found")

    # Check if this db is already training
    if db_name in operations:
        status = operations[db_name]
        if status == "in progress":
            raise HTTPException(
                status_code=400, detail="This database is in the process of training already")

    thread = threading.Thread(target=train_db, kwargs={"db_name": db_name})
    thread.start()

    return {"status": "training successfully initiated"}


@app.get("/db/{db_name}/train")
def get_operation_status(db_name: str):
    if db_name not in operations:
        return {"status": "not started"}

    return {"status": operations[db_name]}


@app.post("/db/{db_name}/query")
def query(db_name: str, query_input: QueryInput):
    db = databases.get(db_name, operations=operations)
    if db == None:
        raise HTTPException(status_code=404, detail="Database not found")

    results = db.query(
        query_vector=query_input.query_vector, preliminary_top_k=query_input.preliminary_top_k, final_top_k=query_input.final_top_k
    )
    ids = results['ids']
    metadata = results['metadata']
    cosine_similarity = results['cosine_similarity']
    return QueryOutput(metadata=metadata, ids=ids, cosine_similarity=cosine_similarity)


@app.post("/db/{db_name}/save")
def save_db(db_name: str):
    db = databases.get(db_name, operations=operations)
    if db == None:
        raise HTTPException(status_code=404, detail="Database not found")
    db.save()
    return {"message": "Database saved successfully"}


@app.post("/db/{db_name}/reload")
def reload_db(db_name: str):
    if db_name not in databases:
        raise HTTPException(status_code=404, detail="Database not found")
    try:
        db = load_db(db_name, db_path)
        databases[db_name] = db
        return {"message": "Database reloaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/db/{db_name}/delete")
def delete_db(db_name: str):
    db = databases.get(db_name, operations=operations)
    if (db == None):
        raise HTTPException(status_code=404, detail="Database not found")
    db.delete()
    # Remove the db from the cache
    databases.remove(db_name)
    
    # If this db was in the operations dictionary, remove it
    if db_name in operations:
        del operations[db_name]
    return {"message": "Database deleted successfully"}


def train_indexes():

    for db_name in training_queue:
        if db_name not in operations or (db_name in operations and operations[db_name] == "complete"):
            print ("training ", db_name)
            try:
                # We can't have this throw an error, so we wrap it in a try/except
                success = train_db(db_name)
            except:
                # TODO: Alert us that something went wrong
                pass
            training_queue.remove(db_name)
        else:
            print ("training already in progress for ", db_name)
            continue


@app.get("/db/find_indexes_to_train")
def find_indexes_to_train():
    # Loop through all of the dbs and find the ones that need to be trained

    if not (os.path.exists(db_path)):
        os.mkdir(db_path)
    db_names = os.listdir(db_path)

    if len(training_queue) > 0:
        # If indexes are already being trained, return the training queue
        # We can't have multiple indexes being trained at the same time in different threads, nor is it necessary
        return {"training_queue": training_queue}

    for db_name in db_names:
        db = databases.get(db_name, operations=operations)
        with initial_training_queue_lock:
            # If this db is in the initial training queue, skip it
            if db_name in initial_training_queue:
                continue
        needs_training = training_utils.check_needs_training(
            db_name, db.num_vectors, operations, db.trained_index_coverage_ratio)
        if needs_training:
            training_queue.append(db_name)
    
    # Start training the indexes in a new thread, if there are any
    if (len(training_queue) > 0):
        thread = threading.Thread(target=train_indexes)
        thread.start()

    return {"training_queue": training_queue}
    


@app.get("/db/get_initial_training_queue")
def get_initial_training_queue():
    # Loop through all of the dbs and find the ones that need to be trained
    return {"initial_training_queue": initial_training_queue}


@app.get("/db/view_cache")
def view_cache():
    # We don't want to actually return the cache, just the keys
    cache_keys = [key for key in databases.cache]
    current_memory_usage = databases.current_memory_usage
    max_memory_usage = databases.max_memory_usage

    return {
        "cache_keys": cache_keys,
        "current_memory_usage": current_memory_usage,
        "max_memory_usage": max_memory_usage
    }

@app.post("/db/{db_name}/remove_from_cache")
def remove_from_cache(db_name: str):
    databases.remove(db_name)
    return {"message": "Database removed from cache"}


@app.post("/db/update_max_memory_usage")
def update_max_memory_usage(max_memory_usage: MaxMemoryInput):
    databases.update_max_memory_usage(max_memory_usage = max_memory_usage.max_memory_usage, operations=operations)
    return {"message": "Max memory usage updated successfully"}

"""
Usage


# Run the server using Uvicorn
# uvicorn app:app --host 0.0.0.0 --port 8000
"""
