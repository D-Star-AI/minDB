import numpy as np
import lmdb
import os
import json
from typing import Callable


"""try:
    MAP_SIZE = int(os.environ["LMDB_MAP_SIZE"]) # 100 * 1024 * 1024 * 1024 # 100GB
except KeyError:
    MAP_SIZE = 100 * 1024 * 1024 * 1024 # 100GB
print ("MAP_SIZE", MAP_SIZE)"""

def create_lmdb(lmdb_path: str, db_name: str) -> str:
    # Create the LMDB for the vectors
    db_path = os.path.join(lmdb_path, db_name)
    os.makedirs(db_path, exist_ok=True)
    env = lmdb.open(db_path)
    env.close()
    return db_path


def add_items_to_lmdb(db_path: str, items: list, ids: list, encode_fn: Callable, LMDB_MAP_SIZE: int = 100*1024*1024*1024) -> None:
    # Add the text to LMDB
    env = lmdb.open(db_path, map_size=LMDB_MAP_SIZE)
    with env.begin(write=True) as txn:
        for i, item in enumerate(items):
            if isinstance(item, dict):
                item = json.dumps(item)
            txn.put(str(ids[i]).encode('utf-8'), encode_fn(item))
    env.close()
    # TODO: handle the case where the text upload fails


def remove_from_lmdb(db_path: str, ids: list, LMDB_MAP_SIZE: int = 100*1024*1024*1024):
    # remove the vectors to the LMDB
    env = lmdb.open(db_path, map_size=LMDB_MAP_SIZE)
    ids_deleted = []
    with env.begin(write=True) as txn:
        for id in ids:
            if txn.get(str(id).encode('utf-8')) is not None:
                txn.delete(str(id).encode('utf-8'))
                ids_deleted.append(id)
    env.close()
    return ids_deleted


def get_ranked_vectors(uncompressed_vectors_lmdb_path: str, I: np.ndarray) -> tuple[np.ndarray, dict]:
    # query lmdb for the vectors
    corpus_vectors = []
    position_to_id_map = {}
    env = lmdb.open(uncompressed_vectors_lmdb_path)
    with env.begin() as txn:
        for i, id in enumerate(I[0]):
            value = txn.get(str(id).encode('utf-8'))
            value = np.frombuffer(value, dtype=np.float32)
            corpus_vectors.append(value)
            position_to_id_map[i] = id
    env.close()
    # Convert the list to a numpy array
    corpus_vectors = np.array(corpus_vectors)
    return corpus_vectors, position_to_id_map


def get_lmdb_index_ids(db_path: str) -> list:
    env = lmdb.open(db_path)
    # Get the ids from the LMDB
    with env.begin() as txn:
        # decode the keys from bytes to strings
        keys = [key.decode('utf-8') for key in txn.cursor().iternext(keys=True, values=False)]
    env.close()
    return keys


def get_lmdb_metadata_by_ids(metadata_lmdb_path: str, ids: list) -> list:
    env = lmdb.open(metadata_lmdb_path)
    # Get the ids from the LMDB
    with env.begin() as txn:
        metadata = []
        for id in ids:
            value = txn.get(str(id).encode('utf-8'))
            # Convert from bytes to string
            value = value.decode('utf-8')
            # Convert from json string to dict
            value = json.loads(value)
            metadata.append(value)
    env.close()
    return metadata


def get_lmdb_vectors_by_ids(uncompressed_vectors_lmdb_path: str, ids: list) -> np.ndarray:
    try:
        env = lmdb.open(uncompressed_vectors_lmdb_path)
    except FileNotFoundError as e:
        print("error", e)
        return None
    # Get the ids from the LMDB
    with env.begin() as txn:
        vectors = []
        for id in ids:
            value = txn.get(str(id).encode('utf-8'))
            value = np.frombuffer(value, dtype=np.float32)
            vectors.append(value)
    env.close()
    vectors = np.array(vectors)
    return vectors


def get_db_count(db_path: str) -> int:
    env = lmdb.open(db_path)
    with env.begin() as txn:
        count = txn.stat()['entries']
    env.close()
    return count
