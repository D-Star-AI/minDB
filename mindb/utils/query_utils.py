import numpy as np

def calculate_cosine_similarity(query_vector: np.ndarray, comparison_vectors: np.ndarray) -> np.ndarray:
    all_cosine_similarity = []
    for vector in comparison_vectors:
        cosine_similarity = np.dot(query_vector, vector)
        all_cosine_similarity.append(cosine_similarity)
    return all_cosine_similarity