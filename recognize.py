import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def recognize_face(embedding, known_embeddings=None, threshold=0.4):
    if embedding is None:
        return "unknown"

    if known_embeddings is None:
        with open("default_class.pkl", "rb") as f:
            known_embeddings = pickle.load(f)

    similarities = {
        person_id: cosine_similarity([embedding], [known])[0][0]
        for person_id, known in known_embeddings.items()
    }

    best_match = max(similarities.items(), key=lambda x: x[1], default=(None, -1))

    if best_match[1] > threshold:
        return best_match[0]
    return "unknown"
