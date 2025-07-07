import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def recognize_face(embedding, known_embeddings=None):
    if embedding is None:
        return "unknown"

    # Load default if none provided
    if known_embeddings is None:
        with open("default_class.pkl", "rb") as f:
            known_embeddings = pickle.load(f)

    threshold = 0.4
    max_sim = -1
    matched_name = "unknown"

    for name, known_embed in known_embeddings.items():
        sim = cosine_similarity([embedding], [known_embed])[0][0]
        if sim > max_sim and sim > threshold:
            max_sim = sim
            matched_name = name

    return matched_name
