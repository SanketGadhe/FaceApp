from flask import Blueprint, request, jsonify, url_for
import os, cv2, uuid
import numpy as np
from embeddings import process_images, get_face_embedding,process_faces_from_urls
from recognize import recognize_face
from face_utils import detect_and_crop_faces, UNKNOWN_DIR
import pickle

memorysnap_bp = Blueprint("memorysnap", __name__)

@memorysnap_bp.route("/train-embeddings", methods=["POST"])
def train_embeddings_from_faces():
    data = request.get_json()
    trip_id = data.get("tripId")
    faces = data.get("faces")

    if not trip_id or not faces:
        return jsonify({"error": "Missing tripId or faces"}), 400

    output_pkl = f"TrainedModels/Events/{trip_id}.pkl"

    try:
        process_faces_from_urls(faces, output_pkl)
        return jsonify({
            "message": f"Model trained successfully for trip {trip_id}",
            "embeddingPath": output_pkl
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@memorysnap_bp.route("/api/memorysnap/recognize", methods=["POST"])
def recognize_memorysnap():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No image provided"}), 400

        # Clear unknowns folder for fresh session
        for f in os.listdir(UNKNOWN_DIR):
            os.remove(os.path.join(UNKNOWN_DIR, f))

        file = request.files['file']
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({"error": "Invalid image"}), 400

        faces = detect_and_crop_faces(img)

        response_faces = []
        for face in faces:
            uid = str(uuid.uuid4())
            filename = f"{uid}.jpg"
            filepath = os.path.join(UNKNOWN_DIR, filename)
            cv2.imwrite(filepath, face)

            image_url = url_for('static', filename=filename, _external=True)
            response_faces.append({
                "id": uid,
                "imageUrl": image_url
            })

        return jsonify({
            "faces": response_faces,
            "count": len(response_faces)
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@memorysnap_bp.route("/classify-faces", methods=["POST"])
def classify_faces_in_images():
    try:
        data = request.get_json()
        trip_id = data.get("tripId")
        embedding_path = data.get("embeddingPath")
        image_paths = data.get("imagePaths")
        print("Images Path",image_paths)

        if not trip_id or not embedding_path or not image_paths:
            return jsonify({"error": "tripId, embeddingPath, and imagePaths are required"}), 400

        if not os.path.exists(embedding_path):
            return jsonify({"error": f"Embedding file not found at {embedding_path}"}), 404

        # Load embeddings for the trip
        with open(embedding_path, 'rb') as f:
            known_embeddings = pickle.load(f)

        results = []

        for image_path in image_paths:
            if not os.path.exists(image_path):
                print(f"⚠️ Skipping missing file: {image_path}")
                continue

            image = cv2.imread(image_path)
            if image is None:
                print(f"⚠️ Unable to read: {image_path}")
                continue

            faces = detect_and_crop_faces(image)
            recognized_ids = []

            for face in faces:
                embedding = get_face_embedding(face)
                if embedding is None:
                    continue

                name = recognize_face(embedding, known_embeddings)
                if name.lower() != "unknown":
                    recognized_ids.append(name)

            results.append({
                "imagePath": image_path,
                "recognized": recognized_ids
            })

        return jsonify({"results": results}), 200

    except Exception as e:
        print(f"❌ Error in /classify-faces: {e}")
        return jsonify({"error": str(e)}), 500