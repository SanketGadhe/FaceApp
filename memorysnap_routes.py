from flask import Blueprint, request, jsonify, url_for
import os, cv2, uuid
import numpy as np
from embeddings import process_images, get_face_embedding
from recognize import recognize_face
from face_utils import detect_and_crop_faces, UNKNOWN_DIR

memorysnap_bp = Blueprint("memorysnap", __name__)

@memorysnap_bp.route("/api/memorysnap/train/<event_id>", methods=["POST"])
def train_memorysnap_model(event_id):
    try:
        input_dir = f"MemorySnapData/{event_id}"
        output_pkl = f"TrainedModels/Events/{event_id}.pkl"

        if not os.path.exists(input_dir):
            return jsonify({"error": "Folder for this event does not exist"}), 404

        process_images(input_dir, output_pkl)
        return jsonify({"message": f"Model trained for event {event_id}"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@memorysnap_bp.route("/api/memorysnap/recognize/<event_id>", methods=["POST"])
def recognize_memorysnap(event_id):
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
