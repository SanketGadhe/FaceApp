from flask import Blueprint, request, jsonify, url_for
import os, cv2, uuid
from embeddings import process_images, get_face_embedding
from recognize import recognize_face
from face_utils import detect_and_crop_faces, UNKNOWN_DIR
import numpy as np

attendance_bp = Blueprint("attendance", __name__)

@attendance_bp.route("/api/train-model/<classID>", methods=["POST"])
def train_attendance_model(classID):
    department = request.json.get("department")
    year = request.json.get("year")
    IMAGE_PATH = f"StudentData/{department}/{year}/{classID}"
    EMBEDDING_PATH = f"TrainedModels/{department}/{year}/{classID}.pkl"
    if not os.path.exists(IMAGE_PATH):
        return jsonify({"error": "Image path does not exist"}), 404
    process_images(IMAGE_PATH, EMBEDDING_PATH)
    return jsonify({"message": "Model trained successfully"})

@attendance_bp.route("/api/recognize_attendance", methods=["POST"])
def recognize_attendance():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    # Clear Unknown_Faces folder
    for f in os.listdir(UNKNOWN_DIR):
        os.remove(os.path.join(UNKNOWN_DIR, f))

    file = request.files['file']
    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

    faces = detect_and_crop_faces(image)
    recognized, unknown = [], []

    for face in faces:
        embedding = get_face_embedding(face)
        name = recognize_face(embedding)
        if name.lower() == "unknown":
            uid = str(uuid.uuid4())
            filepath = os.path.join(UNKNOWN_DIR, f"{uid}.jpg")
            cv2.imwrite(filepath, face)
            unknown.append({"id": uid, "imageUrl": url_for("static", filename=f"{uid}.jpg", _external=True)})
        else:
            recognized.append(name)

    return jsonify({"recognized": list(set(recognized)), "unknown": unknown, "no_face_present": len(faces)})
