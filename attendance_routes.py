from flask import Blueprint, request, jsonify, url_for
import os, cv2, uuid
from embeddings import process_images, get_face_embedding,process_faces_from_urls
from recognize import recognize_face
from face_utils import detect_and_crop_faces
import numpy as np
from utils.s3_utils import upload_image_array_to_s3 ,upload_file_to_s3


attendance_bp = Blueprint("attendance", __name__)

@attendance_bp.route("/api/train-model/<classID>", methods=["POST"])
def train_attendance_model(classID):
    department = request.json.get("department")
    year = request.json.get("year")
    IMAGE_PATH = f"StudentData/{department}/{year}/{classID}"
    EMBEDDING_PATH = f"/tmp/TrainedModels/{department}_{year}_{classID}.pkl"  # Use /tmp for temp storage

    S3_BUCKET_ATTENDANCE_EMBEDDINGS = os.environ.get('S3_BUCKET_NAME_FOR_ATTENDANCE_EMBEDDINGS')
    if not S3_BUCKET_ATTENDANCE_EMBEDDINGS:
        return jsonify({"error": "S3_BUCKET_NAME_FOR_ATTENDANCE_EMBEDDINGS not set"}), 500

    if not os.path.exists(IMAGE_PATH):
        return jsonify({"error": "Image path does not exist"}), 404

    os.makedirs(os.path.dirname(EMBEDDING_PATH), exist_ok=True)

    try:
        process_faces_from_urls(IMAGE_PATH, EMBEDDING_PATH)
        # Upload the embedding file to S3
        s3_key = f"{department}_{year}_{classID}.pkl"
        s3_url = upload_file_to_s3(EMBEDDING_PATH, S3_BUCKET_ATTENDANCE_EMBEDDINGS, s3_key)
        os.remove(EMBEDDING_PATH)
        return jsonify({
            "message": "Model trained successfully",
            "embeddingPath": s3_url
        }), 200
    except Exception as e:
        print(f"Error in train_attendance_model: {str(e)}")
        return jsonify({"error": str(e)}), 500

@attendance_bp.route("/api/recognize_attendance", methods=["POST"])
def recognize_attendance():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

    faces = detect_and_crop_faces(image)
    recognized, unknown = [], []

    S3_BUCKET_UNKNOWN_FACES = os.environ.get('S3_BUCKET_NAME_FOR_UNKNOWN_FACES')
    if not S3_BUCKET_UNKNOWN_FACES:
        return jsonify({"error": "S3_BUCKET_NAME_FOR_UNKNOWN_FACES not set"}), 500

    for face in faces:
        embedding = get_face_embedding(face)
        name = recognize_face(embedding)
        if name.lower() == "unknown":
            uid = str(uuid.uuid4())
            s3_key = f"unknown_faces/{uid}.jpg"
            s3_url = upload_image_array_to_s3(face, S3_BUCKET_UNKNOWN_FACES, s3_key)
            unknown.append({"id": uid, "imageUrl": s3_url})
        else:
            recognized.append(name)

    return jsonify({"recognized": list(set(recognized)), "unknown": unknown, "no_face_present": len(faces)})