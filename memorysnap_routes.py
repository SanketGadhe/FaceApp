# memorysnap_bp.py
from flask import Blueprint, request, jsonify
import os
import uuid
import pickle
import requests # Still needed for downloading images/files from URLs (used by s3_utils)

# Import the new S3 utility functions
from utils.s3_utils import upload_image_array_to_s3, upload_file_to_s3, download_image_from_s3_url, download_file_from_s3

# Assuming these are adapted to handle direct image data (NumPy arrays) or URLs
from embeddings import process_images, get_face_embedding, process_faces_from_urls
from recognize import recognize_face
from face_utils import detect_and_crop_faces # Ensure this returns NumPy arrays, not local paths

memorysnap_bp = Blueprint("memorysnap", __name__)

# Define S3 bucket names from environment variables
# Ensure these are set in your Flask app's environment (e.g., Docker, ECS)
S3_BUCKET_CROPPED_FACES = os.environ.get('S3_BUCKET_NAME_FOR_CROPPED_FACES')
S3_BUCKET_EMBEDDINGS = os.environ.get('S3_BUCKET_NAME_FOR_EMBEDDINGS')

if not all([S3_BUCKET_CROPPED_FACES, S3_BUCKET_EMBEDDINGS]):
    raise EnvironmentError("S3_BUCKET_NAME_FOR_CROPPED_FACES and S3_BUCKET_NAME_FOR_EMBEDDINGS must be set in environment variables.")


@memorysnap_bp.route("/train-embeddings", methods=["POST"])
def train_embeddings_from_faces():
    data = request.get_json()
    trip_id = data.get("tripId")
    faces_urls = data.get("faces") # Expecting a list of S3 URLs to cropped face images

    if not trip_id or not faces_urls:
        return jsonify({"error": "Missing tripId or faces (S3 URLs)"}), 400

    # Temporary local path to save the generated .pkl file before uploading to S3
    local_pkl_path = f"/tmp/embeddings/{trip_id}.pkl"
    os.makedirs(os.path.dirname(local_pkl_path), exist_ok=True)

    try:
        # `process_faces_from_urls` must now download faces from `faces_urls`
        # and then process them to create the embedding file locally.
        # Ensure process_faces_from_urls uses `download_image_from_s3_url` internally
        # or expects URLs and handles downloading itself.
        process_faces_from_urls(faces_urls, local_pkl_path)

        # Upload the generated .pkl file to S3
        s3_embedding_key = f"{trip_id}.pkl"
        s3_embedding_url = upload_file_to_s3(local_pkl_path, S3_BUCKET_EMBEDDINGS, s3_embedding_key)

        # Clean up the local temporary file
        os.remove(local_pkl_path)

        return jsonify({
            "message": f"Model trained successfully for trip {trip_id}",
            "embeddingPath": s3_embedding_url # Return the S3 URL of the .pkl file
        }), 200
    except Exception as e:
        print(f"Error in /train-embeddings: {str(e)}")
        return jsonify({"error": str(e)}), 500


@memorysnap_bp.route("/api/memorysnap/recognize", methods=["POST"])
def recognize_memorysnap():
    try:
        data = request.get_json()
        image_url = data.get('imageUrl') # Get the S3 URL of the selfie image
        trip_id = data.get('tripId') # Get the trip ID
        if not image_url:
            return jsonify({"error": "No imageUrl provided"}), 400

        # Download the image from the S3 URL using s3_utils
        img = download_image_from_s3_url(image_url)

        faces_cropped_images = detect_and_crop_faces(img) # This should return NumPy arrays of cropped faces

        response_faces = []
        for face_img_array in faces_cropped_images:
            # Generate a unique key for the S3 object
            uid = str(uuid.uuid4())
            s3_key = f"{trip_id}/{uid}.jpg" # Example S3 path for cropped faces

            # Upload the cropped face directly to S3 using s3_utils
            s3_cropped_face_url = upload_image_array_to_s3(face_img_array, S3_BUCKET_CROPPED_FACES, s3_key)

            response_faces.append({
                "id": uid,
                "imageUrl": s3_cropped_face_url # Return the S3 URL of the cropped face
            })

        return jsonify({
            "faces": response_faces,
            "count": len(response_faces),
            "originalImageUrl": image_url
        }), 200
    except requests.exceptions.RequestException as req_err:
        print(f"Error downloading image from S3: {req_err}")
        return jsonify({"error": f"Failed to download image from URL: {str(req_err)}"}), 500
    except Exception as e:
        print(f"Error in /api/memorysnap/recognize: {str(e)}")
        return jsonify({"error": str(e)}), 500


@memorysnap_bp.route("/classify-faces", methods=["POST"])
def classify_faces_in_images():
    try:
        data = request.get_json()
        trip_id = data.get("tripId")
        embedding_s3_url = data.get("embeddingPath") # Expecting S3 URL to the .pkl file
        image_urls = data.get("imageUrls") # Expecting array of S3 URLs to images
        print("Images URLs from Node.js:", image_urls)

        if not trip_id or not embedding_s3_url or not image_urls:
            return jsonify({"error": "tripId, embeddingPath (S3 URL), and imageUrls are required"}), 400

        # --- Download Embedding File from S3 ---
        local_embedding_file = f"/tmp/embeddings/{trip_id}.pkl" # Use /tmp for temporary storage in Docker/Linux
        os.makedirs(os.path.dirname(local_embedding_file), exist_ok=True)

        download_file_from_s3(S3_BUCKET_EMBEDDINGS, f"{trip_id}.pkl", local_embedding_file)
        # Assuming the s3_key for embeddings is "trip_embeddings/{trip_id}.pkl" as per train-embeddings

        if not os.path.exists(local_embedding_file):
            return jsonify({"error": f"Embedding file not found locally after download at {local_embedding_file}"}), 404

        # Load embeddings for the trip
        with open(local_embedding_file, 'rb') as f:
            known_embeddings = pickle.load(f)

        results = []

        for image_url in image_urls:
            try:
                # Download image from S3 URL using s3_utils
                image = download_image_from_s3_url(image_url)

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
                    "imageUrl": image_url,
                    "recognized": recognized_ids
                })

            except requests.exceptions.RequestException as req_err:
                print(f"⚠️ Skipping image due to download error from {image_url}: {req_err}")
                results.append({"imageUrl": image_url, "error": "Failed to download"})
            except Exception as e:
                print(f"⚠️ Error processing image from {image_url}: {e}")
                results.append({"imageUrl": image_url, "error": f"Processing error: {str(e)}"})

        # Clean up downloaded embedding file (important for stateless containers)
        if os.path.exists(local_embedding_file):
            os.remove(local_embedding_file)

        return jsonify({"results": results}), 200

    except Exception as e:
        print(f"❌ Error in /classify-faces: {e}")
        return jsonify({"error": str(e)}), 500