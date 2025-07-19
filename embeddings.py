import os
import pickle
import cv2
import numpy as np
from mtcnn import MTCNN
from keras_facenet import FaceNet

# Initialize MTCNN and FaceNet
detector = MTCNN()
embedder = FaceNet()

# Get 512-d face embedding from image
def get_face_embedding(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    detections = detector.detect_faces(image_rgb)

    if not detections:
        return None

    x, y, width, height = detections[0]['box']
    x, y = max(0, x), max(0, y)  # Ensure no negative values
    face = image_rgb[y:y+height, x:x+width]
    face = cv2.resize(face, (160, 160))

    # Get 512-d embedding
    embedding = embedder.embeddings(np.expand_dims(face, axis=0))[0]
    return embedding


def process_images(input_dir, output_pkl):
    face_data = {}
    print("🔍 Starting embedding process...")

    for person_name in os.listdir(input_dir):
        person_path = os.path.join(input_dir, person_name)
        if not os.path.isdir(person_path):
            continue

        embeddings = []
        print(f"📂 Processing: {person_name}")

        for file in os.listdir(person_path):
            img_path = os.path.join(person_path, file)
            img = cv2.imread(img_path)

            if img is None:
                print(f"  ⚠️ Skipping unreadable: {file}")
                continue

            embedding = get_face_embedding(img)
            if embedding is not None:
                embeddings.append(embedding)
                print(f"  ✅ Processed: {file}")
            else:
                print(f"  ❌ No face detected: {file}")

        if embeddings:
            avg_embedding = np.mean(embeddings, axis=0)
            face_data[person_name] = avg_embedding
        else:
            print(f"  ⚠️ No valid embeddings for {person_name}")

    os.makedirs(os.path.dirname(output_pkl), exist_ok=True)

    try:
        with open(output_pkl, 'wb') as f:
            pickle.dump(face_data, f)
            print(f"✅ Embeddings saved at: {output_pkl}")
    except Exception as e:
        print(f"❌ Error saving file: {e}")
import requests
import numpy as np
import cv2
import pickle
import os
from collections import defaultdict
from embeddings import get_face_embedding  # keep this function as-is

def process_faces_from_urls(face_data_list, output_pkl):
    face_data = defaultdict(list)
    print("🔍 Starting embedding process from URLs...\n")

    for face in face_data_list:
        person_id = face.get("person_id")
        image_url = face.get("imageUrl")

        if not person_id or not image_url:
            print(f"⚠️ Skipping incomplete entry: {face}")
            continue

        try:
            # Download the image
            response = requests.get(image_url)
            if response.status_code != 200:
                print(f"❌ Failed to fetch image for {person_id}")
                continue

            image_array = np.frombuffer(response.content, np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

            if image is None:
                print(f"❌ Invalid image for {person_id}")
                continue

            # Generate embedding
            embedding = get_face_embedding(image)
            if embedding is not None:
                face_data[person_id].append(embedding)
                print(f"✅ Embedded image for {person_id}")
            else:
                print(f"❌ No face detected for {person_id}")

        except Exception as e:
            print(f"⚠️ Error processing {person_id}: {e}")

    # Average embeddings
    averaged_data = {}
    for person_id, embeddings in face_data.items():
        if embeddings:
            avg_embedding = np.mean(embeddings, axis=0)
            averaged_data[person_id] = avg_embedding
        else:
            print(f"⚠️ No valid embeddings for {person_id}")

    os.makedirs(os.path.dirname(output_pkl), exist_ok=True)

    try:
        with open(output_pkl, 'wb') as f:
            pickle.dump(averaged_data, f)
            print(f"\n✅ Embeddings saved to: {output_pkl}")
    except Exception as e:
        print(f"❌ Error saving embeddings: {e}")
