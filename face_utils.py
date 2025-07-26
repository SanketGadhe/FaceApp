import numpy as np
import cv2
from keras_facenet import FaceNet
from ultralytics import YOLO
from embeddings import get_face_embedding
from recognize import recognize_face
import os
import uuid
from flask import url_for

facenet_model = FaceNet()
YOLO_MODEL_PATH = "models\yolov8n-face-lindevs.pt"  # Path to your YOLO model
yolo_model = YOLO(YOLO_MODEL_PATH)

# UNKNOWN_DIR = "Unknown_Faces"
# os.makedirs(UNKNOWN_DIR, exist_ok=True)

def detect_and_crop_faces(image):
    results = yolo_model(image)
    boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
    faces = []
    for box in boxes:
        x1, y1, x2, y2 = box
        margin_x = int((x2 - x1) * 0.2)
        margin_y = int((y2 - y1) * 0.2)
        sx = max(0, x1 - margin_x)
        sy = max(0, y1 - margin_y)
        ex = min(image.shape[1], x2 + margin_x)
        ey = min(image.shape[0], y2 + margin_y)
        face = image[sy:ey, sx:ex]
        if face.size == 0:
            continue
        face = cv2.resize(face, (160, 160), interpolation=cv2.INTER_CUBIC)
        kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
        face = cv2.filter2D(face, -1, kernel)
        faces.append(face)
    return faces
