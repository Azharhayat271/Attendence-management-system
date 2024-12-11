# main.py
from fastapi import FastAPI, UploadFile, Form, HTTPException
from pymongo import MongoClient
import boto3
import os
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
from pillow_heif import register_heif_opener

# FastAPI app
app = FastAPI()

# Register HEIC format
register_heif_opener()

# MongoDB setup
MONGO_URI = os.getenv("MONGO_URI", "mongodb://azharhayat271:root@cluster0-shard-00-00.gsfhe.mongodb.net:27017,cluster0-shard-00-01.gsfhe.mongodb.net:27017,cluster0-shard-00-02.gsfhe.mongodb.net:27017/?replicaSet=atlas-tmgaqk-shard-0&ssl=true&authSource=admin")
mongo_client = MongoClient(MONGO_URI)
db = mongo_client['attendance']
students_collection = db['students']

# Supabase S3 storage
SUPABASE_S3_ENDPOINT = os.getenv("SUPABASE_S3_ENDPOINT", "https://oeotkzssxlrxksfnemcm.supabase.co/storage/v1/s3")
ACCESS_KEY = os.getenv("SUPABASE_ACCESS_KEY", "7f9a61644903b40b7d95bcaaf9b23f32")
SECRET_KEY = os.getenv("SUPABASE_SECRET_KEY", "5d678a1c88138687b7ff9543926260962a6685ebe6e07aa91e170de6eba98ed6")
BUCKET_NAME = os.getenv("SUPABASE_BUCKET_NAME", "boost")

s3_client = boto3.client(
    's3',
    endpoint_url=SUPABASE_S3_ENDPOINT,
    aws_access_key_id=ACCESS_KEY,
    aws_secret_access_key=SECRET_KEY,
    region_name='us-east-1',
)

# MediaPipe setup
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

def extract_face_features(image_path):
    """Extract facial features using MediaPipe."""
    face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
    image = cv2.imread(image_path)

    if image is None:
        return None

    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_detection.process(image_rgb)

    if not results.detections:
        return None

    for detection in results.detections:
        bboxC = detection.location_data.relative_bounding_box
        return np.array([bboxC.xmin, bboxC.ymin, bboxC.width, bboxC.height])

    return None

def upload_to_supabase(file_path, file_name):
    """Upload image to Supabase Storage."""
    with open(file_path, 'rb') as file:
        s3_client.upload_fileobj(file, BUCKET_NAME, file_name)

@app.post("/register")
async def register_student(
    name: str = Form(...),
    reg_no: str = Form(...),
    file: UploadFile = None
):
    if file is None:
        raise HTTPException(status_code=400, detail="No image uploaded")

    # Save the uploaded file locally
    file_path = f"temp/{file.filename}"
    with open(file_path, "wb") as f:
        f.write(await file.read())

    # Convert HEIC if necessary
    if file.filename.lower().endswith(".heic"):
        with Image.open(file_path) as img:
            converted_path = file_path.replace(".HEIC", ".jpg")
            img.convert("RGB").save(converted_path, "JPEG")
            file_path = converted_path

    # Extract face features
    face_features = extract_face_features(file_path)
    if face_features is None:
        os.remove(file_path)
        raise HTTPException(status_code=400, detail="No face detected in the image")

    # Upload to Supabase
    upload_to_supabase(file_path, file.filename)

    # Save to MongoDB
    student_data = {
        "name": name,
        "registration_number": reg_no,
        "face_features": face_features.tolist(),
        "image_url": f"{SUPABASE_S3_ENDPOINT}/{BUCKET_NAME}/{file.filename}"
    }
    students_collection.insert_one(student_data)
    os.remove(file_path)

    return {"message": "Student registered successfully", "student": student_data}

@app.post("/attendance")
async def mark_attendance(file: UploadFile = None):
    if file is None:
        raise HTTPException(status_code=400, detail="No image uploaded")

    # Save the uploaded file locally
    file_path = f"temp/{file.filename}"
    with open(file_path, "wb") as f:
        f.write(await file.read())

    # Convert HEIC if necessary
    if file.filename.lower().endswith(".heic"):
        with Image.open(file_path) as img:
            converted_path = file_path.replace(".HEIC", ".jpg")
            img.convert("RGB").save(converted_path, "JPEG")
            file_path = converted_path

    # Extract face features
    input_face_features = extract_face_features(file_path)
    if input_face_features is None:
        os.remove(file_path)
        raise HTTPException(status_code=400, detail="No face detected in the image")

    # Compare with registered students
    registered_students = students_collection.find()
    for student in registered_students:
        registered_face = np.array(student['face_features'])
        if np.linalg.norm(input_face_features - registered_face) < 0.1:  # Tolerance threshold
            os.remove(file_path)
            return {"message": f"Attendance marked for {student['name']} (Reg. No: {student['registration_number']})"}

    os.remove(file_path)
    return {"message": "No matching face found. Attendance not marked."}
