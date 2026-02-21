from fastapi import FastAPI, UploadFile, File
import cv2
import numpy as np
from ultralytics import YOLO

app = FastAPI()

# This automatically downloads the YOLOv8-pose weights and loads them onto the A10G
model = YOLO('yolov8n-pose.pt') 

@app.post("/analyze-pose")
async def analyze_pose(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    
    results = model(frame)[0]
    
    if results.keypoints is not None and len(results.keypoints.xyn) > 0:
        keypoints = results.keypoints.xyn.cpu().numpy().tolist()
        return {"status": "success", "keypoints": keypoints}
    
    return {"status": "no_person_detected"}