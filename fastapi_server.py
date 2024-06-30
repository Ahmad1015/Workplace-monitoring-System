from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from pymongo import MongoClient
import os
from datetime import datetime
from video_recording import main_record_and_process, start_worker
import threading

app = FastAPI()
video_counter = 1
script_dir = os.path.dirname(os.path.abspath(__file__))
filename = ""

# MongoDB connection setup
client = MongoClient("mongodb://localhost:27017")

class RecordVideoRequest(BaseModel):
    duration: int = 10
    fps: int = 30
    ip_address: str
    port: int

class ScreenshotDetectionRecord(BaseModel):
    name: str
    screenshot_path: str
    timestamp: datetime

class VideoDetectionRecord(BaseModel):
    name: str
    video_path: str
    timestamp: datetime

class GunDetectionRecord(BaseModel):
    name: str
    video_path: str
    timestamp: datetime

@app.on_event("startup")
def startup_event():
    print("Starting worker at startup")
    worker_thread = threading.Thread(target=start_worker, daemon=True)
    worker_thread.start()
    print("Worker thread should be started")

@app.post("/record_video/")
async def record_video_endpoint(background_tasks: BackgroundTasks, request: RecordVideoRequest):
    global video_counter
    global filename
    filename = os.path.join(script_dir, f'output_{video_counter}.mp4')
    print(f"Adding task to record video: {filename}")
    ip_address = f"http://{request.ip_address}:{request.port}/video"
    background_tasks.add_task(main_record_and_process, filename, request.duration, request.fps, ip_address)
    video_counter += 1
    return {"info": f"Video recording started: {filename}"}

@app.post("/face_detection/")
async def save_detection(record: ScreenshotDetectionRecord):
    db = client["face_detection"]
    collection = db["detections"]
    collection.insert_one(record.dict())
    return {"info": "Detection record saved"}

@app.post("/fight_detection/")
async def save_detection(record: VideoDetectionRecord):
    db = client["fight_detection"]
    collection = db["detections"]
    collection.insert_one(record.dict())
    return {"info": "Detection record saved"}

@app.post("/gun_detection/")
async def save_detection(record: GunDetectionRecord):
    db = client["gun_detection"]
    collection = db["detections"]
    collection.insert_one(record.dict())
    return {"info": "Detection record saved"}

def create_app():
    return app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(create_app(), host="0.0.0.0", port=8000)
