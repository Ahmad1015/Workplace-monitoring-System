from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from pymongo import MongoClient
import os
from datetime import datetime
from video_recording import main_record_and_process

app = FastAPI()
video_counter = 1
script_dir = os.path.dirname(os.path.abspath(__file__))
filename = ""

# MongoDB connection setup
client = MongoClient("mongodb://localhost:27017")
db = client["face_detection_db"]
collection = db["detections"]

class RecordVideoRequest(BaseModel):
    duration: int = 10
    fps: int = 30
    ip_address: str
    port: int

class DetectionRecord(BaseModel):
    name: str
    screenshot_path: str
    timestamp: datetime

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

@app.post("/save_detection/")
async def save_detection(record: DetectionRecord):
    collection.insert_one(record.dict())
    return {"info": "Detection record saved"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
