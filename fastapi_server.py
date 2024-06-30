from fastapi import FastAPI, BackgroundTasks 
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pymongo import MongoClient
import os
from datetime import datetime
from bson import ObjectId
from video_recording import main_record_and_process, start_worker
import threading
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to your needs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
video_counter = 1
script_dir = os.path.dirname(os.path.abspath(__file__))
filename = ""
app.mount("/screenshots", StaticFiles(directory="D:/Workplace-monitoring-System/screenshots"), name="screenshots")
app.mount("/Videos", StaticFiles(directory="D:/Workplace-monitoring-System/Videos"), name="Videos")

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

class WritingDetectionRecord(BaseModel):
    filename: str
    original_file: str
    start_time: int
    end_time: int
    confidence: float
    video_path: str

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

@app.post("/writing_detection/")
async def save_detection(record: WritingDetectionRecord):
    db = client["writing_detection"]
    collection = db["detections"]
    collection.insert_one(record.dict())
    return {"info": "Detection record saved"}

@app.get("/get-face-screenshots")
async def get_face_screenshots():
    try:
        db = client["face_detection"]
        collection = db["detections"]
        records = collection.find()
        
        screenshots = []
        for record in records:
            record["_id"] = str(record["_id"])
            # Convert absolute path to relative path
            if "screenshot_path" in record:
                record["screenshot_path"] = "/screenshots/" + os.path.basename(record["screenshot_path"])
            screenshots.append(record)
        
        return screenshots
    except Exception as e:
        return {"error": str(e)}

@app.get("/get-unauthorized-face-screenshots")
async def get_unauthorized_face_screenshots():
    try:
        db = client["face_detection"]
        collection = db["detections"]
        records = collection.find()
        
        screenshots = []
        for record in records:
            # Extract name and timestamp from record
            name = record.get("name", "")
            timestamp = record.get("timestamp", "")
            
            # Append name and timestamp to screenshots list
            screenshots.append({
                "name": name,
                "timestamp": timestamp
            })
        
        return screenshots
    except Exception as e:
        return {"error": str(e)}
    
@app.get("/get-videos")
async def get_videos():
    try:
        db = client["fight_detection"]
        collection = db["detections"]
        records = collection.find()
        
        videos = []
        for record in records:
            record["_id"] = str(record["_id"])
            # Convert absolute path to relative path
            if "video_path" in record:
                record["video_path"] = "/Videos/" + os.path.basename(record["video_path"])
            videos.append(record)
            print(videos)
        return videos
    except Exception as e:
        return {"error": str(e)}



def create_app():
    return app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(create_app(), host="0.0.0.0", port=8000)
