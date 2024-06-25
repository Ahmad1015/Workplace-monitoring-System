from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
import os
from video_recording import record_video

app = FastAPI()
video_counter = 1
script_dir = os.path.dirname(os.path.abspath(__file__))
filename = ""

class RecordVideoRequest(BaseModel):
    duration: int = 60
    fps: int = 30
    ip_address: str
    port: int

@app.post("/record_video/")
async def record_video_endpoint(background_tasks: BackgroundTasks, request: RecordVideoRequest):
    global video_counter
    global filename
    filename = os.path.join(script_dir, f'output_{video_counter}.mp4')
    print(f"Adding task to record video: {filename}")
    ip_address = f"http://{request.ip_address}:{request.port}/video"
    background_tasks.add_task(record_video, filename, request.duration, request.fps, ip_address)
    video_counter += 1
    return {"info": f"Video recording started: {filename}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
