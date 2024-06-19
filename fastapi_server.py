from fastapi import FastAPI, BackgroundTasks
import os
from Fight_Model.video_recording import record_video
from Fight_Model.fight_detection import fight_detection
from Writing_Activity_Model.ActivityRecPyTorchVideo import ActivityRecognition

app = FastAPI()
video_counter = 1
script_dir = os.path.dirname(os.path.abspath(__file__))
filename =""

@app.post("/record_video/")
async def record_video_endpoint(background_tasks: BackgroundTasks, duration: int = 60, fps: int = 30):
    global video_counter
    global filename
    filename = os.path.join(script_dir, f'output_{video_counter}.mp4')
    print(f"Adding task to record video: {filename}")
    background_tasks.add_task(record_video, filename, duration, fps, video_counter)
    video_counter += 1
    return {"info": f"Video recording started: {filename}"}

@app.post("/process_fight_detection/")
async def process_fight_detection_endpoint(background_tasks: BackgroundTasks, filename: str):
    print(f"Adding task to process fight detection on video: {filename}")
    background_tasks.add_task(fight_detection, filename)
    return {"info": f"Fight detection started on: {filename}"}

@app.post("/process_activity_recognition/")
async def process_activity_recognition_endpoint(background_tasks: BackgroundTasks, filename: str):
    print(f"Adding task to process activity recognition on video: {filename}")
    background_tasks.add_task(ActivityRecognition, filename)
    return {"info": f"Activity recognition started on: {filename}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
