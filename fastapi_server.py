from fastapi import FastAPI, BackgroundTasks
import os
from Fight_Model.video_recording import record_video

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




if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
