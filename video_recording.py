import cv2
import time
import os
import asyncio
from concurrent.futures import ThreadPoolExecutor
from face_recog import face_detection  
from fight_detection import fight_detection 
import shutil
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
import uvicorn

# Define the Output Videos directory relative to the current script's location
output_dir = os.path.dirname(__file__)

def record_video(filename, duration=10, fps=30, ip_address=0):
    print(f"Started recording video: {filename}")

    # Check if IP camera is provided
    cap = cv2.VideoCapture(ip_address)

    if not cap.isOpened():
        print("Error: Camera not accessible")
        return

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_filepath = os.path.join(output_dir, filename)
    out = cv2.VideoWriter(output_filepath, fourcc, fps, (int(cap.get(3)), int(cap.get(4))))

    start_time = time.time()
    frame_count = 0
    total_frames = duration * fps

    while frame_count < total_frames:
        ret, frame = cap.read()
        if ret:
            out.write(frame)
            frame_count += 1
        else:
            print("Failed to read frame from camera")
            break

    cap.release()
    out.release()

    print(f"Finished recording video: {filename}")

    temp = filename.split("\\")[-1]

    copy1 = os.path.join(output_dir, f"copy1_{temp}")
    copy2 = os.path.join(output_dir, f"copy2_{temp}")

    try:
        shutil.copy(output_filepath, copy1)
        shutil.copy(output_filepath, copy2)
    except Exception as e:
        print(f"Error copying files: {e}")

    return copy1, copy2

async def process_video(model, filename):
    if model == 'face_detection':
        print(f"Started face detection on: {filename}")
        await face_detection(filename)
    elif model == 'fight_detection':
        print(f"Started fight detection on: {filename}")
        await fight_detection(filename)

def delete_video(filename):
    try:
        os.remove(filename)
        print(f"Deleted video: {filename}")
    except OSError as e:
        print(f"Error deleting video {filename}: {e}")

async def run_process_video(model, filename):
    await process_video(model, filename)

async def main_record_and_process(filename, duration, fps, ip_address):
    copy1, copy2 = record_video(filename, duration, fps, ip_address)
    print("Going into the main function")
    with ThreadPoolExecutor() as executor:
        loop = asyncio.get_event_loop()
        tasks = [
            loop.run_in_executor(executor, asyncio.run, run_process_video('face_detection', copy1)),
            loop.run_in_executor(executor, asyncio.run, run_process_video('fight_detection', copy2)),
        ]
        await asyncio.gather(*tasks)
    
    # Optionally delete the videos after processing
    delete_video(copy1)
    delete_video(copy2)