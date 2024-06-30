import cv2
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from face_recog import face_detection
from fight_detection import fight_detection
import threading
import asyncio
import queue

# Define the Output Videos directory relative to the current script's location
output_dir = os.path.dirname(__file__)
video_queue = queue.Queue()

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
        os.remove(filename)
    except Exception as e:
        print(f"Error copying files: {e}")

    return copy1, copy2

import asyncio

def process_video(model, filename):
    print(f"Inside Process Video for model {model} on file {filename}")
    if model == 'face_detection':
        print(f"Started face detection on: {filename}")
        asyncio.run(face_detection(filename))
        print(f"Completed face detection on: {filename}")
    elif model == 'fight_detection':
        print(f"Started fight detection on: {filename}")
        asyncio.run(fight_detection(filename))
        print(f"Completed fight detection on: {filename}")


def delete_video(filename):
    print(f"Attempting to delete video: {filename}")
    try:
        os.remove(filename)
        print(f"Deleted video: {filename}")
    except OSError as e:
        print(f"Error deleting video {filename}: {e}")

def main_record_and_process(filename, duration, fps, ip_address):
    print("Recording video")
    copy1, copy2 = record_video(filename, duration, fps, ip_address)
    print("Recording completed. Adding videos to queue")
    video_queue.put((copy1, 'face_detection'))
    video_queue.put((copy2, 'fight_detection'))
    print(f"Queue now contains: {list(video_queue.queue)}")  # Print current queue status

def worker(executor):
    print("Worker started")
    while True:
        print("Worker waiting for items in the queue")
        filename, model = video_queue.get()
        try:
            print(f"Worker processing {filename} with model {model}")
            future = executor.submit(process_video, model, filename)
            future.result()  # Wait for the processing to complete
        finally:
            video_queue.task_done()
            delete_video(filename)
            print(f"Worker finished processing {filename} with model {model}")

def start_worker():
    print("Starting worker task")
    executor = ThreadPoolExecutor(max_workers=2)  # Number of concurrent workers
    worker_thread = threading.Thread(target=worker, args=(executor,), daemon=True)
    worker_thread.start()
    print("Worker task created")

# Make sure to start the worker when the script runs
if __name__ == "__main__":
    start_worker()
