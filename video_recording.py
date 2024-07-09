import cv2
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from ActivityRecPyTorchVideo import ActivityRecognition
import threading
import asyncio
import queue
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)

output_dir = os.path.dirname(__file__)
video_queue = queue.Queue()

async def record_video(filename, duration=10, fps=30, ip_address=0):
    logging.info(f"Started recording video: {filename}")
    cap = cv2.VideoCapture(ip_address)

    if not cap.isOpened():
        logging.error("Error: Camera not accessible")
        return None

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
            logging.error("Failed to read frame from camera")
            break

    cap.release()
    out.release()
    logging.info(f"Finished recording video: {filename}")

    temp = filename.split("\\")[-1]
    copy2 = os.path.join(output_dir, f"copy2_{temp}")

    try:
        shutil.copy(output_filepath, copy2)
        os.remove(output_filepath)
    except Exception as e:
        logging.error(f"Error copying files: {e}")
        return None

    return copy2

async def process_video(model, filename):
    logging.info(f"Inside Process Video for model {model} on file {filename}")
    logging.info(f"Started activity detection on: {filename}")
    await ActivityRecognition(filename)  
    logging.info(f"Completed activity detection on: {filename}")

def delete_video(filename):
    logging.info(f"Attempting to delete video: {filename}")
    try:
        os.remove(filename)
        logging.info(f"Deleted video: {filename}")
    except OSError as e:
        logging.error(f"Error deleting video {filename}: {e}")

async def main_record_and_process(filename, duration, fps, ip_address):
    logging.info("Recording video")
    copy2 = await record_video(filename, duration, fps, ip_address)
    if copy2:
        logging.info("Recording completed. Adding videos to queue")
        video_queue.put((copy2, 'activity_detection'))
        logging.info(f"Queue now contains: {list(video_queue.queue)}")
    else:
        logging.error("Recording failed. Not adding to queue")

def worker(executor):
    logging.info("Worker started")
    while True:
        logging.info("Worker waiting for items in the queue")
        filename, model = video_queue.get()
        try:
            logging.info(f"Worker processing {filename} with model {model}")
            future = executor.submit(asyncio.run, process_video(model, filename))
            future.result()
        finally:
            video_queue.task_done()
            delete_video(filename)
            logging.info(f"Worker finished processing {filename} with model {model}")

def start_worker():
    logging.info("Starting worker task")
    executor = ThreadPoolExecutor(max_workers=2)
    worker_thread = threading.Thread(target=worker, args=(executor,), daemon=True)
    worker_thread.start()
    logging.info("Worker task created")

if __name__ == "__main__":
    start_worker()
