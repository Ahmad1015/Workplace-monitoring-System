import cv2
import threading
import time
import os
import asyncio
from concurrent.futures import ThreadPoolExecutor
from face_recog import face_detection  
from fight_detection import fight_detection 
import shutil    

# Define the Output Videos directory relative to the current script's location
output_dir = os.path.join(os.path.dirname(__file__))

def record_video(filename, duration=60, fps=30, ip_address=0):
    print(f"Started recording video: {filename}")

    # Check if IP camera is provided
    cap = cv2.VideoCapture(ip_address)

    if not cap.isOpened():
        print("Error: Camera not accessible")
        return

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(os.path.join(output_dir, filename), fourcc, fps, (int(cap.get(3)), int(cap.get(4))))

    start_time = time.time()
    while (time.time() - start_time) < duration:
        ret, frame = cap.read()
        if ret:
            out.write(frame)
        else:
            print("Failed to read frame from camera")
            break

    cap.release()
    out.release()

    print(f"Finished recording video: {filename}")

    copy1 = os.path.join(output_dir, f"copy1_{filename}")
    copy2 = os.path.join(output_dir, f"copy2_{filename}")

    shutil.copy(os.path.join(output_dir, filename), copy1)
    shutil.copy(os.path.join(output_dir, filename), copy2)

    return copy1, copy2

async def process_video(model, filename):
    if model == 'face_detection':
        print(f"Started face detection on: {filename}")
        face_detection(filename)
    elif model == 'fight_detection':
        print(f"Started fight detection on: {filename}")
        fight_detection(filename)

def delete_video(filename):
    try:
        os.remove(filename)
        print(f"Deleted video: {filename}")
    except OSError as e:
        print(f"Error deleting video {filename}: {e}")

async def main(filename):
    copy1, copy2 = record_video(filename)
    
    with ThreadPoolExecutor() as executor:
        loop = asyncio.get_event_loop()
        tasks = [
            loop.run_in_executor(executor, process_video, 'face_detection', copy1),
            loop.run_in_executor(executor, process_video, 'fight_detection', copy2),
        ]
        await asyncio.gather(*tasks)
    
    # Optionally delete the videos after processing
    delete_video(copy1)
    delete_video(copy2)

# Usage example
if __name__ == '__main__':
    filename = "video.mp4"
    asyncio.run(main(filename))
