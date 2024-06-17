import cv2
import threading
import time
import os
from .fight_detection import fight_detection  # Relative import

def record_video(filename, duration=60, fps=30, video_counter=1):
    print(f"Started recording video: {filename}")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Camera not accessible")
        return
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, (int(cap.get(3)), int(cap.get(4))))

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
    process_recorded_video(filename)


def process_recorded_video(filename):
    print(f"Started processing: {filename}")
    fight_detection(filename)
