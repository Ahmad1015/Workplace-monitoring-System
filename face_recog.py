import numpy as np
import face_recognition
import cv2
import time
import requests
from datetime import datetime
import os


output_dir = os.path.dirname(__file__)

async def face_detection(filename, frame_skip=10):
    # Load a sample picture and learn how to recognize it.
    try:
        known_image = face_recognition.load_image_file("Official_photo.jpeg")
        known_face_encoding = face_recognition.face_encodings(known_image)[0]
        print("Known face encoding loaded successfully.")
    except Exception as e:
        print(f"Error loading known image: {e}")
        return

    # Create arrays of known face encodings and their names
    known_face_encodings = [known_face_encoding]
    known_face_names = ["Ahmad"]

    # Initialize the video capture
    video_capture = cv2.VideoCapture(filename)

    if not video_capture.isOpened():
        print("Error: Could not open video file.")
        return

    print("Video file opened successfully.")

    # Initialize variables for time-based screenshot saving
    screenshot_interval = 5  # 5 seconds
    last_screenshot_time = time.time()
    screenshot_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "screenshots")

    # Create the directory if it doesn't exist
    if not os.path.exists(screenshot_directory):
        os.makedirs(screenshot_directory)

    frame_count = 0

    while video_capture.isOpened():
        # Grab a single frame of video
        ret, frame = video_capture.read()
        if not ret:
            print("End of video file reached or failed to capture image.")
            break

        frame_count += 1

        # Skip frames to reduce processing load
        if frame_count % frame_skip != 0:
            continue

        current_time = time.time()

        # Resize frame of video to 1/2 size for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

        # Convert the image from BGR color (OpenCV) to RGB color (face_recognition)
        rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])

        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        print(f"Found {len(face_locations)} face(s) in the current frame.")

        face_detected = False

        # Loop through each face found in the current frame of video
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # Use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
                face_detected = True

            # Scale back up face locations since the frame we detected in was scaled to 1/2 size
            top *= 2
            right *= 2
            bottom *= 2
            left *= 2

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        # Save screenshot if the interval has passed and a face was detected
        if face_detected and (current_time - last_screenshot_time) >= screenshot_interval:
            screenshot_filename = os.path.join(screenshot_directory, f"screenshot_{int(current_time)}.jpg")
            cv2.imwrite(screenshot_filename, frame)
            print(f"Saved screenshot: {screenshot_filename}")
            last_screenshot_time = current_time

            # Send the detection record to the FastAPI endpoint
            detection_record = {
                "name": name,
                "screenshot_path": screenshot_filename,
                "timestamp": datetime.now().isoformat()
            }
            response = requests.post("http://localhost:8000/face_detection/", json=detection_record)
            if response.status_code == 200:
                print("Detection record saved successfully.")
            else:
                print("Failed to save detection record.")

        # Debug message for each frame
        print("Frame processed.")

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting...")
            break

    # Release handle to the video file
    video_capture.release()
    cv2.destroyAllWindows()
    print("Video file and windows released/closed successfully.")
    
    

    try:
        os.remove(filename)
    except Exception as e:
        print(f"Error removing video file: {e}")



    
    

