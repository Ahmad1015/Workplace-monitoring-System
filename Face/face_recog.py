import numpy as np
import face_recognition
import cv2

# Load a sample picture and learn how to recognize it.
try:
    known_image = face_recognition.load_image_file("Official_photo.jpeg")
    known_face_encoding = face_recognition.face_encodings(known_image)[0]
    print("Known face encoding loaded successfully.")
except Exception as e:
    print(f"Error loading known image: {e}")
    exit()

# Create arrays of known face encodings and their names
known_face_encodings = [known_face_encoding]
known_face_names = ["Ahmad"]

# Initialize some variables
video_capture = cv2.VideoCapture("http://192.168.10.6:4747/video")

if not video_capture.isOpened():
    print("Error: Could not open IP camera stream.")
    exit()

print("IP camera stream opened successfully.")

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    # Resize frame of video to 1/2 size for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

    # Convert the image from BGR color (OpenCV) to RGB color (face_recognition)
    rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])

    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    print(f"Found {len(face_locations)} face(s) in the current frame.")

    # Print face encodings for debugging
    for face_encoding in face_encodings:
        print("Face encoding:", face_encoding)

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

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Debug message for each frame
    print("Frame displayed.")

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting...")
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
print("IP camera and windows released/closed successfully.")
