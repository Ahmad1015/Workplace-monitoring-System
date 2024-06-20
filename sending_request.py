import requests
import time
import subprocess

# Full path to detect.py
detect_script = r"F:\Policy-Based-Presence-Tracker-UI\Writing_Activity_Model\yolo\detect.py"

# Full path to weights file
weights_path = r"F:\Policy-Based-Presence-Tracker-UI\Writing_Activity_Model\yolo\best.pt"


# Define the command and arguments
command = ["python", detect_script, "--weights", weights_path, "--source", 'processed_output_1.mp4']

# Run the command
result = subprocess.run(command, capture_output=True, text=True)

# Print the output
print("Output from Yolo:")
print(result.stdout)
print(result.stderr)
    
