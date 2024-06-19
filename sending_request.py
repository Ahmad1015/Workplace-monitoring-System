import requests
import time

for _ in range(3):
    response_record = requests.post("http://localhost:8000/record_video/")
    print(response_record.json())
    time.sleep(60)
    response_fight_detection = requests.post("http://localhost:8000/process_fight_detection/")
    print(response_fight_detection.json())
