import requests
import time
# Define the URL and the data payload
url = 'http://localhost:8000/record_video/'
data = {
    "duration": 10,
    "fps": 20,
    "ip_address": "10.135.8.107:4747/video",
    "port": 4747
}
for i in range(2):
    # Send the POST request with JSON data
    response = requests.post(url, json=data)
    print(response)
    time.sleep(10)

