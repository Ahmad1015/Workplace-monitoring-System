import requests
import time
# Define the URL and the data payload
url = 'http://localhost:8000/record_video/'
data = {
    "duration": 10,
    "fps": 13,
    "ip_address": "192.168.10.3",
    "port": 4747
}
for i in range(3):
    # Send the POST request with JSON data
    response = requests.post(url, json=data)
    print(response)
    time.sleep(10)

