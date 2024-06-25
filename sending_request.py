import requests

# Define the URL and the data payload
url = 'http://localhost:8000/record_video/'
data = {
    "duration": 60,
    "fps": 30,
    "ip_address": "192.168.10.3",
    "port": 4747
}

# Send the POST request with JSON data
response = requests.post(url, json=data)

# Print the response
print(response.json())
