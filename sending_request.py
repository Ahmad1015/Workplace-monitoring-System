import requests

response = requests.post("http://localhost:8000/record_video/")
print(response.json())
