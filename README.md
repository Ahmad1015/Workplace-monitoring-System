# Workplace Monitoring System

## Project Overview
This project is a Workplace Monitoring System designed to monitor various activities in a workspace using video footage from IP cameras. The system is built using FastAPI for backend processing and React for the frontend interface. It is capable of detecting activities such as face detection, fight detection, gun detection, and writing detection in real-time video streams.

## Features
- Real-time Video Recording: Capture video from IP cameras for specified durations.
- Face Detection: Detect faces in video frames and log the results.
- Fight Detection: Analyze video for potential fights using advanced activity recognition models.
- Gun Detection: Detect the presence of guns in videos.
- Writing Detection: Identify writing activities in video clips.
- Asynchronous Processing: Utilize FastAPI's asynchronous capabilities to handle video recording and detection concurrently.
- Persistence: Store detection records in MongoDB for later analysis and review.
- React Frontend:
    - Display screenshots from video recordings.
    - Save video recordings and screenshots to directories.
    - Navigate through screenshots using next and previous buttons.
    - Future plans to integrate video playback functionality on the frontend.
## Technology Stack
- Backend: FastAPI, Python, PyTorch, OpenCV
- Frontend: React.js
- Database: MongoDB
- Machine Learning: Detectron2, PyTorchVideo, Yolov7
  
## Setup Instructions
### Prerequisites
- Python 3.8+
- Node.js and npm
- MongoDB
## Setup
Either Setup Using Conda or Manually install Libraries
## Conda
### Setting Up the Conda Environment

To simplify setting up your development environment, we provide a Conda environment file named `environment.yaml`. This file contains all the necessary Python packages and their versions required for this project.

### Instructions:

1. **Install Conda**: Ensure that Conda is installed on your system. You can download it from [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html).

2. **Create the Conda Environment**: Open a terminal or command prompt and navigate to the directory where the `environment.yaml` file is located. Run the following command to create and activate the Conda environment:

    ```bash
    conda env create -f environment.yaml
    ```

    This command will create a Conda environment with all the dependencies specified in the `environment.yaml` file.

3. **Activate the Environment**: After the environment is created, activate it using the command:

    ```bash
    conda activate <environment_name>
    ```

    Replace `<environment_name>` with the name specified in the `environment.yaml` file.

4. **Start Working**: You are now ready to start working on the project with all necessary packages installed.

Using the provided `environment.yaml` file ensures that you have all the required packages and dependencies installed in one step, saving you the effort of installing each package individually.
## Manual
### Backend Setup
1. Clone the repository:
```bash
git clone https://github.com/yourusername/workplace-monitoring-system.git
cd workplace-monitoring-system
```
2. Install the required Python packages:
```bash
pip install -r requirements.txt
```
3. Set up MongoDB:
Ensure MongoDB is running locally. You may need to configure the connection string in the FastAPI server code if your MongoDB setup differs.
4. Run the FastAPI server:
FastAPI server will start on port 8000.
### Frontend Setup
Navigate to the frontend directory , install required packages and starting the React development server:
```bash
cd frontend
npm install
npm start
```
This command will start the React development server on port 3000. You can access the frontend at http://localhost:3000.
Usage Guide
Recording Videos
To start recording videos, send a POST request to the /record_video/ endpoint with the following JSON payload:
```json
{
    "duration": 10,
    "fps": 24,
    "ip_address": "192.168.10.3",
    "port": 4747
}
```
Remember to write the correct IP address and port number for the IP camera.
The backend will record a video of the specified duration from the given IP camera and process it for various detections.

## Viewing Screenshots
- Open the React frontend in your browser at http://localhost:3000.
- Navigate through the screenshots using the "Next" and "Previous" buttons.
- Screenshots are displayed from the recordings, allowing you to review captured images.
## Viewing Detection Results
- Face Detection: Results are saved in the face_detection collection in MongoDB.
- Fight Detection: Results are saved in the fight_detection collection in MongoDB.
- Gun Detection: Results are saved in the gun_detection collection in MongoDB.
- Writing Detection: Results are saved in the writing_detection collection in MongoDB.
You can query these collections to access detection records and analyze them further.

## Future Enhancements
- Video Playback on Frontend: Integrate video playback functionality to view recorded videos directly in the React frontend.
- Enhanced Detection Models: Improve detection accuracy and add new detection capabilities.
## Contributing
Contributions are welcome! If you have any improvements or bug fixes, please submit a pull request.

## Acknowledgments
- [FastAPI](https://fastapi.tiangolo.com/)
- [React](https://react.dev/)
- [PyTorch](https://pytorch.org/)
- [Detectron2](https://github.com/facebookresearch/detectron2)
- [PyTorchVideo](https://pytorchvideo.org/)
- [Yolov7](https://github.com/WongKinYiu/yolov7)
