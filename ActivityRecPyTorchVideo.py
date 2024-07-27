import os
import shutil
import torch
import json
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import CenterCropVideo, NormalizeVideo
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.transforms import ApplyTransformToKey, ShortSideScale, UniformTemporalSubsample
import urllib
import gc
import cv2
import requests
import tempfile

def ActivityRecognition(filename):
    # Clear cache and free up memory
    gc.collect()
    torch.cuda.empty_cache()
    
    # Device selection
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load the pretrained SlowFast model
    model_name = "slowfast_r50"
    model = torch.hub.load("facebookresearch/pytorchvideo", model=model_name, pretrained=True)
    model = model.to(device)
    model = model.eval()

    # Load class names
    json_url = "https://dl.fbaipublicfiles.com/pyslowfast/dataset/class_names/kinetics_classnames.json"
    json_filename = "kinetics_classnames.json"
    try:
        urllib.request.urlretrieve(json_url, json_filename)
        with open(json_filename, "r") as f:
            kinetics_classnames = json.load(f)
    except Exception as e:
        print(f"Error downloading class names: {e}")
        return

    kinetics_id_to_classname = {v: k for k, v in kinetics_classnames.items()}

    # Define transforms
    side_size = 256
    mean = [0.45, 0.45, 0.45]
    std = [0.225, 0.225, 0.225]
    crop_size = 256
    num_frames = 32
    sampling_rate = 2
    frames_per_second = 30
    alpha = 4

    class PackPathway(torch.nn.Module):
        def forward(self, frames: torch.Tensor):
            fast_pathway = frames
            slow_pathway = torch.index_select(
                frames,
                1,
                torch.linspace(0, frames.shape[1] - 1, frames.shape[1] // alpha).long(),
            )
            return [slow_pathway, fast_pathway]

    transform = ApplyTransformToKey(
        key="video",
        transform=Compose([
            UniformTemporalSubsample(num_frames),
            Lambda(lambda x: x / 255.0),
            NormalizeVideo(mean, std),
            ShortSideScale(size=side_size),
            CenterCropVideo(crop_size),
            PackPathway()
        ]),
    )

    # Verify the video file exists
    if not os.path.exists(filename):
        print(f"Video file not found: {filename}")
        return

    # Load the video
    video = EncodedVideo.from_path(filename)
    total_duration = video.duration
    total_frames = int(total_duration * frames_per_second)

    # Sliding window parameters
    window_size = 6  # seconds per segment
    overlap = 1  # seconds of overlap

    # Function to predict and aggregate results
    def predict_windowed(video, window_size, overlap):
        clip_duration = (num_frames * sampling_rate) / frames_per_second
        post_act = torch.nn.Softmax(dim=1)
        actions = []

        for window_start in range(0, int(total_duration - window_size) + 1, window_size - overlap):
            window_end = window_start + window_size

            # Load video segment
            try:
                video_data = video.get_clip(start_sec=window_start, end_sec=window_end)
                video_data = transform(video_data)
                inputs = video_data["video"]
                inputs = [i.to(device)[None, ...] for i in inputs]

                # Predict actions
                with torch.no_grad():
                    preds = model(inputs)
                    preds = post_act(preds)
                    pred_classes = preds.topk(k=5).indices[0].cpu().numpy()
                    pred_probs = preds.topk(k=5).values[0].cpu().numpy()

                # Store the segment's predictions
                for i in range(len(pred_classes)):
                    class_name = kinetics_id_to_classname.get(int(pred_classes[i]), "Unknown")
                    actions.append((window_start, window_end, class_name, pred_probs[i]))

            except Exception as e:
                print(f"Error processing window {window_start}-{window_end}s: {e}")

        return actions

    # Predict actions in windowed segments
    predicted_actions = predict_windowed(video, window_size, overlap)

    # Process detected actions
    for start, end, label, prob in predicted_actions:
        if label == "writing" and prob > 0.1:
            print(f"Writing detected from {start}s to {end}s (confidence: {prob:.2f})")
            
            try:
                # Save the clip with annotation
                cap = cv2.VideoCapture(filename)
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                out_filename = f"{filename}_writing_{start}_{end}.avi"
                out = cv2.VideoWriter(out_filename, fourcc, frames_per_second, (int(cap.get(3)), int(cap.get(4))))

                cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
                while cap.get(cv2.CAP_PROP_POS_MSEC) <= end * 1000:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    cv2.putText(frame, 'Writing Detection', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                    out.write(frame)

                cap.release()
                out.release()

                # Move the video to the desired directory
                script_dir = os.path.dirname(__file__)
                video_output_dir = os.path.join(script_dir, 'Videos')
                os.makedirs(video_output_dir, exist_ok=True)
                final_path = os.path.join(video_output_dir, os.path.basename(out_filename))
                shutil.move(out_filename, final_path)

                # Post clip info to API
                clip_info = {
                    "filename": os.path.basename(final_path),
                    "original_file": os.path.basename(filename),
                    "start_time": start,
                    "end_time": end,
                    "confidence": float(prob),  # Convert to a JSON serializable type
                    "video_path": final_path
                }

                response = requests.post("http://localhost:8000/writing_detection/", json=clip_info)
                if response.status_code == 200:
                    print(f"Clip information saved successfully: {clip_info}")
                else:
                    print(f"Failed to save clip information: {response.text}")

            except Exception as e:
                print(f"Error saving clip: {e}")


