import torch
import json
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import CenterCropVideo, NormalizeVideo
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.transforms import ApplyTransformToKey, ShortSideScale, UniformTemporalSubsample
import urllib
import subprocess
from Writing_Activity_Model.yolo import detect  



def ActivityRecognition(filename):
    print(filename)
    # Device on which to run the model
    device = "cpu"

    # Load the pretrained model
    model_name = "slowfast_r50"
    model = torch.hub.load("facebookresearch/pytorchvideo", model=model_name, pretrained=True)
    model = model.to(device)
    model = model.eval()

    # Load class names
    try:
        json_url = "https://dl.fbaipublicfiles.com/pyslowfast/dataset/class_names/kinetics_classnames.json"
        json_filename = "kinetics_classnames.json"
        urllib.request.urlretrieve(json_url, json_filename)
    except:
        pass
    with open(json_filename, "r") as f:
        kinetics_classnames = json.load(f)

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
        def __init__(self):
            super().__init__()

        def forward(self, frames: torch.Tensor):
            fast_pathway = frames
            slow_pathway = torch.index_select(
                frames,
                1,
                torch.linspace(0, frames.shape[1] - 1, frames.shape[1] // alpha).long(),
            )
            frame_list = [slow_pathway, fast_pathway]
            return frame_list

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

    # Load the example video
    video_path = f"{filename}"
    video = EncodedVideo.from_path(video_path)

    # Define sliding window parameters
    window_size = 4  # seconds per segment
    overlap = 2  # seconds of overlap

    # Duration and total frames
    total_duration = video.duration
    total_frames = int(total_duration * frames_per_second)

    # Function to predict and aggregate results
    def predict_windowed(video, start_sec, end_sec, window_size, overlap):
        clip_duration = (num_frames * sampling_rate) / frames_per_second
        post_act = torch.nn.Softmax(dim=1)
        actions = []

        for window_start in range(0, int(total_duration - window_size) + 1, window_size - overlap):
            window_end = window_start + window_size

            # Load video segment
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

        return actions

    # Predict actions in windowed segments
    predicted_actions = predict_windowed(video, 0, total_duration, window_size, overlap)

    # Display predictions
    for start, end, label, prob in predicted_actions:
        print(f"From {start}s to {end}s: {label} (confidence: {prob:.2f})")
    
    with open(f"{filename}.txt", "w") as file:
        for start, end, label, prob in predicted_actions:
            if prob > 0.7:  # Only write actions with confidence above 0.7
                file.write(f"From {start}s to {end}s: {label} (confidence: {prob:.2f})\n")
    # Full path to detect.py
    detect_script = r"F:\Policy-Based-Presence-Tracker-UI\Writing_Activity_Model\yolo\detect.py"

    # Full path to weights file
    weights_path = r"F:\Policy-Based-Presence-Tracker-UI\Writing_Activity_Model\yolo\best.pt"

    
    # Define the command and arguments
    command = ["python", detect_script, "--weights", weights_path, "--source", filename]

    # Run the command
    result = subprocess.run(command, capture_output=True, text=True)

    # Print the output
    print("Output from Yolo:")
    print(result.stdout)
    print(result.stderr)

    

if __name__ == "__main__":
    ActivityRecognition("processed_output_1.mp4")