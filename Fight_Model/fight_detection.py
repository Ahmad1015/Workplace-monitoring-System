import cv2
import os
import torch
import numpy as np
import detectron2
from detectron2.config import get_cfg
from Writing_Activity_Model.ActivityRecPyTorchVideo import ActivityRecognition
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
import pytorchvideo
from pytorchvideo.transforms.functional import (
    uniform_temporal_subsample,
    short_side_scale_with_boxes,
    clip_boxes_to_image,
)
from torchvision.transforms._functional_video import normalize
from pytorchvideo.data.ava import AvaLabeledVideoFramePaths
from pytorchvideo.models.hub import slow_r50_detection  # Another option is slowfast_r50_detection
import matplotlib.pyplot as plt

def show_frame(frame):
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.title("Input Frame")
    plt.show()

def fight_detection(filename):
    print("Inside detection video file")
    
    # Define the Output Videos directory relative to the current script's location
    output_dir = os.path.join(os.path.dirname(__file__))
    print(f"Processed video saved at: {os.path.abspath(os.path.join(output_dir, f'processed_{filename}'))}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    video_model = slow_r50_detection(True)  # Another option is slowfast_r50_detection
    video_model = video_model.eval().to(device)

    cfg = get_cfg()
    cfg.MODEL.DEVICE = device
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.55
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    predictor = DefaultPredictor(cfg)

    def get_person_bboxes(inp_img, predictor):
        # Get predictions from the predictor
        predictions = predictor(inp_img.cpu().detach().numpy())['instances'].to('cpu')
        
        # Extract boxes, scores, and classes from the predictions
        boxes = predictions.pred_boxes if predictions.has("pred_boxes") else None
        scores = predictions.scores if predictions.has("scores") else None
        classes = np.array(predictions.pred_classes.tolist() if predictions.has("pred_classes") else None)
        
        # Convert scores to numpy array for compatible operation
        scores = scores.numpy() if scores is not None else None
        
        # Apply the mask for person class (class == 0) and high confidence scores
        if scores is not None and classes is not None:
            mask = (classes == 0) & (scores > 0.75)
            predicted_boxes = boxes[mask].tensor.cpu()
        else:
            predicted_boxes = torch.empty((0, 4))  # No predictions
        
        print(f"Detected {len(predicted_boxes)} prediction(s)")
        return predicted_boxes

    def ava_inference_transform(clip, boxes, num_frames=4, crop_size=256, data_mean=[0.45, 0.45, 0.45], data_std=[0.225, 0.225, 0.225], slow_fast_alpha=None):
        boxes = np.array(boxes)
        ori_boxes = boxes.copy()

        clip = uniform_temporal_subsample(clip, num_frames)
        clip = clip.float()
        clip = clip / 255.0

        height, width = clip.shape[2], clip.shape[3]
        boxes = clip_boxes_to_image(boxes, height, width)

        clip, boxes = short_side_scale_with_boxes(clip, size=crop_size, boxes=boxes)
        clip = normalize(clip, np.array(data_mean, dtype=np.float32), np.array(data_std, dtype=np.float32))
        boxes = clip_boxes_to_image(boxes, clip.shape[2], clip.shape[3])

        if slow_fast_alpha is not None:
            fast_pathway = clip
            slow_pathway = torch.index_select(clip, 1, torch.linspace(0, clip.shape[1] - 1, clip.shape[1] // slow_fast_alpha).long())
            clip = [slow_pathway, fast_pathway]

        return clip, torch.from_numpy(boxes), ori_boxes

    # Get the directory of the current script
    script_dir = os.path.dirname(__file__)
    # Construct the absolute path to the 'ava_action_list.pbtxt' file
    label_map_file = os.path.join(script_dir, 'ava_action_list.pbtxt')

    # Use the absolute path to read the label map
    label_map, allowed_class_ids = AvaLabeledVideoFramePaths.read_label_map(label_map_file)

    def process_video(video_path, output_path):
        try:
            # Video capture for reading the input video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Failed to open video file: {video_path}")

            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            print(f"Input Video FPS: {fps}, Frame Size: {frame_width}x{frame_height}")

            # Extract directory and filename
            output_dir = os.path.dirname(video_path)
            original_filename = os.path.basename(video_path)
            
            # Construct new filename with processed_ prefix
            processed_filename = f'processed_{original_filename}'
            
            # Construct the full output path
            processed_output_path = os.path.join(output_dir, processed_filename)

            print(f"Output video will be saved to: {processed_output_path}")

            # Initialize VideoWriter
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4 files
            output_video = cv2.VideoWriter(processed_output_path, fourcc, fps, (frame_width, frame_height))

            if not output_video.isOpened():
                raise ValueError(f"Failed to open VideoWriter for {processed_output_path}. Check codec, path, and permissions.")

            print("VideoWriter initialized successfully.")

            # First pass: collect all predictions
            encoded_vid = pytorchvideo.data.encoded_video.EncodedVideo.from_path(video_path)
            video_duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
            time_stamp_range = range(0, int(video_duration))

            predictions = []

            for time_stamp in time_stamp_range:
                inp_imgs = encoded_vid.get_clip(
                    time_stamp - 0.5,
                    time_stamp + 0.5
                )

                if inp_imgs is None or inp_imgs['video'] is None:
                    continue

                inp_imgs = inp_imgs['video']
                inp_img = inp_imgs[:, inp_imgs.shape[1] // 2, :, :]
                inp_img = inp_img.permute(1, 2, 0)

                predicted_boxes = get_person_bboxes(inp_img, predictor)
                if len(predicted_boxes) == 0:
                    continue

                inputs, inp_boxes, _ = ava_inference_transform(inp_imgs, predicted_boxes.numpy())
                inp_boxes = torch.cat([torch.zeros(inp_boxes.shape[0], 1), inp_boxes], dim=1)

                preds = video_model(inputs.unsqueeze(0).to(device), inp_boxes.to(device))
                preds = preds.to('cpu')
                preds = torch.cat([torch.zeros(preds.shape[0], 1), preds], dim=1)

                frame_predictions = []
                for i, box in enumerate(predicted_boxes):
                    scores = preds[i]
                    predicted_classes = [label_map[idx] for idx, score in enumerate(scores) if score > 0.5]
                    if predicted_classes:
                        frame_predictions.append(predicted_classes)
                
                predictions.append((time_stamp, frame_predictions))

            print("Finished generating predictions.")

            # Second pass: save annotated frames to output video
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            font = cv2.FONT_HERSHEY_SIMPLEX
            thickness = 2
            font_scale = 1
            color = (0, 255, 0)

            for time_stamp, frame_predictions in predictions:
                frame_number = int(time_stamp * fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

                for _ in range(int(fps)):
                    ret, frame = cap.read()
                    if not ret:
                        break

                    if frame_predictions:
                        combined_classes = set()
                        for predicted_classes in frame_predictions:
                            combined_classes.update(predicted_classes)

                        combined_classes_str = ', '.join(combined_classes)
                        cv2.putText(frame, combined_classes_str, (10, 30), font, font_scale, color, thickness, cv2.LINE_AA)

                    output_video.write(frame)

            cap.release()
            output_video.release()
            print(f"Processed video saved to {processed_output_path}")
            try:
                os.remove(video_path)
                print(f"The file '{video_path}' has been deleted successfully.")
            except FileNotFoundError:
                print(f"The file '{video_path}' does not exist.")

        except Exception as e:
            print(f"Error during video processing: {e}")

    process_video(filename, os.path.join(output_dir, f'processed_{filename}'))
        # Split the path into directory and filename
    directory, original_filename = os.path.split(filename)

    # Add 'processed_' before the original filename
    new_filename = 'processed_' + original_filename

    # Combine the directory path and the new filename
    processed_filename = os.path.join(directory, new_filename)
    print("Hello")
    print(processed_filename)
    ActivityRecognition(processed_filename)
