import cv2
import os
import torch
import numpy as np
import shutil
import logging
import requests
from datetime import datetime
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from pytorchvideo.transforms.functional import (
    uniform_temporal_subsample,
    short_side_scale_with_boxes,
    clip_boxes_to_image,
)
from torchvision.transforms._functional_video import normalize
from pytorchvideo.data.ava import AvaLabeledVideoFramePaths
from pytorchvideo.models.hub import slow_r50_detection
from pytorchvideo.data.encoded_video import EncodedVideo
from ActivityRecPyTorchVideo import ActivityRecognition
from detect import run_detection
import gc

output_dir = os.path.dirname(__file__)
# Configure logging
logging.basicConfig(filename='fight_detection.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

async def fight_detection(filename):
    print("Inside Fight Detection")
    gc.collect()
    torch.cuda.empty_cache()
    # Print GPU memory status
    reserved_memory = torch.cuda.memory_reserved()
    allocated_memory = torch.cuda.memory_allocated()
    print(f"Reserved memory: {reserved_memory / 1e9:.2f} GB")
    print(f"Allocated memory: {allocated_memory / 1e9:.2f} GB")

    logging.info(f"Started fight detection on: {filename}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    video_model = slow_r50_detection(True)
    video_model = video_model.eval().to(device)
    logging.info(f"Using device: {device}")

    cfg = get_cfg()
    cfg.MODEL.DEVICE = device
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.55
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    predictor = DefaultPredictor(cfg)

    def get_person_bboxes(inp_img, predictor):
        predictions = predictor(inp_img.cpu().detach().numpy())['instances'].to('cpu')
        boxes = predictions.pred_boxes if predictions.has("pred_boxes") else None
        scores = predictions.scores if predictions.has("scores") else None
        classes = np.array(predictions.pred_classes.tolist() if predictions.has("pred_classes") else None)
        scores = scores.numpy() if scores is not None else None

        if scores is not None and classes is not None:
            mask = (classes == 0) & (scores > 0.75)
            predicted_boxes = boxes[mask].tensor.cpu()
        else:
            predicted_boxes = torch.empty((0, 4))

        logging.info(f"Detected {len(predicted_boxes)} person(s) in frame.")
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

    script_dir = os.path.dirname(__file__)
    label_map_file = os.path.join(script_dir, 'ava_action_list.pbtxt')
    label_map, allowed_class_ids = AvaLabeledVideoFramePaths.read_label_map(label_map_file)

    def process_video(video_path):
        try:
            # Ensure the output directory exists
            output_dir = os.path.dirname(video_path)
            fight_detection_dir = os.path.join(output_dir, 'Videos')
            if not os.path.exists(fight_detection_dir):
                os.makedirs(fight_detection_dir)

            # Make a copy of the original video
            original_video_copy = os.path.join(output_dir, f'copy_{os.path.basename(video_path)}')
            shutil.copyfile(video_path, original_video_copy)
            logging.info(f"Made a copy of the original video: {original_video_copy}")

            cap = cv2.VideoCapture(original_video_copy)
            if not cap.isOpened():
                raise ValueError(f"Failed to open video file: {original_video_copy}")

            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            logging.info(f"Input Video FPS: {fps}, Frame Size: {frame_width}x{frame_height}")

            # Output video file path
            processed_filename = f'processed_{os.path.basename(video_path)}'
            processed_output_path = os.path.join(output_dir, processed_filename)
            logging.info(f"Output video will be saved to: {processed_output_path}")

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4 files
            output_video = cv2.VideoWriter(processed_output_path, fourcc, fps, (frame_width, frame_height))

            if not output_video.isOpened():
                raise ValueError(f"Failed to open VideoWriter for {processed_output_path}. Check codec, path, and permissions.")

            logging.info("VideoWriter initialized successfully.")

            # Using EncodedVideo to read the video
            encoded_vid = EncodedVideo.from_path(original_video_copy)
            video_duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
            time_stamp_range = range(0, int(video_duration))

            predictions = []
            fight_detected = False

            for time_stamp in time_stamp_range:
                inp_imgs = encoded_vid.get_clip(time_stamp - 0.5, time_stamp + 0.5)

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

                # Check if 'fight' is detected
                for predicted_classes in frame_predictions:
                    if 'fight/hit (a person)' in predicted_classes:
                        fight_detected = True
                        break
                if fight_detected:
                    break  # We can stop early if a fight is detected

            logging.info("Finished generating predictions.")

            if fight_detected:
                logging.info("Fight detected. Preparing to save the video.")

                cap.release()  # Release the video capture object

                cap = cv2.VideoCapture(original_video_copy)
                frame_count = 0

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # Overlay text on all frames if fight detected
                    cv2.putText(frame, 'Fight Detected', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

                    output_video.write(frame)
                    frame_count += 1

                cap.release()  # Release the video capture object again

                # Release the video writer object
                output_video.release()
                logging.info("Released video writer resources.")

                # Move the processed video to the fight detection folder
                new_video_path = os.path.join(fight_detection_dir, processed_filename)
                shutil.move(processed_output_path, new_video_path)

                # Send fight detection record to FastAPI
                detection_record = {
                    "name": f"Fight detected in {os.path.basename(video_path)}",
                    "video_path": new_video_path,
                    "timestamp": datetime.now().isoformat()
                }

                fastapi_url = "http://localhost:8000/fight_detection/"
                try:
                    response = requests.post(fastapi_url, json=detection_record)
                    response.raise_for_status()
                    logging.info(f"Fight detection sent to FastAPI. Response: {response.json()}")
                except requests.exceptions.RequestException as e:
                    logging.error(f"Failed to send fight detection to FastAPI: {e}")

            else:
                logging.info("No fight detected in the video.")

            # Ensure all resources are released before attempting to delete the file
            cap.release()
            output_video.release()

            # Remove the original video copy
            if os.path.exists(original_video_copy):
                os.remove(original_video_copy)
                logging.info(f"Deleted the original video copy: {original_video_copy}")

        except PermissionError as e:
            logging.error(f"PermissionError: {e}")
            raise
        except Exception as e:
            logging.error(f"Error processing video {video_path}: {e}")
            raise

    process_video(filename)
    logging.info(f"Fight detection completed for: {filename}")
    gc.collect()
    torch.cuda.empty_cache()
    # Print GPU memory status
    reserved_memory = torch.cuda.memory_reserved()
    allocated_memory = torch.cuda.memory_allocated()
    print(f"Reserved memory: {reserved_memory / 1e9:.2f} GB")
    print(f"Allocated memory: {allocated_memory / 1e9:.2f} GB")
    print("Going for Yolo Detection")
    # Call YOLOv7 detection on the original video file and save output in script directory
    run_detection(
        weights='best.pt',
        source=filename,
        device='cuda',
        view_img=True,
        save_txt=True,
        save_conf=True,
        nosave=False,
        classes=None,
        agnostic_nms=False,
        augment=False,
        project=output_dir,  # Save output in the script directory
        name='exp',
        exist_ok=True,
        no_trace=True,
    )
    gc.collect()
    torch.cuda.empty_cache()
    print("Going into Activity Recognition")
    ActivityRecognition(filename)
    try:
        os.remove(filename)
    except:
        print("Error removing video file")
