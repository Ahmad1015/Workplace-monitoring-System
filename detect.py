import argparse
import time
from pathlib import Path
import shutil
import requests
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import os
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

output_dir = os.path.dirname(__file__)

def detect(opt):
    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    save_img = not opt.nosave and not source.endswith('.txt')
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)
    videos_dir = Path('Videos')
    videos_dir.mkdir(parents=True, exist_ok=True)

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'

    # Load model
    model = attempt_load(weights, map_location=device)
    stride = int(model.stride.max())
    imgsz = check_img_size(imgsz, s=stride)

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()
    gun_detected = False
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():
            pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):
            if webcam:
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)
            save_path = str(save_dir / p.name)
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "

                if any(int(cls) == 0 for *xyxy, conf, cls in det):
                    gun_detected = True

                for *xyxy, conf, cls in reversed(det):
                    if save_txt:
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)

            print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

            if gun_detected:
                cv2.putText(im0, "Gun Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)

            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    print(f"The image with the result is saved in: {save_path}")
                else:
                    if vid_path != save_path:
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()
                        if vid_cap:
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        print(f"Results saved to {save_dir}")
        if save_txt:
            print(f'Labels saved to {save_dir / "labels"}')

    if save_img:
        for file in save_dir.glob('*.mp4'):
            if isinstance(vid_writer, cv2.VideoWriter):
                vid_writer.release()
            shutil.move(str(file), str(videos_dir / file.name))
            print(f"Moved {file.name} to {videos_dir}")

    print(f'Done. ({time.time() - t0:.3f}s)')
    shutil.rmtree("exp")
    detection_record = {
        "name": "Gun Detection",
        "video_path": output_dir+str(videos_dir)+"/"+str(file.name),
        "timestamp": time.time()
    }
    response = requests.post("http://localhost:8000/gun_detection/", json=detection_record)
    if response.status_code == 200:
        print("Detection record saved successfully.")
    else:
        print("Failed to save detection record.")

def run_detection(weights, source, img_size=640, conf_thres=0.10, iou_thres=0.45, device='cuda', view_img=False, save_txt=False, save_conf=False, nosave=False, classes=None, agnostic_nms=False, augment=False, project='runs/detect', name='Videos', exist_ok=False, no_trace=True):
    class Opt:
        def __init__(self, weights, source, img_size, conf_thres, iou_thres, device, view_img, save_txt, save_conf, nosave, classes, agnostic_nms, augment, project, name, exist_ok, no_trace):
            self.weights = weights
            self.source = source
            self.img_size = img_size
            self.conf_thres = conf_thres
            self.iou_thres = iou_thres
            self.device = device
            self.view_img = view_img
            self.save_txt = save_txt
            self.save_conf = save_conf
            self.nosave = nosave
            self.classes = classes
            self.agnostic_nms = agnostic_nms
            self.augment = augment
            self.project = project
            self.name = name
            self.exist_ok = exist_ok
            self.no_trace = no_trace

    opt = Opt(weights, source, img_size, conf_thres, iou_thres, device, view_img, save_txt, save_conf, nosave, classes, agnostic_nms, augment, project, name, exist_ok, no_trace)
    detect(opt)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.75, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        if opt.update:
            for opt.weights in ['yolov7.pt']:
                detect(opt)
                strip_optimizer(opt.weights)
        else:
            detect(opt)
