from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadImages, LoadWebcam
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.plots import plot_one_box
from utils.torch_utils import time_synchronized

# Source path video or 0 to camera
source = r"C:\Users\angel\Desktop\Video Dimauro\Video cellula sola\Trai-test video\sola (3).mp4"

# Yolov5 variables
weights = r"C:\Users\angel\Desktop\Video Dimauro\Dataset 1\Results\train yolov5s ep700 imgsize416 sd1\results\content\yolov5\runs\train\yolov5s_results\weights\best.pt"
img_size = 416
conf_thres = 0.25
iou_thres = 0.45

# Get source properties
vidcap = cv2.VideoCapture(source)
fps = vidcap.get(cv2.CAP_PROP_FPS)

# Initalize
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Torch device used
half = device.type != 'cpu'  # Half precision  only supported on CUDA

# Load model
model = attempt_load(weights, map_location=device)  # load FP32 model
imgsz = check_img_size(img_size, s=model.stride.max())  # Check img_size
if half:
    model.half()  # To FP16 (only supported on CUDA)

# Extract names and assign a color RGB to each name
names = model.module.names if hasattr(model, 'module') else model.names
colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

# Load video
if str(device) == '0':
    cudnn.benchmark = True  # Set True to speed up constant image size inference
    video = LoadWebcam(img_size=img_size)
else:
    video = LoadImages(source, img_size=img_size)

# Detect for each frame in video
for path, img, im0s, vid_cap in video:
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0

    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    t1 = time_synchronized()
    pred = model(img, augment=False)[0]

    # Apply NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes=None, agnostic=False)
    t2 = time_synchronized()

    # Process detections predicted
    for i, det in enumerate(pred):
        if str(device) == '0':  # Batch_size >= 1
            p, s, im0 = Path(path[i]), '%g: ' % i, im0s[i].copy()
        else:
            p, s, im0 = Path(path), '', im0s

        s += '%gx%g ' % img.shape[2:]  # Print string

        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # Detections for class
                s += '%g %ss, ' % (n, names[int(c)])  # Add number and name of detection to string

            # Write results on im0
            for *xyxy, conf, cls in reversed(det):
                label = '%s %.2f' % (names[int(cls)], conf)  # Create label for detection
                plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3) # Add plot (xy,xy) to img

        # Stream results
        cv2.imshow(str(p), im0)
        if cv2.waitKey(1) & 0xFF == 27:  # Press esc to exit
            raise StopIteration

        print('%sDone. (%.3fs)' % (s, t2 - t1))
