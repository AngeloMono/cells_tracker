from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random, vstack, append, empty

from models.experimental import attempt_load
from utils.datasets import LoadImages, LoadWebcam
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.plots import plot_one_box
from utils.torch_utils import time_synchronized

from sort import Sort


def get_unmatched_reliable_tracks(np_matched_tracks, sort_object: Sort, time_update_conf: int):
    return_tracks = empty((0, 5))
    for track in sort_object.trackers:
        if track.id + 1 not in np_matched_tracks[:, -1]:
            # Check the reliable conditions
            is_video_start = sort_object.frame_count <= sort_object.min_hits
            is_in_time_update_conf = track.time_since_update <= time_update_conf
            is_age_track_greater_than_min_hits = track.age >= sort_object.min_hits
            if is_video_start or (is_in_time_update_conf and is_age_track_greater_than_min_hits):
                # Append track to return_tracks
                row_to_append = append(track.get_state()[0], track.id + 1)
                return_tracks = vstack([return_tracks, row_to_append])
    return return_tracks


# Source path video or 0 to camera
source = r"C:\Users\angel\Desktop\Video Dimauro\Video cellule multiple in movimento\Train-test video\multi (2).mp4"

debug = True
wait_key = 0 if debug else 1
proportion_vid = 0.25 if debug else 0.5

# Yolov5 variables
weights_yolo = r"C:\Users\angel\Desktop\Video Dimauro\Dataset 1\Results\train yolov5s ep700 imgsize416 sd1\results\content\yolov5\runs\train\yolov5s_results\weights\best.pt"
img_size_yolo = 416
conf_thres_yolo = 0.40
iou_thres_yolo = 0.3

# SORT variables
max_age_sort = 20  # Maximum number of frames to keep alive a track without associated detections
min_hits_sort = 5  # Minimum number of associated detections before track is initialised
max_num_frame_reliable_sort = 3  # Max confidence to keep valid a detection not matched
iou_thres_sort = 0.00000000000000005

# Create instance of SORT
mot_tracker = Sort(max_age=max_age_sort, min_hits=min_hits_sort, iou_threshold=iou_thres_sort)
matched_tracks = empty((0, 5))

# Get source properties
vidcap = cv2.VideoCapture(source)
fps = vidcap.get(cv2.CAP_PROP_FPS)

with torch.no_grad():
    # Initalize
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Torch device used
    half = device.type != 'cpu'  # Half precision  only supported on CUDA

    # Load model
    model = attempt_load(weights_yolo, map_location=device)  # load FP32 model
    imgsz = check_img_size(img_size_yolo, s=model.stride.max())  # Check img_size
    if half:
        model.half()  # To FP16 (only supported on CUDA)

    # Extract names and assign a color RGB to each name
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Load video
    if str(source) == '0':
        cudnn.benchmark = True  # Set True to speed up constant image size inference
        video = LoadWebcam(img_size=img_size_yolo, verbose=debug)
    else:
        video = LoadImages(source, img_size=img_size_yolo, verbose=debug)

    # Process each frame in video
    for path, img, im0s, vid_cap in video:  # And print (tot_frame\current_frame) while call the __next__method
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0

        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        detections = model(img, augment=False)[0]

        # Apply NMS
        detections = non_max_suppression(detections, conf_thres_yolo, iou_thres_yolo, classes=None, agnostic=False)
        detections = detections[0]
        t2 = time_synchronized()

        # Processing detections predicted
        p, s, im0s_to_show = Path(path), '', im0s.copy()
        s += '%gx%g ' % img.shape[2:]  # Print string

        if len(detections):
            # Rescale boxes from img_size to im0 size
            detections[:, :4] = scale_coords(img.shape[2:], detections[:, :4], im0s_to_show.shape).round()

            # SORT tracker application
            np_det = detections.cpu().numpy()
            matched_tracks = mot_tracker.update(np_det[:, 0:-1])

            # Get unmatched reliable tracks
            unmatched_reliable_tracks = get_unmatched_reliable_tracks(matched_tracks, mot_tracker,
                                                                      max_num_frame_reliable_sort)

            # Print results
            for c in detections[:, -1].unique():
                n = (detections[:, -1] == c).sum()  # Detections for class
                s += '%g %ss, ' % (n, names[int(c)])  # Add number and name of detection to string

        else:
            matched_tracks = mot_tracker.update()
            unmatched_reliable_tracks = get_unmatched_reliable_tracks(matched_tracks, mot_tracker,
                                                                      max_num_frame_reliable_sort)

        tracks_to_show = vstack([matched_tracks, unmatched_reliable_tracks])

        # TODO applicare Optical Flow ad ogni traccia da mostrare

        # Write tracks on im0s_to_show
        for x1, y1, x2, y2, id_track in tracks_to_show:
            label = f'{names[0]} n. {int(id_track)}'
            plot_one_box((x1, y1, x2, y2), im0s_to_show, label=label, color=colors[0], line_thickness=3)

        print(f'{s}Done. ({(t2 - t1):.3f}s) nt({len(tracks_to_show)})')

        # Stream results
        if debug:
            # Write yolo detections on im0_yolo
            im0s_yolo = im0s.copy()
            for *xyxy, conf, cls in detections:
                label = '%s %.2f' % (names[int(cls)], conf)
                plot_one_box(xyxy, im0s_yolo, label=label, color=colors[int(cls)], line_thickness=3)

            # Write all tracks from SORT tracker on im0_all_tracks
            im0s_all_tracks = im0s.copy()
            for t in mot_tracker.trackers:
                state = t.get_state()[0]
                id_track = t.id
                plot_one_box(state, im0s_all_tracks, label=str(id_track+1), color=colors[int(cls)], line_thickness=3)

            # Write matched and reliable tracks on im0s_matched_and_reliable_tracks
            im0s_matched_and_reliable_tracks = im0s.copy()
            for x1, y1, x2, y2, id_track in matched_tracks:
                plot_one_box((x1, y1, x2, y2), im0s_matched_and_reliable_tracks, label=f'Matched:{int(id_track)}', color=colors[0], line_thickness=3)
            for x1, y1, x2, y2, id_track in unmatched_reliable_tracks:
                plot_one_box((x1, y1, x2, y2), im0s_matched_and_reliable_tracks, label=f'Reliable:{int(id_track)}', color=(0, 0, 0), line_thickness=3)

            # Stream  debug results img
            im0s_yolo = cv2.resize(im0s_yolo, (0, 0), fx=proportion_vid, fy=proportion_vid)
            cv2.imshow('Yolo detection', im0s_yolo)

            im0s_all_tracks = cv2.resize(im0s_all_tracks, (0, 0), fx=proportion_vid, fy=proportion_vid)
            cv2.imshow('All tracks', im0s_all_tracks)

            im0s_matched_and_reliable_tracks = cv2.resize(im0s_matched_and_reliable_tracks, (0, 0), fx=proportion_vid, fy=proportion_vid)
            cv2.imshow('Matched and reliable tracks', im0s_matched_and_reliable_tracks)

        # Stream out-put results img
        im0s_to_show = cv2.resize(im0s_to_show, (0, 0), fx=proportion_vid, fy=proportion_vid)
        cv2.imshow(str(p), im0s_to_show)
        if cv2.waitKey(wait_key) & 0xFF == 27:  # Press esc to exit
            raise StopIteration
