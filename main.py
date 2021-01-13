from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random, vstack, append, empty, zeros, concatenate, uint8, setdiff1d

from models.experimental import attempt_load
from utils.datasets import LoadImages, LoadWebcam
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.plots import plot_one_box
from utils.torch_utils import time_synchronized

from sort import Sort

from math import sqrt, acos, asin, degrees


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


def correct_the_tracks(tracks, img_x_shape: int, img_y_shape: int):
    for t in tracks:
        t[0] = 0 if t[0] < 0 else t[0]
        t[1] = 0 if t[1] < 0 else t[1]
        t[2] = img_x_shape if t[2] > img_x_shape else t[2]
        t[3] = img_y_shape if t[3] > img_y_shape else t[3]


# Source path video or 0 to camera
source = r"C:\Users\angel\Desktop\Video Dimauro\Video cellula sola\Validation video\sola (6).AVI"

debug = True
wait_key = 0 if debug else 1
proportion_vid = 0.25 if debug else 0.5
optical_flow_debug_tracks_id = [1]

# Yolov5 variables
weights_yolo = r"C:\Users\angel\Desktop\Video Dimauro\Dataset 1\Results\train yolov5s ep700 imgsize416 sd1\results\content\yolov5\runs\train\yolov5s_results\weights\best.pt"
img_size_yolo = 416
conf_thres_yolo = 0.45
iou_thres_yolo = 0.3

# SORT variables
max_age_sort = 20  # Maximum number of frames to keep alive a track without associated detections
min_hits_sort = 5  # Minimum number of associated detections before track is initialised
max_num_frame_reliable_sort = 3  # Max confidence to keep valid a detection not matched
iou_thres_sort = 0.00000000000000005

# Optical Flow variables
lk_params = {'status': None,
             'err': None,
             'winSize': (30, 30),
             'maxLevel': 2,
             'criteria': (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
             }
max_points_to_track = 200
min_distance_points = 10

# Create instance of SORT
mot_tracker = Sort(max_age=max_age_sort, min_hits=min_hits_sort, iou_threshold=iou_thres_sort)
matched_tracks = empty((0, 5))

# Create dictionary of currents ROI, previous ROI and movements
previous_frame = None
previous_tracks = empty((0, 5))
currents_p0 = dict()
previous_p0 = dict()
movements = dict()

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
        video_frame_count = video.count if isinstance(video, LoadWebcam) else video.frame

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

        # Marge and correct matched_tracks and unmatched_reliable_tracks
        tracks_to_show = vstack([matched_tracks, unmatched_reliable_tracks]).astype(int)
        correct_the_tracks(tracks_to_show, im0s.shape[1], im0s.shape[0])

        # Apply Optical Flow for each tracks_to_show
        for x1, y1, x2, y2, id_track in tracks_to_show:
            if id_track in previous_tracks[:, -1]:
                previous_roi = previous_frame.copy()[y1:y2, x1:x2]
                previous_gray_roi = cv2.cvtColor(previous_roi, cv2.COLOR_BGR2GRAY)
                current_roi = im0s.copy()[y1:y2, x1:x2]
                current_gray_roi = cv2.cvtColor(current_roi, cv2.COLOR_BGR2GRAY)

                # Apply Optical Flow TODO apply a mask at goodFeaturesToTrack
                previous_p0[id_track] = cv2.goodFeaturesToTrack(previous_gray_roi, max_points_to_track,
                                                                0.1, min_distance_points)
                currents_p0[id_track], status, err = cv2.calcOpticalFlowPyrLK(previous_gray_roi, current_gray_roi,
                                                                              previous_p0[id_track].copy(), None,
                                                                              **lk_params)

                # Sum of  movements
                mov = {'y': 0,
                       'x': 0,
                       'angle_to_x': 0,
                       'angle_to_y': 0,
                       'module': 0
                       }
                for i, (start, end) in enumerate(zip(previous_p0[id_track], currents_p0[id_track])):
                    x_start, y_start = start.ravel()
                    x_end, y_end = end.ravel()
                    mov['x'] = mov['x'] + (x_end - x_start)
                    mov['y'] = mov['y'] + (y_start - y_end)
                # Save movement value in mov[id_track][frame]
                mov['module'] = sqrt(mov['x']**2 + mov['y']**2)
                mov['angle_to_x'] = degrees(acos(mov['x'] / mov['module']))
                mov['angle_to_y'] = degrees(asin(mov['x'] / mov['module']))
                if movements.__contains__(id_track) is False:
                    movements[id_track] = {}
                movements[id_track].update({video_frame_count: mov})

        # Write output on im0s_to_show
        for x1, y1, x2, y2, id_track in tracks_to_show:
            label = f'{names[0]} n. {int(id_track)}'
            plot_one_box((x1, y1, x2, y2), im0s_to_show, label=label, color=colors[0], line_thickness=3)

        print(f'{s}Done. ({(t2 - t1):.3f}s) nt({len(tracks_to_show)})')

        # Stream debug results
        if debug:
            # Stream yolo detections on im0_yolo
            im0s_yolo = im0s.copy()
            for *xyxy, conf, cls in detections:
                label = '%s %.2f' % (names[int(cls)], conf)
                plot_one_box(xyxy, im0s_yolo, label=label, color=colors[int(cls)], line_thickness=3)
            im0s_yolo = cv2.resize(im0s_yolo, (0, 0), fx=proportion_vid, fy=proportion_vid)
            cv2.imshow('Yolo detection', im0s_yolo)

            # Stream all tracks from SORT tracker on im0_all_tracks
            im0s_all_tracks = im0s.copy()
            for t in mot_tracker.trackers:
                state = t.get_state()[0]
                id_track = t.id
                plot_one_box(state, im0s_all_tracks, label=str(id_track + 1), color=colors[int(cls)], line_thickness=3)
            im0s_all_tracks = cv2.resize(im0s_all_tracks, (0, 0), fx=proportion_vid, fy=proportion_vid)
            cv2.imshow('All tracks', im0s_all_tracks)

            # Stream matched and reliable tracks on im0s_matched_and_reliable_tracks
            im0s_matched_and_reliable_tracks = im0s.copy()
            for x1, y1, x2, y2, id_track in matched_tracks:
                plot_one_box((x1, y1, x2, y2), im0s_matched_and_reliable_tracks, label=f'Matched:{int(id_track)}',
                             color=colors[0], line_thickness=3)
            for x1, y1, x2, y2, id_track in unmatched_reliable_tracks:
                plot_one_box((x1, y1, x2, y2), im0s_matched_and_reliable_tracks, label=f'Reliable:{int(id_track)}',
                             color=(0, 0, 0), line_thickness=3)
            im0s_matched_and_reliable_tracks = cv2.resize(im0s_matched_and_reliable_tracks, (0, 0), fx=proportion_vid,
                                                          fy=proportion_vid)
            cv2.imshow('Matched and reliable tracks', im0s_matched_and_reliable_tracks)

            # Stream Optical Flow movement on roi_to_show
            for x1, y1, x2, y2, id_track in tracks_to_show:
                if id_track in optical_flow_debug_tracks_id:
                    if id_track in previous_tracks[:, -1]:
                        im0s_roi_copy = im0s.copy()[y1:y2, x1:x2]
                        # Add movement point on im0s_roi_copy
                        for i, (start, end) in enumerate(zip(previous_p0[id_track], currents_p0[id_track])):
                            x_start, y_start = start.astype(int).ravel()
                            x_end, y_end = end.astype(int).ravel()
                            cv2.circle(im0s_roi_copy, (x_end, y_end), 1, (0, 0, 255), 5)
                            cv2.arrowedLine(im0s_roi_copy, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2, 0)
                        # Create white_roi for movement results
                        shape = im0s_roi_copy.shape
                        white_roi = zeros(shape, dtype=uint8)
                        white_roi.fill(255)
                        center = {'x': int(shape[1]/2),
                                  'y': int(shape[0]/2)}
                        mov = movements[id_track][video_frame_count]
                        cv2.circle(white_roi, (center['x'], center['y']), 1, (0, 0, 0), 3)
                        cv2.arrowedLine(white_roi, (center['x'], center['y']),
                                        (center['x'] + round(mov['x']), center['y'] - round(mov['y'])),
                                        (0, 255, 0), 2, 0)
                        # Add text results to white_roi
                        cv2.putText(white_roi, f"Module: {mov['module']}", (1, shape[0]-2),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color=(0, 0, 0))
                        cv2.putText(white_roi, f"Angle to X: {mov['angle_to_x']}", (1, shape[0] - 22),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color=(0, 0, 0))
                        cv2.putText(white_roi, f"Angle to Y: {mov['angle_to_y']}", (1, shape[0] - 42),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color=(0, 0, 0))
                        cv2.putText(white_roi, f"Y: {mov['y']}", (1, shape[0] - 62),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color=(0, 0, 0))
                        cv2.putText(white_roi, f"X: {mov['x']}", (1, shape[0] - 82),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color=(0, 0, 0))
                        # Concatenate im0s_roi_copy and white_roi to stream the results
                        im0s_movement = concatenate((im0s_roi_copy, white_roi), axis=1)
                        cv2.imshow(f'Movement track n. {id_track}', im0s_movement)
            # Destroy dead windows 'Movement track n. {id_track}'
            for id_track in setdiff1d(previous_tracks[:, -1], tracks_to_show[:, -1]):
                cv2.destroyWindow(f'Movement track n. {id_track}')

        # Update previous_frame, previous_p0 and previous_tracks
        previous_frame = im0s.copy()
        previous_p0 = currents_p0.copy()
        previous_tracks = tracks_to_show.copy()

        # Stream out-put results img
        im0s_to_show = cv2.resize(im0s_to_show, (0, 0), fx=proportion_vid, fy=proportion_vid)
        cv2.imshow(str(p), im0s_to_show)
        if cv2.waitKey(wait_key) & 0xFF == 27:  # Press esc to exit
            raise StopIteration
