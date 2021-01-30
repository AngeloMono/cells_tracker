from pathlib import Path
import os
import shutil

import cv2
import torch
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
from warnings import filterwarnings

from numpy import random, vstack, append, empty, zeros, concatenate, uint8, setdiff1d, arctan, ndarray
from math import sqrt, degrees, radians, cos, sin, tan

from models.experimental import attempt_load
from yolo_utils.datasets import LoadImages, LoadWebcam
from yolo_utils.general import check_img_size, non_max_suppression, scale_coords
from yolo_utils.plots import plot_one_box
from yolo_utils.torch_utils import time_synchronized

from sort import Sort

from yolact import Yolact
from yolact_utils.augmentations import FastBaseTransform
from yolact_utils.functions import SavePath
from data import cfg, set_cfg
from layers.output_utils import postprocess


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


def get_mask_bbox_and_score(yolact_net: Yolact, img, threshold=0.0, max_predictions=1):
    with torch.no_grad():
        frame = torch.from_numpy(img).cuda().float()
        batch = FastBaseTransform()(frame.unsqueeze(0))
        preds = yolact_net(batch)

        h, w, _ = img.shape

        save = cfg.rescore_bbox
        cfg.rescore_bbox = True
        t = postprocess(preds, w, h, visualize_lincomb=False, crop_masks=True)
        cfg.rescore_bbox = save

        idx = t[1].argsort(0, descending=True)[:max_predictions]
        classes, scores, boxes, masks = [x[idx].cpu().numpy() for x in t[:]]

        num_dets_to_consider = min(max_predictions, classes.shape[0])
        # Remove dets below the threshold
        for j in range(num_dets_to_consider):
            if scores[j] < threshold:
                num_dets_to_consider = j
                break
        masks_to_return = boxes_to_return = scores_to_return = None
        if num_dets_to_consider > 0:
            masks = masks[:num_dets_to_consider, :, :, None]
            masks_to_return = []
            boxes_to_return = []
            scores_to_return = []
            for m, b, s in zip(masks, boxes, scores):
                masks_to_return.append(m)
                boxes_to_return.append(b)
                scores_to_return.append(s)
            if len(masks_to_return) == 1:
                masks_to_return = masks_to_return[0]
            if len(boxes_to_return) == 1:
                boxes_to_return = boxes_to_return[0]
            if len(scores_to_return) == 1:
                scores_to_return = scores_to_return[0]
        return masks_to_return, boxes_to_return, scores_to_return


def get_straight_line_from_2_points(p1, p2):
    """
    :param p1: point1
    :param p2: point2
    :return: m1, m2, and q of the given equation "(m1)x - [(m2)y] + q = 0" that passes through the given points
    """
    if p1[0] == p2[0] and p1[1] == p2[1]:
        raise ValueError("Infinite rette")
    elif p1[0] == p2[0]:
        return -1, 0, p1[0]
    m = float(p2[1] - p1[1]) / (p2[0] - p1[0])
    q = p1[1] - (m * p1[0])
    return m, 1, q


def get_degrees_from_the_x_axis(vector_x, vector_y):
    if vector_x == 0 and vector_y == 0:
        return None
    elif vector_x == 0:
        return 90 if vector_y > 0 else 270
    elif vector_y == 0:
        return 0 if vector_x > 0 else 180
    else:
        deg = degrees(arctan(vector_y / vector_x))
        if vector_x < 0 and vector_y < 0:
            deg = 180 + deg
        elif vector_x < 0 and vector_y > 0:
            deg = 180 + deg
        elif vector_x > 0 and vector_y < 0:
            deg = 360 + deg
    return deg


def remove_unmatched_vals(vals: list, dict: dict):
    dict_to_return = dict.copy()
    for id_dict in dict:
        if id_dict not in vals:
            dict_to_return.pop(id_dict)
    return dict_to_return


def get_avg_frames_period(mov_dict: dict, previous_avg, n_frame_to_consider: int, actual_frame: int,
                          frame_step: int = 1):
    start_frame = actual_frame - n_frame_to_consider
    count = 0
    # Assign start frame and count = 0
    if start_frame != 0:
        while mov_dict.__contains__(start_frame - count) and (
                mov_dict[start_frame] * mov_dict[start_frame - count]) > 0:
            count = count + frame_step
        start_frame = start_frame - count + frame_step
    else:
        while mov_dict.__contains__(start_frame) is False:
            start_frame = start_frame + frame_step
        while mov_dict.__contains__(start_frame + count) and (
                mov_dict[start_frame] * mov_dict[start_frame + count]) > 0:
            count = count + frame_step
        start_frame = start_frame + count
    count = 0

    # Calculate the distinct periods
    list_calculated_period = list()
    is_in_other_half_of_start = False
    while start_frame + count <= actual_frame:
        if mov_dict.__contains__(start_frame + count):
            if is_in_other_half_of_start:
                if mov_dict[start_frame] * mov_dict[start_frame + count] > 0:
                    # Check reliability
                    if mov_dict.__contains__(start_frame + count + frame_step) and mov_dict[
                        start_frame + count + frame_step] * mov_dict[start_frame] > 0:
                        list_calculated_period.append(count)
                    start_frame = start_frame + count
                    count = 0
                    is_in_other_half_of_start = False
                else:
                    count = count + frame_step
            else:
                if mov_dict[start_frame] * mov_dict[start_frame + count] < 0:
                    is_in_other_half_of_start = True
                count = count + frame_step
        else:
            start_frame = start_frame + count + frame_step
            count = 0
            is_in_other_half_of_start = False

    # Get the avg of calculated periods
    if len(list_calculated_period) != 0:
        avg_to_return = sum(list_calculated_period) / len(list_calculated_period)
        if previous_avg != 0 and previous_avg is not None:
            avg_to_return = (avg_to_return + previous_avg) / 2
    elif previous_avg != 0 and previous_avg is not None:
        avg_to_return = previous_avg
    else:
        avg_to_return = 0
    return avg_to_return


def get_bpm_from_frames_period(frames_period, fps_video):
    return round(60 * fps_video / frames_period, 2)


# Source path video or 0 to camera
source = r"C:\Users\angel\Desktop\Video Dimauro\Video cellula sola\Train-test video\sola (1).mp4"

debug = False
wait_key = 0 if debug else 1
save_output = True
proportion_vid = 0.25 if debug else 0.5
optical_flow_debug_tracks_id = [1]

# Yolov5 variables
yolo_weights = r"C:\Users\angel\Desktop\Video Dimauro\Dataset Yolo\Results\train yolov5s ep700 imgsize416 sd1\results\content\yolov5\runs\train\yolov5s_results\weights\best.pt"
yolo_img_size = 416
yolo_conf_thres = 0.45
yolo_iou_threshold = 0.3

# SORT variables
max_age_sort = 20  # Maximum number of frames to keep alive a track without associated detections
min_hits_sort = 5  # Minimum number of associated detections before track is initialised
max_num_frame_reliable_sort = 3  # Max confidence to keep valid a detection not matched
iou_thres_sort = 0.00000000000000005

# Yolact variables
yolact_weight = r"C:\Users\angel\Desktop\Video Dimauro\Dataset Yolact\Results\content\yolact\weights\yolact_resnet50_cilia_133_5614_interrupt.pth"
yolact_model_path = SavePath.from_str(yolact_weight)
yolact_config = yolact_model_path.model_name + '_config'
yolact_threshold = 0
yolact_num_max_predictions = 1

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
angle_movements_axis_to_x_axis = dict()
cilia_movements = dict()
avg_frames_period = dict()
bpm = dict()
if debug:
    currents_masks = dict()
    currents_bbox_masks = dict()
    movements_to_show = dict()
if save_output:
    img_video_output = []
    video_output_size = None

# Ignore UserWarning created by yolact
filterwarnings("ignore", category=UserWarning)

with torch.no_grad():
    # Initalize
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Torch device used
    cuda = device.type != 'cpu'  # Half precision  only supported on CUDA

    # Load Yolo  and Yolact models
    yolo_model = attempt_load(yolo_weights, map_location=device)  # load FP32 model
    yolo_img_size = check_img_size(yolo_img_size, s=yolo_model.stride.max())  # Check img_size
    set_cfg(yolact_config)
    if cuda:
        yolo_model.half()  # To FP16 (only supported on CUDA)
        cudnn.fastest = True
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    yolact_model = Yolact()
    yolact_model.load_weights(yolact_weight)
    yolact_model.eval()
    if cuda:
        yolact_model = yolact_model.cuda()
    yolact_model.detect.use_fast_nms = True
    cfg.mask_proto_debug = False

    # Extract names and assign a color RGB to each name
    names = yolo_model.module.names if hasattr(yolo_model, 'module') else yolo_model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Load video
    if str(source) == '0':
        cudnn.benchmark = True  # Set True to speed up constant image size inference
        video = LoadWebcam(img_size=yolo_img_size, verbose=debug)
    else:
        video = LoadImages(source, img_size=yolo_img_size, verbose=debug)
    fps = round(video.cap.get(cv2.CAP_PROP_FPS))

    # Process each frame in video
    for path, img, im0s, vid_cap in video:  # And print (tot_frame\current_frame) while call the __next__ method
        video_frame_count = video.count if isinstance(video, LoadWebcam) else video.frame

        img = torch.from_numpy(img).to(device)
        img = img.half() if cuda else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0

        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        yolo_detections = yolo_model(img, augment=False)[0]

        # Apply NMS
        yolo_detections = non_max_suppression(yolo_detections, yolo_conf_thres, yolo_iou_threshold, classes=None,
                                              agnostic=False)
        yolo_detections = yolo_detections[0]
        t2 = time_synchronized()

        # Processing detections predicted
        p, s, im0s_to_show = Path(path), '', im0s.copy()
        s += '%gx%g ' % img.shape[2:]  # Print string

        if len(yolo_detections):
            # Rescale boxes from img_size to im0 size
            yolo_detections[:, :4] = scale_coords(img.shape[2:], yolo_detections[:, :4], im0s_to_show.shape).round()

            # SORT tracker application
            np_det = yolo_detections.cpu().numpy()
            matched_tracks = mot_tracker.update(np_det[:, 0:-1])

            # Get unmatched reliable tracks
            unmatched_reliable_tracks = get_unmatched_reliable_tracks(matched_tracks, mot_tracker,
                                                                      max_num_frame_reliable_sort)

            # Print results
            for c in yolo_detections[:, -1].unique():
                n = (yolo_detections[:, -1] == c).sum()  # Detections for class
                s += '%g %ss, ' % (n, names[int(c)])  # Add number and name of detection to string
        else:
            matched_tracks = mot_tracker.update()
            unmatched_reliable_tracks = get_unmatched_reliable_tracks(matched_tracks, mot_tracker,
                                                                      max_num_frame_reliable_sort)

        # Marge and correct matched_tracks and unmatched_reliable_tracks
        tracks_to_show = vstack([matched_tracks, unmatched_reliable_tracks]).astype(int)
        correct_the_tracks(tracks_to_show, im0s.shape[1], im0s.shape[0])

        # Calculate the movement of the cilia
        for x1, y1, x2, y2, id_track in tracks_to_show:
            if id_track in previous_tracks[:, -1]:
                previous_roi = previous_frame.copy()[y1:y2, x1:x2]
                previous_gray_roi = cv2.cvtColor(previous_roi, cv2.COLOR_BGR2GRAY)
                current_roi = im0s.copy()[y1:y2, x1:x2]
                current_gray_roi = cv2.cvtColor(current_roi, cv2.COLOR_BGR2GRAY)

                # Create mask using Yolact
                mask, bbox_mask, _ = get_mask_bbox_and_score(yolact_model, previous_roi, threshold=yolact_threshold,
                                                             max_predictions=yolact_num_max_predictions)

                # Apply Optical Flow
                mask = mask.astype(uint8) if isinstance(mask, ndarray) else None
                previous_p0[id_track] = cv2.goodFeaturesToTrack(previous_gray_roi, max_points_to_track,
                                                                0.1, min_distance_points, mask=mask)
                if previous_p0.__contains__(id_track) and previous_p0[id_track] is not None:
                    currents_p0[id_track], status, err = cv2.calcOpticalFlowPyrLK(previous_gray_roi, current_gray_roi,
                                                                                  previous_p0[id_track].copy(), None,
                                                                                  **lk_params)
                # Define axis for movement if it doesn't already exist
                if angle_movements_axis_to_x_axis.__contains__(id_track) is False and bbox_mask is not None:
                    # Get (a)x - (b)y + (c) = 0 to calculate the perpendicular line where b = 0[line x=q] or 1[line y=mx+q]
                    center_roi = (int((x2 - x1) / 2), int((y2 - y1) / 2))
                    center_mask = ((int(bbox_mask[0] + bbox_mask[2] / 2)), int((bbox_mask[1] + bbox_mask[3]) / 2))
                    ax, by, c = get_straight_line_from_2_points(center_roi, center_mask)
                    if by == 0:  # The line passing through the points is parallel to the y axis; x = q
                        # The perpendicular line is parallel to the x axis; y = q
                        angle_movements_axis_to_x_axis[id_track] = 90
                    elif ax == 0:
                        angle_movements_axis_to_x_axis[id_track] = 0
                    else:
                        angle = get_degrees_from_the_x_axis(1, -1 / ax)
                        if angle is not None:
                            angle_movements_axis_to_x_axis[id_track] = 360 - angle  # in [90,0] U ]270,360[

                # Sum of  movements
                mov = {'y': 0,
                       'x': 0,
                       'degrees_to_x': 0,
                       'module': 0
                       }
                for i, (start, end) in enumerate(zip(previous_p0[id_track], currents_p0[id_track])):
                    x_start, y_start = start.ravel()
                    x_end, y_end = end.ravel()
                    mov['x'] = mov['x'] + (x_end - x_start)
                    mov['y'] = mov['y'] + (y_start - y_end)
                mov['module'] = sqrt(mov['x'] ** 2 + mov['y'] ** 2)
                mov['degrees_to_x'] = get_degrees_from_the_x_axis(mov['x'], mov['y'])

                # Save movement value in cilia_movements[id_track][frame]
                if mov['degrees_to_x'] is not None and angle_movements_axis_to_x_axis.__contains__(id_track):
                    if cilia_movements.__contains__(id_track) is False:
                        cilia_movements[id_track] = {}
                    m = mov['module'] * cos(
                        radians(abs(angle_movements_axis_to_x_axis[id_track] - mov['degrees_to_x'])))
                    cilia_movements[id_track].update({video_frame_count: m})

                # Save mask, bbox and movement values to show in debug
                if debug and id_track in optical_flow_debug_tracks_id:
                    currents_masks[id_track] = mask
                    currents_bbox_masks[id_track] = bbox_mask
                    if movements_to_show.__contains__(id_track) is False:
                        movements_to_show[id_track] = {}
                    movements_to_show[id_track] = mov

        # Stream debug results
        if debug:
            # Stream yolo detections on im0_yolo
            im0s_yolo = im0s.copy()
            for *xyxy, conf, cls in yolo_detections:
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
                        shape = im0s_roi_copy.shape
                        center_roi = {
                            'x': int(shape[1] / 2),
                            'y': int(shape[0] / 2)
                        }
                        # Create mask_roi
                        previous_roi = previous_frame.copy()[y1:y2, x1:x2]
                        mask = currents_masks[id_track]
                        bbox_mask = currents_bbox_masks[id_track]
                        center_bbox_mask = {
                            'x': int((bbox_mask[2] - bbox_mask[0]) / 2 + bbox_mask[0]),
                            'y': int((bbox_mask[3] - bbox_mask[1]) / 2 + bbox_mask[1])
                        }
                        mask_roi = cv2.bitwise_and(previous_roi, previous_roi, mask=mask.astype(uint8))
                        cv2.rectangle(mask_roi, (bbox_mask[0], bbox_mask[1]), (bbox_mask[2], bbox_mask[3]), (0, 255, 0),
                                      thickness=2)
                        cv2.circle(mask_roi, (center_roi['x'], center_roi['y']), 2, (0, 0, 255), 5)
                        cv2.circle(mask_roi, (center_bbox_mask['x'], center_bbox_mask['y']), 2, (0, 0, 255), 5)
                        cv2.line(mask_roi, (center_roi['x'], center_roi['y']),
                                 (center_bbox_mask['x'], center_bbox_mask['y']), (0, 0, 255), thickness=2)
                        half_arrow = 50
                        if angle_movements_axis_to_x_axis[id_track] != 0:
                            ax = tan(radians(angle_movements_axis_to_x_axis[id_track]))
                            c_parallel = ax * (-center_roi['x']) + center_roi['y']
                            start_arrow_axis = (center_roi['x'] - half_arrow,
                                                int(ax * (center_roi['x'] + half_arrow) + c_parallel))
                            end_arrow_axis = (center_roi['x'] + half_arrow,
                                              int(ax * (center_roi['x'] - half_arrow) + c_parallel))
                        else:
                            start_arrow_axis = (center_roi['x'] - half_arrow, center_roi['y'])
                            end_arrow_axis = (center_roi['x'] + half_arrow, center_roi['y'])
                        cv2.arrowedLine(mask_roi, start_arrow_axis, end_arrow_axis, (255, 0, 0), thickness=2)
                        # Add movement point on im0s_roi_copy
                        for i, (start, end) in enumerate(zip(previous_p0[id_track], currents_p0[id_track])):
                            x_start, y_start = start.astype(int).ravel()
                            x_end, y_end = end.astype(int).ravel()
                            cv2.circle(im0s_roi_copy, (x_end, y_end), 1, (0, 0, 255), 5)
                            cv2.arrowedLine(im0s_roi_copy, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2, 0)
                        # Create white_roi for movement results
                        white_roi = zeros(shape, dtype=uint8)
                        white_roi.fill(255)
                        mov = movements_to_show[id_track]
                        cv2.circle(white_roi, (center_roi['x'], center_roi['y']), 1, (0, 0, 0), 3)
                        cv2.arrowedLine(white_roi, (center_roi['x'], center_roi['y']),
                                        (center_roi['x'] + round(mov['x']), center_roi['y'] - round(mov['y'])),
                                        (0, 255, 0), 2, 0)
                        if angle_movements_axis_to_x_axis[id_track] != 0:
                            rad = radians(angle_movements_axis_to_x_axis[id_track])
                            m = tan(degrees(angle_movements_axis_to_x_axis[id_track]))
                            end_arrow_mov = (round(center_roi['x'] +
                                                   cilia_movements[id_track][video_frame_count] * abs(cos(rad))),
                                             round(center_roi['y'] -
                                                   cilia_movements[id_track][video_frame_count] * abs(sin(rad))))
                        else:
                            end_arrow_mov = (center_roi['x'] + round(cilia_movements[id_track][video_frame_count]),
                                             center_roi['y'])
                        cv2.arrowedLine(white_roi, (center_roi['x'], center_roi['y']),
                                        end_arrow_mov, (255, 0, 0), 2, 0)
                        cv2.line(white_roi, (center_roi['x'] + round(mov['x']), center_roi['y'] - round(mov['y'])),
                                 end_arrow_mov, (0, 0, 0), 1, 0)
                        # Add text results to white_roi
                        cv2.putText(white_roi, f"Module: {mov['module']}", (1, shape[0] - 2),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color=(0, 0, 0))
                        cv2.putText(white_roi, f"Degrees to X: {mov['degrees_to_x']}", (1, shape[0] - 22),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color=(0, 0, 0))
                        cv2.putText(white_roi, f"Y: {mov['y']}", (1, shape[0] - 42),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color=(0, 0, 0))
                        cv2.putText(white_roi, f"X: {mov['x']}", (1, shape[0] - 62),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color=(0, 0, 0))
                        # Concatenate im0s_roi_copy and white_roi to stream the results
                        im0s_movement = concatenate((mask_roi, im0s_roi_copy, white_roi), axis=1)
                        im0s_movement = cv2.resize(im0s_movement, (0, 0), fx=proportion_vid * 2, fy=proportion_vid * 2)
                        cv2.imshow(f'Movement track n. {id_track}', im0s_movement)

            # Destroy dead windows 'Movement track n. {id_track}'
            for id_track in setdiff1d(previous_tracks[:, -1], tracks_to_show[:, -1]):
                if id_track in optical_flow_debug_tracks_id:
                    cv2.destroyWindow(f'Movement track n. {id_track}')

        # Clean cilia_movements, angle_movements_axis_to_x_axis, avg_frames_period, and bpm every 2 sec
        if video_frame_count % (fps * 2) == 0:
            sort_trackers_id_alive = []
            for track in mot_tracker.trackers:
                sort_trackers_id_alive.append(track.id + 1)
            angle_movements_axis_to_x_axis = remove_unmatched_vals(sort_trackers_id_alive,
                                                                   angle_movements_axis_to_x_axis)
            avg_frames_period = remove_unmatched_vals(sort_trackers_id_alive, avg_frames_period)
            if save_output is False:
                cilia_movements = remove_unmatched_vals(sort_trackers_id_alive, cilia_movements)
                bpm = remove_unmatched_vals(sort_trackers_id_alive, bpm)

        # Calculate avg cilia frequency
        if video_frame_count % fps == 0:
            for id_track in cilia_movements:
                actual_avg_frames_period = avg_frames_period[id_track] if avg_frames_period.__contains__(
                    id_track) else 0
                avg_frames_period[id_track] = get_avg_frames_period(cilia_movements[id_track],
                                                                    actual_avg_frames_period, fps, video_frame_count)
                if avg_frames_period[id_track] != 0:
                    bpm[id_track] = get_bpm_from_frames_period(avg_frames_period[id_track], fps)

        # Update previous_frame, previous_p0 and previous_tracks
        previous_frame = im0s.copy()
        previous_p0 = currents_p0.copy()
        previous_tracks = tracks_to_show.copy()

        # Write output on im0s_to_show
        for x1, y1, x2, y2, id_track in tracks_to_show:
            label = f'{names[0]} {int(id_track)}'
            if bpm.__contains__(id_track):
                label = label + f' - {bpm[id_track]} bpm'
            plot_one_box((x1, y1, x2, y2), im0s_to_show, label=label, color=colors[0], line_thickness=3)
        print(f'{s}Done. ({(t2 - t1):.3f}s) nt({len(tracks_to_show)})')

        # Append frame to save in img_video_output
        if save_output:
            img_video_output.append(im0s_to_show)
            if video_output_size is None:
                height, width, _ = im0s_to_show.shape
                video_output_size = (width, height)

        # Stream out-put results img
        im0s_to_show = cv2.resize(im0s_to_show, (0, 0), fx=proportion_vid, fy=proportion_vid)
        cv2.imshow(str(p), im0s_to_show)
        if cv2.waitKey(wait_key) & 0xFF == 27:  # Press esc to exit
            raise StopIteration

    # Save output results
    if save_output and len(img_video_output) != 0:
        # Make dir in Results
        ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
        file_name = os.path.basename(path)
        results_path = os.path.join(ROOT_DIR, 'results', os.path.splitext(file_name)[0])
        if os.path.exists(results_path):
            shutil.rmtree(results_path)
            # os.rmdir(results_path)
        os.mkdir(results_path)

        # Save plt movements
        try:
            for id_track in cilia_movements:
                frames = cilia_movements[id_track].keys()
                m = cilia_movements[id_track].values()
                plt.plot(frames, m)
                plt.axes().spines['bottom'].set_position(('data', 0))
                subtitle = f'Movement cellula n. {id_track}'
                if bpm.__contains__(id_track):
                    subtitle = subtitle + f' - {bpm[id_track]} bpm'
                plt.suptitle(subtitle)
                plt_path = os.path.join(results_path, f'Movement cellula n. {id_track}.png')
                plt.savefig(plt_path)
            print('\nGrafici salvati con successo.')
        except:
            print('Impossibile salvare i grafici elaborati.')

        # Save video
        try:
            video_path = os.path.dirname(path)
            video_path = os.path.join(results_path, 'output_' + file_name)
            out_video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, video_output_size)
            for i in range(len(img_video_output)):
                out_video.write(img_video_output[i])
            out_video.release()
            print('Video salvato con successo.')
        except:
            print('Impossibile salvare il video elaborato.')
