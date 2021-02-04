import os
import shutil

import cv2
import torch
import torch.backends.cudnn as cudnn
from warnings import filterwarnings

from numpy import vstack, empty, concatenate, uint8, setdiff1d, ndarray
from math import sqrt, radians, cos
from datetime import datetime

from models.experimental import attempt_load
from yolo_utils.datasets import LoadImages, LoadWebcam
from yolo_utils.general import check_img_size, non_max_suppression, scale_coords
from yolo_utils.plots import plot_one_box

from sort import Sort

from yolact import Yolact
from data import cfg, set_cfg

from configs.yolo_config import yolo_weights, yolo_img_size, yolo_conf_thres, yolo_iou_threshold
from configs.sort_config import max_age_sort, min_hits_sort, max_num_frame_reliable_sort, iou_thres_sort
from configs.yolact_config import yolact_weights, yolact_config, yolact_threshold, yolact_num_max_predictions
from configs.optical_flow_config import lk_params, max_points_to_track, min_distance_points

from utils.tracks_utils import get_unmatched_reliable_tracks, correct_the_tracks
from utils.math_utils import get_degrees_from_the_x_axis, get_straight_line_from_2_points
from utils.segmentation_utils import get_mask_bbox_and_score
from utils.period_utils import get_bpm_from_frames_period, get_avg_frames_period
from utils.debug_utils import plot_movements_points, create_mask_roi_img, create_white_roi_movement_img, \
    plot_all_tracks, plot_matched_and_reliable_tracks, plot_yolo_detections
from utils.save_results_utils import save_plts_results, save_video_output


def remove_unmatched_vals(vals: list, dictionary: dict):
    dict_to_return = dictionary.copy()
    for id_dict in dictionary:
        if id_dict not in vals:
            dict_to_return.pop(id_dict)
    return dict_to_return


# Source path video or 0 to camera
source = r"C:\Users\angel\Desktop\Video Dimauro\Video cellula sola\Train-test video\sola (1).mp4"

debug = True
visualize = True
wait_key = 0 if debug else 1
save_output = True
save_video = True
save_plt = True
if save_output:
    save_video = True
    save_plt = True
proportion_vid = 0.25 if debug else 0.5
if source == 0:
    visualize = True
optical_flow_debug_tracks_id = [1]

# Yolov5 variables
# yolo_weights = r"C:\Users\angel\Desktop\Video Dimauro\Dataset Yolo\Results\train yolov5s ep700 imgsize416 sd1\results\content\yolov5\runs\train\yolov5s_results\weights\best.pt"
# yolo_img_size = 416
# yolo_conf_thres = 0.45
# yolo_iou_threshold = 0.3

# SORT variables
# max_age_sort = 20  # Maximum number of frames to keep alive a track without associated detections
# min_hits_sort = 5  # Minimum number of associated detections before track is initialised
# max_num_frame_reliable_sort = 3  # Max confidence to keep valid a detection not matched
# iou_thres_sort = 0.00000000000000005

# # Yolact variables
# yolact_weight = r"C:\Users\angel\Desktop\Video Dimauro\Dataset Yolact\Results\content\yolact\weights\yolact_resnet50_cilia_133_5614_interrupt.pth"
# yolact_model_path = SavePath.from_str(yolact_weight)
# yolact_config = yolact_model_path.model_name + '_config'
# yolact_threshold = 0
# yolact_num_max_predictions = 1

# Optical Flow variables
# lk_params = {'status': None,
#              'err': None,
#              'winSize': (30, 30),
#              'maxLevel': 2,
#              'criteria': (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
#              }
# max_points_to_track = 200
# min_distance_points = 10

# Create instance of SORT
mot_tracker = Sort(max_age=max_age_sort, min_hits=min_hits_sort, iou_threshold=iou_thres_sort)
matched_tracks = empty((0, 5))

# Create global variables for motion calculation
previous_frame = None
previous_tracks = empty((0, 5))
currents_p0 = dict()
previous_p0 = dict()
angle_movements_axis_to_x_axis = dict()
cilia_movements = dict()
avg_frames_period = dict()
bpm = dict()

# Create global variables for show debug windows
if debug:
    currents_masks = dict()
    currents_bbox_masks = dict()
    movements_to_show = dict()
#  Create global variables for save outputs
if save_output or save_video:
    img_video_output = []

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
    yolact_model.load_weights(yolact_weights)
    yolact_model.eval()
    if cuda:
        yolact_model = yolact_model.cuda()
    yolact_model.detect.use_fast_nms = True
    cfg.mask_proto_debug = False

    # Extract names and assign a color RGB for the bbox
    names = yolo_model.module.names if hasattr(yolo_model, 'module') else yolo_model.names
    color = (172, 47, 117)

    # Load video
    if str(source) == '0':
        cudnn.benchmark = True  # Set True to speed up constant image size inference
        video = LoadWebcam(img_size=yolo_img_size, verbose=False)
    else:
        video = LoadImages(source, img_size=yolo_img_size, verbose=False)
    fps = round(video.cap.get(cv2.CAP_PROP_FPS))
    print('Stream processed frames')
    print("Press 'esc' to quit")
    if debug:
        print('\nDEBUG MODE ACTIVATE:')
        print("Press 'space bar' to play or stop the video")
        print('Press any other key to go to the next frame')
    print()

    try:
        # Process each frame in video
        for path, img, im0s, vid_cap in video:
            start_time = datetime.now()
            video_frame_count = video.count if isinstance(video, LoadWebcam) else video.frame

            img = torch.from_numpy(img).to(device)
            img = img.half() if cuda else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0

            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            if debug or visualize is False:
                t1 = datetime.now()
            yolo_detections = yolo_model(img, augment=False)[0]

            # Apply NMS
            yolo_detections = non_max_suppression(yolo_detections, yolo_conf_thres, yolo_iou_threshold, classes=None,
                                                  agnostic=False)
            yolo_detections = yolo_detections[0]
            if debug or visualize is False:
                t2 = datetime.now()
                t_yolo = (t2 - t1).total_seconds()

            # Processing detections predicted
            im0s_to_show = im0s.copy()

            if len(yolo_detections):
                # Rescale boxes from img_size to im0 size
                yolo_detections[:, :4] = scale_coords(img.shape[2:], yolo_detections[:, :4], im0s_to_show.shape).round()

                # SORT tracker application
                np_det = yolo_detections.cpu().numpy()
                matched_tracks = mot_tracker.update(np_det[:, 0:-1])

                # Get unmatched reliable tracks
                unmatched_reliable_tracks = get_unmatched_reliable_tracks(matched_tracks, mot_tracker,
                                                                          max_num_frame_reliable_sort)
            else:
                matched_tracks = mot_tracker.update()
                unmatched_reliable_tracks = get_unmatched_reliable_tracks(matched_tracks, mot_tracker,
                                                                          max_num_frame_reliable_sort)

            # Marge and correct matched_tracks and unmatched_reliable_tracks
            tracks_to_show = vstack([matched_tracks, unmatched_reliable_tracks]).astype(int)
            correct_the_tracks(tracks_to_show, im0s.shape[1], im0s.shape[0])

            if debug or visualize is False:
                t_yolact = 0
            # Calculate the movement of the cilia
            for x1, y1, x2, y2, id_track in tracks_to_show:
                if id_track in previous_tracks[:, -1]:
                    previous_roi = previous_frame.copy()[y1:y2, x1:x2]
                    previous_gray_roi = cv2.cvtColor(previous_roi, cv2.COLOR_BGR2GRAY)
                    current_roi = im0s.copy()[y1:y2, x1:x2]
                    current_gray_roi = cv2.cvtColor(current_roi, cv2.COLOR_BGR2GRAY)

                    if debug or visualize is False:
                        t1 = datetime.now()
                    # Create mask using Yolact
                    mask, bbox_mask, _ = get_mask_bbox_and_score(yolact_model, previous_roi, threshold=yolact_threshold,
                                                                 max_predictions=yolact_num_max_predictions)
                    if debug or visualize is False:
                        t2 = datetime.now()
                        t_yolact = t_yolact + (t2 - t1).total_seconds()

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
                            movements_to_show[id_track] = ()
                        movements_to_show[id_track] = (mov['x'], mov['y'])

            if debug or visualize is False:
                finish_time = datetime.now()
                total_time = (finish_time - start_time).total_seconds()

            # Stream debug results
            if debug:
                # Stream yolo detections on im0_yolo
                im0s_yolo = im0s.copy()
                plot_yolo_detections(im0s_yolo, yolo_detections, names, color)
                im0s_yolo = cv2.resize(im0s_yolo, (0, 0), fx=proportion_vid, fy=proportion_vid)
                cv2.imshow('Yolo detection', im0s_yolo)

                # Stream all tracks from SORT tracker on im0_all_tracks
                im0s_all_tracks = im0s.copy()
                plot_all_tracks(im0s_all_tracks, mot_tracker.trackers, color)
                im0s_all_tracks = cv2.resize(im0s_all_tracks, (0, 0), fx=proportion_vid, fy=proportion_vid)
                cv2.imshow('All tracks', im0s_all_tracks)

                # Stream matched and reliable tracks on im0s_matched_and_reliable_tracks
                im0s_matched_and_reliable_tracks = im0s.copy()
                plot_matched_and_reliable_tracks(im0s_matched_and_reliable_tracks, matched_tracks, color,
                                                 unmatched_reliable_tracks, (0, 0, 0))
                im0s_matched_and_reliable_tracks = cv2.resize(im0s_matched_and_reliable_tracks, (0, 0), fx=proportion_vid,
                                                              fy=proportion_vid)
                cv2.imshow('Matched and reliable tracks', im0s_matched_and_reliable_tracks)

                # Stream Optical Flow movement on im0s_movement
                for x1, y1, x2, y2, id_track in tracks_to_show:
                    if id_track in optical_flow_debug_tracks_id:
                        if id_track in previous_tracks[:, -1]:
                            # Create mask_roi
                            previous_roi = previous_frame.copy()[y1:y2, x1:x2]
                            mask_roi = create_mask_roi_img(previous_roi, currents_masks[id_track],
                                                           currents_bbox_masks[id_track],
                                                           angle_movements_axis_to_x_axis[id_track])

                            # Add movement point on im0s_roi_copy
                            im0s_roi_copy = im0s.copy()[y1:y2, x1:x2]
                            plot_movements_points(im0s_roi_copy, previous_p0[id_track], currents_p0[id_track])

                            # Create white_roi for movement results
                            white_roi = create_white_roi_movement_img(im0s_roi_copy.shape, movements_to_show[id_track],
                                                                      angle_movements_axis_to_x_axis[id_track])

                            # Concatenate mask_roi, im0s_roi_copy and white_roi to stream the results
                            im0s_movement = concatenate((mask_roi, im0s_roi_copy, white_roi), axis=1)
                            im0s_movement = cv2.resize(im0s_movement, (0, 0), fx=proportion_vid * 2, fy=proportion_vid * 2)
                            cv2.imshow(f'Movement track n. {id_track}', im0s_movement)

                # Destroy dead windows 'Movement track n. {id_track}'
                for id_track in setdiff1d(previous_tracks[:, -1], tracks_to_show[:, -1]):
                    if id_track in optical_flow_debug_tracks_id:
                        cv2.destroyWindow(f'Movement track n. {id_track}')

            if debug or visualize is False:
                # Print time elaboration
                if isinstance(video, LoadImages):
                    s = f'Frame({video.frame}/{video.nframes}):'
                else:
                    s = f'Frame {video.count}:'
                s = s + f' time elaboration:{total_time.__format__(".4f")} s (yolo time:{t_yolo.__format__(".4f")} s'
                if t_yolact != 0:
                    s = s + f', yolact time:{t_yolact.__format__(".4f")} s'
                s = s + ')'
                print(s)

            # Clean cilia_movements, angle_movements_axis_to_x_axis, avg_frames_period, and bpm every 2 seconds
            if video_frame_count % (fps * 2) == 0:
                sort_trackers_id_alive = []
                for track in mot_tracker.trackers:
                    sort_trackers_id_alive.append(track.id + 1)
                angle_movements_axis_to_x_axis = remove_unmatched_vals(sort_trackers_id_alive,
                                                                       angle_movements_axis_to_x_axis)
                avg_frames_period = remove_unmatched_vals(sort_trackers_id_alive, avg_frames_period)
                if save_output is False and save_plt is False:
                    cilia_movements = remove_unmatched_vals(sort_trackers_id_alive, cilia_movements)
                    bpm = remove_unmatched_vals(sort_trackers_id_alive, bpm)

            # Calculate avg cilia frequency every second
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

            # Append frame to save in img_video_output
            if save_output or save_video:
                img_video_output.append(im0s_to_show)

            if visualize or debug:
                # Write output on im0s_to_show
                for x1, y1, x2, y2, id_track in tracks_to_show:
                    label = f'{names[0]} {int(id_track)}'
                    if bpm.__contains__(id_track):
                        label = label + f' - {bpm[id_track]} bpm'
                    plot_one_box((x1, y1, x2, y2), im0s_to_show, label=label, color=color, line_thickness=3)

                # Stream out-put results img
                im0s_to_show = cv2.resize(im0s_to_show, (0, 0), fx=proportion_vid, fy=proportion_vid)
                cv2.imshow(fr'{path}', im0s_to_show)
                wk_return = cv2.waitKey(wait_key)
                if wk_return & 0xFF == 27:  # Press esc to exit
                    print('\n-- Stop iteration --')
                    raise StopIteration
                elif debug and wk_return & 0xFF == 32:  # Press space in debug mode to stop or play video
                    wait_key = 1 if wait_key == 0 else 0
            if isinstance(video, LoadImages) and video.frame == video.nframes:
                print('\n-- Stop iteration --')
                raise StopIteration
    except StopIteration:
        cv2.destroyAllWindows()

        # Save output results
        if save_output or save_video or save_plt:
            # Make dir in Results
            ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
            file_name = os.path.basename(path)
            results_path = os.path.join(ROOT_DIR, 'results', os.path.splitext(file_name)[0])
            if os.path.exists(results_path):
                shutil.rmtree(results_path)
            os.mkdir(results_path)

            # Save plt movements
            if save_output or save_plt:
                try:
                    save_plts_results(results_path, cilia_movements, bpm)
                    print('\nGraphs saved successfully.')
                except:
                    print('Unable to save processed graphs.')

            # Save video
            if (save_output or save_video) and len(img_video_output) != 0:
                frame_shape = (int(video.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
                fps = video.cap.get(cv2.CAP_PROP_FPS)
                try:
                    save_video_output(results_path, 'output_' + file_name, img_video_output, fps, frame_shape)
                    print('Video saved successfully.')
                except:
                    print('Unable to save processed video.')
