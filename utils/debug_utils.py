from utils.math_utils import get_degrees_from_the_x_axis
from yolo_utils.plots import plot_one_box
from numpy import uint8, zeros
from math import tan, radians, cos, sin, sqrt
import cv2


def plot_yolo_detections(img, detections, names: list, color):
    """
    Plot yolo detections on img with label 'name: conf'
    :param img: nparray img
    :param detections: nparray detections
    :param names: list of names
    :param color: (R, G, B) color of plot and label
    """
    for *xyxy, conf, cls in detections:
        label = '%s %.2f' % (names[int(cls)], conf)
        plot_one_box(xyxy, img, label=label, color=color, line_thickness=3)


def plot_all_tracks(img, tracks, color):
    """
    Plot all tracks on img with label 'id_tracks + 1'
    :param img: nparray img
    :param tracks: Sort.trackers to show
    :type tracks: list[KalmanBoxTracker]
    :param color: color: (R, G, B) color of plot and label
    """
    for t in tracks:
        state = t.get_state()[0]
        id_track = t.id
        plot_one_box(state, img, label=str(id_track + 1), color=color, line_thickness=3)


def plot_matched_and_reliable_tracks(img, matched_tracks, matched_color, reliable_tracks, reliable_color):
    """
    Plot matched and reliable tracks on img with label 'Matched:id_track' or 'Reliable:id_track'
    :param img: nparray img
    :param matched_tracks: 
    :param matched_color: (R, G, B) color of matched tracks plot and label
    :param reliable_tracks: 
    :param reliable_color: (R, G, B) color of reliable tracks plot and label
    """
    for x1, y1, x2, y2, id_track in matched_tracks:
        plot_one_box((x1, y1, x2, y2), img, label=f'Matched:{int(id_track)}',
                     color=matched_color, line_thickness=3)
    for x1, y1, x2, y2, id_track in reliable_tracks:
        plot_one_box((x1, y1, x2, y2), img, label=f'Reliable:{int(id_track)}',
                     color=reliable_color, line_thickness=3)


def create_mask_roi_img(img, mask, bbox_mask, angle_movements_axis_to_x_axis, arrow_length: int = 100):
    """
    Create a nparray  where the matched part between img and mask contains the img area, moreover plot the bbox of
    the mask, a line between bbox mask center and img center and a representative arrow
    of the angle_movements_axis_to_x_axis
    :param img: nparray img
    :param mask: ndarray representing the mask
    :param bbox_mask: [x,y,x,y] od the top left point and bottom rigth point
    :param angle_movements_axis_to_x_axis: degrees of angle between movement axis and x axis
    :param arrow_length: length of the representative arrow of angle_movements_axis_to_x_axis
    :return: nparray img
    """
    center_roi = (int(img.shape[1] / 2), int(img.shape[0] / 2))  # (x, y)
    center_bbox_mask = (int((bbox_mask[2] - bbox_mask[0]) / 2 + bbox_mask[0]),
                        int((bbox_mask[3] - bbox_mask[1]) / 2 + bbox_mask[1]))
    mask_roi = cv2.bitwise_and(img, img, mask=mask.astype(uint8))
    cv2.rectangle(mask_roi, (bbox_mask[0], bbox_mask[1]), (bbox_mask[2], bbox_mask[3]), (0, 255, 0),
                  thickness=2)
    cv2.circle(mask_roi, (center_roi[0], center_roi[1]), 2, (0, 0, 255), 5)
    cv2.circle(mask_roi, (center_bbox_mask[0], center_bbox_mask[1]), 2, (0, 0, 255), 5)
    cv2.line(mask_roi, (center_roi[0], center_roi[1]),
             (center_bbox_mask[0], center_bbox_mask[1]), (0, 0, 255), thickness=2)
    half_arrow = abs(int(arrow_length/2))
    if angle_movements_axis_to_x_axis != 0:
        ax = tan(radians(angle_movements_axis_to_x_axis))
        c_parallel = ax * (-center_roi[0]) + center_roi[1]
        start_arrow_axis = (center_roi[0] - half_arrow,
                            int(ax * (center_roi[0] + half_arrow) + c_parallel))
        end_arrow_axis = (center_roi[0] + half_arrow,
                          int(ax * (center_roi[0] - half_arrow) + c_parallel))
    else:
        start_arrow_axis = (center_roi[0] - half_arrow, center_roi[1])
        end_arrow_axis = (center_roi[0] + half_arrow, center_roi[1])
    cv2.arrowedLine(mask_roi, start_arrow_axis, end_arrow_axis, (255, 0, 0), thickness=2)
    return mask_roi


def plot_movements_points(img, previous_points, currents_points):
    """
    Plot movements between previous_points and currents_points on img
    :param img: nparray img
    :param previous_points: previous point
    :param currents_points: current point of the movement respect to the previous point
    """
    for i, (start, end) in enumerate(zip(previous_points, currents_points)):
        x_start, y_start = start.astype(int).ravel()
        x_end, y_end = end.astype(int).ravel()
        cv2.circle(img, (x_end, y_end), 1, (0, 0, 255), 5)
        cv2.arrowedLine(img, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2, 0)


def create_white_roi_movement_img(shape, motion_vector, degrees_angle_movements_axis_to_x_axis):
    """
    Create a nparray of shape (shape[0], shape[1], 3) filled with white and a green arrow showing the 'motion_vector'
    and a blue arrow showing the projection on the axis of 'degrees_angle_movements_axis_to_x_axis'
    degrees to the x axis.
    Moreover it adds the vector's coordinate, its module and vector degrees to the x axis at the bottom of the img.
    :param shape: shape of the white roi to create (h, w)
    :param motion_vector: the x and y coordinates of the motion vector (x, y)
    :param degrees_angle_movements_axis_to_x_axis: degrees of angle between movement axis and x axis
    :return: nparray img
    """
    # Create white_roi and plot motion_vector
    center_roi = (int(shape[1] / 2), int(shape[0] / 2))  # (x, y)
    white_roi = zeros((shape[0], shape[1], 3), dtype=uint8)
    white_roi.fill(255)
    cv2.circle(white_roi, (center_roi[0], center_roi[1]), 1, (0, 0, 0), 3)
    cv2.arrowedLine(white_roi, (center_roi[0], center_roi[1]),
                    (center_roi[0] + round(motion_vector[0]), center_roi[1] - round(motion_vector[1])),
                    (0, 255, 0), 2, 0)

    # Calculate end plot arrow on movements axis
    module_motion = sqrt(motion_vector[0] ** 2 + motion_vector[1] ** 2)
    degrees_vector_to_x_axis = get_degrees_from_the_x_axis(motion_vector[0], motion_vector[1])
    delta_rad = radians(abs(degrees_angle_movements_axis_to_x_axis - degrees_vector_to_x_axis))
    motion_vector_on_movements_axis = module_motion * cos(delta_rad)
    end_arrow_mov = (round(center_roi[0] +
                           motion_vector_on_movements_axis * cos(radians(degrees_angle_movements_axis_to_x_axis))),
                     round(center_roi[1] -
                           motion_vector_on_movements_axis * sin(radians(degrees_angle_movements_axis_to_x_axis))))
    cv2.arrowedLine(white_roi, (center_roi[0], center_roi[1]), end_arrow_mov, (255, 0, 0), 2, 0)
    if end_arrow_mov != center_roi:
        cv2.line(white_roi, (center_roi[0] + round(motion_vector[0]), center_roi[1] - round(motion_vector[1])),
                 end_arrow_mov, (0, 0, 0), 1, 0)

    # Add text results to white_roi
    cv2.putText(white_roi, f"Module: {module_motion}", (1, shape[0] - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color=(0, 0, 0))
    cv2.putText(white_roi, f"Degrees to X: {degrees_vector_to_x_axis}", (1, shape[0] - 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color=(0, 0, 0))
    cv2.putText(white_roi, f"Y: {motion_vector[1]}", (1, shape[0] - 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color=(0, 0, 0))
    cv2.putText(white_roi, f"X: {motion_vector[0]}", (1, shape[0] - 62),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color=(0, 0, 0))
    return white_roi
