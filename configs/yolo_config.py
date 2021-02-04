from os.path import join, dirname, abspath

"""
Yolov5 configs variables
"""

# Yolov5 weights path (default: ~/weights/yolov5.pt)
yolo_weights = join(dirname(abspath('main.py')), 'weights', 'yolov5.pt')

# Yolo inference size (pixels)
yolo_img_size = 416

# Yolo object confidence threshold
yolo_conf_thres = 0.45

# Yolo IoU threshold for NMS (non_max_suppression)
yolo_iou_threshold = 0.3

# TODO CREARE METODO PER IL CARICAMENTO DELLA RETE