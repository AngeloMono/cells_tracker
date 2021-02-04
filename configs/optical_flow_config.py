from cv2 import TERM_CRITERIA_EPS, TERM_CRITERIA_COUNT

"""
Optical flow configs variables
"""

# Lucas-Kanade method with pyramids parameters
lk_params = {'status': None,
             'err': None,
             'winSize': (30, 30),
             'maxLevel': 2,
             'criteria': (TERM_CRITERIA_EPS | TERM_CRITERIA_COUNT, 10, 0.03)
             }

# Max point to track in optical flow
max_points_to_track = 200

# Minimum distance between two points
min_distance_points = 10
