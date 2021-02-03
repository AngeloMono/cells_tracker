from sort import Sort
from numpy import empty, append, vstack


def get_unmatched_reliable_tracks(np_matched_tracks, sort_object: Sort, time_update_conf: int):
    """
    Return the reliable SORT tracks where track.time_since_update <= time_update_conf and track.age >= track.min_hits
    :param np_matched_tracks: tracks matched in Sort.update()
    :param sort_object: Sort object
    :param time_update_conf: reliability threshold
    :return: a ndarray containing the reliable tracks (if there are no reliable tracks return a empty ndarray)
    """
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
    """
    Corrects the given tracks values in range [0, img_x_shape] [0, img_y_shape]
    :param tracks:
    :param img_x_shape:
    :param img_y_shape:
    """
    for t in tracks:
        t[0] = 0 if t[0] < 0 else t[0]
        t[1] = 0 if t[1] < 0 else t[1]
        t[2] = img_x_shape if t[2] > img_x_shape else t[2]
        t[3] = img_y_shape if t[3] > img_y_shape else t[3]
