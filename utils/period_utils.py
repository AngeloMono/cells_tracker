def get_avg_frames_period(mov_dict: dict, previous_avg, n_frame_to_consider: int, actual_frame: int,
                          frame_step: int = 1):
    """
    Calculate the average number of frames in which the sign of movement changes considering the previous frames
    if (actual_frame - n_frame_to_consider) > 0
    :param mov_dict: movements dictionary (id_frame: movement)
    :param previous_avg: previous avg to consider
    :param n_frame_to_consider: number of frames to consider before the current one
    :param actual_frame: number of the current frame
    :param frame_step: number of frames to be considered between one detection and the next
    :return: avg or 0 if avg=None
    """
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
                    is_reliable = mov_dict.__contains__(start_frame + count + frame_step)
                    is_reliable = is_reliable and mov_dict[start_frame + count + frame_step] * mov_dict[start_frame] > 0
                    if is_reliable:
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
    """
    Calculate the beats per minute given the period and the fps of the video
    :param frames_period: periods of the movement
    :param fps_video: fps of the video
    :return: beats per minute
    """
    return round(60 * fps_video / frames_period, 2)
