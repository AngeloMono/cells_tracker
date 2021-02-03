from os import path
from os.path import join, splitext
from matplotlib.pyplot import plot, savefig, axes, suptitle, clf
from cv2 import VideoWriter, VideoWriter_fourcc
from numpy import zeros


def save_plt_result(results_path, movements_dict: dict, id_track, bpm: float = None):
    """
    Save the plot of the movements_movements_dict in the results_path
    :param results_path: path where to save the result
    :param movements_dict: dictionary contains the movement for each frame as {id_frame: movement}
    :param id_track: id_track to write in the subtitle
    :param bpm: bpm to write in the subtitle
    """
    frames = movements_dict.keys()
    values = movements_dict.values()
    all_frames = []
    all_values = []
    unmatched_frames = []
    axes().spines['bottom'].set_position(('data', 0))
    for id_key in range(min(frames), max(frames)+1):
        all_frames.append(id_key)
        if movements_dict.__contains__(id_key) is False:
            unmatched_frames.append(id_key)
            all_values.append(0)
        else:
            all_values.append(movements_dict[id_key])
    subtitle = f'Movement cellula n. {id_track}'
    if bpm is not None:
        subtitle = subtitle + f' - {bpm} bpm'
    suptitle(subtitle)
    plot(frames, values, 'o')
    plot(unmatched_frames, zeros(len(unmatched_frames)), 'o', c='red')
    plot(all_frames, all_values)
    plt_path = path.join(results_path, f'Movement cellula n. {id_track}.png')
    savefig(plt_path)
    clf()


def save_plts_results(results_path, movement_dict, bpm_dict):
    """
    Save the plot of the movements_movements_dict in the results_path for each tracks in movement_dict
    :param results_path: path where to save the result
    :param movement_dict: dictionary containing the movements as {id_track: {id_frame: mov ...}}
    :param bpm_dict: dictionary containing the bpm as {id_track: bpm}
    """
    for id_track in movement_dict:
        calculated_bpm = bpm_dict[id_track] if bpm_dict.__contains__(id_track) else None
        save_plt_result(results_path, movement_dict[id_track], id_track, calculated_bpm)


def save_video_output(results_path, file_name, list_frame: list, fps: float, frame_shape: list):
    """
    Save the frame list as mp4 video in the folder results_path assigning the given name
    :param results_path: path where to save the result
    :param file_name: name with which to save the document
    :param list_frame: ordered list of frames to save
    :param fps: fps of the output video
    :param frame_shape: shape of the frame of the video output
    """
    file_name = splitext(file_name)[0]
    file_name = file_name + ".mp4"
    video_path = join(results_path, file_name)
    out_video = VideoWriter(video_path, VideoWriter_fourcc(*'mp4v'), fps, frame_shape)
    for i in range(len(list_frame)):
        out_video.write(list_frame[i])
    out_video.release()
