import numpy as np
import pandas as pd
from .transform import limit_by_freq, group_by_freq, dict_to_array
import os
from copy import deepcopy
import json
import glob
from .plot import diff_plot

# sqrt(2) with default precision np.float64
_SQRT2 = np.sqrt(2)


# DISTANCE METRICS

def positive_error(x, y):
    """
    :param np.array x:
    :param np.array y:
    :return:
    """
    return np.sum(np.abs(x - y))


def hellinger(x, y):
    """
    :param np.array x:
    :param np.array y:
    :return:
    """
    return np.linalg.norm(np.sqrt(x) / np.sum(x) -
                          np.sqrt(y) / np.sum(y)) / _SQRT2


def l2_norm(x, y):
    """
    L2 norm, adapted to dtw format
    :param x:
    :param y:
    :return: euclidean norm
    """
    return np.linalg.norm(x - y)


def integrate(x, y):
    """
    :param x:
    :param y:
    :return:
    """
    diff = np.abs(x - y)
    return np.trapz(diff)


distance_dict = {'positive': positive_error,
                 'hellinger': hellinger,
                 'l2_norm': l2_norm,
                 'integrate': integrate}


def warp_distance(distance_metric, x, y, warp=200):
    """

    :param str distance_metric:
    :param np.array x:
    :param np.array y:
    :param int warp:
    :return:
    """
    # Selecting the array
    distance_func = distance_dict[distance_metric]
    # Copying the value
    x_copy = deepcopy(x)
    y_copy = deepcopy(y)
    # Starting the warping
    min_diff = distance_func(x, y)
    for i in range(1, warp):
        # Moving forward
        forward_diff = distance_func(x_copy[i:], y_copy[:-i])
        if forward_diff < min_diff:
            min_diff = forward_diff
        # Moving backward
        backward_diff = distance_func(x_copy[:-i], y_copy[i:])
        if backward_diff < forward_diff:
            min_diff = backward_diff
    return min_diff


def pair_distance(freq_x,
                  features_x,
                  freq_y,
                  features_y,
                  warp=None,
                  frames=1,
                  distance_metric='l2_norm'):
    """

    :param freq_x:
    :param features_x:
    :param freq_y:
    :param features_y:
    :param warp:
    :param frames:
    :param distance_metric:
    :return:
    """
    distance = 0.0
    freq_div = np.max(freq_x) / frames
    for i in range(1, frames + 1):
        lower_limit = (i - 1) * freq_div
        upper_limit = i * freq_div
        freq_x_frame, features_x_frame = limit_by_freq(freq_x,
                                                       features_x,
                                                       upper_limit=upper_limit,
                                                       lower_limit=lower_limit)
        features_y_frame = np.interp(freq_x_frame,
                                     freq_y,
                                     features_y)

        if warp is None:
            frame_dist = distance_dict[distance_metric](features_x_frame,
                                                        features_y_frame)
        else:
            frame_dist = warp_distance(distance_metric,
                                       features_x_frame,
                                       features_y_frame,
                                       warp)
        distance += frame_dist / frames

    return distance


def distance_matrix(fourier_folder,
                    warp=None,
                    upper_limit=6000.0,
                    frames=1,
                    distance_metric='l2_norm'):
    """

    :param fourier_folder:
    :param warp:
    :param upper_limit:
    :param frames:
    :param distance_metric:
    :return:
    """
    merged_file = os.path.join(fourier_folder, 'merged_file.json')
    if os.path.isfile(merged_file):
        os.remove(merged_file)
    read_files = glob.glob(os.path.join(fourier_folder, '*.json'))
    with open(merged_file, 'wb') as outfile:
        outfile.write(('[{}]'.format(','.join([open(f, 'r').read()
                                               for f in read_files]))).encode('utf8'))

    with open(os.path.join(fourier_folder,
                           'merged_file.json'), 'r') as f:
        merged_file_list = json.load(f)

    merged_file = merged_file_list[0]
    [merged_file.update(d) for d in merged_file_list]

    # Creating a squared DataFrame as matrix distance
    song_names = list(merged_file.keys())
    df = pd.DataFrame(columns=song_names + ['Songs'])
    df['Songs'] = song_names
    df = df.set_index('Songs')
    max_value = 0.0  # In order to normalize distance values
    for song_x in song_names:
        freq_x, features_x = dict_to_array(merged_file[song_x])
        # Filtering frequencies
        freq_x, features_x = limit_by_freq(freq_x,
                                           features_x,
                                           upper_limit=upper_limit)
        for song_y in song_names:
            if song_x != song_y:
                freq_y, features_y = dict_to_array(merged_file[song_y])
                distance = pair_distance(freq_x=freq_x,
                                         features_x=features_x,
                                         freq_y=freq_y,
                                         features_y=features_y,
                                         warp=warp,
                                         frames=frames,
                                         distance_metric=distance_metric)
                # Save also in reverse
                df.loc[song_y, song_x] = distance
                # Save maximum value
                if distance > max_value:
                    max_value = distance
            else:
                distance = 0.0
            df.loc[song_x, song_y] = distance

    return df.sort_index(axis=0, ascending=True).sort_index(axis=1, ascending=True) / max_value
