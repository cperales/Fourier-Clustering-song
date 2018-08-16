import numpy as np
import pandas as pd
from .transform import limit_by_freq, group_by_freq, dict_to_array
import os
from copy import deepcopy
import json
# from scipy.integrate import quad
# from dtw import dtw, fastdtw

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


def pair_distance(song_x,
                  song_y,
                  warp=None,
                  upper_limit=6000.0,
                  frames=1,
                  distance_metric='l2_norm'):
    """

    :param song_x:
    :param song_y:
    :param warp:
    :param upper_limit:
    :param frames:
    :param distance_metric:
    :return:
    """
    # Extract frequencies and features from JSON
    with open(song_x, 'r') as song_x_json:
        song_x_dict = json.load(song_x_json)
        # Bad indexing
        song_x_dict = song_x_dict[list(song_x_dict.keys())[0]]
        # Transform dict into numpy arrays
        freq_x, features_x = dict_to_array(song_x_dict)

    # Extract frequencies and features
    with open(song_y, 'r') as song_y_json:
        song_y_dict = json.load(song_y_json)
        song_y_dict = song_y_dict[list(song_y_dict.keys())[0]]
        freq_y, features_y = dict_to_array(song_y_dict)

    # FILTERING FREQUENCIES, LESS INFORMATION
    freq_x, features_x = limit_by_freq(freq_x,
                                       features_x,
                                       upper_limit=upper_limit)
    # There is an interpolation in song_y, so there is no need
    # of limiting by frequencies again

    # TIME FRAMES
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
        distance += frame_dist

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
    # Creating a squared DataFrame as matrix distance
    song_names = [os.path.splitext(s)[0] for s in os.listdir(fourier_folder)]
    df = pd.DataFrame(columns=song_names + ['Songs'])
    df['Songs'] = song_names
    df = df.set_index('Songs')
    number_songs = len(song_names)
    max_value = 0.0  # In order to normalize
    for i in range(number_songs):
        for j in range(i, number_songs):
            if i != j:
                full_song_x = os.path.join(fourier_folder, song_names[i]) + '.json'
                full_song_y = os.path.join(fourier_folder, song_names[j]) + '.json'
                distance = pair_distance(song_x=full_song_x,
                                         song_y=full_song_y,
                                         warp=warp,
                                         upper_limit=upper_limit,
                                         frames=frames,
                                         distance_metric=distance_metric)
                # Save also in reverse
                df.loc[song_names[j], song_names[i]] = distance
                # Save maximum value
                if distance > max_value:
                    max_value = distance
            else:
                distance = 0.0
            df.loc[song_names[i], song_names[j]] = distance

    return df.sort_index(axis=0, ascending=True).sort_index(axis=1, ascending=True) / max_value
