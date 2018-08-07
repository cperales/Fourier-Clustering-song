import numpy as np
from .load import load_as_dict, load_interpolator
from .transform import fourier_plot, limit_by_freq, group_by_freq
import os
import pickle
import multiprocessing
from random import shuffle
from copy import deepcopy
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
        # Moving backward
        backward_diff = distance_func(x_copy[:-i], y_copy[i:])
        if forward_diff < min_diff:
            if backward_diff < forward_diff:
                min_diff = backward_diff
            else:
                min_diff = forward_diff
        elif backward_diff < min_diff:
            min_diff = backward_diff
    return min_diff


def pair_distance(song_x,
                  song_y,
                  warp=None,
                  upper_limit=6000,
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
    # FILTERING FREQUENCIES, LESS INFORMATION
    frequencies, features = limit_by_freq(frequencies,
                                          features,
                                          upper_limit=6000)

    # TIME FRAMES
    distance = 0.0
    freq_div = np.max(frequencies) / frames
    for i in range(1, frames + 1):
        bottom_limit = (i - 1) * freq_div
        upper_limit = i * freq_div
        frequencies_copy, features_copy = limit_by_freq(frequencies,
                                                        features,
                                                        upper_limit=upper_limit,
                                                        bottom_limit=bottom_limit)
        features_1_copy = np.interp(frequencies_copy,
                                    frequencies_1,
                                    features_1)

        if warp is None:
            distance_metric = distance_dict[dis]

        frame_dist = positive_error(features_copy, features_1_copy)
        # frame_dist = hellinger(features_copy, features_1_copy)
        # frame_dist = integrate(features_copy, features_1_copy)
        # frame_dist = l2_norm(features_copy, features_1_copy)
        # frame_dist, cost, acc, path = fastdtw(features_copy, features_1_copy, dist='euclidean')
        print('{}: Frame {}, dist = {}'.format(song_name, i, frame_dist))
        distance += frame_dist
        if distance > dist_limit:
            print('{}: Distance with {} exceeded'.format(song_name, song_name_1))
            break

    if distance <= dist_limit:
        print('{}: Total distance with {} = {}'.format(song_name, song_name_1, distance))
        avg_distance = distance
        for full_sub_folder, songs in songs_stored_dict.items():
            if song_name_1 in songs:
                sub_folder = full_sub_folder
                text_file = os.path.join(full_sub_folder, 'list_songs.txt')
                break
        dist_limit = distance - eps  # Adapt the song to the bst

    else:
        avg_distance = 0.0

    return avg_distance
