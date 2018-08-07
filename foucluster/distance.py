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
    return np.linalg.norm(x / np.linalg.norm(x) -
                          y / np.linalg.norm(y))


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


def pair_distance_old(pickle_folder,
                  song_folder,
                  distance_limit=0.15,
                  n_frames=1):
    def _check_song(p):
        dist_limit = distance_limit * n_frames
        song_name = os.path.splitext(p)[0]
        print('Song name:', song_name)
        list_folders = os.listdir(song_folder)
        if len(list_folders) == 0:  # First song
            sub_folder = os.path.join(song_folder, '0')
            os.makedirs(sub_folder)
            with open(os.path.join(sub_folder, 'list_songs.txt'), 'w') as f:
                f.write(''.join([song_name, '\n']))
        else:
            already_stored = False
            songs_stored = list()
            songs_stored_dict = dict()
            for sub_folder in list_folders:
                full_sub_folder = os.path.join(song_folder, sub_folder)
                text_file = os.path.join(full_sub_folder, 'list_songs.txt')

                with open(text_file, 'r') as f:
                    more_songs = [f_l.split()[0] for f_l in f.readlines()]
                    songs_stored += more_songs
                    songs_stored_dict.update({full_sub_folder: more_songs})

                # Searching the song in the folder
                if song_name in more_songs:
                    already_stored = True
                    break

            # If it's already stored in any folder, don't compute this part
            if already_stored is False:
                # Now and only now it loads the pickle
                p_name = os.path.join(pickle_folder, p)
                with open(p_name, 'rb') as p_file:
                    p_file.seek(0)
                    tuple_to_load = pickle.load(p_file)
                frequencies = tuple_to_load[0]
                features = tuple_to_load[1]

                # FILTERING FREQUENCIES, LESS INFORMATION
                frequencies, features = limit_by_freq(frequencies,
                                                      features,
                                                      upper_limit=6000)

                frequencies, features = group_by_freq(frequencies,
                                                      features)

                avg_d = np.inf

                shuffle(songs_stored)  # Avoiding enlarging first clusters
                # Loading the other songs
                for song_stored in songs_stored:
                    song = load_interpolator(pickle_folder=pickle_folder,
                                             pickle_file=song_stored)
                    song_name_1 = list(song.keys())[0]
                    features_1 = song[song_name_1]['features']
                    frequencies_1 = song[song_name_1]['frequencies']

                    # TIME FRAMES
                    distance = 0.0
                    freq_div = np.max(frequencies) / n_frames
                    for i in range(1, n_frames + 1):
                        bottom_limit = (i - 1) * freq_div
                        upper_limit = i * freq_div
                        frequencies_copy, features_copy = limit_by_freq(frequencies,
                                                                        features,
                                                                        upper_limit=upper_limit,
                                                                        bottom_limit=bottom_limit)
                        features_1_copy = np.interp(frequencies_copy,
                                                    frequencies_1,
                                                    features_1)

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
                        avg_d = distance
                        for full_sub_folder, songs in songs_stored_dict.items():
                            if song_name_1 in songs:
                                sub_folder = full_sub_folder
                                text_file = os.path.join(full_sub_folder, 'list_songs.txt')
                                break
                        dist_limit = distance - eps  # Adapt the song to the bst

            else:
                avg_d = 0.0

            if avg_d <= eps:  # Already in a folder
                print('Song already in cluster {}'.format(sub_folder))
            elif avg_d <= (dist_limit + eps):  # Similar to a song in that folder
                with open(text_file, 'a') as f:
                    f.write(''.join([song_name, '\n']))
                print('Song {} moved to cluster {}'.format(song_name, sub_folder))
                fourier_plot(frequencies,
                             features,
                             folder=full_sub_folder,
                             filename=song_name)
                print('Cluster distance {}'.format(avg_d))
            else:
                count_folder = np.max(np.array(os.listdir(song_folder), dtype=int)) + 1
                sub_folder = os.path.join(song_folder, str(count_folder))
                os.makedirs(sub_folder)
                with open(os.path.join(sub_folder, 'list_songs.txt'), 'w') as f:
                    f.write(''.join([song_name, '\n']))
                print('New cluster {}'.format(sub_folder))
                fourier_plot(frequencies,
                             features,
                             folder=sub_folder,
                             filename=song_name)
        print()

    cpu_count = 0
    process_list = list()
    for p_i in os.listdir(pickle_folder):
        process = multiprocessing.Process(target=_check_song, args=(p_i,))
        process_list.append(process)
        process.start()
        if cpu_count >= cpus:
            for process in process_list:
                process.join()
            process_list = list()
            cpu_count = 0
        cpu_count += 1
