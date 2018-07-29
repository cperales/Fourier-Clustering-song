import multiprocessing as mp
import os
import pickle
import subprocess
import matplotlib.pyplot as plt
import numpy as np
from scipy.io.wavfile import read

# Enable multiprocessing
cpus = mp.cpu_count()


def removing_spaces(source_folder):
    for song in os.listdir(source_folder):
        new_song = list()
        string_list = song.split() if len(song.split()) > 1 else song.split('_')
        for string in string_list:
            if string != '-' and not string.isdigit():
                new_song.append(string)
        new_song = '_'.join(new_song)
        file = os.path.join(source_folder, song)
        new_file = os.path.join(source_folder, new_song)
        os.rename(file, new_file)


def transform_wav(mp3_file, wav_file):
    """
    Transform mp3 file into wav format using mpg123
    or ffmpeg.

    :param str mp3_file:
    :param str wav_file:
    :return:
    """
    if not os.path.isfile(wav_file):
        try:
            bash_command = ['mpg123', '-w', wav_file, mp3_file]
            subprocess.run(bash_command)
        except Exception as e:
            print(e)
            print('Trying with ffmpeg...')
            alt_command = ['ffmpeg', '-i', mp3_file, wav_file]
            subprocess.run(alt_command)


def fourier_song(wav_file,
                 rate_limit=6000.0):
    rate, aud_data = read(wav_file)
    # Should be mono
    if len(aud_data) != len(aud_data.ravel()):
        aud_data = np.mean(aud_data, axis=1)

    # Zero padding
    len_data = len(aud_data)
    channel_1 = np.zeros(2 ** (int(np.ceil(np.log2(len_data)))))
    channel_1[0:len_data] = aud_data

    # Fourier analysis
    fourier = np.abs(np.fft.fft(channel_1))
    w = np.linspace(0, rate, len(fourier))

    w, fourier_to_plot = limit_by_freq(w, fourier, upper_limit=rate_limit)
    w, fourier_to_plot = group_by_freq(w, fourier_to_plot)

    # a = np.mean(fourier_to_plot)
    fourier_to_plot[np.argmax(fourier_to_plot)] = 0.0
    a = np.max(fourier_to_plot) / 20.0  # Max frequency will be 20.0
    fourier_to_plot = fourier_to_plot / a

    return w, fourier_to_plot


def group_by_freq(freq, features, max_rate=None, min_freq=1):
    """

    :param freq:
    :param features:
    :param max_rate:
    :param min_freq:
    :return:
    """
    if max_rate is None:
        max_rate = np.max(freq)
    final_length = int(max_rate / min_freq)
    new_freq = np.empty(final_length)
    new_features = np.empty(final_length)
    for i in range(final_length):
        mask_1 = freq >= i
        mask_2 = freq < (i + min_freq)
        mask = mask_1 * mask_2
        new_freq[i] = np.mean(freq[mask])
        new_features[i] = np.mean(features[mask])
    return new_freq, new_features


def limit_by_freq(freq, features, upper_limit, bottom_limit=None):
    """

    :param freq:
    :param features:
    :param upper_limit:
    :param bottom_limit:
    :return:
    """
    freq = np.array(freq[:], dtype=np.float)
    features = np.array(features[:], dtype=np.float)
    if bottom_limit is not None:
        bottom_mask = freq >= bottom_limit
        features = features[bottom_mask]
        freq = freq[bottom_mask]
    upper_mask = freq <= upper_limit
    features = features[upper_mask]
    freq = freq[upper_mask]
    return freq, features


def fourier_plot(freq, features,
                 folder=None,
                 filename=None):
    """
    """
    fig = plt.figure(1)
    # Turn interactive plotting off
    plt.ioff()
    plt.plot(freq, features)
    plt.xlabel('frequency')
    plt.ylabel('amplitude')
    if folder is None:
        folder = ''
    if filename is not None:
        plt.savefig(os.path.join(folder,
                                 filename + '.png'))
    plt.close(fig)


def all_songs(source_folder,
              output_folder,
              temp_folder,
              rate_limit=6000.0,
              overwrite=False,
              removing=True,
              plotting=False,
              image_folder=None):
    """
    Transform MP3 into wave and into pickle afterwards.

    :param str source_folder: folder where MP3 files are.
    :param str output_folder: folder where pickle files from
        frequency series are saved.
    :param str temp_folder: folder where wav files are saved.
    :param float rate_limit: maximum frequency of the frequency
        series.
    :param bool overwrite: if True, wav files are overwritten.
    :param bool removing: if True, wav files are removed after
        Fourier transform.
    :param bool plotting: if True, frequency series is plotted.
    :param image_folder: if plotting is True, is the folder
        where the Fourier data is saved.
    :return:
    """
    if not os.path.isdir(temp_folder):
        os.makedirs(temp_folder)

    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)

    if plotting is True and not os.path.isdir(image_folder):
        os.makedirs(image_folder)

    for song in os.listdir(source_folder):
        song_name = os.path.splitext(song)[0]
        pickle_name = os.path.join(output_folder,
                                   song_name + '.pkl')

        if not os.path.isfile(pickle_name) or overwrite is True:
            # Name of files
            mp3_file = os.path.join(source_folder, song)
            wav_file = os.path.join(temp_folder, song_name + '.wav')

            # Transform MP3 into WAV
            transform_wav(mp3_file=mp3_file,
                          wav_file=wav_file)

            # Fourier transformation
            frequencies, fourier_series = fourier_song(wav_file=wav_file,
                                                       rate_limit=rate_limit)

            # Removing wav file
            if removing is True:
                os.remove(wav_file)

            # Save pickle
            with open(pickle_name, 'wb') as output:
                tuple_to_save = frequencies, fourier_series
                pickle.dump(tuple_to_save, output)

            # Plotting
            if plotting is True:
                fourier_plot(freq=frequencies,
                             features=fourier_series,
                             folder=image_folder,
                             filename=song_name)
