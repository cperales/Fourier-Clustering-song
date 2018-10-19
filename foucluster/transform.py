import os
import glob
import json
import subprocess
import multiprocessing as mp
import numpy as np
from scipy.io.wavfile import read
from .plot import fourier_plot, song_plot


def removing_spaces(source_folder):
    for song in os.listdir(source_folder):
        sep = ' ' if ' ' in song else '_'
        new_song = [string for string in song.split(sep)
                    if string != '-' and not string.isdigit()]
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
            bash_command = ['mpg123', '-w', wav_file, '--mono', mp3_file]
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
    freq = np.linspace(0, rate, len(fourier))

    # 20 Hz is the lowest frequency audible by humans
    freq, fourier = limit_by_freq(freq,
                                  fourier,
                                  lower_limit=10,
                                  upper_limit=rate_limit)
    freq, fourier = group_by_freq(freq,
                                  fourier,
                                  step_size=10)

    a = np.max(fourier) / 100.0  # Max frequency will be 100.0
    fourier = fourier / a

    return freq, fourier


def group_by_freq(freq, features, step_size=10):
    """

    :param freq:
    :param features:
    :param step_size:
    :return:
    """
    min_freq = int(np.min(freq))
    max_freq = int(np.max(freq))
    length = int((max_freq - min_freq) / step_size) + 1
    new_freq = np.empty(length, dtype=np.float)
    new_features = np.empty(length, dtype=np.float)
    i = 0
    for freq_i in range(min_freq, max_freq, step_size):
        mask_1 = freq >= freq_i
        mask_2 = freq < freq_i + step_size
        mask = mask_1 * mask_2
        new_freq[i] =  np.mean(freq[mask])
        new_features[i] = np.mean(features[mask])
        i += 1
    new_freq = np.array(new_freq, dtype=np.float)
    new_features = np.array(new_features, dtype=np.float)
    return new_freq, new_features


def limit_by_freq(freq, features, upper_limit, lower_limit=None):
    """
    Limit arrays of frequency and features by maximum frequency and
    bottom frequency.

    :param freq: array of frequencies.
    :param features: array of amplitude.
    :param float upper_limit: maximum frequency.
    :param float lower_limit: minimum frequency.
    :return:
    """
    # Copy into arrays, in order to apply mask
    freq = np.array(freq, dtype=np.float)
    features = np.array(features, dtype=np.float)
    # Mask for bottom limit
    if lower_limit is not None:
        bottom_mask = freq >= lower_limit
        features = features[bottom_mask]
        freq = freq[bottom_mask]
    # Mask for upper limit
    upper_mask = freq <= upper_limit
    features = features[upper_mask]
    freq = freq[upper_mask]
    return freq, features


def dict_to_array(song_dict):
    """

    :param dict song_dict: load form dictionary to array
    :return:
    """
    freq = np.array([k for k in song_dict.keys()], dtype=np.float)
    features = np.array([v for v in song_dict.values()], dtype=np.float)
    return freq, features


def time_to_frequency(song,
                      source_folder,
                      temp_folder,
                      output_folder,
                      rate_limit=6000.0,
                      overwrite=True,
                      plot=True,
                      image_folder=None):
    """
    Transform a MP3 song into WAV format, and then into
    Fourier series.

    :param str song: name of the song, with MP3 extension.
    :param str source_folder: folder where MP3 files are.
    :param str output_folder: folder where pickle files from
        frequency series are saved.
    :param str temp_folder: folder where wav files are saved.
    :param float rate_limit: maximum frequency of the frequency
        series.
    :param bool overwrite:
    :param bool plot: if True, frequency series is plotted.
    :param image_folder: if plotting is True, is the folder
        where the Fourier data is saved.
    :return:
    """
    song_name = os.path.splitext(song)[0]
    json_name = song_name + '.json'

    # Name of files
    mp3_file = os.path.join(source_folder, song)
    wav_file = os.path.join(temp_folder, song_name + '.wav')

    # Transform MP3 into WAV
    transform_wav(mp3_file=mp3_file, wav_file=wav_file)

    full_json_name = os.path.join(output_folder, json_name)
    if not os.path.isfile(full_json_name) or overwrite is True:
        # Fourier transformation
        try:
            try:
                frequencies, fourier_series = \
                    fourier_song(wav_file=wav_file,
                                 rate_limit=rate_limit)
            except MemoryError:
                rate_limit = rate_limit / 2.0
                frequencies, fourier_series = \
                    fourier_song(wav_file=wav_file,
                                 rate_limit=rate_limit)

            # Transform to dict
            freq_dict = {str(x): y for x, y in zip(frequencies, fourier_series)}

            # Save as JSON
            json_to_save = {song: freq_dict}
            with open(full_json_name, 'w') as output:
                json.dump(json_to_save, output)

            # Plotting
            if plot:
                fourier_plot(freq=frequencies,
                             features=fourier_series,
                             folder=image_folder,
                             filename=song_name)
                rate, aud_data = read(wav_file)
                song_plot(features=aud_data,
                          folder=image_folder,
                          filename=song_name)
        except MemoryError:
            print('{} gives MemoryError'.format(song_name))


def all_songs(source_folder,
              output_folder,
              temp_folder,
              rate_limit=6000.0,
              overwrite=True,
              plot=False,
              image_folder=None):
    """
    Transform a directory full of MP3 files
    into WAVE files, and then into Fourier series,
    working with directories.

    :param str source_folder: folder where MP3 files are.
    :param str output_folder: folder where pickle files from
        frequency series are saved.
    :param str temp_folder: folder where wav files are saved.
    :param float rate_limit: maximum frequency of the frequency
        series.
    :param bool overwrite:
    :param bool plot: if True, frequency series is plotted.
    :param image_folder: if plotting is True, is the folder
        where the Fourier data is saved.
    """
    merged_file = os.path.join(output_folder, 'merged_file.json')

    os.makedirs(temp_folder, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)
    if os.path.isfile(merged_file):
        os.remove(merged_file)
    if plot:
        os.makedirs(image_folder, exist_ok=True)

    songs = [(song, source_folder, temp_folder, output_folder, rate_limit,
              overwrite, plot, image_folder)
             for song in os.listdir(source_folder)]

    with mp.Pool(processes=max(int(mp.cpu_count() / 2), 1)) as p:
        p.starmap(time_to_frequency, songs)

    read_files = glob.glob(os.path.join(output_folder, '*.json'))

    with open(merged_file, 'w') as outfile:
        file_contents = [open(f).read() for f in read_files]
        outfile.write('[{}]'.format(','.join(file_contents)))
