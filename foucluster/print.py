import matplotlib.pyplot as plt
import numpy as np
import os


def heatmap_song(song_df):
    plt.pcolor(song_df)
    plt.yticks(np.arange(0.5, len(song_df.index), 1),
               song_df.index)
    plt.xticks(np.arange(0.5, len(song_df.columns), 1),
               song_df.columns)
    plt.show()


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
