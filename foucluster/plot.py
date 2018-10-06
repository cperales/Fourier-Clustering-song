import matplotlib.pyplot as plt
import os
import seaborn as sns


def heatmap_song(song_df,
                 image_name,
                 image_folder=None):
    fig, ax = plt.subplots()
    sns.heatmap(song_df)
    fig.subplots_adjust(left=0.35,
                        bottom=0.35,
                        right=1.0,
                        top=0.95)
    plt.savefig(os.path.join(image_folder, image_name) + '.png')
    plt.close()


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
