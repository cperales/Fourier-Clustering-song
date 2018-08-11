# from foucluster.transform import pair_distance
from foucluster.transform import all_songs
import configparser


config = configparser.ConfigParser()
config.read('config.ini')

source_folder = config['Folder']['Source']
temp_folder = config['Folder']['Temp']
image_folder = config['Folder']['Image']
output_folder = config['Folder']['Output']

all_songs(source_folder=source_folder,
          output_folder=output_folder,
          temp_folder=temp_folder,
          rate_limit=6000,
          overwrite=False,
          plot=False,
          image_folder=None)


# distance_limit = 0.14  # fastdtw
# distance_limit = 70.0  # np.linalg.norm
# distance_limit = 1854.0  # np.trapz
# distance_limit = 0.9  # hellinger
# distance_limit = 0.5  # Warping error
# pair_distance(pickle_folder=pickle_folder,
#               song_folder=output_folder,
#               distance_limit=distance_limit,
#               n_frames=1)
