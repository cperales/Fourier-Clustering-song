from foucluster.transform import all_songs
from foucluster.distance import distance_matrix
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

dist_df = distance_matrix(fourier_folder=output_folder,
                          warp=None,
                          upper_limit=6000.0,
                          frames=1,
                          distance_metric='l2_norm')

dist_df.to_csv('example.csv', sep=';')
