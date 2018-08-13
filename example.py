from foucluster.transform import all_songs
from foucluster.distance import distance_matrix, distance_dict
from foucluster.fouplot import heatmap_song
from foucluster.cluster import discrete_cluster_methods,\
    cluster_methods, discrete_cluster
import configparser
import pandas as pd
import os


config = configparser.ConfigParser()
config.read('config.ini')

source_folder = config['Folder']['Source']
temp_folder = config['Folder']['Temp']
image_folder = config['Folder']['Image']
output_folder = config['Folder']['Output']
distance_folder = config['Folder']['Distance']
cluster_folder = config['Folder']['Cluster']
warp = config['Fourier']['warp']
warp = None if str(warp) == 'None' else int(warp)
frames = int(config['Fourier']['frames'])
rate_limit = float(config['Fourier']['rate'])
metrics = distance_dict.keys()

# all_songs(source_folder=source_folder,
#           output_folder=output_folder,
#           temp_folder=temp_folder,
#           rate_limit=rate_limit,
#           overwrite=False,
#           plot=True,
#           image_folder=image_folder)

for metric in metrics:
    if warp is None:
        name = str(metric)
    else:
        name = str(metric) + '_' + str(warp)
    song_df = distance_matrix(fourier_folder=output_folder,
                              warp=warp,
                              upper_limit=rate_limit,
                              frames=frames,
                              distance_metric=metric)

    song_df.to_csv(os.path.join(distance_folder,
                                name + '.csv'),
                   sep=';')

n_clusters = 3
for discrete_cluster_method in discrete_cluster_methods:
    cluster_df = discrete_cluster(song_df,
                                  method=discrete_cluster_method,
                                  n_clusters=n_clusters)
    cluster_df.to_csv(os.path.join(cluster_folder,
                                   name +
                                   '_' + discrete_cluster_method +
                                   '.csv'),
                      sep=';')

# for metric in metrics:
#     dist_df = pd.read_csv(os.path.join(distance_folder,
#                                        name + '.csv'),
#                           sep=';')
#     dist_df = dist_df.set_index('Songs')
#     heatmap_song(dist_df,
#                  image_name=name,
#                  image_folder=image_folder)
