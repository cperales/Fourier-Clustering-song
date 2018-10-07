from foucluster.transform import all_songs
from foucluster.distance import distance_matrix, distance_dict
from foucluster.plot import heatmap_song
from foucluster.cluster import automatic_cluster, \
    score_cluster, cluster_methods
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

print('Transforming MP3 songs into Fourier series...')
all_songs(source_folder=source_folder,
          output_folder=output_folder,
          temp_folder=temp_folder,
          rate_limit=rate_limit,
          overwrite=False,
          plot=True,
          image_folder=image_folder)

# Distance metric
print('Calculating distance matrix...')
for metric in metrics:
    print(' ', metric)
    song_df = distance_matrix(fourier_folder=output_folder,
                              warp=warp,
                              upper_limit=rate_limit,
                              frames=frames,
                              distance_metric=metric)

    song_df.to_csv(os.path.join(distance_folder,
                                metric + '.csv'),
                   sep=';')

# Heat map
print('Plotting heat maps...')
for metric in metrics:
    print(' ', metric)
    dist_df = pd.read_csv(os.path.join(distance_folder,
                                       metric + '.csv'),
                          sep=';')
    dist_df = dist_df.set_index('Songs')
    heatmap_song(dist_df,
                 image_name=metric,
                 image_folder=image_folder)

# Clustering test
print('Testing cluster methods...')
max_score = 0.0
score_vector = []
metric_vector = []
cluster_method_vector = []

for metric in metrics:
    print(' ', metric)
    song_df = pd.read_csv(os.path.join(distance_folder,
                                       metric + '.csv'),
                          sep=';')
    song_df = song_df.set_index('Songs')
    for cluster_method in cluster_methods:
        print('  ', cluster_method)
        cluster_df = automatic_cluster(dist_df=song_df,
                                       method=cluster_method)
        score = score_cluster(cluster_df)
        cluster_df.to_csv(os.path.join(cluster_folder,
                                       metric + '_' +
                                       cluster_method +
                                       '.csv'),
                          sep=';')
        # Update info
        score_vector.append(score)
        metric_vector.append(metric)
        cluster_method_vector.append(cluster_method)
        # Choosing best methodology
        if score > max_score:
            # print(metric, cluster_method, score)
            max_score = score
            best_metric = metric
            best_cluster_method = cluster_method

test_dict = {'Accuracy': score_vector,
             'Metric': metric_vector,
             'Cluster_method': cluster_method_vector}
df = pd.DataFrame(test_dict)
df.to_csv(os.path.join(cluster_folder,
                       'cluster_test.csv'),
          sep=';', index=False)
print()
print('Best performance ({}) is achieved with {} metric, {} cluster method'.format(max_score,
                                                                                   best_metric,
                                                                                   best_cluster_method))
