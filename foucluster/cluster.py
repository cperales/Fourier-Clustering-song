from copy import deepcopy
from sklearn.cluster import AgglomerativeClustering, SpectralClustering, KMeans
from sklearn.cluster import AffinityPropagation, MeanShift
from scipy.spatial.distance import cdist
import pandas as pd
import numpy as np

eps = 10**(-10)

n_cluster_methods = {'AgglomerativeClustering': AgglomerativeClustering,
                     'SpectralClustering': SpectralClustering,
                     'KMeans': KMeans}

non_n_cluster_methods = {'AffinityPropagation': AffinityPropagation,
                         'MeanShift': MeanShift}


cluster_methods = n_cluster_methods.copy()
cluster_methods.update(non_n_cluster_methods)


def determinist_cluster(dist_df, method, n_clusters):
    """

    :param pd.DataFrame dist_df:
    :param str method:
    :param int n_clusters:
    :return:
    """
    y = n_cluster_methods[method](n_clusters=n_clusters).fit_predict(dist_df)
    cluster_df = deepcopy(dist_df)
    cluster_df['Cluster'] = pd.Series(y, index=cluster_df.index)
    return cluster_df


def automatic_cluster(dist_df, method):
    """

    :param pd.DataFrame dist_df:
    :param str method:
    :return:
    """
    if method in non_n_cluster_methods.keys():
        clf = non_n_cluster_methods[method]().fit(dist_df)
        y = clf.predict(dist_df)
    else:
        n_clusters = jump_method(dist_df=dist_df)
        cluster = n_cluster_methods[method](n_clusters=n_clusters)
        y = cluster.fit_predict(dist_df)

    cluster_df = deepcopy(dist_df)
    cluster_df['Cluster'] = pd.Series(y, index=cluster_df.index)
    return cluster_df


def jump_method(dist_df, n_max=50):
    """
    Method based on information theory to determine best
    number of clusters.

    :param dist_df:
    :param n_max:
    :return:
    """
    dim = len(dist_df.index)
    if n_max > dim:
        n_max = dim
    Y = dim / 2
    distortions = np.empty(n_max + 1)
    jump_vector = np.empty(n_max)
    distortions[0] = 0.0
    for k in range(1, n_max + 1):
        kmean_model = KMeans(n_clusters=k).fit(dist_df)
        distortion = np.min(cdist(dist_df,
                                  kmean_model.cluster_centers_,
                                  'euclidean').ravel()) / dim + eps
        distortions[k] = distortion**(- Y)
        jump_vector[k - 1] = distortions[k] - distortions[k - 1]
    n_cluster = np.argmax(jump_vector) + 1
    return n_cluster


def score_cluster(cluster_df):
    """

    :param cluster_df:
    :return:
    """
    accurate_class = [int(n[0]) for n in cluster_df.index.tolist()]
    accurate_class -= np.unique(accurate_class)[0]
    # Move to 0, 1, ... notation
    accurate_class = np.array(accurate_class, dtype=int)
    cluster_class = np.array(cluster_df['Cluster'].tolist(), dtype=int)
    # Find correspondences between given classes and cluster classes
    correspondence_dict = {}

    for p in np.unique(cluster_class):
        max_c = 0.0
        pos_p = cluster_class == p
        for e in np.unique(accurate_class):
            pos_e = accurate_class == e
            c = (pos_p == pos_e).sum()
            if c > max_c:
                correspondence_dict.update({p: e})
                max_c = c
    # Finding the accuracy
    cluster_class_corrected = [correspondence_dict[p] for p in cluster_class]
    cluster_df['Cluster_corrected'] = pd.Series(cluster_class_corrected,
                                                index=cluster_df.index)
    score_vector = [e == p_c for e, p_c in
                    zip(accurate_class, cluster_class_corrected)]
    return np.average(score_vector)


def party_list(cluster_df, song=None):
    if song is None or song not in cluster_df.columns:
        song = cluster_df.index[0]
    print(cluster_df.sort_values(song)[song])
