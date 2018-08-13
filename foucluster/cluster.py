from sklearn.cluster import AgglomerativeClustering, SpectralClustering, KMeans
from sklearn.cluster import AffinityPropagation, MeanShift
import pandas as pd
from copy import deepcopy


discrete_cluster_methods = {'AgglomerativeClustering': AgglomerativeClustering,
                            'SpectralClustering': SpectralClustering,
                            'KMeans': KMeans}

cluster_methods = {'AffinityPropagation': AffinityPropagation,
                   'MeanShift': MeanShift}


def discrete_cluster(dist_df, method, n_clusters):
    """

    :param dist_df:
    :return:
    """
    y = discrete_cluster_methods[method](n_clusters=n_clusters).fit_predict(dist_df)
    cluster_df = deepcopy(dist_df)
    cluster_df['Cluster'] = pd.Series(y, index=cluster_df.index)
    return cluster_df


def cluster(dist_df, method):
    """

    :param dist_df:
    :return:
    """
    clf = discrete_cluster_methods[method]().fit(dist_df)
    y_kmeans = clf.predict(dist_df)
    cluster_df = deepcopy(dist_df)
    cluster_df['Cluster'] = pd.Series(y_kmeans, index=cluster_df.index)
    return cluster_df
