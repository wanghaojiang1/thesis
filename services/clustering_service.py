from . import matrix, match_service, node_service
from tqdm import tqdm
from scipy.cluster.hierarchy import ward, fcluster, dendrogram
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import pdist
import shutil, os
import numpy as np
import pandas as pd
import json
import matplotlib
matplotlib.use('Agg')

from matplotlib import pyplot as plt

CLUSTERING_THRESHOLD = 1.2
MAX_DISTANCE = 1
MATRIX_LOCATION = './exports/proximity_matrix.csv'
LINKAGE_LOCATION = './exports/linkage_matrix.npy'
DENDROGRAM_LOCATION = './exports/dendrogram.png'

def clusterk(k):
    matrix.initialize_matrix()
    scores = matrix.calculate_score()

    nodes = node_service.get_nodes()
    proximity_matrix = _create_proximity_matrix(nodes, scores)

    cluster = AgglomerativeClustering(n_clusters=k, affinity='euclidean', linkage='ward')
    cluster.fit_predict(proximity_matrix)
    parsed_clusters = _parse_clusters(cluster.labels_)

    make_plot()
    _save_parsed_clusters(parsed_clusters)
    set_clustering_threshold(k)

def cluster():
    matrix.initialize_matrix()
    scores = matrix.calculate_score()

    nodes = node_service.get_nodes()
    proximity_matrix = _create_proximity_matrix(nodes, scores)
    clusters = _hierarchical_clustering(proximity_matrix)
    parsed_clusters = _parse_clusters(clusters)

    make_plot()
    _save_parsed_clusters(parsed_clusters)

def update_cluster(threshold):
    set_clustering_threshold(threshold)
    clusters = cluster_linkage_matrix(get_linkage())
    parsed_clusters = _parse_clusters(clusters)
    make_plot()
    _save_parsed_clusters(parsed_clusters)

def _create_proximity_matrix(nodes, scores):
    # Provided that the scores are in the range of [0, 1]
    inversed_scores = {k: (1 - v) for k, v in scores.items()}

    # N x N matrix where N = |nodes|
    names = list(map(lambda node: node.get('id'), nodes))
    proximity_matrix = pd.DataFrame(np.ones((len(nodes), len(nodes))), index=names, columns=names)
    # Adding distance values
    for (edge_id, distance) in tqdm(inversed_scores.items()):
        relation = node_service.get_match(int(edge_id))
        node_from = relation['from']
        node_to = relation['to']

        proximity_matrix.loc[node_to, node_from] = distance
        proximity_matrix.loc[node_from, node_to] = distance
        # break
    
    # All same tables should have distance 0: for each column
    # QUESTION: should this be 0? might mess up with linking        --> Experiment
    for node in tqdm(nodes):
        node_id = node.get('id')
        proximity_matrix[node_id][node_id] = 0
        siblings = node_service.get_siblings(node_id)

        for sibling in siblings:
            sibling_id = sibling.get('id')

            proximity_matrix[node_id][sibling_id] = MAX_DISTANCE
            proximity_matrix[sibling_id][node_id] = MAX_DISTANCE

    proximity_matrix.to_csv(MATRIX_LOCATION)
    return proximity_matrix

def _parse_clusters(clusters):
    nodes = node_service.get_nodes()
    result_clusters = []
    for cluster in set(clusters):
        indices = [i for i, x in enumerate(clusters) if x == cluster]
        cluster_nodes = list(map(lambda index: nodes[index].get('id'), indices))
        result_clusters.append(cluster_nodes)

    return result_clusters

def _hierarchical_clustering(matrix):
    Z = ward(pdist(matrix))
    save_linkage(Z)

    clusters = cluster_linkage_matrix(Z)

    return clusters

def cluster_linkage_matrix(Z):
    clusters = fcluster(Z, t=CLUSTERING_THRESHOLD, criterion='distance')
    return clusters

####### Code for plotting #######
def make_plot():
    Z = get_linkage()
    show_plot(Z)
    shutil.copy(DENDROGRAM_LOCATION, './static/dendrogram.png')

def show_plot(Z):
    fig, ax = plt.subplots(figsize=(25, 10))
    dn = dendrogram(Z)

    headers = list(get_proximity_matrix().columns)
    del headers[0]

    locs, labels = plt.xticks()
    for label in labels:
        label.set_text(headers[int(label.get_text())])

    ax.set_xticklabels(labels)
    ax.tick_params(axis='x', labelrotation = 90)

    plt.axhline(y=CLUSTERING_THRESHOLD, c='grey', lw=1, linestyle='dashed')
    plt.show()
    fig.tight_layout()
    
    plt.savefig(DENDROGRAM_LOCATION)
    plt.cla()
#################################

def get_proximity_matrix():
    return pd.read_csv(MATRIX_LOCATION)

def get_linkage():
    return np.load(LINKAGE_LOCATION)

def get_clusters():
    if (not _clusters_exists()):
        return {}

    with open('./exports/clusters.json') as json_file:
        result = json.load(json_file)
        return result

def set_clustering_threshold(threshold):
    global CLUSTERING_THRESHOLD 
    CLUSTERING_THRESHOLD = threshold

def save_linkage(Z):
    np.save(LINKAGE_LOCATION, Z)

def _save_parsed_clusters(clusters):
    expert_weights = match_service.get_raw_expert_weights()
    result = {
        'setting': expert_weights,
        'clusters': clusters
    }

    with open("./exports/clusters.json", "w") as outfile:
        json.dump(result, outfile)

def _clusters_exists():
    return os.path.exists('./exports/clusters.json')

def reset():
    remove_file(DENDROGRAM_LOCATION)
    remove_file(LINKAGE_LOCATION)
    remove_file(MATRIX_LOCATION)
    remove_file('./exports/clusters.json')

def remove_file(location):
    if os.path.exists(location):
        os.remove(location)
    else:
        print("The file: {} does not exist".format(location))