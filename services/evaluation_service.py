from . import matrix, match_service, clustering_service, node_service, matching_techniques
from tqdm import tqdm
from scipy.cluster.hierarchy import ward, fcluster, dendrogram
from scipy.spatial.distance import pdist
from itertools import combinations
import os
import numpy as np
import pandas as pd
import json
import matplotlib
import operator
matplotlib.use('Agg')

from matplotlib import pyplot as plt

def create_matchers_best_metric_bar_chart(precision, recall, f_measure):
    ground_truth = get_ground_truth()
    matches = node_service.get_matches()

    precision_map = {}
    recall_map = {}
    f_measure_map = {}

    x = ['Precision', 'Recall', 'F-measure']

    matchers = list(map(lambda matcher: matcher['type'], matching_techniques.VARIANTS))
    for matcher in tqdm(matchers):
        local_matches = list(map(lambda node: {'id': node['id'], 'from': node['from'], 'to': node['to'], 'score': node['scores'][matcher]}, matches))
        local_matches.sort(key=operator.itemgetter('score'), reverse=True)

        local_precision = 0
        local_recall = 0
        local_f_measure = 0

        # for k in range(0, len(local_matches)):
        for k in range(0, 500):
            results = evaluate_relations_at(local_matches, ground_truth, k)
            if results['f-measure'] > local_f_measure:

                local_precision = results['precision']
                local_recall = results['recall']
                local_f_measure = results['f-measure']
        
        precision_map[matcher] = local_precision
        recall_map[matcher] = local_recall
        f_measure_map[matcher] = local_f_measure

    our_name = 'max'
    local_precision = 0
    local_recall = 0
    local_f_measure = 0
    local_matches = list(map(lambda node: {'id': node['id'], 'from': node['from'], 'to': node['to'], 'score': node['scores'][max(node['scores'].keys(), key=(lambda new_k: node['scores'][new_k]))]}, matches))
    local_matches.sort(key=operator.itemgetter('score'), reverse=True)

    for k in range(0, 500):
        results = evaluate_relations_at(local_matches, ground_truth, k)
        if results['f-measure'] > local_f_measure:
            local_precision = results['precision']
            local_recall = results['recall']
            local_f_measure = results['f-measure']

    precision_map[our_name] = local_precision
    recall_map[our_name] = local_recall
    f_measure_map[our_name] = local_f_measure

    our_name = 'weighted_average'
    local_precision = 0
    local_recall = 0
    local_f_measure = 0
    our_matches = match_service.get_ordered_matches()
    for k in range(0, 500):
        results = evaluate_relations_at(our_matches, ground_truth, k)
        if results['f-measure'] > local_f_measure:
            local_precision = results['precision']
            local_recall = results['recall']
            local_f_measure = results['f-measure']
        
    precision_map[our_name] = local_precision
    recall_map[our_name] = local_recall
    f_measure_map[our_name] = local_f_measure


    our_name = 'weighted_average (with clusters)'
    precision_map[our_name] = precision
    recall_map[our_name] = recall
    f_measure_map[our_name] = f_measure

    width = 1.0 / (len(matchers) + 4)
    ind = np.arange(len(x))
    index = 0
    for key, value in precision_map.items():
        total_value = [precision_map[key], recall_map[key], f_measure_map[key]]
        plt.bar(ind + (index * width), total_value, width = width, label=key)

        index = index + 1
    # plt.legend(loc="upper right")
    plt.legend(bbox_to_anchor = (1.05, 0.5))
    plt.axes().set_ylim([0, 1])
    plt.ylabel('Score')
    plt.xlabel('Metric type')
    plt.xticks(ind + (width * (len(matchers) + 3)) / 2, x)
    plt.savefig('exports/matcher_metric_best_f_measure_bar.png', bbox_inches='tight')
    plt.clf()
    return

def create_matchers_metric_graphs():
    ground_truth = get_ground_truth()
    matches = node_service.get_matches()
    precision_map = {}
    recall_map = {}
    f_measure_map = {}

    # x = range(1, len(matches) + 1)
    # with open("./exports/graph/range.txt", "w") as outfile:
    #     outfile.write(str(len(matches) + 1))
    # matchers = list(map(lambda matcher: matcher['type'], matching_techniques.VARIANTS))
    # for matcher in tqdm(matchers):
    #     local_matches = list(map(lambda node: {'id': node['id'], 'from': node['from'], 'to': node['to'], 'score': node['scores'][matcher]}, matches))
    #     local_matches.sort(key=operator.itemgetter('score'), reverse=True)

    #     local_precision = []
    #     local_recall = []
    #     local_f_measure = []

    #     for k in range(0, len(local_matches)):
    #         results = evaluate_relations_at(local_matches, ground_truth, k)
    #         local_precision.append(results['precision'])
    #         local_recall.append(results['recall'])
    #         local_f_measure.append(results['f-measure'])
        
    #     with open("./exports/graph/{}-precision.json".format(matcher), "w") as outfile:
    #         json.dump(local_precision, outfile)
    #     with open("./exports/graph/{}-recall.json".format(matcher), "w") as outfile:
    #         json.dump(local_recall, outfile)
    #     with open("./exports/graph/{}-f-measure.json".format(matcher), "w") as outfile:
    #         json.dump(local_f_measure, outfile)

    #     precision_map[matcher] = local_precision
    #     recall_map[matcher] = local_recall
    #     f_measure_map[matcher] = local_f_measure
    
    our_matches = match_service.get_ordered_matches()
    our_precision = []
    our_recall = []
    our_f_measure = []
    for k in range(0, len(our_matches)):
        results = evaluate_relations_at(our_matches, ground_truth, k)
        our_precision.append(results['precision'])
        our_recall.append(results['recall'])
        our_f_measure.append(results['f-measure'])


    with open("./exports/graph/max-precision.json", "w") as outfile:
            json.dump(our_precision, outfile)
    with open("./exports/graph/max-recall.json", "w") as outfile:
        json.dump(our_recall, outfile)
    with open("./exports/graph/max-f-measure.json", "w") as outfile:
        json.dump(our_f_measure, outfile)

    # our_name = 'weighted_average'
    # precision_map[our_name] = our_precision
    # recall_map[our_name] = our_recall
    # f_measure_map[our_name] = our_f_measure

    # # Precision graph
    # for key, value in precision_map.items():
    #     if key == our_name:
    #         plt.plot(x, value, label=key, alpha=0.9)
    #     else:
    #         plt.plot(x, value, label=key, alpha=0.4)
    # plt.legend(loc="upper right")
    # plt.axes().set_ylim([0, 1])
    # plt.ylabel('Score')
    # plt.xlabel('Number of matches')
    # plt.title("Precision")
    # plt.savefig('exports/matcher_metric_graph_precision.png')

    # Reset 
    plt.clf()

    # Recall
    # for key, value in recall_map.items():
    #     if key == our_name:
    #         plt.plot(x, value, label=key, alpha=0.9)
    #     else:
    #         plt.plot(x, value, label=key, alpha=0.4)
    # plt.legend(loc="lower right")
    # plt.axes().set_ylim([0, 1])
    # plt.ylabel('Score')
    # plt.xlabel('Number of matches')
    # plt.title("Recall")
    # plt.savefig('exports/matcher_metric_graph_recall.png')

    plt.clf()

    # F-measure
    # for key, value in f_measure_map.items():
    #     if key == our_name:
    #         plt.plot(x, value, label=key, alpha=0.9)
    #     else:
    #         plt.plot(x, value, label=key, alpha=0.4)
    # plt.legend(loc="upper right")
    # plt.axes().set_ylim([0, 1])
    # plt.ylabel('Score')
    # plt.xlabel('Number of matches')
    # plt.title("F-measure")
    # plt.savefig('exports/matcher_metric_graph_f_measure.png')
    return


def create_relation_metric_graph(relations, ground_truth):
    max_k = len(node_service.get_matches())
    x = []
    y_precision = []
    y_recall = []
    y_f_measure = []
    for k in range(0, max_k):
        results = evaluate_relations_at(relations, ground_truth, k)
        x.append(k)
        y_precision.append(results['precision'])
        y_recall.append(results['recall'])
        y_f_measure.append(results['f-measure'])

    
    plt.plot(x, y_precision, label='precision', alpha=0.4)
    plt.plot(x, y_recall, label='recall', alpha=0.4)
    plt.plot(x, y_f_measure, label='f-measure', alpha=0.4)
    plt.legend(loc="upper left")
    plt.ylabel('Score')
    plt.xlabel('Number of relations')
    plt.show()
    plt.savefig('exports/metric_graph.png')


def evaluate_relations_at(relations, ground_truth, k):
    relations = relations[:k]
    p = relation_precision(relations, ground_truth)
    r = relation_recall(relations, ground_truth)
    f = f_measure(p, r)
    return {'precision': p, 'recall': r, 'f-measure': f, 'k': k}

def perform_evaluation():
    print("PERFORMING EVALUATION")
    nodes = node_service.get_nodes()
    max = len(nodes)

    print("ITERATING THROUGH EVERY COLUMN")
    for i in tqdm(reversed(range(1, max))):
        clustering_service.clusterk(i)
        clusters = clustering_service.get_clusters()['clusters']
        ground_truth = get_ground_truth()
        evaluation_relations = evaluate_relations(clusters, ground_truth)
        evaluation_clusters = evaluate_clusters(clusters, ground_truth)
        save_evaluation(evaluation_relations)
        save_evaluation(evaluation_clusters)

def evaluate_relations(clusters, ground_truth):
    # Depends on ground_truth actually. If ground_truth does contain the single clusters, then the clusters should too
    clusters = list(filter(lambda cluster: len(cluster) > 1, clusters))

    p = precision(clusters, ground_truth)
    r = recall(clusters, ground_truth)
    f = f_measure(p, r)

    threshold = clustering_service.CLUSTERING_THRESHOLD

    return {'precision': p, 'recall': r, 'f-measure': f, 'threshold': threshold, 'type': 'Relations'}

def evaluate_clusters(clusters, ground_truth):
    # Depends on ground_truth actually. If ground_truth does contain the single clusters, then the clusters should too
    clusters = list(filter(lambda cluster: len(cluster) > 1, clusters))

    p = precision_cluster(clusters, ground_truth)
    r = recall_cluster(clusters, ground_truth)
    f = f_measure(p, r)

    threshold = clustering_service.CLUSTERING_THRESHOLD

    return {'precision': p, 'recall': r, 'f-measure': f, 'threshold': threshold, 'type': 'Clusters'}

# def evaluate_clusters(clusters, ground_truth):
#     # Depends on ground_truth actually. If ground_truth does contain the single clusters, then the clusters should too
#     clusters = list(filter(lambda cluster: len(cluster) > 1, clusters))

#     p = precision_cluster(clusters, ground_truth)
#     r = recall_cluster(clusters, ground_truth)
#     f = f_measure_cluster(clusters, ground_truth)

#     threshold = clustering_service.CLUSTERING_THRESHOLD

#     return {'precision': p, 'recall': r, 'f-measure': f, 'threshold': threshold}

# Brute force best amount of clusters for maximizing f-measure
def best_clusters(pivot):
    nodes = node_service.get_nodes()
    max = len(nodes)
    best = evaluate_cluster(pivot)
    save_evaluation(best)
    index = -1

    for i in range(pivot, max):
        value = evaluate_cluster(i)
        save_evaluation(value)
        if best['f-measure'] < value['f-measure']:
            best = value
            index = i


    return index

def evaluate_cluster(k):
    clustering_service.clusterk(k)
    clusters = clustering_service.get_clusters()['clusters']
    ground_truth = get_ground_truth()
    evaluation = evaluate_clusters(clusters, ground_truth)
    return evaluation


# Find percentage of false positives
def false_positives():
    truth = get_ground_truth_relations()
    all_matches = node_service.get_matches()
    all_matches = list(filter(lambda match: match["id"] not in truth, all_matches))
    false_matches = list(map(lambda match: match["scores"], all_matches))

    false_positives = {}
    for match in false_matches:
        for key, value in match.items():
            if key not in false_positives:
                false_positives[key] = []
            
            if value > 0:
                false_positives[key] = false_positives[key] + [value]

    averages =  {key: (sum(value)/max(len(value), 1)) for key, value in false_positives.items()}
    length =  {key: (len(value)) for key, value in false_positives.items()}

    total_false = len(false_matches)
    false_positives = {k: round(len(v)/total_false*100, 2) for k, v in false_positives.items()}

    return {"percentages": false_positives,
        "averages": averages,
        "length": length,
        "max_length": len(false_matches)}

def true_positives():
    truth = get_ground_truth_relations()
    all_matches = node_service.get_matches()
    all_matches = list(filter(lambda match: match["id"] in truth, all_matches))
    true_matches = list(map(lambda match: match["scores"], all_matches))

    true_positives = {}
    for match in true_matches:
        for key, value in match.items():
            if key not in true_positives:
                true_positives[key] = []
            
            if value > 0:
                true_positives[key] = true_positives[key] + [value]

    averages =  {key: (sum(value)/max(len(value), 1)) for key, value in true_positives.items()}
    length =  {key: (len(value)) for key, value in true_positives.items()}

    total_true = len(true_matches)
    true_positives = {k: round(len(v)/total_true*100, 2) for k, v in true_positives.items()}
    return {"percentages": true_positives,
        "averages": averages,
        "length": length,
        "max_length": len(true_matches)}

# Clusters and ground_truth should be in the form of [[cluster_1], [cluster_2], [cluster_3]]
def relation_precision(relations, ground_truth):                         # I
    relations = parse_relations(relations)
    if len(relations) == 0:
        return 0

    true_relations = parse_clusters(ground_truth)                   # R
    true_positives = relation_intersect(relations, true_relations)  # P
    
    return len(true_positives)/len(relations)

def relation_recall(relations, ground_truth):
    relations = parse_relations(relations)                          # I
    if len(relations) == 0:
        return 0

    true_relations = parse_clusters(ground_truth)                   # R
    true_positives = relation_intersect(relations, true_relations)  # P
    
    return len(true_positives)/len(true_relations)

# Clusters and ground_truth should be in the form of [[cluster_1], [cluster_2], [cluster_3]]
def precision(clusters, ground_truth):
    relations = parse_clusters(clusters)                            # I
    if len(relations) == 0:
        return 0

    true_relations = parse_clusters(ground_truth)                   # R
    true_positives = relation_intersect(relations, true_relations)  # P
    
    return len(true_positives)/len(relations)

def recall(clusters, ground_truth):
    relations = parse_clusters(clusters)                            # I
    true_relations = parse_clusters(ground_truth)                   # R
    if len(true_relations) == 0:
        return 0

    true_positives = relation_intersect(relations, true_relations)  # P

    return len(true_positives)/len(true_relations)

def f_measure(p, r):
    if p + r == 0:
        return 0
    
    return (2 * p * r)/(p + r)

def parse_relations(relations):
    tuples = []

    for relation in relations:
        relations = [relation['from'], relation['to']]
        tuples.append(relations)

    return tuples

def parse_clusters(clusters):
    tuples = []

    for cluster in clusters:
        relations = list(combinations(cluster, 2))
        tuples = tuples + relations

    return tuples

def relation_intersect(relations, truth):
    positive = []

    for relation in relations:
        for true_relation in truth:
            if is_equal_relation(relation, true_relation):
                positive.append(relation)
                break

    return positive

def is_equal_relation(relation, truth):
    return sorted(relation) == sorted(truth)

def precision_cluster(clusters, ground_truth):
    sum = 0

    if len(clusters) == 0:
        return 0

    for cluster in clusters:
        max_precision = 0
        for truth in ground_truth:
            max_precision = max(_single_cluster_precision(cluster, truth), max_precision)

        sum += max_precision

    return sum/len(clusters)

def recall_cluster(clusters, ground_truth):
    sum = 0

    if len(clusters) == 0:
        return 0

    for cluster in clusters:
        max_recall = 0
        for truth in ground_truth:
            max_recall = max(_single_cluster_recall(cluster, truth), max_recall)

        sum += max_recall

    return sum/len(ground_truth)

def f_measure_cluster(clusters, ground_truth):
    p = precision_cluster(clusters, ground_truth)
    r = recall_cluster(clusters, ground_truth)

    if (p + r) == 0:
        return 0

    return (2 * p * r)/(p + r)

def _single_cluster_precision(cluster, ground_truth):
    return len(_intersect(cluster, ground_truth)) / len(cluster)

def _single_cluster_recall(cluster, ground_truth):
    return len(_intersect(cluster, ground_truth)) / len(ground_truth)


def _intersect(list_1, list_2):
    intersection = [value for value in list_1 if value in list_2]  
    return intersection  

def save_evaluation(evaluation):
    result = [evaluation]
    old_results = get_evaluations()

    with open("./exports/evaluations.json", "w") as outfile:
        json.dump(old_results + result, outfile)

def save_ground_truth(cluster):
    new_truth = [cluster]
    truth = get_ground_truth()

    flat_truth = [item for sublist in truth for item in sublist]
    intersect = _intersect(flat_truth, cluster)

    if (len(intersect) > 0):
        return

    with open("./exports/ground_truth.json", "w") as outfile:
        json.dump(truth + new_truth, outfile)

def save_ground_truth_relation(relation):
    new_truth = [relation]
    truth = get_ground_truth_relations()

    flat_truth = [item for sublist in truth for item in sublist]
    intersect = _intersect(flat_truth, relation)

    # Can not save twice
    if (len(intersect) > 0):
        return

    with open("./exports/ground_truth_relations.json", "w") as outfile:
        json.dump(truth + new_truth, outfile)

def get_unlabeled_nodes():
    nodes = list(map(lambda node: node['id'], node_service.get_nodes()))

    truth = get_ground_truth()
    truth = [item for sublist in truth for item in sublist]

    return [item for item in nodes if item not in truth]

def get_evaluations():
    try:
        with open('./exports/evaluations.json') as json_file:
            result = json.load(json_file)
            return result
    except:
        return []

def get_ground_truth():
    try:
        with open('./exports/ground_truth.json') as json_file:
            result = json.load(json_file)
            return result
    except:
        return []

def get_ground_truth_relations():
    try:
        with open('./exports/ground_truth_relations.json') as json_file:
            result = json.load(json_file)
            return result
    except:
        return []

def flatten(list):
    return [item for sublist in list for item in sublist]

def reset_evaluations():
    if os.path.exists("./exports/evaluations.json"):
        os.remove("./exports/evaluations.json")
    else:
        print("The file does not exist")

def reset_ground_truth():
    if os.path.exists("./exports/ground_truth.json"):
        os.remove("./exports/ground_truth.json")
    else:
        print("The file does not exist")

def reset_ground_truth_relations():
    if os.path.exists("./exports/ground_truth_relations.json"):
        os.remove("./exports/ground_truth_relations.json")
    else:
        print("The file does not exist")