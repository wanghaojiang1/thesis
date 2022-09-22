from . import matching_techniques, node_service, matrix, configuration
from tqdm import tqdm
import os
import numpy as np
import pandas as pd
import json
import operator

LEARNING_RATE = 0.1   # Should be <= 0.5
WEIGHT_INIT_VALUE = 1.0

def match_with_all_techniques(df1, df2):
    result = {}

    if (not _expert_weights_is_aligned()):
        _initialize_expert_weights()

    # Match with all defined matching variants
    for matching_tuple in matching_techniques.VARIANTS:
        matching_result = matching_tuple['matcher'](df1, df2, matching_tuple['function'], matching_tuple['arguments'])
        partial_result = _filter_on_threshold(matching_result, matching_tuple['threshold'])

        result[matching_tuple['type']] = partial_result

    return result

def get_expert_weights():
    raw_weights = get_raw_expert_weights()
    return list(raw_weights.values())

def get_raw_expert_weights():
    if (not _expert_weights_exists()):
        return {}

    with open('./exports/expert_weights.json') as json_file:
        result = json.load(json_file)
        return result

# Optimism: Start with thinking all experts are the best
# Punish all wrong decisions made by them
def adjust_weights(edge_id, correct=False):
    edge = node_service.get_match(edge_id)
    expert_weights = get_raw_expert_weights()

    if (not edge):
        return

    for key, value in edge['scores'].items():
        if (key not in expert_weights.keys()):
            continue
        truth = 1 if correct else 0

        # MAYBE: punish experts based on how similar they think it is
        # DONE: the value that the expert gives is considered in the readjustment of the value
        new_weight = _decrease_weight(expert_weights[key], value, truth) if (bool(value > 0) ^ correct) else expert_weights[key]
        expert_weights.update({key: new_weight})
    
    _save_expert_weights(expert_weights)
    
    # TODO: label the edge such that we don't label everything more than once
    node_service.label_edge(edge_id)

# Second reinforcement learning approach: collaborative contribution
def adjust_weights_collab(edge_id, correct=False):
    edge = node_service.get_match(edge_id)
    expert_weights = get_raw_expert_weights()

    if (not edge):
        return
    
    # Calculate weighted score:
    weighted_average = 0
    total_weight = 0
    for key, value in edge['scores'].items():
        weight = expert_weights[key]
        score = value

        weighted_average += weight * score
        total_weight += weight
    weighted_average = weighted_average/total_weight

    # This is ideal case
    truth = 1.0 if correct else 0.0
    
    # Tries to calculate the contribution: score with matcher - score without matcher
    # The score is an indicator if the score would improve to what extend
    # This indicator is used to tweak the weights based on whether it is true or false
    contribution = {}
    for key, value in edge['scores'].items():
        contribution[key] = calculate_contribution(edge['scores'], key, weighted_average)

        if truth == 0:
            contribution[key] = -contribution[key]
        contribution[key] = contribution[key]

        new_weight = _adjust_weight(expert_weights[key], contribution[key])

        expert_weights.update({key: new_weight})

    _save_expert_weights(expert_weights)

    node_service.label_edge(edge_id)


# W_i from paper
def calculate_contribution(nodes, target, score):
    expert_weights = get_raw_expert_weights()

    # With target
    with_target = score

    # Without target
    total_weight = 0
    weighted_average = 0
    for key, value in nodes.items():
        if key == target:
            continue

        score = value
        weight = expert_weights[key]

        weighted_average += weight * score
        total_weight += weight

    weighted_average = weighted_average/total_weight

    # With - Without
    return with_target - weighted_average

def add_truth(edge_id, correct=False):

    scores = matrix.get_matrix()
    expert_weights = matching_techniques.VARIANTS

    scores = scores[str(edge_id)].head(len(expert_weights)).to_dict()

    # edge = node_service.get_match(edge_id)

    result = {
        'edge_id': edge_id,
        'correct': correct,
        'scores': scores
    }

    _save_labelled_edge(result)
    node_service.label_edge(edge_id)

def _save_labelled_edge(edge):
    result = [edge]
    old_results = _get_labelled_edges()

    with open("./exports/truth.json", "w") as outfile:
        json.dump(old_results + result, outfile)

def _get_labelled_edges():
    if (not os.path.exists('./exports/truth.json')):
        return []

    with open('./exports/truth.json') as json_file:
        result = json.load(json_file)
        return result

def _get_collective_labelled_edges():
    if (not os.path.exists('./exports/collective_truth.json')):
        return []

    with open('./exports/collective_truth.json') as json_file:
        result = json.load(json_file)
        return result

def get_ordered_matches():
    matches = node_service.get_matches()
    scores = matrix.get_scores()
    matches = list(map(lambda node: {'id': node['id'], 'from': node['from'], 'to': node['to'], 'score': scores[str(node['id'])]}, matches))

    matches.sort(key=operator.itemgetter('score'), reverse=True)
    return matches

def _adjust_weight(weight, contribution):
    return min(max(0.0, ((contribution * LEARNING_RATE) + weight)), 1.0)

def reset_expert_weights():
    raw_expert_weights = get_raw_expert_weights()
    new_expert_weights = {}
    # new_expert_weights = {k: 1.0 for k, v in raw_expert_weights.items()}
    for matcher in matching_techniques.VARIANTS:
        new_expert_weights[matcher["type"]] = 1.0

    _save_expert_weights(new_expert_weights)

# TODO: come up with better way of adjusting the weights
def _decrease_weight(weight, score, truth):
    return max((1.0 - (LEARNING_RATE * (truth - score))) * weight, 0.0)

# Filter results based on thresholds
def _filter_on_threshold(result_list, threshold = 0):
	result = {}
	for key in result_list.keys():
		if result_list[key] >= threshold:
			result[key] = result_list[key]

	return result

# Use when new experts are added
def _initialize_expert_weights():
    matching_tuples = list(map(lambda variant: {variant['type']: WEIGHT_INIT_VALUE}, matching_techniques.VARIANTS))
    expert_weights = {k: v for d in matching_tuples for k, v in d.items()}

    # Initialise all weights
    if (not _expert_weights_exists):
        _save_expert_weights(expert_weights)
        return

    # Initialise new expert weights only
    #   Prevents losing weight values when adding new variants
    raw_expert_weights = get_raw_expert_weights()
    expert_weights.update(raw_expert_weights)

    _save_expert_weights(expert_weights)

def _expert_weights_exists():
    return os.path.exists('./exports/expert_weights.json')

def _expert_weights_is_aligned():
    variant_types = list(map(lambda variant: variant['type'], matching_techniques.VARIANTS))
    expert_weights = list(get_raw_expert_weights().keys())

    return expert_weights == variant_types

def _save_expert_weights(expert_weights):
    with open("./exports/expert_weights.json", "w") as outfile:
        json.dump(expert_weights, outfile)