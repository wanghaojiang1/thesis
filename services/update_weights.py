from . import matching_techniques, node_service, matrix
import os
import json

LEARNING_RATE = 0.1

# Optimism: Start with thinking all experts are the best
# Punish all wrong decisions made by them
def multiplicative_weight_update(edge_id, correct=False):
    print('DOING IT')
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
    
    print(expert_weights)
    _save_expert_weights(expert_weights)
    print('saved')
    
    # TODO: label the edge such that we don't label everything more than once
    node_service.label_edge(edge_id)

def _decrease_weight(weight, score, truth):
    
    return max((1.0 - (LEARNING_RATE * abs(truth - score))) * weight, 0.0)


# Second reinforcement learning approach: collaborative contribution
def reinforcement_learning(edge_id, correct=False):
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

    print(expert_weights)
    _save_expert_weights(expert_weights)
    print('saved')

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

def _adjust_weight(weight, contribution):
    return max(0.0, ((contribution * LEARNING_RATE) + weight))


def linear_programming(edge_id, correct=False):
    scores = matrix.get_matrix()
    expert_weights = matching_techniques.VARIANTS

    scores = scores[str(edge_id)].head(len(expert_weights)).to_dict()

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



def _save_expert_weights(expert_weights):
    with open("./exports/expert_weights.json", "w") as outfile:
        json.dump(expert_weights, outfile)

def get_raw_expert_weights():
    if (not _expert_weights_exists()):
        return {}

    with open('./exports/expert_weights.json') as json_file:
        result = json.load(json_file)
        return result

def _expert_weights_exists():
    return os.path.exists('./exports/expert_weights.json')