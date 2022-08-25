from . import matching_techniques, node_service, matrix, update_weights

# AGGREGATION
STRATEGIES = ['WEIGHTED_AVERAGE', 'MAX']
COMBINE_STRATEGY = STRATEGIES[0]
NORMALIZE = False

MODULES = ['Multiplicative weight update', 'Reinforcement learning', 'Linear programming', 'Normalized linear programming']
THRESHOLDS = [0, 0.5]

# Training Module
TRAINING_MODULE = MODULES[1]

# Including scores
THRESHOLD = THRESHOLDS[0]



# SETTING CONFIGURATIONS
def set_aggregation_strategy(strategy):
    global COMBINE_STRATEGY
    COMBINE_STRATEGY = strategy

def set_normalization(normalize):
    global NORMALIZE
    NORMALIZE = normalize


def update_weight(edge_id, correct=False):
    if TRAINING_MODULE == MODULES[0]:
        update_weights.multiplicative_weight_update(edge_id, correct)
    elif TRAINING_MODULE == MODULES[1]:
        update_weights.reinforcement_learning(edge_id, correct)
    elif TRAINING_MODULE == MODULES[2] or TRAINING_MODULE == MODULES[3]:
        update_weights.linear_programming(edge_id, correct)
