from . import matching_techniques, node_service, matrix, update_weights
from utils import valentine
from valentine.algorithms import Coma, Cupid, DistributionBased, SimilarityFlooding
import json

# Necessary for Cupid
import nltk
nltk.download('omw-1.4')

# DATA SOURCE

import os

DATA_SPACES = os.listdir('./tables')
DATA_SPACE = ''

# AGGREGATION
STRATEGIES = ['WEIGHTED_AVERAGE', 'MAX']
COMBINE_STRATEGY = STRATEGIES[0]
NORMALIZE = True

MODULES = ['Multiplicative weight update', 'Reinforcement learning', 'Linear programming', 'Normalized linear programming']
THRESHOLDS = [0, 0.5]

# Training Module
TRAINING_MODULE = MODULES[1]

# Including scores
THRESHOLD = THRESHOLDS[0]

# Matching function
VALENTINE = valentine.match_on_function

# All included matchers
VARIANTS = [
	{
		'matcher': VALENTINE,
		'type': 'coma_schema',
		'function': Coma,
		'arguments': {
			'strategy': "COMA_OPT"
			# 'strategy': "COMA_OPT_INST"
		},
		'threshold': 0
	},
    {
		'matcher': VALENTINE,
		'type': 'coma_instance',
		'function': Coma,
		'arguments': {
			# 'strategy': "COMA_OPT"
			'strategy': "COMA_OPT_INST"
		},
		'threshold': 0
	},
	{
		'matcher': VALENTINE,
		'type': 'cupid',
		'function': Cupid,
		'arguments': {},
		'threshold': 0
	},
	{
		'matcher': VALENTINE,
		'type': 'distribution_based',
		'function': DistributionBased,
		'arguments': {},
		'threshold': 0
	},
	{
		'matcher': VALENTINE,
		'type': 'similarity_flooding',
		'function': SimilarityFlooding,
		'arguments': {},
		'threshold': 0
	}]

def set_data_space(space):
    global DATA_SPACE
    DATA_SPACE = DATA_SPACES[space]

def set_matchers(matchers):
    global VARIANTS
    parsed  = []

    for matcher in matchers:
        parsed.append(VARIANTS[matcher])

    matching_techniques.set_variants(parsed)

def set_threshold(threshold):
    global THRESHOLD
    THRESHOLD = threshold

def set_training_module(module):
    global TRAINING_MODULE
    if (module == 3):
        # Machine Learning
        TRAINING_MODULE = -1
    else:
        TRAINING_MODULE = MODULES[module]

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

def get_configuration():

	if os.path.exists("./exports/configuration.json"):
		import_configurations()

	configuration = {}

	configuration['method'] = 'Machine Learning' if TRAINING_MODULE == -1 else TRAINING_MODULE
	configuration['cutoff'] = THRESHOLD

	matchers = list(map(lambda matcher: matcher['type'], matching_techniques.VARIANTS))
	configuration['matchers'] = matchers
	configuration['dataspace'] = DATA_SPACE

	return configuration

def export_configurations():
	result = get_configuration()

	with open("./exports/configuration.json", "w") as outfile:
		json.dump(result, outfile)

def import_configurations():
	global TRAINING_MODULE
	global THRESHOLD
	global DATA_SPACE

	with open('./exports/configuration.json') as json_file:
		result = json.load(json_file)

		TRAINING_MODULE = result['method']
		DATA_SPACE = result['dataspace']
		THRESHOLD = result['cutoff']
		
		matchers = result['matchers']
		
		variants = []
		for matcher in matchers:
			for variant in VARIANTS:
				if matcher == variant['type']:
					variants.append(variant)

		matching_techniques.set_variants(variants)



