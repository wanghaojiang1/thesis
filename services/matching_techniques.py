from utils import valentine
from valentine.algorithms import Coma, Cupid, DistributionBased, SimilarityFlooding

# Necessary for Cupid
import nltk
nltk.download('omw-1.4')

# Matching function
VALENTINE = valentine.match_on_function

# All included matchers
VARIANTS = [{
		'matcher': VALENTINE,
		'type': 'coma',
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