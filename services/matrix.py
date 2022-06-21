import pandas as pd
from . import matching_techniques, match_service, node_service

MATRIX_LOCATION = './exports/matrix.csv'
COMBINE_STRATEGY = 'MAX'

def set_aggregation_strategy(strategy):
    global COMBINE_STRATEGY
    COMBINE_STRATEGY = strategy


def initialize_matrix():
    matching_edges = node_service.get_matches()
    matching_experts = list(map(lambda variant: variant['type'], matching_techniques.VARIANTS))

    # Matrix:     EDGES x EXPERTS
    data = list(map(lambda edge: { edge['id']: list(edge['scores'].values()) }, matching_edges))
    data = {k:v for d in data for k,v in d.items()}

    matrix = pd.DataFrame(data, index=matching_experts)
    matrix.to_csv(MATRIX_LOCATION)

# Disclaimer: function assumes that the expert_weights and the scores in the matix are in the same order
# TODO: Come up with a better function of combining those signals into one score
def calculate_score():
    matrix = pd.read_csv(MATRIX_LOCATION, index_col=0)
    matching_edges = node_service.get_matches()
    expert_weights = match_service.get_expert_weights()

    # Check if score row already exists
    if 'score' in matrix.index:
        matrix.drop(['score'], inplace = True)

    score_row = {}
    for edge in matching_edges:
        data = matrix[str(edge['id'])].head(len(expert_weights))

        score = 0
        if COMBINE_STRATEGY == 'MAX':
            score = max_score(data)
        elif COMBINE_STRATEGY == 'WEIGHTED_AVERAGE':
            score = weighted_score(data)

        score_row[str(edge['id'])] = score
    
    score_frame = pd.DataFrame(score_row, index=['score'])
    matrix = matrix.append(score_frame)
    matrix.to_csv(MATRIX_LOCATION)

    return score_row

def max_score(scores):
    score = 0

    for index in range(0, len(scores)):
            score = max(scores.iloc[index], score)

    return score

def weighted_score(scores):
    expert_weights = match_service.get_expert_weights()
    score = 0

    for index in range(0, len(scores)):
            score += expert_weights[index] * scores.iloc[index]

    return score/sum(expert_weights)


def get_scores():
    matrix = pd.read_csv(MATRIX_LOCATION, index_col=0)
    scores = matrix.loc['score']
    return scores