import pandas as pd
from . import matching_techniques, match_service, node_service, configuration
from sklearn import preprocessing
import numpy as np
import os
from tqdm import tqdm

MATRIX_LOCATION = './exports/matrix.csv'

def initialize_matrix():
    matching_edges = node_service.get_matches()
    matching_experts = list(map(lambda variant: variant['type'], matching_techniques.VARIANTS))

    if configuration.NORMALIZE:
        print("NORMALIZING")
        for expert in matching_experts:
            corresponding = np.array(list(map(lambda x: x['scores'][expert], matching_edges)))

            xmin = min(corresponding) 
            xmax = max(corresponding)
            for i, x in enumerate(corresponding):
                matching_edges[i]['scores'][expert] = (x-xmin) / (xmax-xmin)

    # Matrix:     EDGES x EXPERTS
    data = list(map(lambda edge: { edge['id']: list(edge['scores'].values()) }, matching_edges))
    data = {k:v for d in data for k,v in d.items()}

    matrix = pd.DataFrame(data, index=matching_experts)
    matrix.to_csv(MATRIX_LOCATION)

# Disclaimer: function assumes that the expert_weights and the scores in the matix are in the same order
# TODO: Come up with a better function of combining those signals into one score
def calculate_score():
    print("CALCULATING SCORE")
    print(configuration.COMBINE_STRATEGY)
    matrix = pd.read_csv(MATRIX_LOCATION, index_col=0)
    matching_edges = node_service.get_matches()
    expert_weights = match_service.get_expert_weights()

    # Check if score row already exists
    if 'score' in matrix.index:
        matrix.drop(['score'], inplace = True)

    score_row = {}
    print("CALCULATING SCORES")
    for edge in tqdm(matching_edges):
        data = matrix[str(edge['id'])].head(len(expert_weights))

        score = 0
        if configuration.COMBINE_STRATEGY == configuration.STRATEGIES[1]:
            print("MAX SCORE")
            score = max_score(data)
        elif configuration.COMBINE_STRATEGY == configuration.STRATEGIES[0]:
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

def get_matrix():
    matrix = pd.read_csv(MATRIX_LOCATION, index_col=0)
    return matrix

def reset():
    if os.path.exists(MATRIX_LOCATION):
        os.remove(MATRIX_LOCATION)
    else:
        print("The file: {} does not exist".format(MATRIX_LOCATION))