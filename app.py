from flask import Flask, Response, request, redirect, render_template
from utils import database
import os
import numpy
import operator
import sys
import json
import pandas as pd
from services import node_service, match_service, clustering_service, evaluation_service, matrix, linear_programming_module, circos_module, configuration

app = Flask('app')

# ACTIAVTE ENVIRONMENT: source ./newvenv/bin/activate

DATA_SPACE = "TPC-H"

@app.after_request
def add_header(response):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    response.headers['X-UA-Compatible'] = 'IE=Edge,chrome=1'
    response.headers['Cache-Control'] = 'public, max-age=0'
    return response


import subprocess
@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/test")
def test():
    # support_vector_machine_module.evaluate()
    # circos_module.export_graph()
    # print(node_service.get_matches())
    # node_service.get_matches()
    configuration.update_weight(12145, True)
    return "<p>Hello, World!</p>"


@app.route("/get-matcher-performance")
def get_matcher_performance():
    # relations = match_service.get_ordered_matches()
    evaluation_service.create_matchers_metric_graphs()
 
    return 'OK'

@app.route("/get-best-performance-per-matcher")
def get_best_performance_per_matcher():
    evaluation_service.create_matchers_best_metric_bar_chart(0.48, 0.23, 0.32)
    return 'OK'

@app.route("/get-weights/<t>")
def get_weights(t):
    threshold = int(t)/100.0
    linear_programming_module.solve(threshold)
    return "DONE LINEAR PROGRAMMING WITH: threshold=" + str(threshold)

@app.route("/backup")
def backup():
    subprocess.run('pg_dump -h localhost -U root -d thesis -Fc > /mnt/e/Educational/Thesis/Code/thesis/exports/backups/dump.sql', shell=True)
    return "OK"

@app.route("/restore")
def restore():
    database.dropTables()
    subprocess.run('pg_restore -h localhost -U root -d thesis -vcC --clean < /mnt/e/Educational/Thesis/Code/thesis/exports/backups/dump.sql', shell=True)
    return "Restored"

@app.route("/get-matches")
def get_matches():
    return render_template("matches.html", nodes=match_service.get_ordered_matches())

@app.route("/metric-graph")
def metrics_graph():
    relations = match_service.get_ordered_matches()
    ground_truth = evaluation_service.get_ground_truth()
    evaluation_service.create_relation_metric_graph(relations, ground_truth)
 
    return 'OK'

@app.route("/metrics-at/<k>")
def metrics_at(k):
    relations = match_service.get_ordered_matches()
    ground_truth = evaluation_service.get_ground_truth()
    result = evaluation_service.evaluate_relations_at(relations, ground_truth, int(k))
 
    return json.dumps(result)

@app.route("/set-aggregation/<strategy>")
def set_aggregation_strategy(strategy):
    matrix.set_aggregation_strategy(strategy)
    
    return f"OK, new aggregation strategy: {matrix.COMBINE_STRATEGY}"

@app.route("/set-normalization/<strategy>")
def set_normalization(strategy):
    matrix.set_normalization(strategy == 'True')
    
    return f"OK, new normalization strategy: {matrix.NORMALIZE}"

@app.route("/purge")
def purge():
    node_service.delete_nodes()
    node_service.delete_matches()
    node_service.delete_edges()

    return "<p>Succesfully purged!</p>"

@app.route("/use/<dataspace>")
def use(dataspace):
    global DATA_SPACE
    DATA_SPACE = dataspace
    return f"OK, new data space: {DATA_SPACE}"

@app.route("/nodes")
def get_nodes():
    return f"The nodes in PostgreSQL are: {node_service.get_nodes()}"

@app.route("/siblings/<table>/<column>")
def get_siblings(table, column):
    column_id = table + '/' + column
    return f"The nodes in PostgreSQL are: {node_service.get_siblings(column_id)}"

@app.route("/match/<edge_id>")
def get_match(edge_id):
    return f"The nodes in PostgreSQL are: {node_service.get_match(edge_id)}"

@app.route("/initialise")
def initialise_tables():
    done = []
    for filename in os.listdir('./tables/{}'.format(DATA_SPACE)):
        f = os.path.join('./tables/{}'.format(DATA_SPACE), filename)
        df = pd.read_csv(f, nrows=10)

        for column in df.columns:
            column_id = filename + "/" + column
            source = filename

            node_service.insert_node(column_id, source)
        
        database.commit()
        done.append(filename)

    return f"Done for: {done}"

@app.route("/profile")
def profile():
    tables = node_service.get_tables()
    print("TABLES: ", tables)
    for x in range(0, len(tables)):
        table1_path = os.path.join('./tables/{}'.format(DATA_SPACE), tables[x])
        print('reading x')
        df1 = pd.read_csv(table1_path, engine='python', on_bad_lines='skip')
        for y in range(x + 1, len(tables)):
            table2_path = os.path.join('./tables/{}'.format(DATA_SPACE), tables[y])
            print('reading y')
            df2 = pd.read_csv(table2_path, engine='python', on_bad_lines='skip')
            print(" ---- MATCHING:" + tables[x] + " " + tables[y])
            matches = match_service.match_with_all_techniques(df1, df2)
            distinct_matches = set([y for x in list(matches.values()) for y in x])

            for key in distinct_matches:
                ((_, col_from), (_, col_to)) = key
                column_id_from = tables[x] + "/" + col_from
                column_id_to = tables[y] + "/" + col_to

                edge_id = node_service.insert_edge(column_id_from, column_id_to)
                for matching_type in matches.keys():
                    value =  matches[matching_type][key] if key in matches[matching_type].keys() else 0
                    node_service.insert_match(edge_id, matching_type, value)

            database.commit()
    return "Done profiling"

@app.route('/label-relations')
def label():
    matching_edges = node_service.get_unlabelled_matches()

    return render_template("label_relations.html", relationships=matching_edges)

@app.route('/label-relations', methods = ['POST'])
def label_post():
    edges_to_label = request.form.getlist('relationID')
    correct = True if request.form.get('correct') == 'true' else False

    for relation in edges_to_label:
        relationId = int(relation)
        # match_service.adjust_weights_collab(relationId, correct)
        # match_service.add_truth(relationId, correct)
        configuration.update_weight(relationId, correct)

    matching_edges = node_service.get_matches()

    return render_template("label_relations.html", relationships=matching_edges, success=True)

@app.route('/cluster')
def cluster():
    expert_weights = match_service.get_raw_expert_weights()
    return render_template("hierarchical_clustering.html", weights=expert_weights)

@app.route('/cluster', methods = ['POST'])
def cluster_post():
    clustering_service.cluster()
    expert_weights = match_service.get_raw_expert_weights()
    evaluation_service.reset_evaluations()
    
    # clusters = clustering_service.get_clusters()['clusters']
    # ground_truth = evaluation_service.get_ground_truth()
    # evaluation = evaluation_service.evaluate_clusters(clusters, ground_truth)
    # evaluation_service.save_evaluation(evaluation)

    return render_template("hierarchical_clustering.html", weights=expert_weights, success=True)

@app.route('/results')
def results():
    clusters = clustering_service.get_clusters()
    evaluations = evaluation_service.get_evaluations()

    return render_template("results.html", setting=clusters['setting'], clusters=clusters['clusters'], threshold=clustering_service.CLUSTERING_THRESHOLD, evaluations=evaluations)

@app.route('/best-cluster/<number>')
def best_clusters(number):
    clusters = evaluation_service.best_clusters(int(number))
    return json.dumps(clusters)

@app.route('/perform-evaluation')
def perform_evaluation():
    evaluation_service.perform_evaluation()

    return "OK DONE EVALUATING"


@app.route('/cluster-k/<number>')
def cluster_k(number):

    clustering_service.clusterk(int(number))
    expert_weights = match_service.get_raw_expert_weights()
    # traversal.evaluation_service.reset_evaluations()
    
    clusters = clustering_service.get_clusters()['clusters']
    ground_truth = evaluation_service.get_ground_truth()
    evaluation = evaluation_service.evaluate_clusters(clusters, ground_truth)
    evaluation_service.save_evaluation(evaluation)

    return redirect("/results")

@app.route('/update-threshold', methods = ['POST'])
def update_threshold():
    threshold = float(request.form.get('threshold'))

    if threshold:
        clustering_service.update_cluster(threshold)

        clusters = clustering_service.get_clusters()['clusters']
        ground_truth = evaluation_service.get_ground_truth()
        evaluation = evaluation_service.evaluate_clusters(clusters, ground_truth)
        evaluation_service.save_evaluation(evaluation)

    return redirect("/results")

@app.route('/label-truth')
def label_truth():
    nodes = evaluation_service.get_unlabeled_nodes()
    clusters = evaluation_service.get_ground_truth()

    return render_template("submit_ground_truth.html", nodes=nodes, clusters=clusters)

@app.route('/label-truth', methods=['POST'])
def label_truth_post():
    cluster = request.form.getlist('columnID')
    evaluation_service.save_ground_truth(cluster)

    nodes = evaluation_service.get_unlabeled_nodes()
    clusters = evaluation_service.get_ground_truth()
    return render_template("submit_ground_truth.html", nodes=nodes, clusters=clusters, success=True)

from itertools import combinations
@app.route('/label-relations-from-ground')
def label_relations_from_ground():
    clusters = evaluation_service.get_ground_truth()

    edges = node_service.get_unlabelled_matches()
    for cluster in clusters:
        combination = list(combinations(cluster, 2))
        for (fromTable, toTable) in combination:
            edge = next((item for item in edges if (item['from'] == fromTable or item['from'] == toTable) and (item['to'] == fromTable or item['to'] == toTable)), None)

            if edge:
                # match_service.add_truth(edge['id'], True)
                configuration.update_weight(edge['id'], True)

    return "OK"

# export FLASK_ENV=development
app.run(debug=True)