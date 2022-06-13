from flask import Flask, Response, request, redirect, render_template
from utils import database
import os
import numpy
import pandas as pd
from services import node_service, match_service, clustering_service, evaluation_service

app = Flask('app')

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

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/test")
def test():
    match_service.adjust_weights_collab(1830, True)
    return "OK"

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
        df = pd.read_csv(f)

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
    for x in range(0, len(tables)):
        for y in range(x + 1, len(tables)):
            table1_path = os.path.join('./tables/{}'.format(DATA_SPACE), tables[x])
            table2_path = os.path.join('./tables/{}'.format(DATA_SPACE), tables[y])

            df1 = pd.read_csv(table1_path)
            df2 = pd.read_csv(table2_path)
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
    matching_edges = node_service.get_matches()

    return render_template("label_relations.html", relationships=matching_edges)

@app.route('/label-relations', methods = ['POST'])
def label_post():
    edges_to_label = request.form.getlist('relationID')
    correct = True if request.form.get('correct') == 'true' else False

    for relation in edges_to_label:
        relationId = int(relation)
        match_service.adjust_weights_collab(relationId, correct)

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

# export FLASK_ENV=development
app.run(debug=True)