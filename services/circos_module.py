# Pandas is gonna be used to read the csv file stored on the web:
import pandas as pd
import holoviews as hv
from holoviews import opts, dim
from bokeh.sampledata.les_mis import data
from bokeh.plotting import show, output_file
from . import matrix, match_service, node_service, clustering_service, evaluation_service

def export_graph():
    print('YES')
    hv.extension('bokeh')
    hv.output(size=200)

    # [[], [], [], []] .....
    clusters = clustering_service.get_clusters()['clusters']
    truth = evaluation_service.get_ground_truth()

    nodes = []
    lookup = {}
    group_lookup = {}
    # print(data['nodes'])
    groupCount = 1
    nodeCount = 0
    for cluster in truth:
        for column in cluster:
            nodes.append({'name': 'TRUTH_' + column, 'group': groupCount})
            lookup[column] = nodeCount
            group_lookup[column] = groupCount
            nodeCount += 1
        groupCount += 1

    links = []
    for cluster in clusters:
        start = nodeCount
        for column in cluster:
            nodes.append({'name': column, 'group': groupCount})

            if column in lookup:
                links.append({'source': lookup[column], 'target': nodeCount, 'value': 10, 'groupCount':group_lookup[column]})
            else:
                links.append({'source': start, 'target': nodeCount, 'value': 1, 'groupCount': -1})
            nodeCount += 1
        groupCount += 1
    
    name_links = list(map(lambda x: x['source'], links))
    for column, node_number in lookup.items():
        if node_number not in name_links:
            prev = group_lookup[group_lookup.Keys.ToList()[node_number - 1]]
            next =  group_lookup[group_lookup.Keys.ToList()[node_number + 1]]
            
            if prev == group_lookup[column]:
                links.append({'source': node_number, 'target': node_number - 1, 'value': 1, 'groupCount': group_lookup[column]})
            else:
                links.append({'source': node_number, 'target': node_number + 1, 'value': 1, 'groupCount': group_lookup[column]})
        else:
            print('k')
    
    links = pd.DataFrame(links)
    nodes = hv.Dataset(pd.DataFrame(nodes), 'index')

    chord = hv.Chord((links, nodes))
    chord.opts(opts.Chord(labels='name', 
                          cmap='Category20', 
                          edge_cmap='Category20', 
                          edge_color=dim('groupCount').str(), 
                          node_color=dim('group').str(), 
                          edge_line_width=4,
                          edge_alpha=0.3))
    hv.save(chord, 'diagram.html')
    print("SHOULD BE GENERATED")

    # print(data['links'])
    # print(data['nodes'])

    # links = pd.DataFrame(data['links'])
    # nodes = hv.Dataset(pd.DataFrame(data['nodes']), 'index')
    # print(pd.DataFrame(data['nodes']))
    # print(links)
    # print(nodes)