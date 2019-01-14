import csv
import networkx as nx
import os.path
import numpy as np

hubs = {}
authorities = {}

def computeHITS(fileName, node_info, training_set):

    IDs = [element[0] for element in node_info]

    global G
    G = nx.DiGraph()
    edges = [(element[0], element[1]) for element in training_set if element[2] == "1"]
    # nodes = [int(id) for id in IDs]
    nodes = IDs
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    hubs, authorities = nx.hits(G)
    with open(fileName, 'w') as f:
        csv_out = csv.writer(f)
        csv_out.writerow(["id", "hub", "authority"])
        for id in nodes:
            csv_out.writerow([id, hubs[id], authorities[id]])

    return hubs, authorities



def initNodesHITS(node_info, training_set):
    global hubs, authorities
    fileName = "./data/hits_infos.csv"
    if not os.path.isfile(fileName):
        print("File ", fileName, " not found. Computing HITS values for nodes...")
        computeHITS(fileName, node_info, training_set)
        nodesHits = np.loadtxt(fileName, delimiter=',', skiprows=1)
        for n in nodesHits:
            id = n[0]
            hubs[id] = float(n[1])
            authorities[id] = float(n[2])
        print("Finished computing HITS features for nodes.")
    else:
        print("File ", fileName, " found: loading HITS nodes values from there")
        nodesHits = np.loadtxt(fileName, delimiter=',', skiprows=1)
        for n in nodesHits:
            id = n[0]
            hubs[id] = float(n[1])
            authorities[id] = float(n[2])

def analyze_pairs(pairsSet):
    sourcesHubs = []
    targetsAuthorities = []
    totalScores = []

    for i in range(len(pairsSet)):
        source = pairsSet[i][0] # an ID of edges
        target = pairsSet[i][1] # an ID of edges

        source_hub = float(hubs[int(source)])
        target_authority = float(authorities[int(target)])
        total_score = source_hub + target_authority

        sourcesHubs.append(source_hub)
        targetsAuthorities.append(target_authority)
        totalScores.append(total_score)

    return sourcesHubs, targetsAuthorities, totalScores

def write_HITS_features(pairsSet, fileName):
    if not os.path.isfile(fileName):
        print("File ", fileName, " not found, HITS features for pairs will be computed now")
        sourcesHubs, targetsAuthorities, totalScores = analyze_pairs(pairsSet)
        lst = list(zip(sourcesHubs, targetsAuthorities, totalScores))

        with open(fileName, 'w') as f:
            csv_out = csv.writer(f)
            csv_out.writerow(["source hub", "target authority", "sum"])
            for row in lst:
                csv_out.writerow(row)
        print("Written file ", fileName)

