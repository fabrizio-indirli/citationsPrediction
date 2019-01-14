import numpy as np
import csv
import os.path
import networkx as nx
import sys
from time import clock

# You can use this script to calculate all features and write them to .txt or .csv files.
# If the files with the computed features already exist, they will not be calculated another time.

sys.path.insert(0, './scripts/')
import jaccard_coefficient as jc
import closeness_centrality as cc
import new_features as nf
import number_paths as nu
import overlap_title as oti
import same_journal as sj
import temp_diff as td
import common_neighbours as cn
import HITS as hits
import TFIDFsims as ts
import authorsTools as at
import authorsCitations as ac

# import papers
with open("node_information.csv", "r") as file:
    reader = csv.reader(file)
    node_info = list(reader)

print("Shape of node_information: ", np.asarray(node_info).shape)

print("One element of node_information: ", node_info[0])

IDs = [element[0] for element in node_info]

print("Some elements of IDs: ", IDs[0:5])

# import training set
with open("training_set.txt", "r") as file:
    reader = csv.reader(file)
    training_set = list(reader)

training_set = [element[0].split(" ") for element in training_set]

print("Some elements of training_set: ", training_set[0:5])

# import testing set
with open("testing_set.txt", "r") as file:
    reader = csv.reader(file)
    testing_set = list(reader)

testing_set = [element[0].split(" ") for element in testing_set]

print("Some elements of testing_set: ", testing_set[0:5])

# calculate indexes of nodes of training_set in IDs
if not os.path.isfile("./data/index_training.txt"):

    with open("./data/index_training.txt", 'w') as file:
        for i in range(0, len(training_set)):
            file.write(str(IDs.index(training_set[i][0])) + " " + str(IDs.index(training_set[i][1])) + "\n")

with open("./data/index_training.txt", "r") as file:
    reader = csv.reader(file)
    index_training = list(reader)

index_training = [element[0].split(" ") for element in index_training]
index_training = [[int(element[0]), int(element[1])] for element in index_training]

print("Some elements of index_training:", index_training[0:5])
print("Length of index_training: ", len(index_training))

# calculate indexes of nodes of testing_set in IDs
if not os.path.isfile("./data/index_testing.txt"):
    print("Computing indexes...")
    with open("./data/index_testing.txt", 'w') as file:
        for i in range(0, len(testing_set)):
            file.write(str(IDs.index(testing_set[i][0])) + " " + str(IDs.index(testing_set[i][1])) + "\n")

with open("./data/index_testing.txt", "r") as file:
    reader = csv.reader(file)
    index_testing = list(reader)

index_testing = [element[0].split(" ") for element in index_testing]
index_testing = [[int(element[0]), int(element[1])] for element in index_testing]

print("Some elements of index_testing:", index_testing[0:5])
print("Length of index_testing: ", len(index_testing))

# define the graph where the nodes are the papers and the edges represent citations between the papers
G = nx.Graph()

# add a list of nodes:
G.add_nodes_from(IDs)

print("Some nodes of the graph: ")
print(list(G.nodes())[0:5])

# add a list of edges
edges = [(element[0], element[1]) for element in training_set if element[2] == "1"]
G.add_edges_from(edges)

print("Some edges of the graph: ")
print(list(G.edges())[0:5])


# compute features

# abstract features:

# abstracts TFIDF cosine similarities
ts.abstracts_sims(training_set, "./data/training_sims.csv")
ts.abstracts_sims(testing_set, "./data/testing_sims.csv")


# author features

# number of common authors
at.computeCommonAuthors(training_set, index_training, "./data/commonAuths_training.txt", node_info)
at.computeCommonAuthors(testing_set, index_testing, "./data/commonAuths_testing.txt", node_info)

# authors-to-authors and authors-to-journals citations
if (not os.path.isfile("./data/tot_auth_cits_training.csv")
        or not os.path.isfile("./data/tot_auth_cits_testing.csv")
        or not os.path.isfile("./data/auths_cits_journal_training.csv")
        or not os.path.isfile("./data/auths_cits_journal_testing.csv")):
    authsCitsTable, journalsCitsTable = ac.buildAuthsCitsTables(training_set, node_info, index_training)
    ac.computeAuthorsToAuthorsCits("./data/tot_auth_cits_training.csv", training_set, index_training, authsCitsTable, node_info)
    ac.computeAuthorsToAuthorsCits("./data/tot_auth_cits_testing.csv", testing_set, index_testing, authsCitsTable, node_info)
    ac.computeAuthorsToJournalsCits("./data/auths_cits_journal_training.csv", training_set, index_training, journalsCitsTable, node_info)
    ac.computeAuthorsToJournalsCits("./data/auths_cits_journal_testing.csv", testing_set, index_testing, journalsCitsTable, node_info)


# graph features

# sum and difference in closeness centrality:
cc.closeness_centrality_nodes(G)

# import closeness centralities of the nodes
with open("./data/closeness_centrality_nodes.txt", "r") as file:
    reader = csv.reader(file)
    closeness_centralities = list(reader)

closeness_centralities = [element[0].split(" ") for element in closeness_centralities]

print("Some elements of closeness_centralities:", closeness_centralities[0:5])

cc.sum_and_difference_closeness_centrality(closeness_centralities, index_training, index_testing)

# jaccard coefficient:
jc.jaccard_coefficients(IDs, edges, training_set, "training")
jc.jaccard_coefficients(IDs, edges, testing_set, "testing")

# resource allocation
nf.resource_allocation_index(G, training_set, testing_set)

# academic adar index
nf.academic_adar_index(G, training_set, testing_set)

# preferential attachment
nf.preferential_attachment(G, training_set, testing_set)

# number of paths of length 1 up to 3
nu.number_paths_limited_length(G, training_set, testing_set, 3)

# number of common neighbours
cn.common_neighbours(IDs, edges, training_set, "training")
cn.common_neighbours(IDs, edges, testing_set, "testing")

# compute HITS features
hits.initNodesHITS(node_info, training_set)
hits.write_HITS_features(training_set, "./data/hitsFeatures_training.csv")
hits.write_HITS_features(testing_set, "./data/hitsFeatures_testing.csv")


# journal features

# same journal
sj.same_journal(node_info, index_training, index_testing)


# publication year features

# temporal difference
td.temporal_difference(node_info, index_training, index_testing)


# title features

# number of common words in titles
oti.overlap_title(node_info, index_training, index_testing)


print("Terminated building features.")
