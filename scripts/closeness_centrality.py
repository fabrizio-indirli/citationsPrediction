import numpy as np
import csv
import networkx as nx
from time import clock
import os.path

# This script calculates two features: Sum and difference of closeness centrality.


# calculates closeness centrality of the nodes
def closeness_centrality_nodes(graph):
    if not os.path.isfile("./data/closeness_centrality_nodes.txt"):

        print("Closeness centralities of the nodes are calculated now.")

        t0 = clock()

        closeness_centralities = []

        graph_nodes = list(graph.nodes)

        for i in range(0, len(graph_nodes)):
            closeness_centralities.append(nx.closeness_centrality(graph, u=graph_nodes[i], distance=None))
            if i % 1000 == 0:
                print("Calculated " + str(i) + " samples.")

        with open("closeness_centrality_nodes.txt", 'w') as file:
            for i in range(0, len(closeness_centralities)):
                file.write(str(closeness_centralities[i]) + "\n")

        print("runtime: ", clock() - t0)   # for 100 samples: 43.43 seconds  for all 27'770 samples: a few hours (on my computer)


# calculates sum and difference of closeness centrality for training and testing data
def sum_and_difference_closeness_centrality(centralities, index_train, index_test):

    if not os.path.isfile("./data/sum_closeness_centrality_training.txt"):
        print("Sums of closeness centralities for the training data are calculated now.")
        with open("./data/sum_closeness_centrality_training.txt", 'w') as f1:
            for i in range(0, len(index_train)):
                f1.write(str(float(centralities[int(index_train[i][0])][0]) + float(centralities[int(index_train[i][1])][0])) + "\n")

    if not os.path.isfile("./data/difference_closeness_centrality_training.txt"):
        print("Differences of closeness centralities for the training data are calculated now.")
        with open("./data/difference_closeness_centrality_training.txt", 'w') as f2:
            for i in range(0, len(index_train)):
                f2.write(str(abs(float(centralities[int(index_train[i][0])][0]) - float(centralities[int(index_train[i][1])][0]))) + "\n")

    if not os.path.isfile("./data/sum_closeness_centrality_testing.txt"):
        print("Sums of closeness centralities for the testing data are calculated now.")
        with open("./data/sum_closeness_centrality_testing.txt", 'w') as f3:
            for i in range(0, len(index_test)):
                f3.write(str(float(centralities[int(index_test[i][0])][0]) + float(centralities[int(index_test[i][1])][0])) + "\n")

    if not os.path.isfile("./data/difference_closeness_centrality_testing.txt"):
        print("Differences of closeness centralities for the testing data are calculated now.")
        with open("./data/difference_closeness_centrality_testing.txt", 'w') as f4:
            for i in range(0, len(index_test)):
                f4.write(str(abs(float(centralities[int(index_test[i][0])][0]) - float(centralities[int(index_test[i][1])][0]))) + "\n")
