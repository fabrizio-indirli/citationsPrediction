import numpy as np
import csv
import networkx as nx
from time import clock
import os.path

# This script calculates one feature: The number of paths of lengths 1 up to m (an integer >= 1) between the two nodes

# calculates the number of paths of lengths 1 up to m between the nodes of training and testing set
def number_paths_limited_length(graph, train_set, test_set, m):

    # training set
    if not os.path.isfile("./data/number_paths_training.txt"):
        print("Number of paths of length up to " + str(m) + " is calculated now for the training data.")
        t0 = clock()
        n = len(train_set)
        results = np.zeros((n, m))

        for i in range(0, n):
            v = np.zeros((1, m))
            for j in range(0, m):
                v[0, j] = len(list(nx.all_simple_paths(graph, train_set[i][0], train_set[i][1], cutoff=j+1)))
            results[i, 0] = v[0, 0]
            for j in range(1, m):
                results[i, j] = v[0, j] - v[0, j-1]
            if i % 50 == 0:
                print(i)

        print("Some rows of results:", results[0:5])

        with open("./data/number_paths_training.txt", 'w') as file:
            for i in range(0, results.shape[0]):
                output = ""
                for j in range(0, len(results[i])-1):
                    output += str(results[i, j])
                    output += " "
                output += str(results[i, -1])
                file.write(output + "\n")

        print(clock() - t0)

    # testing set
    if not os.path.isfile("./data/number_paths_testing.txt"):
        print("Number of paths of length up to " + str(m) + " is calculated now for the testing data.")
        t0 = clock()
        n = len(test_set)
        results = np.zeros((n, m))

        for i in range(0, n):
            v = np.zeros((1, m))
            for j in range(0, m):
                v[0, j] = len(list(nx.all_simple_paths(graph, test_set[i][0], test_set[i][1], cutoff=j+1)))
            results[i, 0] = v[0, 0]
            for j in range(1, m):
                results[i, j] = v[0, j] - v[0, j-1]
            if i % 50 == 0:
                print(i)

        print("Some rows of results:", results[0:5])

        with open("./data/number_paths_testing.txt", 'w') as file:
            for i in range(0, results.shape[0]):
                output = ""
                for j in range(0, len(results[i])-1):
                    output += str(results[i, j])
                    output += " "
                output += str(results[i, -1])
                file.write(output + "\n")

        print(clock() - t0)
