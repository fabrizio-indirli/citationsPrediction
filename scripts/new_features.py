import numpy as np
import csv
import networkx as nx
from time import clock
import os.path

# This script calculates three features: resource allocation index, academic adar index and preferential attachment


# calculates resource allocation index for training and testing set
def resource_allocation_index(graph, train_set, test_set):
    if not os.path.isfile("./data/resource_allocation_training.csv"):
        print("Computing training resource-allocation index")

        t0 = clock()
        results = []

        for i in range(0, len(train_set)):
            preds = nx.resource_allocation_index(graph, [(train_set[i][0], train_set[i][1])])
            results.append([float(p) for u, v, p in preds])
            if i % 5000 == 0:
                print(i)

        print("Some elements of results:", results[0:5])

        with open("./data/resource_allocation_training.csv", 'w') as file:
            csv_out = csv.writer(file)
            for i in range(0, len(train_set)):
                csv_out.writerow(results[i])
                if i % 100000 == 0:
                    print(i)

        print(clock() - t0)

    if not os.path.isfile("./data/resource_allocation_testing.csv"):
        print("Computing testing resource-allocation index")
        t0 = clock()
        results = []

        for i in range(0, len(test_set)):
            preds = nx.resource_allocation_index(graph, [(test_set[i][0], test_set[i][1])])
            results.append([float(p) for u, v, p in preds])
            if i % 5000 == 0:
                print(i)

        print("Some elements of results:", results[0:5])

        with open("./data/resource_allocation_testing.csv", 'w') as file:
            csv_out = csv.writer(file)
            for i in range(0, len(test_set)):
                csv_out.writerow(results[i])
                if i % 100000 == 0:
                    print(i)

        print(clock() - t0)


# calculates academic adar index for training and testing set
def academic_adar_index(graph, train_set, test_set):
    if not os.path.isfile("./data/adamic_adar_index_training.csv"):
        print("Computing training adamic-adar index")
        t0 = clock()
        results = []

        for i in range(0, len(train_set)):
            preds = nx.adamic_adar_index(graph, [(train_set[i][0], train_set[i][1])])
            results.append([float(p) for u, v, p in preds])
            if i % 5000 == 0:
                print(i)

        print("Some elements of results:", results[0:5])

        with open("./data/adamic_adar_index_training.csv", 'w') as file:
            csv_out = csv.writer(file)
            for i in range(0, len(train_set)):
                csv_out.writerow(results[i])
                if i % 100000 == 0:
                    print(i)

        print(clock() - t0)

    if not os.path.isfile("./data/adamic_adar_index_testing.csv"):
        print("Computing testing resource-allocation index")
        t0 = clock()
        results = []

        for i in range(0, len(test_set)):
            preds = nx.adamic_adar_index(graph, [(test_set[i][0], test_set[i][1])])
            results.append([float(p) for u, v, p in preds])
            if i % 5000 == 0:
                print(i)

        print("Some elements of results:", results[0:5])

        with open("./data/adamic_adar_index_testing.csv", 'w') as file:
            csv_out = csv.writer(file)
            for i in range(0, len(test_set)):
                csv_out.writerow(results[i])
                if i % 100000 == 0:
                    print(i)

        print(clock() - t0)


# calculates preferential attachment for training and testing set
def preferential_attachment(graph, train_set, test_set):
    if not os.path.isfile("./data/preferential_attachment_training.csv"):

        t0 = clock()
        results = []

        for i in range(0, len(train_set)):
            preds = nx.preferential_attachment(graph, [(train_set[i][0], train_set[i][1])])
            results.append([float(p) for u, v, p in preds])
            if i % 5000 == 0:
                print(i)

        print("Some elements of results:", results[0:5])

        with open("./data/preferential_attachment_training.csv", 'w') as file:
            csv_out = csv.writer(file)
            for i in range(0, len(train_set)):
                csv_out.writerow(results[i])
                if i % 100000 == 0:
                    print(i)

        print(clock() - t0)

    if not os.path.isfile("./data/preferential_attachment_testing.csv"):

        t0 = clock()
        results = []

        for i in range(0, len(test_set)):
            preds = nx.preferential_attachment(graph, [(test_set[i][0], test_set[i][1])])
            results.append([float(p) for u, v, p in preds])
            if i % 5000 == 0:
                print(i)

        print("Some elements of results:", results[0:5])

        with open("./data/preferential_attachment_testing.csv", 'w') as file:
            csv_out = csv.writer(file)
            for i in range(0, len(test_set)):
                csv_out.writerow(results[i])
                if i % 100000 == 0:
                    print(i)

        print(clock() - t0)
