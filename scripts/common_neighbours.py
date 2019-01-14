import numpy as np
import igraph
import csv
import os.path


# writes the number of common neighbours of the elements of "pairs" with respect to the graph defined by
# "nodes" and "edges" into a text file named common_neighbours_"filename".txt
def common_neighbours(nodes, edges, pairs, filename):

    if not os.path.isfile("./data/common_neighbours_"+filename+".txt"):

        print("Number of common neighbours for " + filename + " data is calculated now.")

        # create graph
        graph = igraph.Graph()
        graph.add_vertices(nodes)
        graph.add_edges(edges)

        # create adjacent list
        AdjList = graph.get_adjlist()

        neigh = np.zeros(len(pairs))

        for i in range(0, len(pairs)):
            s1 = set(AdjList[nodes.index(pairs[i][0])])
            s2 = set(AdjList[nodes.index(pairs[i][1])])
            if len(s1.union(s2)) != 0:
                neigh[i] = (len(s1.intersection(s2)))
            if i % 10000 == 0:
                print("Calculated " + str(i) + " samples.")

        with open("./data/common_neighbours_"+filename+".txt", 'w') as file:
            for i in range(0, len(pairs)):
                file.write(str(neigh[i]) + "\n")

        print("Maximal number of common neighbours: ", int(np.max(neigh)))

    return None
