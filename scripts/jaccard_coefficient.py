import numpy as np
import igraph
import csv
import os.path


# writes the jaccard coefficients of the elements of "pairs" with respect to the graph defined by "nodes" and "edges"
# into a text file named jaccard_coefficients_"filename".txt
def jaccard_coefficients(nodes, edges, pairs, filename):

    if not os.path.isfile("./data/jaccard_coefficients_" + filename + ".txt"):

        print("Jaccard coefficients for " + filename + " data are calculated now.")

        # create graph
        graph = igraph.Graph()
        graph.add_vertices(nodes)
        graph.add_edges(edges)

        # create adjacent list
        AdjList = graph.get_adjlist()

        jac = np.zeros(len(pairs))

        for i in range(0, len(pairs)):
            s1 = set(AdjList[nodes.index(pairs[i][0])])
            s2 = set(AdjList[nodes.index(pairs[i][1])])
            if len(s1.union(
                    s2)) != 0:  # the jaccard coefficient of a pair of nodes both having zero neighbours is set to zero
                jac[i] = (len(s1.intersection(s2)) / len(s1.union(s2)))
            if i % 10000 == 0:
                print("Calculated jaccard_coeff of " + str(i) + " samples.")

        with open("./data/jaccard_coefficients_" + filename + ".txt", 'w') as file:
            for i in range(0, jac.shape[0]):
                file.write(str(jac[i]) + "\n")

        print("Maximal Jaccard coefficient: ", np.max(jac))

    return None
