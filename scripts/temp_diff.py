import numpy as np
import csv
import os.path

# This script calculates one feature: The temporal difference between the publication of two papers.


# calculates the temporal distance between papers for training and testing set
def temporal_difference(info, index_train, index_test):

    # calculates the temporal distance between papers for training set
    if not os.path.isfile("./data/temp_diff_training.txt"):

        print("Temporal difference is calculated for the training data.")

        with open("./data/temp_diff_training.txt", 'w') as file:
            for i in range(0, len(index_train)):
                file.write(str(abs(int(info[index_train[i][0]][1]) - int(info[index_train[i][1]][1]))) + "\n")

    # calculates the temporal distance between papers for testing set
    if not os.path.isfile("./data/temp_diff_testing.txt"):

        print("Temporal difference is calculated for the testing data.")

        with open("./data/temp_diff_testing.txt", 'w') as file:
            for i in range(0, len(index_test)):
                file.write(str(abs(int(info[index_test[i][0]][1]) - int(info[index_test[i][1]][1]))) + "\n")
