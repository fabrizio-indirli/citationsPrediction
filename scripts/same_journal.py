import numpy as np
import csv
import os.path

# This script caculates one feature: same_journal
# For a pair of papers this feature is equal one if and only if the two papers werde published in the same journal and
# is equal zero if and only if the two papers were not published in the same journal.


# checks whether papers were published in the same journal for training and testing set
def same_journal(info, index_train, index_test):

    # check whether papers were published in the same journal for training_set
    if not os.path.isfile("./data/same_journal_training.txt"):

        print("Same journal is now calculated for the training data.")

        with open("./data/same_journal_training.txt", 'w') as file:
            for i in range(0, len(index_train)):
                file.write(str(int(info[index_train[i][0]][4].lower() == info[index_train[i][1]][4].lower())) + "\n")

    # check whether papers were published in the same journal for testing_set
    if not os.path.isfile("./data/same_journal_testing.txt"):

        print("Same journal is now calculated for the testing data.")

        with open("./data/same_journal_testing.txt", 'w') as file:
            for i in range(0, len(index_test)):
                file.write(str(int(info[index_test[i][0]][4].lower() == info[index_test[i][1]][4].lower())) + "\n")

