import numpy as np
import csv
import nltk
import os.path

# This script calculates one feature: The number of common words in the titles of two papers.

# calculates the number of overlapping words in the titles for training and testing set
def overlap_title(info, index_train, index_test):

    # calculate number of overlapping words in title for training_set
    if not os.path.isfile("./data/overlap_title_training.txt"):

        print("Overlap title is now calculated for the training data.")

        nltk.download('punkt')  # for tokenization
        nltk.download('stopwords')
        stpwds = set(nltk.corpus.stopwords.words("english"))
        stemmer = nltk.stem.PorterStemmer()

        overlap_title_training = []

        for i in range(0, len(index_train)):
            # convert to lowercase and tokenize
            # ex: ['corrections', 'to', 'the', 'abelian', 'born-infeld', 'action', 'arising', 'from']
            source_title = info[index_train[i][0]][2].lower().split(" ")
            target_title = info[index_train[i][1]][2].lower().split(" ")

            # remove stopwords
            source_title = [token for token in source_title if token not in stpwds]
            source_title = [stemmer.stem(token) for token in source_title]

            target_title = [token for token in target_title if token not in stpwds]
            target_title = [stemmer.stem(token) for token in target_title]

            overlap_title_training.append(len(set(source_title).intersection(set(target_title))))

        with open("./data/overlap_title_training.txt", 'w') as file:
            for i in range(0, len(index_train)):
                file.write(str(overlap_title_training[i]) + "\n")

    # calculate number of overlapping words in title for testing_set
    if not os.path.isfile("./data/overlap_title_testing.txt"):

        print("Overlap title is now calculated for the testing data.")

        nltk.download('punkt')  # for tokenization
        nltk.download('stopwords')
        stpwds = set(nltk.corpus.stopwords.words("english"))
        stemmer = nltk.stem.PorterStemmer()

        overlap_title_testing = []

        for i in range(0, len(index_test)):
            # convert to lowercase and tokenize
            # ex: ['corrections', 'to', 'the', 'abelian', 'born-infeld', 'action', 'arising', 'from']
            source_title = info[index_test[i][0]][2].lower().split(" ")
            target_title = info[index_test[i][1]][2].lower().split(" ")

            # remove stopwords
            source_title = [token for token in source_title if token not in stpwds]
            source_title = [stemmer.stem(token) for token in source_title]

            target_title = [token for token in target_title if token not in stpwds]
            target_title = [stemmer.stem(token) for token in target_title]

            overlap_title_testing.append(len(set(source_title).intersection(set(target_title))))

        with open("./data/overlap_title_testing.txt", 'w') as file:
            for i in range(0, len(index_test)):
                file.write(str(overlap_title_testing[i]) + "\n")

