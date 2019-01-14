from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import csv
import os.path



def abstracts_sims(pairsSet, fileName):
    if not os.path.isfile(fileName):

        # Import node information
        with open("./node_information.csv", "r") as f:
            reader = csv.reader(f)
            node_info = list(reader)
        IDs = [element[0] for element in node_info]

        print("Cosine similarities for the abstracts are calculated.")

        corpus = [element[5] for element in node_info]
        vectorizer = TfidfVectorizer(stop_words="english")
        # each row is a node in the order of node_info
        features_TFIDF = vectorizer.fit_transform(corpus)  # compute TFIDF vector of each paper's abstract
        abstractsSimilarities = cosine_similarity(features_TFIDF)

        sims = []

        for i in range(len(pairsSet)):
            source = pairsSet[i][0]  # an ID of edges
            target = pairsSet[i][1]  # an ID of edges

            ## find an index maching to the source and target ID
            index_source = IDs.index(source)
            index_target = IDs.index(target)

            sims.append(abstractsSimilarities[index_source, index_target])

        with open(fileName, 'w') as f:
            csv_out = csv.writer(f)
            for row in sims:
                csv_out.writerow([row])
