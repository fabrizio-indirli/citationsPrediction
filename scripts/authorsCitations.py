import csv
import numpy as np
from authorsTools import processArticleAuths, cleanAuthors, getAllAuthors
import re
import os.path

authToId = {}
authToRow = {}
journalToColumn = {}

def cleanJournal(str):
    ns = re.sub("('|``)","",str)
    ns = re.sub('(""|")',"",ns)
    return ns
    
def buildAuthsCitsTables(training_set, node_info, ind):
    IDs = [element[0] for element in node_info]

    # for authors-to-authors citations
    authors = getAllAuthors(node_info)
    iToSourceAuths = {}
    iToTargetAuths = {}
    i = 0

    for a in authors:
        authToId[a] = i
        i += 1
    authsCitsTable = np.zeros((len(authors), len(authors)))
    ###########################################################

    # for authors-to-journals citations
    journalsList = (np.array(node_info))[:, 4]
    cleanedJournals = []

    for j in journalsList:
        cleanedJournals.append(cleanJournal(j))
    journals = set(cleanedJournals)
    c = 0

    for j in journals:
        journalToColumn[j] = c
        c += 1
    i = 0

    for a in authors:
        authToRow[a] = i
        i += 1

    journalsCitsTable = np.zeros((len(authors), len(journals)))
    #########################################################
    for i in range(len(training_set)):
        source = training_set[i][0]  # an ID of edges
        target = training_set[i][1]  # an ID of edges

        ## find an index maching to the source ID
        index_source = ind[i][0]
        index_target = ind[i][1]

        ## find info corresponding to the ID
        source_info = node_info[index_source]
        target_info = node_info[index_target]

        source_auth = processArticleAuths(source_info[3])
        iToSourceAuths[i] = source_auth
        target_auth = processArticleAuths(target_info[3])
        iToTargetAuths[i] = target_auth
        target_journal = target_info[4]
        target_journal = cleanJournal(target_journal)

        for sa in source_auth:
            if not sa: continue
            # for authors-to-journals citations
            saRow = authToRow[sa]
            tjCol = journalToColumn[target_journal]
            journalsCitsTable[saRow, tjCol] += 1
            # for authors-to-authors citations
            saID = authToId[sa]
            for ta in target_auth:
                if not ta: continue
                # for authors-to-authors citations
                taID = authToId[ta]
                authsCitsTable[saID, taID] += 1

        if (i % 2000 == 0): print("Building authors citations tables: analyzed ", i, " training samples.")
    print("Authors citations tables built successfully.")
    return authsCitsTable, journalsCitsTable


def authsToAuthsCits(source_info, target_info, authsCitsTable):
    source_auth = processArticleAuths(source_info[3])
    target_auth = processArticleAuths(target_info[3])
    tot = 0
    for sa in source_auth:
        if not sa: continue
        saID = authToId[sa]
        for ta in target_auth:
            if not ta: continue
            taID = authToId[ta]
            tot += authsCitsTable[saID, taID]
    return tot

def authsToJournalsCits(source_info, target_info, journalsCitsTable):
    source_auth = processArticleAuths(source_info[3])
    target_journal = cleanJournal(target_info[4])
    tot = 0
    for sa in source_auth:
        if not sa: continue
        saRow = authToRow[sa]
        tjCol = journalToColumn[target_journal]
        tot += journalsCitsTable[saRow, tjCol]
    return tot

def computeAuthorsToAuthorsCits(fileName, pairsSet, ind, authsCitsTable, node_info):
    if not os.path.isfile(fileName):
        print("File ", fileName, " not found. Computing it...")
        authsToAuths = []
        IDs = [element[0] for element in node_info]

        for i in range(len(pairsSet)):
            source = pairsSet[i][0]  # an ID of edges
            target = pairsSet[i][1]  # an ID of edges

            ## find an index maching to the source ID
            index_source = ind[i][0]
            index_target = ind[i][1]

            ## find info corresponding to the ID
            source_info = node_info[index_source]
            target_info = node_info[index_target]

            authsToAuths.append(authsToAuthsCits(source_info, target_info, authsCitsTable))
            if (i % 2000 == 0):
                print("Computed authors-to-authors citations of ", i, " samples")

        with open(fileName, 'w') as f:
            csv_out = csv.writer(f)
            for row in authsToAuths:
                csv_out.writerow([row])
        print("File ", fileName, " written")


def computeAuthorsToJournalsCits(fileName, pairsSet, ind, journalsCitsTable, node_info):
    if not os.path.isfile(fileName):
        print("File ", fileName, " not found. Computing it...")
        authsToJournals = []
        IDs = [element[0] for element in node_info]

        for i in range(len(pairsSet)):
            source = pairsSet[i][0]  # an ID of edges
            target = pairsSet[i][1]  # an ID of edges

            ## find an index maching to the source ID
            index_source = ind[i][0]
            index_target = ind[i][1]

            ## find info corresponding to the ID
            source_info = node_info[index_source]
            target_info = node_info[index_target]

            authsToJournals.append(authsToJournalsCits(source_info, target_info, journalsCitsTable))
            if (i % 2000 == 0):
                print("Computed authors-to-journals citations of ", i, " samples")

        with open(fileName, 'w') as f:
            csv_out = csv.writer(f)
            for row in authsToJournals:
                csv_out.writerow([row])
        print("File ", fileName, " written")




