import re
import numpy as np
import os.path
import csv

def cleanAuthors(lst):
    # cleans a list of strings
    nl = []
    for str in lst:
        ns = re.sub('\([^)]*\)','',str)
        ns = re.sub('\([^\)]*$', '', ns)
        ns = re.sub('^(\s+)','',ns)
        ns = re.sub('( $)', '', ns)
        nl.append(ns)
    return nl

def cleanAuthor(s):
    # cleans a single string
    ns = re.sub('\([^)]*\)', '', s)
    ns = re.sub('\([^\)]*$', '', ns)
    ns = re.sub('^(\s+)', '', ns)
    ns = re.sub("(\\~|\\')", '', ns)
    ns = re.sub('\\"', '', ns)
    return ns


def processArticleAuths(str):
    # takes as input a string of one article's authors
    # returns the list of authors, cleaned from special characters
    a = cleanAuthor(str)
    lst = a.split(",")
    lst = cleanAuthors(lst)
    return lst


def getCommonAuthorsNum(source_info, target_info):
    # returns the number of common authors of the 2 articles
    source_auths = processArticleAuths(source_info[3])
    target_auths = processArticleAuths(target_info[3])
    return len(set(source_auths).intersection(set(target_auths)))

def getAllAuthors(node_info):
    articlesAuthors = np.array(node_info)
    articlesAuthors = articlesAuthors[:, 3]

    splittedAuthors = []
    for el in articlesAuthors:
        splittedAuthors = splittedAuthors + processArticleAuths(el)

    splittedAuthors = [el for el in splittedAuthors if el]  # removes empty strings
    return set(splittedAuthors)

def computeCommonAuthors(pairsSet, ind, fileName, node_info):
    IDs = [element[0] for element in node_info]
    if not os.path.isfile(fileName):
        print("File ", fileName, " not found, common_authors will be computed now")
        res = []

        for i in range(len(pairsSet)):
            source = pairsSet[i][0]  # an ID of edges
            target = pairsSet[i][1]  # an ID of edges

            ## find an index maching to the source and target ID
            index_source = ind[i][0]
            index_target = ind[i][1]

            source_info = node_info[index_source]
            target_info = node_info[index_target]

            res.append(getCommonAuthorsNum(source_info, target_info))

        with open(fileName, 'w') as f:
            csv_out = csv.writer(f)
            for row in res:
                csv_out.writerow([row])
        print("Written file ", fileName)

