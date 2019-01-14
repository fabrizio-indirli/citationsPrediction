import random
import numpy as np
import csv
import os.path
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from matplotlib import pyplot
import keras as K
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import ClassifierChain
from sklearn import neighbors
from sklearn.tree import DecisionTreeClassifier
from time import clock
from sklearn import svm

# set mode = 0 to try out a certain configuration of features and model parameters and set mode = 1 to create a submission
mode = 1

# decide which features dealing with the abstracts you would like to use
use_cos_abstract = 1        # cosine similarities of abstracts

# decide which features dealing with the authors you would like to use
use_nca = 1                 # number of common authors
use_ncta = 1                # number of citations of author to author
use_nctj = 1                # number of citations of author to journal

# decide which features dealing with the graph you would like to use
use_ncn = 1                 # number of common neighbours
use_np = 1                  # number of paths of length 3
use_jac = 1                 # jaccard coefficients
use_ra = 1                  # resource allocation
use_aa = 1                  # adamic adar index
use_pa = 1                  # preferential attachment
use_scc = 1                 # sum of closeness centralities
use_dcc = 1                 # difference of closeness centralities
use_hit1 = 1                # source hub
use_hit2 = 1                # target authority

# decide which features dealing with the journals you would like to use
use_sj = 1                  # same journal

# decide which features dealing with the titles you would like to use
use_ncw_title = 1           # number of common words of titles

# decide which features dealing with the publication years you would like to use
use_td = 1                  # temporal difference


# define lists which will be filled with the features chosen above
features_training_list = []
features_testing_list = []

# import papers
with open("node_information.csv", "r") as file:
    reader = csv.reader(file)
    node_info = list(reader)

print("One element of node_information: ", node_info[0])
print("Shape of node_information: ", np.asarray(node_info).shape)

IDs = [element[0] for element in node_info]

print("Some elements of IDs: ", IDs[0:5])

# import training set
with open("training_set.txt", "r") as file:
    reader = csv.reader(file)
    training_set = list(reader)

training_set = [element[0].split(" ") for element in training_set]

print("Some elements of training_set: ", training_set[0:5])
print("Length of training_set: ", len(training_set))

# import testing set
with open("testing_set.txt", "r") as file:
    reader = csv.reader(file)
    testing_set = list(reader)

testing_set = [element[0].split(" ") for element in testing_set]

print("Some elements of testing_set: ", testing_set[0:5])
print("Length of testing_set: ", len(testing_set))

# import features

# abstract features:

# import cosine similarities for abstracts
if use_cos_abstract == 1:
    with open("./data/training_sims.csv", 'r') as file:
        reader = csv.reader(file)
        cos_sim_abstract_train = list(reader)

    cos_sim_abstract_train = [float(element[0]) for element in cos_sim_abstract_train]

    print("Some elements of cos_sim_abstract_train: ", cos_sim_abstract_train[0:5])
    print(len(cos_sim_abstract_train))
    features_training_list.append(cos_sim_abstract_train)

    with open("./data/testing_sims.csv", 'r') as file:
        reader = csv.reader(file)
        cos_sim_abstract_test = list(reader)

    cos_sim_abstract_test = [float(element[0]) for element in cos_sim_abstract_test]

    print("Some elements of cos_sim_abstract_test: ", cos_sim_abstract_test[0:5])
    print(len(cos_sim_abstract_test))
    features_testing_list.append(cos_sim_abstract_test)

# author features:

# import number of common authors
if use_nca == 1:
    with open("./data/commonAuths_training.txt", "r") as f:
        reader = csv.reader(f)
        comm_auth_train = list(reader)

    comm_auth_train = [float(element[0]) for element in comm_auth_train]

    print("Some elements of comm_auth_train: ", comm_auth_train[0:5])
    print(len(comm_auth_train))
    features_training_list.append(comm_auth_train)

    with open("./data/commonAuths_testing.txt", "r") as f:
        reader = csv.reader(f)
        comm_auth_test = list(reader)

    comm_auth_test = [float(element[0]) for element in comm_auth_test]

    print("Some elements of comm_auth_test: ", comm_auth_test[0:5])
    print(len(comm_auth_test))
    features_testing_list.append(comm_auth_test)

# import number of citations of author to author
if use_ncta == 1:
    with open("./data/tot_auth_cits_training.csv", 'r') as file:
        reader = csv.reader(file)
        tot_auth_cits_train = list(reader)

    tot_auth_cits_train = [float(element[0]) for element in tot_auth_cits_train]

    print("Some elements of tot_auth_cits_train: ", tot_auth_cits_train[0:5])
    print(len(tot_auth_cits_train))
    features_training_list.append(tot_auth_cits_train)

    with open("./data/tot_auth_cits_testing.csv", 'r') as file:
        reader = csv.reader(file)
        tot_auth_cits_test = list(reader)

    tot_auth_cits_test = [float(element[0]) for element in tot_auth_cits_test]

    print("Some elements of tot_auth_cits_test: ", tot_auth_cits_test[0:5])
    print(len(tot_auth_cits_test))
    features_testing_list.append(tot_auth_cits_test)

# import number of citations of author to journal
if use_nctj == 1:
    with open("./data/auths_cits_journal_training.csv", "r") as f:
        reader = csv.reader(f)
        auths_cits_journal_train = list(reader)

    auths_cits_journal_train = [float(element[0]) for element in auths_cits_journal_train]

    print("Some elements of auths_cits_journal_train: ", auths_cits_journal_train[0:5])
    print(len(auths_cits_journal_train))
    features_training_list.append(auths_cits_journal_train)

    with open("./data/auths_cits_journal_testing.csv", "r") as f:
        reader = csv.reader(f)
        auths_cits_journal_test = list(reader)

    auths_cits_journal_test = [float(element[0]) for element in auths_cits_journal_test]

    print("Some elements of auths_cits_journal_test: ", auths_cits_journal_test[0:5])
    print(len(auths_cits_journal_test))
    features_testing_list.append(auths_cits_journal_test)


# graph features:

# import number of common neighbours (implemented in common_neighbours.py)
if use_ncn == 1:
    with open("./data/common_neighbours_training.txt", "r") as f:
        reader = csv.reader(f)
        com_neigh_train = list(reader)

    com_neigh_train = [float(element[0]) for element in com_neigh_train]

    print("Some elements of com_neigh_train: ", com_neigh_train[0:5])
    print(len(com_neigh_train))
    features_training_list.append(com_neigh_train)

    with open("./data/common_neighbours_testing.txt", "r") as f:
        reader = csv.reader(f)
        com_neigh_test = list(reader)

    com_neigh_test = [float(element[0]) for element in com_neigh_test]

    print("Some elements of com_neigh_test: ", com_neigh_test[0:5])
    print(len(com_neigh_test))
    features_testing_list.append(com_neigh_test)

# import number of paths of length 3 between nodes (implemented in number_paths.py)
if use_np == 1:
    with open("./data/number_paths_training.txt", "r") as f:
        reader = csv.reader(f)
        number_paths_train = list(reader)

    number_paths_train = [element[0].split(" ") for element in number_paths_train]
    number_paths_train = [float(element[2]) for element in number_paths_train]

    print("Some elements of number_paths_train: ", number_paths_train[0:5])
    print(len(number_paths_train))
    features_training_list.append(number_paths_train)

    with open("./data/number_paths_testing.txt", "r") as f:
        reader = csv.reader(f)
        number_paths_test = list(reader)

    number_paths_test = [element[0].split(" ") for element in number_paths_test]
    number_paths_test = [float(element[2]) for element in number_paths_test]

    print("Some elements of number_paths_test: ", number_paths_test[0:5])
    print(len(number_paths_test))
    features_testing_list.append(number_paths_test)

# import jaccard coefficients (implemented in jaccard_coefficient.py)
if use_jac == 1:
    with open("./data/jaccard_coefficients_training.txt", "r") as file:
        reader = csv.reader(file)
        jac_coeffs_train = list(reader)

    jac_coeffs_train = [float(element[0]) for element in jac_coeffs_train]

    print("Some elements of jac_coeffs_train: ", jac_coeffs_train[0:5])
    print(len(jac_coeffs_train))
    features_training_list.append(jac_coeffs_train)

    with open("./data/jaccard_coefficients_testing.txt", "r") as file:
        reader = csv.reader(file)
        jac_coeffs_test = list(reader)

    jac_coeffs_test = [float(element[0]) for element in jac_coeffs_test]

    print("Some elements of jac_coeffs_test: ", jac_coeffs_test[0:5])
    print(len(jac_coeffs_test))
    features_testing_list.append(jac_coeffs_test)

# import resource allocation (implemented in new_features.py)
if use_ra == 1:
    with open("./data/resource_allocation_training.csv", 'r') as file:
        reader = csv.reader(file)
        resource_allocation_train = list(reader)

    resource_allocation_train = [float(element[0]) for element in resource_allocation_train]

    print("Some elements of resource_allocation_train: ", resource_allocation_train[0:5])
    print(len(resource_allocation_train))
    features_training_list.append(resource_allocation_train)

    with open("./data/resource_allocation_testing.csv", 'r') as file:
        reader = csv.reader(file)
        resource_allocation_test = list(reader)

    resource_allocation_test = [float(element[0]) for element in resource_allocation_test]

    print("Some elements of resource_allocation_test: ", resource_allocation_test[0:5])
    print(len(resource_allocation_test))
    features_testing_list.append(resource_allocation_test)

# import adamic adar index (implemented in new_features.py)
if use_aa == 1:
    with open("./data/adamic_adar_index_training.csv", 'r') as file:
        reader = csv.reader(file)
        adamic_adar_index_train = list(reader)

    adamic_adar_index_train = [float(element[0]) for element in adamic_adar_index_train]

    print("Some elements of adamic_adar_index_train: ", adamic_adar_index_train[0:5])
    print(len(adamic_adar_index_train))
    features_training_list.append(adamic_adar_index_train)

    with open("./data/adamic_adar_index_testing.csv", 'r') as file:
        reader = csv.reader(file)
        adamic_adar_index_test = list(reader)

    adamic_adar_index_test = [float(element[0]) for element in adamic_adar_index_test]

    print("Some elements of adamic_adar_index_test: ", adamic_adar_index_test[0:5])
    print(len(adamic_adar_index_test))
    features_testing_list.append(adamic_adar_index_test)

# import preferential attachment (implemented in new_features.py)
if use_pa == 1:
    with open("./data/preferential_attachment_training.csv", 'r') as file:
        reader = csv.reader(file)
        preferential_attachment_train = list(reader)

    preferential_attachment_train = [float(element[0]) for element in preferential_attachment_train]

    print("Some elements of preferential_attachment_train: ", preferential_attachment_train[0:5])
    print(len(preferential_attachment_train))
    features_training_list.append(preferential_attachment_train)

    with open("./data/preferential_attachment_testing.csv", 'r') as file:
        reader = csv.reader(file)
        preferential_attachment_test = list(reader)

    preferential_attachment_test = [float(element[0]) for element in preferential_attachment_test]

    print("Some elements of preferential_attachment_test: ", preferential_attachment_test[0:5])
    print(len(preferential_attachment_test))
    features_testing_list.append(preferential_attachment_test)

# import sum of closeness centralities (implemented in closeness_centrality.py)
if use_scc == 1:
    with open("./data/sum_closeness_centrality_training.txt", 'r') as file:
        reader = csv.reader(file)
        close_cent_sum_train = list(reader)

    close_cent_sum_train = [float(element[0]) for element in close_cent_sum_train]

    print("Some elements of close_cent_sums_train: ", close_cent_sum_train[0:5])
    print(len(close_cent_sum_train))
    features_training_list.append(close_cent_sum_train)

    with open("./data/sum_closeness_centrality_testing.txt", 'r') as file:
        reader = csv.reader(file)
        close_cent_sum_test = list(reader)

    close_cent_sum_test = [float(element[0]) for element in close_cent_sum_test]

    print("Some elements of close_cent_sums_test: ", close_cent_sum_test[0:5])
    print(len(close_cent_sum_test))
    features_testing_list.append(close_cent_sum_test)

# import difference of closeness centralities (implemented in closeness_centrality.py)
if use_dcc == 1:
    with open("./data/difference_closeness_centrality_training.txt", 'r') as file:
        reader = csv.reader(file)
        close_cent_dif_train = list(reader)

    close_cent_dif_train = [float(element[0]) for element in close_cent_dif_train]

    print("Some elements of close_cent_dif_train: ", close_cent_dif_train[0:5])
    print(len(close_cent_dif_train))
    features_training_list.append(close_cent_dif_train)

    with open("./data/difference_closeness_centrality_testing.txt", 'r') as file:
        reader = csv.reader(file)
        close_cent_dif_test = list(reader)

    close_cent_dif_test = [float(element[0]) for element in close_cent_dif_test]

    print("Some elements of close_cent_dif_test: ", close_cent_dif_test[0:5])
    print(len(close_cent_dif_test))
    features_testing_list.append(close_cent_dif_test)

# import source hub
if use_hit1 == 1:
    with open("./data/hitsFeatures_training.csv", 'r') as file:
        reader = csv.reader(file)
        hits_train_0 = list(reader)

    hits_train_0 = hits_train_0[1:len(hits_train_0)]
    hits_train_0 = [float(element[0]) for element in hits_train_0]

    print("Some elements of hits_train_0: ", hits_train_0[0:5])
    print(len(hits_train_0))
    features_training_list.append(hits_train_0)

    with open("./data/hitsFeatures_testing.csv", 'r') as file:
        reader = csv.reader(file)
        hits_test_0 = list(reader)

    hits_test_0 = hits_test_0[1:len(hits_test_0)]
    hits_test_0 = [float(element[0]) for element in hits_test_0]

    print("Some elements of hits_test_0: ", hits_test_0[0:5])
    print(len(hits_test_0))
    features_testing_list.append(hits_test_0)

# import target authority
if use_hit2 == 1:
    with open("./data/hitsFeatures_training.csv", 'r') as file:
        reader = csv.reader(file)
        hits_train_1 = list(reader)

    hits_train_1 = hits_train_1[1:len(hits_train_1)]
    hits_train_1 = [float(element[1]) for element in hits_train_1]

    print("Some elements of hits_train_1: ", hits_train_1[0:5])
    print(len(hits_train_1))
    features_training_list.append(hits_train_1)

    with open("./data/hitsFeatures_testing.csv", 'r') as file:
        reader = csv.reader(file)
        hits_test_1 = list(reader)

    hits_test_1 = hits_test_1[1:len(hits_test_1)]
    hits_test_1 = [float(element[1]) for element in hits_test_1]

    print("Some elements of hits_test_1: ", hits_test_1[0:5])
    print(len(hits_test_1))
    features_testing_list.append(hits_test_1)


# journal features:

# import same journal (0 or 1) (implemented in same_journal.py)
if use_sj == 1:
    with open("./data/same_journal_training.txt", "r") as f:
        reader = csv.reader(f)
        same_journal_train = list(reader)

    same_journal_train = [float(element[0]) for element in same_journal_train]

    print("Some elements of same_journal_train: ", same_journal_train[0:5])
    print(len(same_journal_train))
    features_training_list.append(same_journal_train)

    with open("./data/same_journal_testing.txt", "r") as f:
        reader = csv.reader(f)
        same_journal_test = list(reader)

    same_journal_test = [float(element[0]) for element in same_journal_test]

    print("Some elements of same_journal_test: ", same_journal_test[0:5])
    print(len(same_journal_test))
    features_testing_list.append(same_journal_test)


# publication year features:

# import temporal difference (implemented in temp_diff.py)
if use_td == 1:
    with open("./data/temp_diff_training.txt", "r") as f:
        reader = csv.reader(f)
        temp_diff_train = list(reader)

    temp_diff_train = [float(element[0]) for element in temp_diff_train]

    print("Some elements of temp_diff_train: ", temp_diff_train[0:5])
    print(len(temp_diff_train))
    features_training_list.append(temp_diff_train)

    with open("./data/temp_diff_testing.txt", "r") as f:
        reader = csv.reader(f)
        temp_diff_test = list(reader)

    temp_diff_test = [float(element[0]) for element in temp_diff_test]

    print("Some elements of temp_diff_test: ", temp_diff_test[0:5])
    print(len(temp_diff_test))
    features_testing_list.append(temp_diff_test)


# title features:

# import number of common words in titles (implemented in overlap_abstract_title.py)
if use_ncw_title == 1:
    with open("./data/overlap_title_training.txt", "r") as f:
        reader = csv.reader(f)
        overlap_title_train = list(reader)

    overlap_title_train = [float(element[0]) for element in overlap_title_train]

    print("Some elements of overlap_title_train: ", overlap_title_train[0:5])
    print(len(overlap_title_train))
    features_training_list.append(overlap_title_train)

    with open("./data/overlap_title_testing.txt", "r") as f:
        reader = csv.reader(f)
        overlap_title_test = list(reader)

    overlap_title_test = [float(element[0]) for element in overlap_title_test]

    print("Some elements of overlap_title_test: ", overlap_title_test[0:5])
    print(len(overlap_title_test))
    features_testing_list.append(overlap_title_test)


if len(features_training_list) > 0:  # we can only continue if at least one feature has been chosen

    # compose feature matrices
    features_train = np.array(features_training_list).T
    features_test = np.array(features_testing_list).T

    # convert labels into floats then into column array
    labels = [float(element[2]) for element in training_set]
    labels = np.array(list(labels))

    if mode == 0:

        # Using this mode you can try out a certain configuration of features and model parameters.

        # do cross validation of a certain configuration

        m = 5   # number of parts in which the training set is divided, one part is used for testing, the others for training

        # create and shuffle index set
        index = np.arange(0, len(training_set))
        np.random.shuffle(index)

        # split index set into m parts
        r = []

        for i in range(1, m):
            r.append(int(i * len(training_set)/m))

        S = np.split(index, r)

        # define lists for scores: one for the score of the training set and one for the score of the testing set

        score_train_xgb = np.zeros(m)
        score_train_train_xgb = np.zeros(m)     # In the case of XGBoost classifier we split the training set into a training and a validation part.
        score_validate_xgb = np.zeros(m)        # We will calculate the score for the training set, the validation set, the union of training and validation set and the testing set.
        score_test_xgb = np.zeros(m)

        score_train_nn = np.zeros(m)
        score_test_nn = np.zeros(m)

        score_train_ovr = np.zeros(m)
        score_test_ovr = np.zeros(m)

        score_train_nbr = np.zeros(m)
        score_test_nbr = np.zeros(m)

        score_train_dtc = np.zeros(m)
        score_test_dtc = np.zeros(m)

        score_train_svm = np.zeros(m)
        score_test_svm = np.zeros(m)

        # define lists for runtimes

        t_xgb = np.zeros(m)
        t_nn = np.zeros(m)
        t_ovr = np.zeros(m)
        t_nbr = np.zeros(m)
        t_dtc = np.zeros(m)
        t_svm = np.zeros(m)

        # do cross validation

        for i in range(0, m):

            print("Started round number %d of %d of cross validation." %(i+1, m))

            # create lists of indexes of training and testing samples
            index_training_0 = []
            for j in range(0, m):
                if j != i:
                    index_training_0.append(S[j])
                else:
                    index_testing = S[i]

            # write list of indexes of training samples in right way
            index_training = []
            for j in range(0, len(index_training_0)):
                for h in range(0, len(index_training_0[j])):
                    index_training.append(index_training_0[j][h])

            X_train = features_train[index_training, :]
            X_test = features_train[index_testing, :]

            y_train = labels[index_training]
            y_test = labels[index_testing]

            # do a train / validate split
            X_train_train, X_train_validate, y_train_train, y_train_validate = train_test_split(X_train, y_train, test_size=0.2, shuffle=True)

            # definition of the metric used for the XGBoost classifier
            def xgboost_f1(y, v):
                v = v.get_label()
                y_binary = [1. if y_continuous > 0.5 else 0. for y_continuous in y]
                return 'f1', 1.0 - f1_score(v, y_binary)

            # classifier 1: XGBoost

            t_xgb[i] = clock()

            parameters = {
                "n_estimators": 100,
                "learning_rate": 0.1,
                "max_depth": 3,
                "silent": True,
                "n_jobs": -1
            }

            model = XGBClassifier(**parameters)

            model.fit(
                X_train_train,
                y_train_train,
                verbose=True,
                early_stopping_rounds=20,
                eval_set=[(X_train_validate, y_train_validate)],
                eval_metric=xgboost_f1)

            pred_train = np.array(model.predict(X_train))
            score_train_xgb[i] = f1_score(y_train, pred_train)

            pred_train_train = np.array(model.predict(X_train_train))
            score_train_train_xgb[i] = f1_score(y_train_train, pred_train_train)

            pred_validate = np.array(model.predict(X_train_validate))
            score_validate_xgb[i] = f1_score(y_train_validate, pred_validate)

            pred_test = np.array(model.predict(X_test))
            score_test_xgb[i] = f1_score(y_test, pred_test)

            t_xgb[i] = clock() - t_xgb[i]

            # plot feature importance
            '''
            pyplot.bar(range(len(model.feature_importances_)), model.feature_importances_)
            pyplot.show()
            '''

        # classifier 2: Neural Network

            t_nn[i] = clock()

            X_train_nn = preprocessing.scale(X_train)
            X_test_nn = preprocessing.scale(X_test)
            X_train_train_nn = preprocessing.scale(X_train_train)
            X_train_validate_nn = preprocessing.scale(X_train_validate)

            X_train_nn = X_train
            X_test_nn = X_test
            X_train_train_nn = X_train_train
            X_train_validate_nn = X_train_validate

            model = K.Sequential()
            model.add(K.layers.Dense(128, input_dim=X_train_nn.shape[1], activation='sigmoid'))
            model.add(K.layers.Dense(64, activation='sigmoid'))
            model.add(K.layers.Dense(32, activation='sigmoid'))
            model.add(K.layers.Dense(1))
            model.summary()

            model.compile(optimizer='adam',
                          loss='mean_squared_error',
                          metrics=['mean_squared_error'],
                          loss_weights=None,
                          sample_weight_mode=None,
                          weighted_metrics=None,
                          target_tensors=None)

            model.fit(x=X_train_nn,
                      y=y_train,
                      batch_size=None,
                      epochs=15,
                      verbose=1,
                      callbacks=None,
                      validation_split=0.0,
                      validation_data=None,
                      shuffle=True,
                      class_weight=None,
                      sample_weight=None,
                      initial_epoch=0,
                      steps_per_epoch=None,
                      validation_steps=None)

            pred_train = np.array(model.predict(X_train_nn))
            print(len(pred_train))

            for j in range(0, len(pred_train)):
                if pred_train[j] < 0.5:
                    pred_train[j] = 0
                else:
                    pred_train[j] = 1

            score_train_nn[i] = f1_score(y_train, pred_train)

            pred_test = np.array(model.predict(X_test_nn))

            for j in range(0, len(pred_test)):
                if pred_test[j] < 0.5:
                    pred_test[j] = 0
                else:
                    pred_test[j] = 1

            score_test_nn[i] = f1_score(y_test, pred_test)

            t_nn[i] = clock() - t_nn[i]

        # classifier 3: One Versus Rest Classifier using logistic regression (used with standard configuration)

            t_ovr[i] = clock()
            
            model = OneVsRestClassifier(LogisticRegression())
            model.fit(X_train, y_train)
            
            score_train_ovr[i] = f1_score(model.predict(X_train), y_train)
            score_test_ovr[i] = f1_score(model.predict(X_test), y_test)
                              
            t_ovr[i] = clock() - t_ovr[i]

        # classifier 4: KNeighbors Classifier (used with standard configuration)

            t_nbr[i] = clock()
            
            model = neighbors.KNeighborsClassifier()
            model.fit(X_train, y_train)
            
            score_train_nbr[i] = f1_score(model.predict(X_train), y_train)
            score_test_nbr[i] = f1_score(model.predict(X_test), y_test)
                              
            t_nbr[i] = clock() - t_nbr[i]

        # classifier 5: Decision Tree Classifier (used with standard configuration)

            t_dtc[i] = clock()
            
            model = DecisionTreeClassifier()
            model.fit(X_train, y_train)
            
            score_train_dtc[i] = f1_score(model.predict(X_train), y_train)
            score_test_dtc[i] = f1_score(model.predict(X_test), y_test)
                              
            t_dtc[i] = clock() - t_dtc[i]

        # classifier 6: SVM (used with standard configuration)

            t_svm[i] = clock()

            model = svm.LinearSVC()
            model.fit(X_train, y_train)

            score_train_svm[i] = f1_score(model.predict(X_train), y_train)
            score_test_svm[i] = f1_score(model.predict(X_test), y_test)

            t_svm[i] = clock() - t_svm[i]

        # print results

        # XGBoost
        print("score_train_xgb: ", score_train_xgb)
        print("average: ", np.mean(score_train_xgb))

        print("score_train_train_xgb: ", score_train_train_xgb)
        print("average: ", np.mean(score_train_train_xgb))

        print("score_validate_xgb: ", score_validate_xgb)
        print("average: ", np.mean(score_validate_xgb))

        print("score_test_xgb: ", score_test_xgb)
        print("average: ", np.mean(score_test_xgb))

        print("runtimes xgb: ", t_xgb)
        print("average: ", np.mean(t_xgb))

        # Neural Network
        print("score_train_nn: ", score_train_nn)
        print("average: ", np.mean(score_train_nn))

        print("score_test_nn: ", score_test_nn)
        print("average: ", np.mean(score_test_nn))

        print("runtimes nn: ", t_nn)
        print("average: ", np.mean(t_nn))

        # One Versus Rest Classifier
        print("score_train_ovr: ", score_train_ovr)
        print("average: ", np.mean(score_train_ovr))

        print("score_test_ovr: ", score_test_ovr)
        print("average: ", np.mean(score_test_ovr))

        print("runtimes ovr: ", t_ovr)
        print("average: ", np.mean(t_ovr))

        # KNeighbors Classifier
        print("score_train_nbr: ", score_train_nbr)
        print("average: ", np.mean(score_train_nbr))

        print("score_test_nbr: ", score_test_nbr)
        print("average: ", np.mean(score_test_nbr))

        print("runtimes nbr: ", t_nbr)
        print("average: ", np.mean(t_nbr))

        # Decision Tree Classifier
        print("score_train_dtc: ", score_train_dtc)
        print("average: ", np.mean(score_train_dtc))

        print("score_test_dtc: ", score_test_dtc)
        print("average: ", np.mean(score_test_dtc))

        print("runtimes dtc: ", t_dtc)
        print("average: ", np.mean(t_dtc))

        # SVM
        print("score_train_svm: ", score_train_svm)
        print("average: ", np.mean(score_train_svm))

        print("score_test_svm: ", score_test_svm)
        print("average: ", np.mean(score_test_svm))

        print("runtimes svm: ", t_svm)
        print("average: ", np.mean(t_svm))

    if mode == 1:

        # for creating a submission:

        # do a train / validate split
        X_train, X_validate, y_train, y_validate = train_test_split(features_train, labels, test_size=0.2, shuffle=True)

        # classifier 1: XGBoost

        def xgboost_f1(y, v):
                v = v.get_label()
                y_binary = [1. if y_continuous > 0.5 else 0. for y_continuous in y]
                return 'f1', 1.0 - f1_score(v, y_binary)

        parameters = {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 3,
            "silent": True,
            "n_jobs": -1
        }

        model = XGBClassifier(**parameters)

        model.fit(
            X_train,
            y_train,
            verbose=True,
            early_stopping_rounds=20,
            eval_set=[(X_validate, y_validate)],
            eval_metric=xgboost_f1)

        pred_train = np.array(model.predict(X_train))
        score_train_xgb = f1_score(y_train, pred_train)
        print("xgb score of X_train: ", score_train_xgb)
        print("Length of pred_train_: ", len(pred_train))
        print("Sum of elements of pred_train: ", np.sum(pred_train))

        pred_validate = np.array(model.predict(X_validate))
        score_validate_xgb = f1_score(y_validate, pred_validate)
        print("xgb score of X_validate: ", score_validate_xgb)
        print("Length of pred_validate: ",len(pred_validate))
        print("Sum of elements of pred_validate: ", np.sum(pred_validate))

        pred = np.array(model.predict(features_test))
        print("Length of pred: ", len(pred))
        print("Sum of elements of pred: ", np.sum(pred))

        if not os.path.isfile("./predictions/predictions_xgb.csv"):
            with open('./predictions/predictions_xgb.csv', 'w') as csvfile:
                fieldnames = ['id', 'category']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                writer.writeheader()
                for i in range(0, len(pred)):
                    writer.writerow({'id': str(i), 'category': str(int(pred[i]))})

        '''
        # plot feature importance
        pyplot.bar(range(len(model.feature_importances_)), model.feature_importances_)
        pyplot.show()
        '''

        # classifier 2: Neural Network

        features_train = preprocessing.scale(features_train)
        features_test = preprocessing.scale(features_test)
        X_train = preprocessing.scale(X_train)
        X_validate = preprocessing.scale(X_validate)

        model = K.Sequential()
        model.add(K.layers.Dense(128, input_dim=features_train.shape[1], activation='sigmoid'))
        model.add(K.layers.Dense(64, activation='sigmoid'))
        model.add(K.layers.Dense(32, activation='sigmoid'))
        model.add(K.layers.Dense(1))
        model.summary()

        model.compile(optimizer='adam',
                      loss='mean_squared_error',
                      metrics=['mean_squared_error'],
                      loss_weights=None,
                      sample_weight_mode=None,
                      weighted_metrics=None,
                      target_tensors=None)

        model.fit(x=features_train,
                  y=labels,
                  batch_size=None,
                  epochs=15,
                  verbose=1,
                  callbacks=None,
                  validation_split=0.0,
                  validation_data=None,
                  shuffle=True,
                  class_weight=None,
                  sample_weight=None,
                  initial_epoch=0,
                  steps_per_epoch=None,
                  validation_steps=None)

        pred_train = np.array(model.predict(features_train))

        for i in range(0, len(pred_train)):
            if pred_train[i] < 0.5:
                pred_train[i] = 0
            else:
                pred_train[i] = 1

        score_train_nn = f1_score(labels, pred_train)
        print("nn score of features_train: ", score_train_nn)
        print("Length of pred_train: ", len(pred_train))
        print("Sum of elements of pred_train: ", np.sum(pred_train))

        pred_train_reduced = np.array(model.predict(X_train))

        for i in range(0, len(pred_train_reduced)):
            if pred_train_reduced[i] < 0.5:
                pred_train_reduced[i] = 0
            else:
                pred_train_reduced[i] = 1

        score_train_reduced_nn = f1_score(y_train, pred_train_reduced)
        print("nn score of X_train: ", score_train_reduced_nn)
        print("Length of pred_train_reduced: ", len(pred_train_reduced))
        print("Sum of elements of pred_train_reduced: ", np.sum(pred_train_reduced))

        pred_validate = np.array(model.predict(X_validate))

        for i in range(0, len(pred_validate)):
            if pred_validate[i] < 0.5:
                pred_validate[i] = 0
            else:
                pred_validate[i] = 1

        score_validate_nn = f1_score(y_validate, pred_validate)
        print("nn score of X_validate: ", score_validate_nn)
        print("Length of pred_validate: ", len(pred_validate))
        print("Sum of elements of pred_validate: ", np.sum(pred_validate))

        pred = np.array(model.predict(features_test))

        for i in range(0, len(pred)):
            if pred[i] < 0.5:
                pred[i] = 0
            else:
                pred[i] = 1

        print("Length of pred: ", len(pred))
        print("Sum of elements of pred: ", np.sum(pred))

        if not os.path.isfile("./predictions/predictions_nn.csv"):
            with open('./predictions/predictions_nn.csv', 'w') as csvfile:
                fieldnames = ['id', 'category']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                writer.writeheader()
                for i in range(0, len(pred)):
                    writer.writerow({'id': str(i), 'category': str(int(pred[i]))})

else:
    print("Please choose at least one feature.")
