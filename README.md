# Links prediction in Citation Networks
Repository for the project of the INF554 Machine Learning 1 course
This project was developed by Fabrizio Indirli, Leon Kloten, Martin Wohlfender and Seongbin Lim

### Introduction
The project consisted in predicting missing links in a citation network.
In particular, some edges were removed from a citation network constructed upon a dataset with 27770 articles: the aim of the project was to build a model that could predict whether an edge was part of the original network or not.

To do so, 17 features have been extracted from the given data and various supervised learning models have been developed,
obtaining an F1 score of almost 0.97.
Among all the features, 6 of them largely affect the predictions performance: *number of paths, TFIDF cosine similarities of
abstracts, source hub score, target authority score* and the *Resource Allocation index*.
Between the various classifiers that were tried, **XGBoost** and **Neural Network** produced the best results.

### Folders structure
- Folder **scripts** contains the feature-generation scripts that are called by *create_features.py*
- Folder **data** contains the CSV/TXT data files generated running *create_features.py*
- *node_information.csv* is a spreadsheet that contains the datas about the 27770 articles of the citation network: the columns are *Node_ID, Publication Year, Title, Authors, Journal, Abstract*
- *training_set.txt* contains the known datas on the edges of the citation network: each line of this file
has the form *Source_Node_ID, Target_Node_ID, Exist* and the associated edge *(Source_Node_ID, Target_Node_ID)* exists in the citation network only if the value of *Exists* = 1
- *testing_set.txt* contains possible edges *Source_Node_ID, Target_Node_ID* that have to be classified
- *report.pdf* is a short (4 pages) report of the project
- *presentation.pdf* is the longer presentation of the project

### How to run the code
1. Default *training_set.txt, testing_set.txt* and *node_information.csv* files are already provided. <br>
**If you want to use yours**, put them (with the same names and structure!) in the root folder where the *create_features.py* and *model.py* files are, replacing the default ones, and delete all the content of the **data** folder.
2. Run create_features.py to generate the files containing the features

    **WARNING**: 
	The calculation of one of the features (TFIDF cosine similarity) requires about 13 GB of ram on the provided dataset and might fail with "MemoryError" if not enough memory is available.
    This is why we provide this feature already calculated for the default dataset in the *"training_sims.csv"* and *"testing_sims.csv"* files inside the **data** folder, but it can be computed from scratch on a computer with enough memory

    **REMARKS**:
    * As it takes a lot of time to create them, the following feature files are already provided, too:
	*closeness_centrality_nodes.txt*, *number_paths_testing.txt, number_paths_training.txt, hits_infos.csv* <br>

	 * The files containing the calculated features are stored in the **data** folder.
	
	 * The scripts that calculate the features are stored in the **scripts** folder.
 	
2. Run model.py to generate the predictions
    
    **REMARK**: 
	The predictions will be stored in the folder **predictions**.
