"""
Principal function module for handling, formatting and graph of zoo data.

todo: improvements on function parameters definition.

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def create_comparation_table_clustering(pred, X, labels, dict_class_types, class_label):
    ''' Create a dataframe with the values of all the features (denoted by X) and the label results
    of the clustering algorithm, plus class names (table denoted by yhat). Also, returns a comparision table
    with the presence of each class type with each cluster type, if present.
    
    Args:
        - pred (pd.DataFrame): class prediction result.
        - X (pd.DataFrame): features dataset, input for prediction.
        - labels (pd.Series): true labels of the original dataset.
        - dict_class_types (dict): dict with keys being the true class number and values with names.
        - class_label (pd.DataFrame): dataframe with information about data classes (number and name).
        
    Returns:
        - yhat (pd.DataFrame): dataframe with all the feature values plus the results of the model.
        - comp_table (pd.DataFrame): dataframe with the results of the clustering model, plus calculations of the presence.
        
        
    '''    
    
    # Create dataframe with features plus prediction values
    yhat = X.copy()
    yhat['cluster_label'] = pred
    yhat['class_label'] = labels
    yhat['class_name'] = yhat['class_label'].replace(dict_class_types)
    
    # Create comparision table with results
    check_labels_w_names = yhat.groupby(['class_label','cluster_label']).size().reset_index()
    check_labels_w_names['class_name'] = check_labels_w_names['class_label'].replace(dict_class_types)
    check_labels_w_names = check_labels_w_names.set_index(['class_name', 'class_label'])
    check_labels_w_names = pd.merge(check_labels_w_names.reset_index(), class_label[['Class_Number', 'Number_Of_Animal_Species_In_Class']], 
            'left', left_on = 'class_label', right_on = 'Class_Number')
    check_labels_w_names = check_labels_w_names.rename({'Number_Of_Animal_Species_In_Class': 'class_total_number',
                                                        0: 'cluster_total_number'}, axis = 1)
    check_labels_w_names['presence_relative_to_class'] = np.round(check_labels_w_names['cluster_total_number']/check_labels_w_names['class_total_number'],3)
    comp_table = check_labels_w_names.set_index(['class_name', 'class_label'])

    return yhat, comp_table


def compute_distance_matrix(centroids_clusters, centroids_classes):
    """ 
    Calculation of the distance between two vectors.
    
    Args:
        - centroids_clusters (pd.DataFrame): matrix of the centroids for each cluster.
        - centroids_classes (pd.DataFrame): matrix of the centroids for each of the classes.
        
    Returns:
        - dmatrix (pd.DataFrame): distance matrix between centroids of each cluster with each class.
    
    """
    

    distance_matrix = []
    for cluster in centroids_clusters.index:
        distance_cluster_list = []
        for class_centroid in centroids_classes.index:
            eu_distance = np.linalg.norm(centroids_clusters.loc[cluster] - centroids_classes.loc[class_centroid])
            distance_cluster_list.append(eu_distance)
        distance_matrix.append(distance_cluster_list)
        
    dmatrix = pd.DataFrame(distance_matrix, 
                           index = list(centroids_clusters.index),
                           columns = list(centroids_classes.index))
    
    return dmatrix

def classification_metrics(yhat, centroids_clusters,assignment):
    """
    Computation of the a regular classification model based on a confusion matrix
    
    Args:
        - yhat (pd.DataFrame): dataframe with all the feature values plus the results of the model.
        - assignment (dict): dictionary with the cluster number (keys) and their corresponding true class name (values). 
    
    Returns:
        - metrics_result_df (pd.DataFrame): result table with metrics.
    
    """
    
    cc = centroids_clusters.copy()
    metrics_result_df = pd.DataFrame(columns = ["class_name","Accuracy", "Precision", "Specificity", "Recall", "F1"])
    
    for cluster_number, class_name in assignment.items():
        predicted_bool_col = yhat["cluster_label"] == cluster_number
        true_bool_col = yhat["class_name"] == class_name
        acc_sc = accuracy_score(true_bool_col, predicted_bool_col)
        pre_sc = precision_score(true_bool_col, predicted_bool_col)
        specificity_sc = recall_score(true_bool_col, predicted_bool_col, pos_label = 0)
        recall_sc = recall_score(true_bool_col, predicted_bool_col)
        f1_sc = f1_score(true_bool_col, predicted_bool_col)
        
        metrics_result_df.loc[cluster_number] = [class_name, acc_sc, pre_sc, specificity_sc, recall_sc, f1_sc]
        
    cc["class"] = cc["label_cluster"].replace(assignment)
    return metrics_result_df, cc 


def assign_clusters_distance(dmatrix):
    
    """
    Assign based on Euclidean distance between centroids the most similar cluster
    
    Args:
        - dmatrix (pd.DataFrame): dataframe with cluster numbers as indexes and class labels as columns. Values are the distances
                                previously computed.
    
    Returns:
        - cluster_assignment (pd.Series):  assignment of each cluster for each class label.
    
    """
    
    dm = dmatrix.copy()
    assign_tuples = []

    while len(dm) > 0:
        
        # We get the class that has the min distance computed
        minimum_distance_class = dm.min().idxmin()
        # Get the cluster index relative to that min value
        min_index = dm[minimum_distance_class].idxmin()

        # Save the tuple (index, class)
        assignment_tuple = (int(min_index), minimum_distance_class)
        assign_tuples.append(assignment_tuple)
        dm = dm.drop(min_index)
        dm = dm.drop(minimum_distance_class, axis = 1)
        
    return dict(assign_tuples)

def plot_chars_heatmap(data, title = None):
    """
    Plot features from cluster centroids into a heatmap
    
    Args:
        - data (pd.DataFrame): data with values between 0 and 1 representing the presence of each feature for each class.
        - title (str): title of the graph.
    
    Returns:
        - None (only plots a graph).
    
    """
    
    plt.figure(figsize = (7,7))

    ax = sns.heatmap(data = data, annot = True)
    ax.set_title(title)

    plt.show()
    
def aggregate_features_cluster_centers(zoo_data, columns, cluster_centers = None, index = None):
    """
    Aggregate features based for each centroid of clusters predicted. This function also applies a specific formatting to the returned dataframe.
    
    Args:
        - zoo_data (pd.DataFrame): original dataset with each row as an animal and columns as features. Values are 1 or 0 (presence or not of feature).
        - columns (pd.DataFrame): columns names of the cluster centroid dataset (features).
        - cluster_centers (list): dataframe with cluster centroids.
        - index (list): index names of the cluster centroids dataset (predicted class names).
    
    Returns:
        - agg_clusters: dataframe with the cluster centroids.
        - centroids_classes: dataframe with the classes centroids.
    
    """
    
    # --- For graphs ---
    agg_clusters = pd.DataFrame(cluster_centers , index = index,columns = columns).astype('float32').reset_index().rename({'index': 'label_cluster'}, axis = 1)
    # --- For computation
    centroids_classes = zoo_data.drop(['class_type', 'animal_name'], axis = 1).groupby(['class_names']).mean()
    
    return agg_clusters, centroids_classes

def create_centroids_clusters(yhat, zm, X, create_centroids = [True, None]):
    """
    Aggregate features based for each centroid of clusters predicted. This function also applies a specific formatting to the returned dataframe.
    
    Args:
        - yhat (pd.DataFrame): result dataframe with predictions and features.
        - zm (pd.DataFrame): original dataset with each row as an animal and columns as features. Values are 1 or 0 (presence or not of feature).
        - X (pd.DataFrame): input dataframe for models with only features.
        - create_centroids (list): list with 2 values. First position checks if we want to create centroids for our clusters (in case a particular model
                                 does not return centroids), and creates it in case of True. If False, the second value should be a list containing
                                 the centroids of our clusters.
    
    Returns:
        - centroids_clusters (pd.DataFrame): dataframe with the cluster centroids.
        - centroids_classes (pd.DataFrame): dataframe with the classes centroids.
    
    """
    
    if create_centroids[0]:
        agg_centroids_clusters = yhat.drop(["class_name"], axis = 1).groupby(["cluster_label"]).mean()[X.columns]
        agg_centroids_clusters = pd.DataFrame(data = np.array(agg_centroids_clusters), 
                    index = list(agg_centroids_clusters.index),
                    columns = agg_centroids_clusters.columns)
        
        centroids_clusters, centroids_classes = aggregate_features_cluster_centers(zm, X.columns, 
                                                                            np.array(agg_centroids_clusters),
                                                                            list(agg_centroids_clusters.index))
    else:
        centroids_clusters, centroids_classes = aggregate_features_cluster_centers(zm, 
                                                                                   X.columns, 
                                                                                   create_centroids[1])
    return centroids_clusters, centroids_classes


def result_pipeline(y_pred, zm, X, labels, dict_class_types, class_label, create_centroids = [True, None]):
    """
    Main pipeline for the results of the clustering model applied. Handles results and returns metrics and useful information.
    
    Args:
        - y_pred (pd.DataFrame): class prediction result.
        - zm (pd.DataFrame): original dataset with each row as an animal and columns as features. Values are 1 or 0 (presence or not of feature).
        - X (pd.DataFrame): features dataset, input for prediction.
        - labels (pd.Series): true labels of the original dataset.
        - dict_class_types (dict): dict with keys being the true class number and values with names.
        - class_label (pd.DataFrame): dataframe with information about data classes (number and name).
        - create_centroids (list): list with 2 values. First position checks if we want to create centroids for our clusters (in case a particular model
                                 does not return centroids), and creates it in case of True. If False, the second value should be a list containing
                                 the centroids of our clusters.
    
    Returns:
        - yhat (pd.DataFrame): dataframe with all the feature values plus the results of the model.
        - comp_table (pd.DataFrame): dataframe with the results of the clustering model, plus calculations of the presence.
        - dmatrix (pd.DataFrame): distance matrix between centroids of each cluster with each class.
        - gb_clusters (pd.DataFrame): cluster centroid dataframe with asignment of cluster label to true class by proximity.
        - assignment (dict): assignment dict of each cluster label for each true class.
        - assignment_new_label (dict): same as assignment dict but each class is replaced by its name (for instance, 1: Mammal, 2: Reptile...).
        - centroids_clusters (pd.DataFrame): dataframe with the cluster centroids.
    
    """

    # Merges prediction result with features dataframe and create a comparation table
    yhat, comp_table = create_comparation_table_clustering(y_pred, X, labels, dict_class_types, class_label)

    # Create centroids (in case a model doesn't compute centroids) for each cluster and true class labels
    centroids_clusters, centroids_classes = create_centroids_clusters(yhat, zm, X, create_centroids)

    dmatrix = compute_distance_matrix(centroids_clusters.set_index("label_cluster"), centroids_classes)
    assignment = assign_clusters_distance(dmatrix)

    assignment_new_label = dict(zip(assignment.keys(),[f'{x[0]} ({x[1]})' for x in assignment.items()]))
    gb_clusters = centroids_clusters.drop(["legs", "label_cluster"], axis = 1)
    gb_clusters.index = [assignment_new_label[i] for i in centroids_clusters["label_cluster"]]
    

    return yhat, comp_table, dmatrix, gb_clusters, assignment, assignment_new_label, centroids_clusters