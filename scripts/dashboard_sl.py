from os import listdir
from os.path import isfile, join
import pandas as pd
import numpy as np
import streamlit as st
import altair as alt


path_metrics = "../data/result_metrics"
metric_files = [f for f in listdir(path_metrics) if isfile(join(path_metrics, f))]
model_name_metric = [x.split("_result_metrics.pkl")[0] for x in metric_files]
data_list = [pd.read_pickle(f'{path_metrics}/{x}') for x in metric_files]


path_descr = "../data/description_clusters"
descr_files = [f for f in listdir(path_descr) if isfile(join(path_descr, f))]
model_name_descr = [x.split("_descr.pkl")[0] for x in descr_files]
descr_list = [pd.read_pickle(f'{path_descr}/{x}') for x in descr_files]

# Create classes list
animal_classes = list(descr_list[0]["class"].unique())

# Create dataframe with models
# Title and Introduction

st.title("Classifying animals using unsupervised Clustering models")
st.image("https://media.npr.org/assets/img/2015/08/07/animal-eyes_wide-c87870d497894b4fe68d37c80c1511da4f302ddf.jpg?s=1400&c=100&f=jpeg",
         caption = "Top row: iStockphoto; bottom row: Flickr")
st.header("📖What if...?")

st.write("If we are given a set of characteristics relative to an animal, are we able to exactly determine which \
    class it corresponds? We could probably check the presence of unique class features, like the presence of milk-producing glands in the case of mammals \
    or fins in fishes. Maybe we could see the overall most important features and then make a decision. \
      \n  \
    \n  In this project, we use an unsupervised approach to classify each animal from a dataset based on the presence of different features. \
    The results shown below compare between different clustering methods the level of accuracy given a selected animal class.")


st.header("❔Confusion and more confusion")
st.write("To get a better understanding of the results and also to standardize measured results, we applied the concept of **confusion matrix** for each class. \
         This concept is often used in binary classification problems, but we can adapt our data in such a way that we can access this useful measuring tool. \
         For each label (either predicted or true) of a particular class X we can create a new variable that has the value 1 when it corresponds \
        to the class, or 0 if it doesn't. Then we compute the confusion matrix and this way, we can get our desired metrics")

st.image("https://glassboxmedicine.com/wp-content/uploads/2019/02/confusion-matrix.png?w=816", caption = "2x2 Confusion Matrix")

st.write("All of the metrics used are computed using the values of the confusion matrix. \
    To have a better understanding of these metrics, here is a brief explanation:")
st.markdown("\n  - **_Accuracy_**: of total predictions, what percentage is right?\n\
            \n  - **_Precision_**: of the positives predicted, what percentage is truly positive?\n\
            \n  - **_Specificity_**: how well is the model at predicting negative results?\n\
            \n  - **_Recall_**: how well is the model at predicting positive results?\n\
            \n  - **_F1_**: tells us how balanced accuracy and precision is. Do we have a lower or higher trade-off between these two metrics?")

st.sidebar.title("Select a class")
options_classes_metrics = st.sidebar.selectbox("Class", animal_classes)

cfmx_metrics = ["Accuracy", "Precision", "Specificity", "Recall", "F1"]

st.subheader("📈Metrics results")
# options_classes_metrics = st.selectbox("Class", animal_classes)
metric_filter = st.pills("Select metric(s)",cfmx_metrics, default = "Accuracy")
st.write("If we see only accuracy, then overall, we could say that we have good models. The main ones are DBSCAN, Aglomerative \
    Clustering and KMeans.\
      \n\n  For the mammal class KMeans is the strongest, which makes sense since this class is the one with the highest number \
    of points. Both DBSCAN (density based) and Ag. Clustering (hierarchical based) behave well in other classes, where the number of points is \
    very reduced.\
      \n\n  ")

df_option_class_metrics = []
for e, name_model in enumerate(model_name_metric):
    index_model = model_name_metric.index(name_model)
    itdf = data_list[index_model]
    dft = itdf[itdf['class_name'] == options_classes_metrics].drop(["class_name"], axis = 1).reset_index(drop = True).copy()
    dft["Model"] = name_model
    df_option_class_metrics.append(dft)
    
df_option_class_metrics = pd.concat(df_option_class_metrics).reset_index(drop = True)

metric_filtered_df = df_option_class_metrics.sort_values(metric_filter, ascending = False)
max_value_best_model = metric_filtered_df[metric_filter].max()
metric_filtered_df["Type"] = np.where(metric_filtered_df[metric_filter] == max_value_best_model,
         "Best model", "Lesser model")
    

c = alt.Chart(metric_filtered_df).mark_bar().encode(
    x = alt.X("Model:O", sort='-y', title="Model", axis = alt.Axis(labelAngle=-45)),
    y = alt.Y(f'{metric_filter}:Q', scale = alt.Scale(domain = [0,1])),
    color = alt.Color(f'Type:O', legend=alt.Legend(title="Status")
)).properties(
    title = "Performance of different models based on selected metric",
    width = 1200,
    height = 500
)

st.altair_chart(c, use_container_width=True)



st.header("🐺A wolf in sheep's clothing")
st.write("If our model says that an animal is a mammal, then what does he understand as a mammal?\n\
         \n  Using the results of each model we can compute the centroids and group all the points in one characteristic vector. \
             For models that don't have a proper centroid (DBSCAN, for instance) we aggregated the data for each cluster and then \
            each centroid was calculated. \n\
        \n  To pair a cluster to an animal class, for each model the distance between each cluster centroid and each class centroid \
            is computed. The assignment is made based on the minimum distance of cluster/class centroid pairs. When one class is assigned to a cluster \
            then it cannot be assigned again to another cluster.")

# char
st.image("https://miro.medium.com/v2/resize:fit:1400/1*lzpTuWUXS53y0XdLK8htig.png", caption = "KMeans graphical example")
st.subheader("Profiling classes")

st.write("Since we have a large amount of features, graphical representation of the clusters is a bit complex. Hence, we'll check the presence of \
            each feature in every centroid.")

# Esto debe filtrar por fila
# option_class = st.selectbox("Class", animal_classes)
options_models = st.pills("Select model(s)", model_name_descr, default = "kmeans", selection_mode="multi")

# For que vaya por cada dataframe y luego filtre, juntar todos, plotear

df_option_class = []

for e, name_model in enumerate(options_models):
    index_model = model_name_descr.index(name_model)
    itdf = descr_list[index_model]
    dft = itdf[itdf['class'] == options_classes_metrics].drop(["legs", "label_cluster", "class"], axis = 1).reset_index(drop = True).copy()
    dft["Model"] = name_model
    df_option_class.append(dft)
    
df_option_class = pd.concat(df_option_class)
df_option_class = pd.melt(df_option_class, ['Model'])

bp = alt.Chart(df_option_class).mark_bar().encode(
    y = alt.Y("variable:O", title = "Feature"),
    x = alt.X('value:Q', title = "Presence" ,scale = alt.Scale(domain = [0,1])),
    color = alt.Color(f'Model:N', legend=alt.Legend(title="Model")),
    yOffset= 'Model:N'
    ).properties(
    title = "Barplot with hue",
    width = 1200,
    height = 500
)

st.altair_chart(bp, use_container_width=True)

st.write("As we can see, not all models understand a class the same way. For instance, if we check the feature profile \
        of Reptile, our DBSCAN model also identifies 'fins', 'hair' and even 'milk' as features of some reptilians, which is completely off. \
        The mammal class seems to have a better profile than any other class, independent of model.")