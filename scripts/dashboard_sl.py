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

st.title("Result Dashboard on Animal Classifcation")
st.write("This is the main dashboard for the results on the animal classification project.\
            We are given data about a zoo with multiple animals and its features, from which we \
            need to develop a clustering algorithm to predict each of the animal class without \
            'looking' the data.")


st.sidebar.title("EUUU")
st.sidebar.write("ctm")
options_classes_metrics = st.sidebar.selectbox("Class", animal_classes)

cfmx_metrics = ["Accuracy", "Precision", "Specificity", "Recall", "F1"]

st.subheader("Metrics results")
# options_classes_metrics = st.selectbox("Class", animal_classes)
metric_filter = st.pills("Metrics", cfmx_metrics, default = "Accuracy")

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
         "Best model", "Worse model")
    

c = alt.Chart(metric_filtered_df).mark_bar().encode(
    x = alt.X("Model:O", sort='-y', title="Category"),
    y = alt.Y(f'{metric_filter}:Q', scale = alt.Scale(domain = [0,1])),
    color = alt.Color(f'Type:O', legend=alt.Legend(title="Status")
)).properties(
    title = "Barplot with hue",
    width = 1200,
    height = 500
)

st.altair_chart(c, use_container_width=True)



# char
st.subheader("Caracterización por clase")

# Esto debe filtrar por fila
# option_class = st.selectbox("Class", animal_classes)
options_models = st.pills("Models", model_name_descr, default = "kmeans", selection_mode="multi")

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


# st.bar_chart(df_option_class, x="variable", y="value", color="Model", stack=False,
#              width = 1800, height = 500, horizontal = True)


bp = alt.Chart(df_option_class).mark_bar().encode(
    y = alt.Y("variable:O"),
    x = alt.X('value:Q', scale = alt.Scale(domain = [0,1])),
    color = alt.Color(f'Model:N', legend=alt.Legend(title="Model")),
    yOffset= 'Model:N'
    ).properties(
    title = "Barplot with hue",
    width = 1200,
    height = 500
)

st.altair_chart(bp, use_container_width=True)