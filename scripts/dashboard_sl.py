from os import listdir
from os.path import isfile, join
import pandas as pd
import numpy as np
import streamlit as st


path_metrics = "../data/result_metrics"
metric_files = [f for f in listdir(path_metrics) if isfile(join(path_metrics, f))]
model_name_metric = [x.split("_result_metrics.pkl")[0] for x in metric_files]
data_list = [pd.read_pickle(f'{path_metrics}/{x}') for x in metric_files]


path_descr = "../data/description_clusters"
descr_files = [f for f in listdir(path_descr) if isfile(join(path_descr, f))]
model_name_descr = [x.split("_descr.pkl")[0] for x in descr_files]
descr_list = [pd.read_pickle(f'{path_descr}/{x}') for x in descr_files]

# Create dataframe with models




# Title and Introduction
st.title("Result Dashboard on Animal Classifcation")
st.write("This is the main dashboard for the results on the animal classification project.\
            We are given data about a zoo with multiple animals and its features, from which we \
            need to develop a clustering algorithm to predict each of the animal class without \
            'looking' the data.")

st.subheader("Metrics results")
options = st.multiselect("hola", model_name_metric)

# metrics
st.write(data_list[0])
st.bar_chart(data_list[0], x="class_name", y="Accuracy", stack=False)

# char
st.subheader("Caracterización por clase")

animal_classes = list(descr_list[0]["class"].unique())

# Esto debe filtrar por fila
option_class = st.selectbox("Classes", animal_classes)
options_models = st.pills("Models", model_name_descr, default = "kmeans", selection_mode="multi")

# For que vaya por cada dataframe y luego filtre, juntar todos, plotear

df_option_class = []

for e, name_model in enumerate(options_models):
    index_model = model_name_descr.index(name_model)
    itdf = descr_list[index_model]
    dft = itdf[itdf['class'] == option_class].drop(["legs", "label_cluster", "class"], axis = 1).reset_index(drop = True).copy()
    dft["Model"] = name_model
    df_option_class.append(dft)
    
df_option_class = pd.concat(df_option_class)
df_option_class = pd.melt(df_option_class, ['Model'])


st.bar_chart(df_option_class, x="variable", y="value", color="Model", stack=False,
             width = 1800, height = 500, horizontal = True)