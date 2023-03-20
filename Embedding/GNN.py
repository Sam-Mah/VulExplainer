from json2graph import *
from sklearn.metrics import classification_report
from keras.callbacks import EarlyStopping

import re
import math

import stellargraph as sg
from stellargraph.mapper import PaddedGraphGenerator
from stellargraph.layer import DeepGraphCNN
from stellargraph import StellarGraph
#from stellargraph import datasets

import pandas as pd
import numpy as np
import os

from sklearn import model_selection
from IPython.display import display, HTML

from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Conv1D, MaxPool1D, Dropout, Flatten
from tensorflow.keras.losses import binary_crossentropy
import tensorflow as tf

print("Libraries imported")

file_name_list = []
graphs = []
graph_labels = []


file_directory = 'Output_embedding'
n_file = len([name for name in os.listdir(file_directory) if os.path.isfile(os.path.join(file_directory, name))])
print("no of files: ", n_file)
count_file = 1
substring = "good"
label = 0
count_good = 0
for file_name in os.listdir(file_directory):
    f = os.path.join(file_directory, file_name)
    if os.path.isfile(f):
        square_node_features = create_graph(file_directory, file_name)
        if substring in file_name:
            label = 0
            count_good = count_good + 1
        else:
            label = 1
        file_name_list.append(file_name)
        graphs.append(square_node_features)
        graph_labels.append(label)

        if count_file % 10 == 0:
            print(count_file, "/", n_file, " graph created!")
        count_file = count_file + 1
print("total no. of Good: ", count_good)
print("total no. of Bad: ", count_file - count_good)
AsmGraph = pd.DataFrame()
AsmGraph["file_name"] = file_name_list
AsmGraph["graphs"] = graphs
AsmGraph["graph_labels"] = graph_labels

print(graphs[0].info())

summary = pd.DataFrame(
    [(g.number_of_nodes(), g.number_of_edges()) for g in graphs],
    columns=["nodes", "edges"],
)
summary.describe().round(1)

graph_labels = pd.get_dummies(graph_labels, drop_first=True)

generator = PaddedGraphGenerator(graphs=graphs)

k = 35
# k = 20
layer_sizes = [32, 32, 32, 1]

dgcnn_model = DeepGraphCNN(
    layer_sizes=layer_sizes,
    activations=["tanh", "tanh", "tanh", "tanh"],
    k=k,
    bias=False,
    generator=generator,
)
x_inp, x_out = dgcnn_model.in_out_tensors()

x_out = Conv1D(filters=16, kernel_size=sum(layer_sizes), strides=sum(layer_sizes))(x_out)
x_out = MaxPool1D(pool_size=2)(x_out)

x_out = Conv1D(filters=32, kernel_size=5, strides=1)(x_out)

x_out = Flatten()(x_out)

x_out = Dense(units=128, activation="relu")(x_out)
x_out = Dropout(rate=0.5)(x_out)

predictions = Dense(units=2, activation="sigmoid")(x_out)

model = Model(inputs=x_inp, outputs=predictions)

model.compile(
    optimizer=Adam(lr=0.005), loss=binary_crossentropy, metrics=["acc"])

train_graphs, test_graphs = model_selection.train_test_split(
    graph_labels, train_size=0.8, test_size=None, stratify=graph_labels)

gen = PaddedGraphGenerator(graphs=graphs)

train_gen = gen.flow(
    list(train_graphs.index - 1),
    targets=train_graphs.values,
    batch_size=20,
    symmetric_normalization=False,
)

test_gen = gen.flow(
    list(test_graphs.index - 1),
    targets=test_graphs.values,
    batch_size=20,
    symmetric_normalization=False,
)

epochs = 300
#es = EarlyStopping(monitor='val_acc', mode='max', verbose=1, patience=500)
history = model.fit(
    train_gen, epochs=epochs, verbose=1, validation_data=test_gen, shuffle=True,
)

test_metrics = model.evaluate(test_gen)
print("\nTest Set Metrics:")
for name, val in zip(model.metrics_names, test_metrics):
    print("\t{}: {:0.4f}".format(name, val))
y = model.predict(test_gen)
print(classification_report(np.array(test_graphs.values).flatten(), np.argmax(y, axis=-1)))
AsmGraph_prediction = pd.DataFrame()
AsmGraph_prediction["actual"] = np.array(test_graphs.values).flatten()
AsmGraph_prediction["prediction"] = np.argmax(y, axis=-1)
AsmGraph_prediction.to_csv('AsmGraph_prediction.csv')
