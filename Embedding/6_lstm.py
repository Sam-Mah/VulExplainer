# NOTE: This is only a starter template. Wherever additional changes are required, please feel free modify/update.

import pickle
import os
import pandas as pd
from nltk import PorterStemmer
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
import itertools

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
import re
import matplotlib.pyplot as plt
import string
import json
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torchtext as tt
from sklearn.metrics import accuracy_score, f1_score, precision_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import torch.optim as optim
import time


# TODO: Feel free to improve the model
class Vulnerability_Detection(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_class, num_layers, word_embeddings, bidirectional):
        # Constructor
        super(Vulnerability_Detection, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.embedding.weight = nn.Parameter(word_embeddings, requires_grad=False)
        self.lstm = nn.LSTM(input_size=embed_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            bidirectional=bidirectional,
                           # dropout=dropout,
                           batch_first=True
                            )
        self.fc = nn.Linear(hidden_size*2, num_class)
        # self.act = nn.Sigmoid()
        self.act = nn.Softmax(dim=1)

    def forward(self, text, text_lengths):
        embeddings = self.embedding(text)
        # packed sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embeddings, text_lengths, batch_first=True)

        packed_output, (hidden, cell) = self.lstm(packed_embedded)

        # Concat the final forward and backward hidden state
        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)

        # hidden = [batch size, hid dim * num directions]
        dense_outputs = self.fc(hidden)

        # Final activation function
        out = self.act(dense_outputs)

        # Classic LSTM
        # embeddings = self.embedding(text)
        # hidden_out = self.lstm(embeddings)
        # dense_output = self.fc(hidden_out[0])
        # output = self.act(dense_output)
        # # output = output[:, -1]
        # out = torch.mean(output, 1)
        return out
# Step #0: Load data
def load_data(path: str) -> list:
    """Load Pickle files"""

    start_time = time.time()
    with open(path, 'rb') as f:
        data_list = pickle.load(f)
    return data_list
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Step #1: Analyse data
def preprocess_data(path_f) -> None:
    # Opening JSON file
    curr_dir = os.path.join(os.getcwd(), path_f)
    substring = "good"
    data_out = []
    for filename in os.listdir(curr_dir):
        with open(os.path.join(curr_dir, filename), 'r') as f:
            # Returns JSON object as
            data = json.load(f)
            # Iterating through the json
            src_list = []
            dict = {}
            for x in data.keys():
                src_list.append(data[x]['src'])

            # src_list1 = list(itertools.chain.from_iterable(src_list))
            flat_src = list(itertools.chain.from_iterable(src_list))
            src_str = ' '.join(flat_src)
            dict['src'] = src_str
            if substring in filename:
                dict['label'] = 'good'
            else:
                dict['label'] = 'bad'
            data_out.append(dict)

    training_data, testing_data = train_test_split(data_out, test_size=0.3, random_state=25)

    print(f"No. of training examples: {len(training_data)}")
    print(f"No. of testing examples: {len(testing_data)}")

    return training_data, testing_data
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Step #2: Define data fields
def data_fields() -> dict:
    # Type of fields on Data
    SRC = tt.legacy.data.Field(sequential=True,
                                    batch_first=True,
                                     init_token='<sos>',
                                     eos_token='<eos>',
                                     lower=True,
                                     stop_words=stop_words,  # Remove English stop words
                                     tokenize=tt.legacy.data.utils.get_tokenizer("basic_english"))
    LABEL = tt.legacy.data.Field(sequential=False,
                                      use_vocab=False,
                                      unk_token=None,
                                      is_target=True)

    fields = [('src', SRC), ('label', LABEL)]

    return fields, SRC, LABEL

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Step #2: Clean data
def data_clean(data: list, fields: dict) -> list:
    """A data cleaning routine."""

    clean_data = []
    for curr_data in data:
        #Tokenize the data
        tokenized_data = tt.legacy.data.Example.fromlist(list(curr_data.values()), fields)
        clean_data.append(tokenized_data)
    return clean_data
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Step #2: Prepare data
def data_prepare(data: list, fields: dict, val_percent: int) -> list:
    """A data preparation routine."""

    clean_train, clean_val = tt.legacy.data.Dataset(data, fields).split(split_ratio=val_percent)

    return clean_train, clean_val

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Step #3: Extract features
def extract_features(X_train, X_valid, SRC: tt.legacy.data.Field, LABEL: tt.legacy.data.Field,
                     batch_s) :
    train_iter, val_iter = [], []
    if X_train:
        #Initilize with glove embeddings
        SRC.build_vocab(X_train, vectors="glove.6B.100d")
        LABEL.build_vocab(X_train)
        train_iter = tt.legacy.data.BucketIterator(X_train, batch_size=batch_s, sort_key=lambda x: len(x.src),
                                                   device=device, sort=True, sort_within_batch=True)

    if X_valid:
        val_iter = tt.legacy.data.BucketIterator(X_valid, batch_size=batch_s, sort_key=lambda x: len(x.src),
                                                 device=device, sort=True, sort_within_batch=True)

    print(list(SRC.vocab.stoi.items()))
    # No. of unique tokens in text
    print("Size of SRC vocabulary:", len(SRC.vocab))

    # No. of unique tokens in label
    print("Size of LABEL vocabulary:", len(LABEL.vocab))

    # Commonly used words
    print("Commonly used words:", SRC.vocab.freqs.most_common(10))

    # Word dictionary
    print(LABEL.vocab.stoi)

    return train_iter, val_iter, SRC, LABEL
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# define accuracy metric
def binary_accuracy(preds, y):
    # round predictions to the closest integer
    if (torch.argmax(preds)==y):
        correct=1
    else:
        correct=0
    return correct
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
def one_hot_vector(label, num_class):
        # Get the actual labels and return one-hot vectors
        st = np.zeros((num_class))
        st[label] = 1
        return st
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Step #4: Train model
def train_model(classification_model: Vulnerability_Detection, SRC: tt.legacy.data.Field, LABEL: tt.legacy.data.Field, train_iter: tt.legacy.data.BucketIterator, optimizer,
                loss_func, num_class) :
    """Create a training loop"""
    # Initialize every epoch
    epoch_loss = 0
    epoch_acc = 0

    # Set the model in training phase
    classification_model.train()
    train_iter.create_batches()
    for batch in train_iter.batches:
        # Resets the gradients after every batch
        optimizer.zero_grad()
        batch_loss = 0
        batch_acc = 0
        for data_point in batch:
            x = data_point.src
            # Convert to integer sequence
            indexed = [SRC.vocab.stoi[t] for t in x]
            # Compute no. of words
            length = [len(indexed)]
            # Convert to tensor
            tensor = torch.LongTensor(indexed).to(device)
            tensor = tensor.unsqueeze(1).T
            length_tensor = torch.LongTensor(length)
            y = LABEL.vocab.stoi[data_point.label]
            # Convert to 1d tensor
            predictions = classification_model(tensor, length_tensor).squeeze()
            y = torch.LongTensor([y])
            predictions = torch.reshape(predictions, (1, num_class))
            loss = loss_func(predictions, y)
            acc = binary_accuracy(predictions, y)
            # Backpropage the loss and compute the gradients
            loss.backward()
            # Update the weights
            optimizer.step()
            # Keep track of loss and accuracy of each batch
            batch_loss += loss.item()
            batch_acc += acc
        # keep track of loss and accuracy of each epoch
        epoch_loss += (batch_loss/len(batch))
        epoch_acc += (batch_acc/len(batch))

    return classification_model, epoch_loss / len(train_iter), epoch_acc / len(train_iter)

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
def evaluate_model(classification_model: Vulnerability_Detection, SRC: tt.legacy.data.Field, LABEL: tt.legacy.data.Field, val_iter: tt.legacy.data.BucketIterator,
                   loss_func, num_class) :
    # initialize every epoch
    epoch_loss = 0
    epoch_acc = 0

    classification_model.eval()
    val_iter.create_batches()
    # Deactivates autograd
    with torch.no_grad():
        for batch in val_iter.batches:
            batch_loss = 0
            batch_acc = 0
            for data_point in batch:
                x = data_point.src
                # Convert to integer sequence
                indexed = [SRC.vocab.stoi[t] for t in x]
                # Compute no. of words
                length = [len(indexed)]
                # Convert to tensor
                tensor = torch.LongTensor(indexed).to(device)
                tensor = tensor.unsqueeze(1).T
                # Convert to tensor
                length_tensor = torch.LongTensor(length)
                y = LABEL.vocab.stoi[data_point.label]
                # Convert to 1d tensor
                predictions = classification_model(tensor, length_tensor).squeeze()
                y = torch.LongTensor([y])
                predictions = torch.reshape(predictions, (1, num_class))
                loss = loss_func(predictions, y)
                acc = binary_accuracy(predictions, y)
                # keep track of loss and accuracy of each batch
                batch_loss += loss.item()
                batch_acc += acc
            # keep track of loss and accuracy of each epoch
            epoch_loss += (batch_loss / len(batch))
            epoch_acc += (batch_acc / len(batch))

    return classification_model, epoch_loss / len(val_iter), epoch_acc / len(val_iter)

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Step #5: Stand-alone Test data & Compute metrics
def compute_metrics(classification_model: Vulnerability_Detection, test_data: list, SRC: tt.legacy.data.Field, LABEL: tt.legacy.data.Field, num_class) -> None:
    test_iter = tt.legacy.data.BucketIterator(test_data, batch_size=len(test_data), sort_key=lambda x: len(x.src),
                                               device=device, sort=True, sort_within_batch=True)

    classification_model.eval()
    test_iter.create_batches()
    predictions = []
    true_labels = []
    # For the whole test samples
    for sample in test_iter.batches:
        for data_point in sample:
            x = data_point.src
            # Convert to integer sequence
            indexed = [SRC.vocab.stoi[t] for t in x]
            # Compute no. of words
            length = [len(indexed)]
            # convert to tensor
            tensor = torch.LongTensor(indexed).to(device)
            tensor = tensor.unsqueeze(1).T
            # Convert to tensor
            length_tensor = torch.LongTensor(length)
            y = LABEL.vocab.stoi[data_point.label]
            y = torch.FloatTensor(one_hot_vector(y, num_class))
            true_labels.append(y)
            # Convert to 1d tensor
            prediction = classification_model(tensor, length_tensor).squeeze()
            predictions.append(prediction)

    # Compute Performance Metrics
    lbls = [torch.argmax(t) for t in true_labels]
    preds = [torch.argmax(t) for t in predictions]
    ACC = accuracy_score(lbls, preds)
    PR = precision_score(lbls, preds, average='weighted',  labels=np.unique(preds))
    F1 = f1_score(lbls, preds, average='weighted', labels=np.unique(preds))

    # Save metrics into a CSV file
    data_pd = [['Accuracy', ACC], ['Precision', PR], ['F1_Score', F1]]
    df = pd.DataFrame(data_pd, columns=['Measure', 'Percentage'])
    np.savetxt('./Metric_Values_Test.csv', df, delimiter=',', fmt='%s')

    return None

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
def main(data_path: str) -> None:
    # Define hyperparameters
    embedding_dim = 100
    num_hidden_nodes = 25
    num_classes = 2
    n_layers = 1
    bidirection = True
    # dropout = 0.2
    N_EPOCHS = 20
    # LOSS_THRESH = 0.001
    batch_size = 25
    # l_rate = [0.001, 0.005, 0.01, 0.1]
    l_rate = 0.01

    ### Perform the following steps and complete the code

    train_data, test_data = preprocess_data(data_path)
    ### Step #0: Load data
    # train_data = load_data(train_data)

    ### Step #1: Analyse data
    # analyse_data(train_data)

    ### Step #2: Clean and prepare data
    fields, SRC, LABEL = data_fields()
    train_data = data_clean(train_data, fields)

    train_ds, val_ds = data_prepare(train_data, fields, val_percent=0.5)

    ### Step #3: Extract features
    train_iter, val_iter, SRC, LABEL = extract_features(train_ds, val_ds, SRC, LABEL, batch_size)
    word_embeds = SRC.vocab.vectors
    vocab_size = len(SRC.vocab.stoi)


    ### Step #4: Train model

    # Initilize the model
    classification_model = Vulnerability_Detection(vocab_size=vocab_size, embed_size=embedding_dim,
                                           hidden_size=num_hidden_nodes, num_class=num_classes,
                                           num_layers=n_layers,word_embeddings=word_embeds, bidirectional=bidirection)

    # Define optimizer and loss function
    optimizer = optim.Adam(classification_model.parameters(), lr=l_rate)
    # loss_func = nn.BCELoss()
    loss_func = nn.CrossEntropyLoss()

    best_valid_loss = float('inf')

    for epoch in range(N_EPOCHS):
        # train the model
        classification_model, train_loss, train_acc = train_model(classification_model, SRC, LABEL, train_iter, optimizer,
                                                                  loss_func, num_classes)
        # Evaluate the model
        classification_model, valid_loss, valid_acc = evaluate_model(classification_model,  SRC, LABEL, val_iter, loss_func, num_classes)

        # Save the best model
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(classification_model.state_dict(), 'saved_weights.pt')

        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')

    # Step #5: Stand-alone Test data & Compute metrics
    test_data = data_clean(test_data, fields)
    compute_metrics(classification_model, test_data, SRC, LABEL, num_classes)

    return 0
    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
if __name__ == "__main__":
    data_path = ""
    main(data_path)
    print("Mission Accomplished")
