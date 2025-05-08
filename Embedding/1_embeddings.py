from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.naive_bayes import GaussianNB

from sklearn.pipeline import Pipeline
import os
import json
import itertools
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn import preprocessing


def preprocess_data(path_f):
    # Opening JSON file
    curr_dir = os.path.join(os.getcwd(), path_f)
    substring = "good"
    data_out = []
    vocabulary = []
    for filename in os.listdir(curr_dir):
        with open(os.path.join(curr_dir, filename), 'r') as f:
            # Returns JSON object as
            data = json.load(f)
            # Iterating through the json
            src_list = []
            dict = {}
            for x in data.keys():
                if (data[x]['src']):
                    blk_src = data[x]['src']
                    src_list.append(blk_src)
                    # flt_blk_src = list(itertools.chain.from_iterable(blk_src))
                    vocabulary.append(' '.join(blk_src))
                    print('vocabulary is:')
                    print(blk_src)
            flat_src = list(itertools.chain.from_iterable(src_list))
            # flat_src = list(itertools.chain.from_iterable(src_list1))
            # src_str = ' '.join(flat_src)
            dict['src'] = flat_src
            if substring in filename:
                dict['label'] = 'good'
            else:
                dict['label'] = 'bad'
            print(dict)
            data_out.append(dict)

    unique_vocab = np.unique(np.array(vocabulary))

    return data_out, unique_vocab


def prepare_date(data):
    training_data, testing_data = train_test_split(data, test_size=0.3, random_state=42)

    print(f"No. of training examples: {len(training_data)}")
    print(f"No. of testing examples: {len(testing_data)}")

    return training_data, testing_data

def classify(train_data, test_data, vocab) -> None:
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    for x in train_data:
        x_train.append(' '.join(x['src']))
        y_train.append(' '.join(x['label']))

    for x in test_data:
        x_test.append(' '.join(x['src']))
        y_test.append(' '.join(x['label']))

    tfidf = TfidfVectorizer(min_df=2, vocabulary=vocab)
    words = tfidf.get_feature_names()
    print(len(words))
    # model = LogisticRegressionCV()
    model = RandomForestClassifier(max_depth=300)  # , random_state=None)
    # model = GradientBoostingClassifier(random_state=None)
    # model = MLPClassifier(hidden_layer_sizes= (100,10),solver='adam', random_state=None)
    # clf = DecisionTreeClassifier(max_depth=100, random_state=None)
    # model = svm.SVC(gamma='scale', decision_function_shape='ovo')
    # model = KNeighborsClassifier(n_neighbors=3)
    # model = GaussianNB()
    x_transformed = tfidf.fit_transform(x_train)
    model.fit(x_transformed, y_train)
    x_test_transformed = tfidf.transform(x_test)
    train_score = model.score(x_transformed, y_train)
    test_score = model.score(x_test_transformed, y_test)


    # return accuracy_score(y, model.predict(X), sample_weight=sample_weight)
    ACC = accuracy_score(y_test, model.predict(x_test_transformed))
    PR = precision_score(y_test, model.predict(x_test_transformed), average='weighted')
    F1 = f1_score(y_test, model.predict(x_test_transformed), average='weighted')
    RC = recall_score(y_test, model.predict(x_test_transformed), average='weighted')

    result_base = "Train Accuracy: {train_acc:<.1%}  Test Accuracy: {test_acc:<.1%}"
    result = result_base.format(train_acc=train_score, test_acc=test_score)
    print(
        f"final train_acc:{train_score}, val_acc: {train_score}, test_acc: {ACC}, test_pr: {PR}, test_f1: {F1}, test_rc: {RC}")

    print(result)

def embedding(vocab, data_path, data):
    output_path = r''
    curr_dir = os.path.join(os.getcwd(), data_path)
    model = LogisticRegressionCV()
    x_data = []
    y_data = []

    for x in data:
        x_data.append(' '.join(x['src']))
        y_data.append(' '.join(x['label']))

    tfidf = TfidfVectorizer(min_df=2, vocabulary=vocab)
    x_transformed = tfidf.fit_transform(x_data)
    model.fit(x_transformed, y_data)
    x_transformed = x_transformed.toarray()

    coef = model.coef_.reshape(-1)
    words = tfidf.get_feature_names_out()
    indx = np.argmax(coef)
    print(words[indx])

    i = 0
    for filename in os.listdir(curr_dir):
        with open(os.path.join(curr_dir, filename), 'r') as f:
            # Returns JSON object as
            data = json.load(f)
            # Iterating through the json
            for x in data.keys():
                if (data[x]['src']):
                    indx = list(vocab).index(' '.join(data[x]['src']))
                    data[x]["embedding"] = x_transformed[i][indx]
                    if (data[x]["embedding"]!=0.0):
                        print("non-zero embedding")
        i += 1

        with open(output_path + "\\" + filename + "_embedding.json", 'w') as outfile:
            json.dump(data, outfile)

def main(data_path: str):
    data, vocab = preprocess_data(data_path)
    # &&&&&&&&&&&&&&&&&&&&&&&Embeddings&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    embedding(vocab, data_path, data)
    # &&&&&&&&&&&&&&&&&&&&&&&Classification&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    # train_data, test_data = prepare_date(data)
    # classify(train_data, test_data, vocab)

if __name__ == '__main__':
    data_path = r''

    main(data_path)
    print("Mission Accomplished")
