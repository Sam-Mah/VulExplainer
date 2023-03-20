import numpy as np
import random
import pickle as pkl
import os

# graph = list()
# features = list()
# labels = list()
d1 =25
num_samples = 1000
# for i in range(100):
    # importing the random module
# d1 = random.randint(10, 20)
graph = np.random.rand(num_samples, d1,d1)
# graph.append(arr1)
features = np.random.rand(num_samples, d1, 10)
# features.append(arr2)
labels = np.random.rand(num_samples,2)
# labels.append(arr3)

out_list = [graph, features, labels]
dir_path = os.path.dirname(os.path.realpath(__file__))
path = dir_path + '/ExplanationEvaluation/datasets/pkls/' + "garbage_dataset" + '.pkl'
open_file = open(path, "wb")
pkl.dump(out_list, open_file)
open_file.close()
