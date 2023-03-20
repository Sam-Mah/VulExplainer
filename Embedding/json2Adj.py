# Convert the graph structure of a Json file to a CSV
import collections
import math
import os.path
import re
import numpy as np
import pandas as pd
import json
import pickle as pkl

out_dir = "Output_Graph_Adj"
path_f = "Output_embedding_withEdge"
curr_dir = os.path.join(os.getcwd(), path_f)

# Get the average number of blocks in files
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
max_len = 0
sum = 0
for filename in os.listdir(curr_dir):
    if filename.endswith('.json'):
        with open(os.path.join(curr_dir, filename), 'r') as f:
            data = json.load(f)
            sum += len(data)
            if (len(data) > max_len):
                max_len = len(data)
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
print(max_len)
num_samples = len(os.listdir(curr_dir))
d1 = math.ceil(sum / num_samples)
# d1 = max_len
len_fet_vec = 9
num_class = 2

graph = np.zeros((num_samples, d1, d1))
features = np.zeros((num_samples, d1, len_fet_vec))
labels = np.zeros((num_samples, num_class))
blk_hash_lst = []
file_lst = []
i = 0
substring = "good"
good_lbl = np.array([0, 1])
bad_lbl = np.array([1, 0])
for filename in os.listdir(curr_dir):
    if filename.endswith('.json'):
        # file_name_slpt = re.split('..', filename)
        # out_path = os.path.join(out_dir, file_name_slpt[0]+"_"+file_name_slpt[1])
        out_path = os.path.join(out_dir, filename)
        # os.makedirs(out_path)
        with open(os.path.join(curr_dir, filename), 'r') as f:
            # df_nodes = pd.DataFrame(columns=['ID', 'Features', 'Src'])
            # df_edges = pd.DataFrame(columns=['Source', 'Destination', 'Features'])
            data = json.load(f)
            blk_dict = {}
            j = 0
            for block in data:
                if (len(data[block]['src']) != 0 and j < d1):
                    # Node DataFrame
                    # Node Features: (Operand Types + TFIDF Value)
                    features[i, j] = np.hstack((data[block]['features'], data[block]['embedding']))
                    blk_dict[j] = block
                    # node_feature.append(data[block]['embedding'])
                    # list_row = {'ID': block, 'Features': node_feature, 'Src': data[block]['src']}
                    # df_nodes = df_nodes.append(list_row, ignore_index=True)
                    for k, v in data[block]['call'].items():
                        # Edge DataFrame
                        # Source, Destination, Features
                        # list_row_2 = {'Source': block, 'Destination': k, 'Features': v}
                        # df_edges = df_edges.append(list_row_2, ignore_index=True)
                        # index of block k in data[block]
                        print(type(data[block]))
                        ordered_data = collections.OrderedDict(data)
                        print(ordered_data.keys())
                        indx = 0
                        for key, value in ordered_data.items():
                            if (key == k) and (indx < d1):
                                # To make the adjacency graph symmetrical
                                graph[i, j, indx] = 1
                                graph[i, indx, j] = 1
                                break
                            indx += 1
                        # t = ordered_data.keys().index(k)
                # df_nodes.to_csv(os.path.join(out_path, filename + "_node.csv"))
                j += 1
            if substring in filename:
                labels[i] = good_lbl
            else:
                labels[i] = bad_lbl

        blk_hash_lst.append(blk_dict)
        file_lst.append(filename)
        i += 1
print("Mission Accomplished")
out_list = [graph, features, labels, blk_hash_lst, file_lst]
dir_path = os.path.dirname(os.path.realpath(__file__))
path = dir_path + "Vulnerability" + '.pkl'
open_file = open(path, "wb")
pkl.dump(out_list, open_file)
open_file.close()
