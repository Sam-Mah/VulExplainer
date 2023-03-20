import json
import pandas as pd
import os
from stellargraph import StellarGraph

'''
def is_embedding(file_directory, file_name):
    file_name = "\\" + file_name
    with open(file_directory + file_name, 'r') as f:
        data = json.load(f)
    if "embedding" in data:
        print('0')
        return 0
    else:
        print('1')
        return 1
'''
vocab_size = 10

def create_graph(file_directory, file_name):
    with open(os.path.join(file_directory, file_name), 'r') as f:
        data = json.load(f)

    #print("------------creating features---------------------")
    embedding_data = pd.DataFrame(columns=['node'] + [i for i in range(vocab_size)])
    for key in list(data.keys()):
        new_row = pd.DataFrame([[key] + data[key]["embedding_doc2vec_without_sum"]], columns=['node'] + [i for i in range(vocab_size)])
        embedding_data = pd.concat([embedding_data, new_row])
    square_node_data = embedding_data.set_index('node')
    #print(square_node_data)

    #print("------------creating edges---------------------")
    edges_data = pd.DataFrame(columns=['source', 'target'])
    for caller in list(data.keys()):
        callee_list = data[caller]["call"]
        for callee in callee_list:
            new_row = pd.DataFrame([[caller, str(callee)]], columns=['source', 'target'])
            edges_data = pd.concat([edges_data, new_row])
    edges_data = edges_data.reset_index(drop=True)
    #print(edges_data)

    #print("------------creating nodes---------------------")
    graph_ = StellarGraph(square_node_data, edges_data)
    return graph_

#file_directory = 'E:\saqib_work1\data\Ashita_Data\sample_files_samaneh'
#file_name = '\cwe119_gcc_32_1157_CWE121_Stack_Based_Buffer_Overflow__CWE193_char_alloca_memcpy_51_bad.o.tmp0.json_graph.json'

#graph = create_graph(file_directory, file_name)
#print(graph.info())