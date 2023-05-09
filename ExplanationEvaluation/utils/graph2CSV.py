import pandas as pd
import os
import json
import csv


def example2CSV(idx, edges, nodes):
    path_f = ""
    curr_dir = os.path.join("I:\XAI_Project\Datasets\Data_VulEx\Output_Graph_CSVs", path_f)
    df_nodes_out = pd.DataFrame(columns=['ID', 'Features', 'Src'])
    df_edges_out = pd.DataFrame(columns=['Source', 'Destination', 'Features'])

    # list_row = {'ID': block, 'Features': node_feature, 'Src': data[block]['src']}
    # list_row_2 = {'Source': block, 'Destination': k, 'Features': v}

    for dir in os.listdir(curr_dir):
        if int(dir) == idx:
            for filename in os.listdir(os.path.join(curr_dir, dir)):
                if filename.find("_edge.csv") != -1:
                    # with open(os.path.join(curr_dir, dir, filename), 'r') as csvfile:
                    df_edges = pd.read_csv(os.path.join(curr_dir, dir, filename), index_col=0, squeeze=True)
                    # csv_edge = df.to_dict()
                    # csv_edge = csv.reader(csvfile, delimiter=' ', quotechar='|')
                else:
                    df_nodes = pd.read_csv(os.path.join(curr_dir, dir, filename), index_col=0, squeeze=True)
                    # csv_node = df.to_dict()
                    # with open(os.path.join(curr_dir, dir, filename), 'r') as csvfile:
                    #     csv_node = csv.reader(csvfile, delimiter=' ', quotechar='|')
            break
    try:
        for k, v in nodes.items():
            # if v in df_nodes.loc[v,'ID']:
            list_row_node = {'ID': v, 'Features': df_nodes.at[k, 'Features'], 'Src': df_nodes.at[k, 'Src']}
            df_nodes_out = df_nodes_out.append(list_row_node, ignore_index=True)

        for src, dest in edges:
            list_row_edge = {'Source': df_nodes.at[src, 'ID'], 'Destination': df_nodes.at[dest, 'ID']}
            df_edges_out = df_edges_out.append(list_row_edge, ignore_index=True)

    except Exception as err:
        print(err)

    print(type(df_nodes_out))
    print(type(df_edges_out))
    df_nodes_out.drop_duplicates
    df_nodes_out.to_csv(filename+"_nodes.csv")
    df_edges_out.drop_duplicates()
    df_edges_out.to_csv(filename+"_edges.csv")

