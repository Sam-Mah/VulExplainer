# Convert the graph structure of a Json file to a CSV
import os.path
import re

import pandas as pd
import json

out_dir = "Output_Graph_CSVs"
path_f = "Output_embedding_withEdge"
curr_dir = os.path.join(os.getcwd(), path_f)

i = 0
for filename in os.listdir(curr_dir):
    if filename.endswith('.json'):
        file_name_slpt = re.split('\\.', filename)
        out_path = os.path.join(out_dir, file_name_slpt[0]+"_"+file_name_slpt[1])
        out_path = os.path.join(out_dir, str(i))
        os.makedirs(out_path)
        with open(os.path.join(curr_dir, filename), 'r') as f:
            df_nodes = pd.DataFrame(columns=['ID', 'Features', 'Src'])
            df_edges = pd.DataFrame(columns=['Source', 'Destination', 'Features'])
            data = json.load(f)
            for block in data:
                if(len(data[block]['src'])!= 0):
                    # Node DataFrame
                    # Node Features: (Operand Types + TFIDF Value)
                    node_feature = data[block]['features']
                    node_feature.append(data[block]['embedding'])
                    list_row = {'ID' : block, 'Features': node_feature, 'Src': data[block]['src']}
                    df_nodes = df_nodes.append(list_row, ignore_index=True)
                    for k, v in data[block]['call'].items():
                        # Edge DataFrame
                        # Source, Destination, Features
                        list_row_2 = {'Source': block, 'Destination': k, 'Features': v}
                        df_edges = df_edges.append(list_row_2, ignore_index=True)

            df_nodes.to_csv(os.path.join(out_path, filename + "_node.csv"))
            df_edges.to_csv(os.path.join(out_path, filename + "_edge.csv"))

    i += 1

print("Mission Accomplished")

