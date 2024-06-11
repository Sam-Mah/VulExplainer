import os
import json
import torch
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
import numpy as np

# Load BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('prajjwal1/bert-mini')
model = BertModel.from_pretrained('prajjwal1/bert-mini')


# Function to generate embeddings using BERT
def generate_embeddings(text):
    tokens = tokenizer.encode(text[:511], add_special_tokens=True)
    input_ids = torch.tensor([tokens])

    with torch.no_grad():
        outputs = model(input_ids)
        embeddings = outputs.last_hidden_state[0, 0, :]
        # embeddings = outputs[0].squeeze(0)

    return embeddings.tolist()


# Directory path containing JSON files
# directory = 'E:\saqib_work1\\data\\Sam\\JSON_BERT\\Output_embedding_withEdge'
# directory = 'I:\XAI_Project\Datasets\Data_VulEx\Output_embedding_withEdge_NDSS_small_Bert'
directory = 'I:\XAI_Project\Datasets\Data_VulEx\Output_embedding_withEdge_NDSS _small_Bert'
# Get a list of JSON file paths in the directory
json_files = [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith('.json')]

# Track progress using tqdm
progress_bar = tqdm(total=len(json_files))

# Iterate over JSON files, generate embeddings, and update JSON data
for json_file in json_files:
    with open(json_file, 'r') as file:
        data = json.load(file)

    for key in data.keys():
        src_block = data[key]['src']
        src_embeddings = generate_embeddings(' '.join(src_block))
        data[key]['embedding'] = list(np.hstack((src_embeddings, data[key]['embedding'])))

    with open(json_file, 'w') as file:
        json.dump(data, file)

    progress_bar.update(1)

progress_bar.close()
