from ExplanationEvaluation.configs.selector import Selector
from ExplanationEvaluation.tasks.training import train_node, train_graph

import torch
import numpy as np

_dataset = 'vul'  # One of: bashapes, bacommunity, treecycles, treegrids, ba2motifs, mutag

# Parameters below should only be changed if you want to run any of the experiments in the supplementary
_folder = 'replication'  # One of: replication, batchnorm
_model = 'gnn' if _folder == 'replication' else 'ori'
# _model = 'pg'
# PGExplainer
config_path = f"./ExplanationEvaluation/configs/{_folder}/models/model_{_model}_{_dataset}.json"

config = Selector(config_path)
extension = (_folder == 'extension')
config = Selector(config_path).args

torch.manual_seed(config.model.seed)
torch.cuda.manual_seed(config.model.seed)
np.random.seed(config.model.seed)
_dataset = config.model.dataset
_explainer = config.model.paper

if _dataset[:3] == "syn":
    train_node(_dataset, _explainer, config.model)
elif _dataset == "ba2" or _dataset == "mutag" or _dataset == "vul":
    train_graph(_dataset, _explainer, config.model)
