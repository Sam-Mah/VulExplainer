import torch
import numpy as np

from ExplanationEvaluation.datasets.dataset_loaders import load_dataset
from ExplanationEvaluation.datasets.utils import adj_to_edge_index

graphs, features, labels, blk_hash_lst, _, _, _ = load_dataset('vul')
# Resize th graph
min = np.shape(graphs[0])[1]
for x in graphs:
    dim = np.shape(x)[1]
    if dim <= min:
        min = dim
for i in range(len(graphs)):
    graphs[i] = np.resize(graphs[i], (2, min))

graphs = torch.tensor(graphs)
features = torch.tensor(features)
labels = torch.tensor(labels)

# Overwrite these models with your own if you want to
# from ExplanationEvaluation.models.GNN_paper import NodeGCN
from ExplanationEvaluation.models.PG_paper import GraphGCN
# model = NodeGCN(10, 2)
# model = GraphGCN(9, 2)
model = GraphGCN(265, 2)
# path = "./ExplanationEvaluation/models/pretrained/GNN/syn1/best_model"
# checkpoint = torch.load(path)
# model.load_state_dict(checkpoint['model_state_dict'])

task = 'graph'
# task = 'node'

from ExplanationEvaluation.explainers.PGExplainer import PGExplainer

explainer = PGExplainer(model, graphs, features, task)

# indices show what part of the graph you want to explain
# indices = range(4001, 6000, 1)
indices = range(0, 120, 1)
# indices = range(2000, 3000, 1)
explainer.prepare(indices)
# A graph to explain
idx = indices[119]  # select a node to explain, this needs to be part of the list of indices
# file_name = file_list[idx]
graph, expl = explainer.explain(idx)
explainer.evaluate(indices)

from ExplanationEvaluation.utils.plotting import plot

with open('file.txt', 'w') as data:
    data.write(str(blk_hash_lst[idx]))

# Need to feed the edge indices
plot(graph, expl, labels, blk_hash_lst[idx], idx, -1, 100, 'vul', show=True)

from ExplanationEvaluation.datasets.ground_truth_loaders import load_dataset_ground_truth

explanation_labels, indices = load_dataset_ground_truth('vul')

from ExplanationEvaluation.evaluation.AUCEvaluation import AUCEvaluation
from ExplanationEvaluation.evaluation.EfficiencyEvaluation import EfficiencyEvluation

auc_evaluation = AUCEvaluation(task, explanation_labels, indices)
inference_eval = EfficiencyEvluation()

from ExplanationEvaluation.tasks.replication import run_experiment

auc, time = run_experiment(inference_eval, auc_evaluation, explainer, indices)

print(auc)
print(time)

for idx in indices:
    graph, expl = explainer.explain(idx)
    plot(graph, expl, labels, idx, 12, 100, 'vul', show=True)
