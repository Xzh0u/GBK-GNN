from torch_geometric.utils.convert import to_networkx
from typing import List
import numpy as np
import operator
from torch.nn import CosineSimilarity
import torch
import random

def compute_cosine_similarity(dataset, edge_index, attri):
    cos = CosineSimilarity(dim=0, eps=1e-6)
    similaity_list = []
    for i in range(len(dataset['graph'])):
        if attri == "label":
            attri = dataset['graph'][i].y
        elif attri == "feature":
            attri = dataset['graph'][i].x
        for item in edge_index.transpose(1, 0):
            similarity = cos(attri[item[0]].float(), attri[item[1]].float())
            similaity_list.append(float(similarity))
    return similaity_list


def compute_parameter(model):
    Total_params = 0
    Trainable_params = 0
    NonTrainable_params = 0

    for param in model.parameters():
        mulValue = np.prod(param.size())
        Total_params += mulValue
        if param.requires_grad:
            Trainable_params += mulValue
        else:
            NonTrainable_params += mulValue

    print(f'Total params: {Total_params}')
    print(f'Trainable params: {Trainable_params}')
    print(f'Non-trainable params: {NonTrainable_params}')


def compute_label_percentage(li: List):
    dict = {}
    for key in li:
        dict[key] = dict.get(key, 0) + 1
    return dict


def compute_smoothness(dataset):
    smooth_edges = 0
    G = to_networkx(dataset)
    adj_dict = dict(G.adj.items())
    for i in range(len(G.nodes)):
        if len(adj_dict[i]) == 0:
            continue
        node_labels = []
        for key in dict(adj_dict[i]):
            node_labels.append(dataset.y[key])
        precent_dict = compute_label_percentage(node_labels)
        prop_max_label = max(precent_dict.items(),
                             key=operator.itemgetter(1))[0]
        if dataset.y[i].equal(prop_max_label):
            smooth_edges += 1

    return smooth_edges / len(G.nodes)


def split_dataset(num_nodes, p=[0.6, 0.2, 0.2]):
    train_mask_list = [True] + [False] * (num_nodes - 1)  # at least one sample
    test_mask_list = [True] + [False] * (num_nodes - 1)
    val_mask_list = [True] + [False] * (num_nodes - 1)
    for i in range(num_nodes - 3):
        p_now = random.uniform(0, 1)
        for j in range(len(p)):
            if (p_now <= p[j]):
                if (j == 0):
                    train_mask_list[i + 1] = True
                elif (j == 1):
                    test_mask_list[i + 1] = True
                elif (j == 2):
                    val_mask_list[i + 1] = True
                break
            p_now -= p[j]
    return torch.tensor(train_mask_list), torch.tensor(test_mask_list), torch.tensor(val_mask_list)
