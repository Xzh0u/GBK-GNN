import torch
from torch_geometric.datasets import *
from data_loader.datasets_source_info import *
from utils.statistic import compute_smoothness, split_dataset


class DatasetSelection:
    def __init__(self, source_name, dataset_name, split, task_type="NodeClasification"):
        task2str = {"NodeClasification": "node_",
                    "EdgeClasification": "edge_",
                    "GraphClasification": "graph_"}
        dataset_dir = data_dir + source_name + "/"
        if (dataset_name == ""):
            dataset = dataset_sources_dic[source_name](dataset_dir)
        else:
            dataset = dataset_sources_dic[source_name](
                dataset_dir, dataset_name)

        self.dataset = {"graph": []}
        smoothness = num_class = num_node = num_edge = 0
        for i in range(len(dataset)):
            num_node += dataset[i].x.shape[0]
            num_edge += dataset[i].edge_index.shape[1]
            if (dataset[i].y.shape == torch.Size([1]) and task_type == "NodeClasification"):
                dataset[i].y.data = dataset[i].x.argmax(dim=1)
                num_class = max(dataset[i].x.shape[1], num_class)
            else:
                if (len(dataset[i].y.shape) != 1):
                    num_class = max(dataset[i].y.shape[1], num_class)
                    dataset[i].y.data = dataset[i].y.argmax(dim=1)
                else:
                    num_class = max(max(dataset[i].y + 1), num_class)
            if not hasattr(dataset[i], 'train_mask'):
                
                data_tmp = dataset[i]
                data_tmp.train_mask, data_tmp.test_mask, data_tmp.val_mask = split_dataset(
                    dataset[i].x.shape[0], split)
                self.dataset["graph"].append(data_tmp)
                
            self.dataset["graph"].append(dataset[i])
            smoothness += compute_smoothness(dataset[i]) * \
                dataset[i].x.shape[0]

        if (type(num_class) != type(1)):
            num_class = num_class.numpy()

        smoothness /= num_node
        self.dataset['num_node'] = num_node
        self.dataset['num_edge'] = num_edge
        self.dataset['num_node_features'] = dataset[0].x.shape[1]
        self.dataset['smoothness'] = smoothness
        self.dataset['num_' + task2str[task_type] + 'classes'] = num_class
        self.dataset['num_classes'] = num_class

    def get_dataset(self):
        return self.dataset
