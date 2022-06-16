import torch
import torch.nn.functional as F
from torch.nn.modules.linear import Linear


class DNN(torch.nn.Module):
    def __init__(self, args):
        super(DNN, self).__init__()
        dim_size = args.dim_size
        dataset = args.dataset
        self.fc1 = Linear(dataset['num_node_features'], dim_size)
        self.fc2 = Linear(dim_size, dataset['num_classes'])

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)
