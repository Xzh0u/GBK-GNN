import torch
from torch.nn import Linear, Sigmoid
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.nn import GCN2Conv
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from typing import Optional, Tuple
from torch_geometric.typing import Adj, OptTensor

import math
import torch
from torch import Tensor
from torch.nn import Parameter
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm

from models.message_passing import MessagePassingNew


class GCN2(torch.nn.Module):
    def __init__(self, args, num_layers=64, alpha=0.1, theta=0.5,
                 shared_weights=True, dropout=0.0):
        super(GCN2, self).__init__()
        dim_size = args.dim_size
        dataset = args.dataset
        self.aug = args.aug
        self.lins = torch.nn.ModuleList()
        self.lins.append(Linear(dataset['num_node_features'], dim_size))
        self.lins.append(Linear(dim_size, dataset['num_classes']))

        self.convs = torch.nn.ModuleList()
        if args.aug == True:
            for layer in range(num_layers):
                self.convs.append(
                    GCN2ConvNew(dim_size, alpha, theta, layer + 1,
                                shared_weights, normalize=True))
        else:
            for layer in range(num_layers):
                self.convs.append(
                    GCN2Conv(dim_size, alpha, theta, layer + 1,
                             shared_weights, normalize=True))
        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, self.dropout, training=self.training)
        x = x_0 = self.lins[0](x).relu()
        sigma_list = []
        if self.aug == True:
            for conv in self.convs:
                x = F.dropout(x, self.dropout, training=self.training)
                x, sigma = conv(x, x_0, edge_index)
                sigma_list.append(sigma)
                x = x.relu()

            x = F.dropout(x, self.dropout, training=self.training)
            x = self.lins[1](x)

            return x.log_softmax(dim=-1), sigma_list

        else:
            for conv in self.convs:
                x = F.dropout(x, self.dropout, training=self.training)
                x = conv(x, x_0, edge_index)
                x = x.relu()

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.lins[1](x)

        return x.log_softmax(dim=-1)


class GCN2ConvNew(MessagePassingNew):

    _cached_edge_index: Optional[Tuple[Tensor, Tensor]]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(self, channels: int, alpha: float, theta: float = None,
                 layer: int = None, shared_weights: bool = True,
                 cached: bool = False, add_self_loops: bool = True,
                 normalize: bool = True, negative_slope: float = 0.2, **kwargs):

        kwargs.setdefault('aggr', 'add')
        super(GCN2ConvNew, self).__init__(**kwargs)

        self.channels = channels
        self.alpha = alpha
        self.beta = 1.
        if theta is not None or layer is not None:
            assert theta is not None and layer is not None
            self.beta = math.log(theta / layer + 1)
        self.cached = cached
        self.normalize = normalize
        self.add_self_loops = add_self_loops
        self.negative_slope = negative_slope

        self._cached_edge_index = None
        self._cached_adj_t = None

        self.lin_l = Linear(channels, channels, bias=False)
        self.lin_r = self.lin_l
        self.att_l = Parameter(torch.Tensor(1, channels))
        self.att_r = Parameter(torch.Tensor(1, channels))
        self.weight_p = Parameter(torch.Tensor(channels, channels))
        self.weight_n = Parameter(torch.Tensor(channels, channels))

        self.weight1 = Parameter(torch.Tensor(channels, channels))

        if shared_weights:
            self.register_parameter('weight2', None)
        else:
            self.weight2 = Parameter(torch.Tensor(channels, channels))

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.lin_l.weight)  # MLP -> MLP([W*h_i||W*h_j])
        glorot(self.lin_r.weight)
        glorot(self.att_l)  # W -> a = MLP([W*h_i||W*h_j])
        glorot(self.att_r)
        glorot(self.weight_p)
        glorot(self.weight_n)
        glorot(self.weight1)
        glorot(self.weight2)
        self._cached_edge_index = None
        self._cached_adj_t = None

    def forward(self, x: Tensor, x_0: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        """"""
        C = self.channels

        x_l: OptTensor = None
        x_r: OptTensor = None
        sigma_l: OptTensor = None
        sigma_r: OptTensor = None
        if isinstance(x, Tensor):
            assert x.dim() == 2, 'Static graphs not supported.'
            x_l = x_r = self.lin_l(x).view(-1, C)
            sigma_l = (x_l * self.att_l).sum(dim=-1)
            sigma_r = (x_r * self.att_r).sum(dim=-1)

        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim), False,
                        self.add_self_loops, dtype=x.dtype)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim), False,
                        self.add_self_loops, dtype=x.dtype)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache

        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        x_p = x @ self.weight_p
        x_n = x @ self.weight_n

        x, sigma = self.propagate(edge_index, x=(
            x_p, x_n), edge_weight=edge_weight, sigma=(sigma_l, sigma_r), size=[None, None])

        x.mul_(1 - self.alpha)
        x_0 = self.alpha * x_0[:x.size(0)]

        if self.weight2 is None:
            out = x.add_(x_0)
            out = torch.addmm(out, out, self.weight1, beta=1. - self.beta,
                              alpha=self.beta)
        else:
            out = torch.addmm(x, x, self.weight1, beta=1. - self.beta,
                              alpha=self.beta)
            out += torch.addmm(x_0, x_0, self.weight2, beta=1. - self.beta,
                               alpha=self.beta)

        return out, sigma

    def message(self, x_j: Tensor, x_i: OptTensor, edge_weight: Tensor, sigma_i: Tensor, sigma_j: OptTensor) -> Tensor:
        sigma = sigma_j if sigma_i is None else sigma_j + sigma_i
        sigma = F.leaky_relu(sigma, self.negative_slope)

        sigmoid = Sigmoid()
        sigma = sigmoid(sigma)

        return edge_weight.view(-1, 1) * x_j * sigma.clone().view(-1, 1) + edge_weight.view(-1, 1) * x_i * (1 - sigma.clone().view(-1, 1)), sigma

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self):
        return '{}({}, alpha={}, beta={})'.format(self.__class__.__name__,
                                                  self.channels, self.alpha,
                                                  self.beta)


def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)


def glorot_orthogonal(tensor, scale):
    if tensor is not None:
        torch.nn.init.orthogonal_(tensor.data)
        scale /= ((tensor.size(-2) + tensor.size(-1)) * tensor.var())
        tensor.data *= scale.sqrt()


def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)
