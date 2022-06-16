import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn import SAGEConv
from typing import Union, Tuple
from torch_geometric.typing import OptPairTensor, Adj, OptTensor, Size
from .message_passing import MessagePassingNew
from torch import Tensor
from torch.nn import Linear
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn.conv import MessagePassing
import math
torch.manual_seed(0)


class GraphSage(torch.nn.Module):
    def __init__(self, args):
        super(GraphSage, self).__init__()
        dim_size = args.dim_size
        dataset = args.dataset
        self.aug = args.aug
        if args.aug == False:
            self.conv1 = SAGEConv(dataset['num_node_features'], dim_size)
            self.conv2 = SAGEConv(dim_size, dataset['num_classes'])
        else:
            self.conv1 = SAGEConvNew(
                dataset['num_node_features'], dim_size)
            self.conv2 = SAGEConvNew(dim_size, dataset['num_classes'])

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        if self.aug == True:
            x, sigma1 = self.conv1(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, training=self.training)
            x, sigma2 = self.conv2(x, edge_index)

            return F.log_softmax(x, dim=1), [sigma1, sigma2]
        else:
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, training=self.training)
            x = self.conv2(x, edge_index)

            return F.log_softmax(x, dim=1)


class SAGEConvNew(MessagePassingNew):
    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, normalize: bool = False,
                 root_weight: bool = True,
                 bias: bool = True, negative_slope: float = 0.2, **kwargs):  # yapf: disable
        kwargs.setdefault('aggr', 'mean')
        super(SAGEConvNew, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.root_weight = root_weight
        self.negative_slope = negative_slope
        self.dim_size = 128
        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        if isinstance(in_channels, int):
            self.lin_1 = Linear(in_channels, out_channels, bias=False)
            self.lin_2 = self.lin_1
        else:
            self.lin_1 = Linear(in_channels[0], out_channels, False)
            self.lin_2 = Linear(in_channels[1], out_channels, False)

        self.att_l = Parameter(torch.Tensor(1, out_channels))
        self.att_r = Parameter(torch.Tensor(1, out_channels))

        if out_channels % 2 == 0:
            self.lin_p = Linear(in_channels[0], int(
                out_channels / 2), bias=bias)
            self.lin_n = Linear(in_channels[1], int(
                out_channels / 2), bias=bias)
        else:
            self.lin_p = Linear(in_channels[0], int(out_channels), bias=bias)
            self.lin_n = Linear(in_channels[1], int(out_channels), bias=bias)

        self.lin_l = Linear(in_channels[0], out_channels, bias=bias)  # changed
        if self.root_weight:
            self.lin_r = Linear(in_channels[1], out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.lin_1.weight)  # MLP -> MLP([W*h_i||W*h_j])
        glorot(self.lin_2.weight)
        glorot(self.lin_p.weight)
        glorot(self.lin_n.weight)
        glorot(self.att_l)  # W -> a = MLP([W*h_i||W*h_j])
        glorot(self.att_r)
        self.lin_l.reset_parameters()
        if self.root_weight:
            self.lin_r.reset_parameters()

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                size: Size = None) -> Tensor:
        """"""
        C = self.out_channels

        x_l: OptTensor = None
        x_r: OptTensor = None
        sigma_l: OptTensor = None
        sigma_r: OptTensor = None
        if isinstance(x, Tensor):
            assert x.dim() == 2, 'Static graphs not supported.'
            x_l = x_r = self.lin_1(x).view(-1, C)
            sigma_l = (x_l * self.att_l).sum(dim=-1)
            sigma_r = (x_r * self.att_r).sum(dim=-1)

        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        out, sigma = self.propagate(edge_index, x=x, sigma=(
            sigma_l, sigma_r), size=[None, None])
        out = self.lin_p(out)

        out_, _ = self.propagate(edge_index, x=x, sigma=(
            sigma_l, sigma_r), size=[None, None], previous=sigma)

        if self.out_channels % 2 == 0:
            out = torch.cat((out, self.lin_n(out_)), 1)
        else:
            out += self.lin_n(out_)

        x_r = x[1]
        if self.root_weight and x_r is not None:
            out += self.lin_r(x_r)

        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)

        return out, sigma

    def message(self, x_j: Tensor, x_i: OptTensor, sigma_i: Tensor, sigma_j: OptTensor) -> Tensor:
        sigma = sigma_j if sigma_i is None else sigma_j + sigma_i
        sigma = F.leaky_relu(sigma, self.negative_slope)
        sigmoid = nn.Sigmoid()
        sigma = sigmoid(sigma)

        return x_j * sigma.clone().view(-1, 1), sigma

    def message_negative(self, x_j: Tensor, x_i: OptTensor, sigma: Tensor) -> Tensor:
        return x_i * (1 - sigma.clone().view(-1, 1)), sigma

    def message_and_aggregate(self, adj_t: SparseTensor,
                              x: OptPairTensor) -> Tensor:
        adj_t = adj_t.set_value(None, layout=None)
        return matmul(adj_t, x[0], reduce=self.aggr)

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


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
