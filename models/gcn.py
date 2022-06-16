import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.linear import Linear
from typing import Optional, Tuple
from torch_geometric.typing import Adj, OptTensor, Size

import math
from torch import Tensor
from torch.nn import Parameter
from torch_scatter import scatter_add
from torch_sparse import SparseTensor, matmul, fill_diag, sum as sparsesum, mul
from torch_geometric.nn.conv import MessagePassing
from .message_passing import MessagePassingNew
from torch_geometric.utils import add_remaining_self_loops, softmax
from torch_geometric.utils.num_nodes import maybe_num_nodes


class GCN(torch.nn.Module):
    def __init__(self, args):
        super(GCN, self).__init__()
        dim_size = args.dim_size
        dataset = args.dataset
        self.aug = args.aug
        if args.aug == True:
            self.conv1 = GCNConvNew(
                dataset['num_node_features'], dim_size)
            self.conv2 = GCNConvNew(
                dim_size, dataset['num_classes'])
        else:
            self.conv1 = GCNConv(
                dataset['num_node_features'], dim_size)
            self.conv2 = GCNConv(
                dim_size, dataset['num_classes'])

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


class GCNConvNew(MessagePassingNew):

    _cached_edge_index: Optional[Tuple[Tensor, Tensor]]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(self, in_channels: int, out_channels: int,
                 improved: bool = False, cached: bool = False,
                 add_self_loops: bool = True, normalize: bool = True,
                 bias: bool = True,
                 negative_slope: float = 0.2, **kwargs):

        kwargs.setdefault('aggr', 'add')
        super(GCNConvNew, self).__init__(node_dim=0, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize
        self.negative_slope = negative_slope
        # self.sim_out_channels = sim_out_channels = 2
        # self.att = Parameter(torch.Tensor(1, out_channels))
        if isinstance(in_channels, int):
            self.lin_l = Linear(in_channels, out_channels, bias=False)
            self.lin_r = self.lin_l
        else:
            self.lin_l = Linear(in_channels[0], out_channels, False)
            self.lin_r = Linear(in_channels[1], out_channels, False)

        self.att_l = Parameter(torch.Tensor(1, out_channels))
        self.att_r = Parameter(torch.Tensor(1, out_channels))

        self._cached_edge_index = None
        self._cached_adj_t = None

        self.weight_p = Parameter(torch.Tensor(in_channels, out_channels))
        self.weight_n = Parameter(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.lin_l.weight)  # MLP -> MLP([W*h_i||W*h_j])
        glorot(self.lin_r.weight)
        glorot(self.weight_p)
        glorot(self.weight_n)
        glorot(self.att_l)  # W -> a = MLP([W*h_i||W*h_j])
        glorot(self.att_r)
        zeros(self.bias)
        self._cached_edge_index = None
        self._cached_adj_t = None

    def forward(self, x: Tensor, edge_index: Adj, edge_weight: OptTensor = None) -> Tensor:
        # self.sigma: OptTensor = None
        C = self.out_channels

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
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache

        x_p = x @ self.weight_p
        x_n = x @ self.weight_n

        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        out, sigma = self.propagate(edge_index, edge_weight=edge_weight, x=(x_p, x_n),
                                    sigma=(sigma_l, sigma_r),
                                    size=[None, None])

        if self.bias is not None:
            out += self.bias

        return out, sigma

    def message(self, x_j: Tensor, x_i: OptTensor, edge_weight: OptTensor, sigma_i: Tensor, sigma_j: OptTensor, index: Tensor) -> Tensor:
        sigma = sigma_j if sigma_i is None else sigma_j + sigma_i
        # sigma = sigma_j.sum(dim=-1) if sigma_i is None else sigma_j.sum(dim=-1) + sigma_i.sum(dim=-1)
        sigma = F.leaky_relu(sigma, self.negative_slope)

        sigmoid = nn.Sigmoid()
        sigma = sigmoid(sigma)
        # sigma = sigma[:, 0] >= sigma[:, 1]
        # sigma = softmax(sigma, index)
        return edge_weight.view(-1, 1) * x_j * sigma.clone().view(-1, 1) + edge_weight.view(-1, 1) * x_i * (1 - sigma.clone().view(-1, 1)), sigma

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


class GCNConv(MessagePassing):
    _cached_edge_index: Optional[Tuple[Tensor, Tensor]]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(self, in_channels: int, out_channels: int,
                 improved: bool = False, cached: bool = False,
                 add_self_loops: bool = True, normalize: bool = True,
                 bias: bool = True, **kwargs):

        kwargs.setdefault('aggr', 'add')
        super(GCNConv, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize

        self._cached_edge_index = None
        self._cached_adj_t = None

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)
        self._cached_edge_index = None
        self._cached_adj_t = None

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        """"""

        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache

        x = x @ self.weight

        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                             size=None)

        if self.bias is not None:
            out += self.bias

        return out

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)

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


@torch.jit._overload
def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, dtype=None):

    fill_value = 2. if improved else 1.

    if isinstance(edge_index, SparseTensor):
        adj_t = edge_index
        if not adj_t.has_value():
            adj_t = adj_t.fill_value(1., dtype=dtype)
        if add_self_loops:
            adj_t = fill_diag(adj_t, fill_value)
        deg = sparsesum(adj_t, dim=1)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
        adj_t = mul(adj_t, deg_inv_sqrt.view(-1, 1))
        adj_t = mul(adj_t, deg_inv_sqrt.view(1, -1))
        return adj_t

    else:
        num_nodes = maybe_num_nodes(edge_index, num_nodes)

        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)

        if add_self_loops:
            edge_index, tmp_edge_weight = add_remaining_self_loops(
                edge_index, edge_weight, fill_value, num_nodes)
            assert tmp_edge_weight is not None
            edge_weight = tmp_edge_weight

        row, col = edge_index[0], edge_index[1]
        deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
