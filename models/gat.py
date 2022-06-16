import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from typing import Union, Tuple, Optional
from torch_geometric.typing import (OptPairTensor, Adj, Size, NoneType,
                                    OptTensor)
import math
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Parameter, Linear
from torch_sparse import SparseTensor, set_diag
from torch_geometric.nn.conv import MessagePassing
from .message_passing import MessagePassingNew
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax


class GAT(torch.nn.Module):
    def __init__(self, args):
        super(GAT, self).__init__()
        dim_size = args.dim_size
        dataset = args.dataset
        self.aug = args.aug
        if args.aug == True:
            self.conv1 = GATConvNew(
                dataset['num_node_features'], dim_size)
            self.conv2 = GATConvNew(
                dim_size, dataset['num_classes'])
        else:
            self.conv1 = GATConv(dataset['num_node_features'], dim_size)
            self.conv2 = GATConv(dim_size, dataset['num_classes'])

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


class GATConvNew(MessagePassingNew):
    _alpha: OptTensor

    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, heads: int = 1, concat: bool = True,
                 negative_slope: float = 0.2, dropout: float = 0.0,
                 add_self_loops: bool = True, bias: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(GATConvNew, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops

        if isinstance(in_channels, int):
            self.lin_l = Linear(in_channels, heads * out_channels, bias=False)
            self.lin_r = self.lin_l
        else:
            self.lin_l = Linear(in_channels[0], heads * out_channels, False)
            self.lin_r = Linear(in_channels[1], heads * out_channels, False)

        self.att_l = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_r = Parameter(torch.Tensor(1, heads, out_channels))

        if isinstance(in_channels, int):
            self.lin1 = Linear(in_channels, out_channels, bias=False)
            self.lin2 = self.lin1
        else:
            self.lin1 = Linear(in_channels[0], out_channels, False)
            self.lin2 = Linear(in_channels[1], out_channels, False)

        self.att1 = Parameter(torch.Tensor(1, out_channels))
        self.att2 = Parameter(torch.Tensor(1, out_channels))
        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self._alpha = None

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.lin_l.weight)
        glorot(self.lin_r.weight)
        glorot(self.att_l)
        glorot(self.att_r)
        glorot(self.lin1.weight)  # MLP -> MLP([W*h_i||W*h_j])
        glorot(self.lin2.weight)
        glorot(self.att1)  # W -> a = MLP([W*h_i||W*h_j])
        glorot(self.att2)
        zeros(self.bias)

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                size: Size = None, return_attention_weights=None):
        # type: (Union[Tensor, OptPairTensor], Tensor, Size, NoneType) -> Tensor  # noqa
        # type: (Union[Tensor, OptPairTensor], SparseTensor, Size, NoneType) -> Tensor  # noqa
        # type: (Union[Tensor, OptPairTensor], Tensor, Size, bool) -> Tuple[Tensor, Tuple[Tensor, Tensor]]  # noqa
        # type: (Union[Tensor, OptPairTensor], SparseTensor, Size, bool) -> Tuple[Tensor, SparseTensor]  # noqa
        r"""
        Args:
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """
        H, C = self.heads, self.out_channels

        x_l: OptTensor = None
        x_r: OptTensor = None
        alpha_l: OptTensor = None
        alpha_r: OptTensor = None
        if isinstance(x, Tensor):
            assert x.dim() == 2, 'Static graphs not supported in `GATConv`.'
            x_l = x_r = self.lin_l(x).view(-1, H, C)
            alpha_l = (x_l * self.att_l).sum(dim=-1)
            alpha_r = (x_r * self.att_r).sum(dim=-1)
        else:
            x_l, x_r = x[0], x[1]
            assert x[0].dim() == 2, 'Static graphs not supported in `GATConv`.'
            x_l = self.lin_l(x_l).view(-1, H, C)
            alpha_l = (x_l * self.att_l).sum(dim=-1)
            if x_r is not None:
                x_r = self.lin_r(x_r).view(-1, H, C)
                alpha_r = (x_r * self.att_r).sum(dim=-1)

        assert x_l is not None
        assert alpha_l is not None

        C_ = self.out_channels

        x1: OptTensor = None
        x2: OptTensor = None
        sigma1: OptTensor = None
        sigma2: OptTensor = None
        if isinstance(x, Tensor):
            assert x.dim() == 2, 'Static graphs not supported.'
            x1 = x2 = self.lin1(x).view(-1, C_)
            sigma1 = (x1 * self.att1).sum(dim=-1)
            sigma2 = (x2 * self.att2).sum(dim=-1)

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                num_nodes = x_l.size(0)
                if x_r is not None:
                    num_nodes = min(num_nodes, x_r.size(0))
                if size is not None:
                    num_nodes = min(size[0], size[1])
                edge_index, _ = remove_self_loops(edge_index)
                edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
            elif isinstance(edge_index, SparseTensor):
                edge_index = set_diag(edge_index)

        # propagate_type: (x: OptPairTensor, alpha: OptPairTensor)
        out, sigma = self.propagate(edge_index, x=(x_l, x_r),
                                    alpha=(alpha_l, alpha_r), sigma=(sigma1, sigma2), size=size)

        alpha = self._alpha
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out += self.bias

        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out, sigma

    def message(self, x_j: Tensor, x_i: OptTensor, alpha_j: Tensor, alpha_i: OptTensor,
                sigma_i: Tensor, sigma_j: OptTensor, index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:
        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        sigma = sigma_j if sigma_i is None else sigma_j + sigma_i
        sigma = F.leaky_relu(sigma, self.negative_slope)
        sigmoid = nn.Sigmoid()
        sigma = sigmoid(sigma)
        return x_j * alpha.unsqueeze(-1) * sigma.clone().view(-1, 1) + x_i * alpha.unsqueeze(-1) * (1 - sigma.clone().view(-1, 1)), sigma

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)


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
