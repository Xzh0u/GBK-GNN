import argparse
from models import dnn, gat, gcn, gin, sage, gcn2
MODEL_CLASSES = {'DNN': dnn.DNN, 'GraphSage': sage.GraphSage,
                 'GIN': gin.GIN, 'GCN2': gcn2.GCN2,
                 'GCN': gcn.GCN, 'GAT': gat.GAT, }


class HyperparameterSetting:
    def __init__(self, experiment_type):
        if (experiment_type == "node classification"):
            parser = argparse.ArgumentParser()
            # training arguments
            parser.add_argument('--log_interval', type=int, default=1,
                                help="log interval while training.")
            parser.add_argument('--epoch', type=int, default=500,
                                help="num of training epoches.")
            parser.add_argument('--seed', type=int, default=1,
                                help="random seed.")
            parser.add_argument('--gpu_id', type=int, default=0,
                                help="gpu id.")
            parser.add_argument('--lr', type=float, default=0.001,
                                help="learning rate.")
            parser.add_argument('--weight_decay', type=float, default=0.01,
                                help="weight decay.")
            parser.add_argument('--save_path', type=str, default="experiment_ans.csv",
                                help="The name of the file to save results.")
            parser.add_argument('--model_path', type=str, default="save.pt",
                                help="The name of the file to save models.")
            # model arguments
            parser.add_argument('--dim_size', type=int, default=16,
                                help="size of hidden state in network.")
            parser.add_argument("--split", nargs="+",
                                default=[0.6, 0.2, 0.2], type=float)
            parser.add_argument('--aug', dest='aug', default=False, action='store_true',
                                help="Whether use our message passing method.")
            parser.add_argument('--lamda', type=float, default=1.84,
                                help="The hypereparameter of regularization term.")
            parser.add_argument("--model_type", default="GraphSage", type=str,
                                help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
            # dataset arguments
            parser.add_argument("--source_name", default="WebKB",
                                type=str, help="source of dataset.")
            parser.add_argument("--dataset_name", default="Texas",
                                type=str, help="name of dataset.")

            self.parser = parser

    def get_args(self):
        return self.parser.parse_args()
