from data_loader.datasets_source_info import get_dataset_category
from torch.utils.data import DataLoader
from utils.statistic import *
from data_loader.dataset_selection import DatasetSelection


class DataLoader(DataLoader):

    def __init__(self, args):
        self.dataset_name = args.dataset_name
        self.source_name = args.source_name
        self.dataset = DatasetSelection(
            self.source_name, self.dataset_name, args.split).get_dataset()

        print("load source:", self.source_name, "category:", get_dataset_category(
            self.source_name, self.dataset_name), "dataset", self.dataset_name)
