from torch_geometric.datasets import *

# datasets save folder directory
data_dir = "./datasets/"

# DataInformation
dataset_sources_dic = {
    "Planetoid": Planetoid,
    "WebKB": WebKB,
    "Actor": Actor,
    "WikipediaNetwork": WikipediaNetwork,
}
dataset_name_dic = {
    "Planetoid": {
        "Citation Network":
            ["Cora", "CiteSeer", "PubMed"]
    },
    "WebKB": {
        "Social Network":
            ["Cornell", "Texas", "Wisconsin"]
    },
    "WikipediaNetwork": {
        "Social Network":
            ["chameleon", "squirrel"]
    },
    "Actor": {
        "Social Network": [""]
    },
}


def get_dataset_category(source_name, dataset_name):
    for category_name in dataset_name_dic[source_name]:
        if (dataset_name in dataset_name_dic[source_name][category_name]):
            return category_name
    return None
