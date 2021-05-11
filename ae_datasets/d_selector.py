import graph
from ae_datasets import d_creator


def retrieve_dataset(dataset_name, partition='train', directed=True):
    '''

    :param dataset_name: 'erdosrenyinodes_0.25_none'
                         'community_ours'
                         'community_overfit'
                         'erdosrenyinodes_0.25_overfit'

    :return:
    '''
    if dataset_name.startswith("erdosrenyinodes"):
        _, p, overfit = dataset_name.split("_")
        dataset = d_creator.DatasetErdosRenyiNodes(p=float(p), partition=partition, overfit=overfit=="overfit")
    elif dataset_name.startswith("erdosrenyi"):
        _, n_samples, n_nodes, n_edges = dataset_name.split("_")
        dataset = d_creator.DatasetErdosRenyi(None, int(n_nodes), int(n_edges), partition, directed)
    elif dataset_name == "community_ours":
        dataset = d_creator.DatasetCommunity(partition=partition)
    elif dataset_name == "community_overfit":
        dataset = d_creator.DatasetCommunity(n_samples=100)
    else:
        raise Exception("Wrong dataset %s" % dataset_name)
    return dataset


if __name__ == "__main__":
    datasets = ['erdosrenyinodes_0.25_none']
    for dataset in datasets:
        print(dataset)
        dataset = retrieve_dataset(dataset, "test")
        for sample in dataset.graphs:
            graph.plot_graph(sample)



