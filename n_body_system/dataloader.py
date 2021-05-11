import random
import torch


class Dataloader():
    def __init__(self, dataset, batch_size=1, slice=[0, 1e8], shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.n_nodes = self.dataset.get_n_nodes()
        self.edges = self.expand_edges(dataset.edges, batch_size, self.n_nodes)
        self.idxs_permuted = list(range(len(self.dataset)))
        self.shuffle = shuffle
        self.slice = slice
        if self.shuffle:
            random.shuffle(self.idxs_permuted)
        self.idx = 0

    def __iter__(self):
        return self

    def expand_edges(self, edges, batch_size, n_nodes):
        edges = [torch.LongTensor(edges[0]), torch.LongTensor(edges[1])]
        if batch_size == 1:
            return edges
        elif batch_size > 1:
            rows, cols = [], []
            for i in range(batch_size):
                rows.append(edges[0] + n_nodes*i)
                cols.append(edges[1] + n_nodes*i)
            edges = [torch.cat(rows), torch.cat(cols)]
        return edges

    def __next__(self):
        if self.idx > len(self.dataset) - self.batch_size:
            self.idx = 0
            #random.shuffle(self.dataset.graphs)
            raise StopIteration  # Done iterating.
        else:
            loc, vel, edge_attr, charges = self.dataset.data
            idx_permuted = self.idxs_permuted[self.idx:self.idx + self.batch_size]
            batched_data = loc[idx_permuted], vel[idx_permuted], edge_attr[idx_permuted], charges[idx_permuted]
            [loc_batch, vel_batch, edge_attr_batch, loc_end_batch, charges_batch] = self.cast_batch(list(batched_data))

            self.idx += self.batch_size
            return loc_batch, vel_batch, edge_attr_batch, loc_end_batch, charges_batch

    def cast_batch(self, batched_data):
        #loc_batch, vel_batch, edges_batch, loc_end_batch = batched_data
        #if self.batch_size > 1:
        #    raise Exception("To implement")
        batched_data = [d.contiguous().view(-1, d.size(2)) for d in batched_data]

        return batched_data
        #else:
        #    return loc_batch[0], vel_batch[0], edges_batch[0], loc_end_batch[0]

    def __len__(self):
        return len(self.dataset)

    def partition(self):
        return self.dataset.partition


if __name__ == "__main__":
    '''
    from dataset_nbody import NBodyDataset

    dataset_train = NBodyDataset(partition='train')
    dataloader_train = Dataloader(dataset_train)
    for i, (loc, vel, edges) in enumerate(dataset_train):
        print(i)
    '''


