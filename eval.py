import torch
import json
import matplotlib.pyplot as plt
import networkx as nx

def adjacency_error(adj_pred, adj_gt):
    n_nodes = adj_gt.size(0)
    adj_pred = (adj_pred > 0.5).type(torch.float32)
    adj_errors = torch.abs(adj_pred - adj_gt)
    wrong_edges = torch.sum(adj_errors)
    adj_error = wrong_edges/ (n_nodes ** 2 - n_nodes)
    return wrong_edges.item(), adj_error.item()

def tp_fp_fn(adj_pred, adj_gt):
    adj_pred = (adj_pred > 0.5).type(torch.float32)
    tp = torch.sum(adj_pred*adj_gt).item()
    fp = torch.sum(adj_pred * (1 - adj_gt)).item()
    fn = torch.sum((1-adj_pred)*adj_gt).item()
    return tp, fp, fn

def graph_edit_distance(adj_pred, adj_gt):
    eye = torch.eye(adj_pred.size(0))
    adj_pred = adj_pred * (1 - eye)
    adj_gt = adj_gt * (1 - eye)

    adj_pred = (adj_pred > 0.5).type(torch.float32)

    g1 = nx.from_numpy_matrix(adj_pred.detach().numpy(), create_using=nx.Graph)
    g2 = nx.from_numpy_matrix(adj_gt.detach().numpy(), create_using=nx.Graph)
    ged = nx.graph_edit_distance(g1, g2)
    return ged

class ProgressReporter:
    def __init__(self, path='', file_name='outputs.json'):
        self.path = path
        self.filepath = path + '/' + file_name
        self.data = {'train': {}, 'val': {}, 'test': {}}

    def add_epoch(self, res, partition='test'):
        for key in res:
            if key not in self.data[partition]:
                self.data[partition][key] = []
            self.data[partition][key].append(res[key])
        self._save()
        if partition == 'test':
            self.plot()

    def _save(self):
        with open(self.filepath, 'w') as outfile:
            json.dump(self.data, outfile)

    def load(self):
        with open(self.filepath) as json_file:
            self.data = json.load(json_file)

    def plot_partition(self, curve, partition='train', line='--'):
        data = self.data[partition]
        #plt.ylim((0, 0.55))
        plt.plot(data['epoch'], data[curve], line, c='b')

    def plot_curve(self, curve='adj_err'):
        self.plot_partition(curve, partition='train', line='--')
        self.plot_partition(curve, partition='test', line='-')
        plt.legend(['train', 'test'])
        plt.savefig(self.path + '/' + curve + '.png')
        plt.clf()

    def plot(self):
        for key in self.data['test']:
            self.plot_curve(key)

def plots_accuracies(exp_names):
    path = 'outputs_vae/%s'
    file_name = 'output.json'
    for exp_name in exp_names:
        pr = ProgressReporter(path=path % exp_name, file_name=file_name)
        pr.load()
        x = pr.data['train']['epoch'][1:50]
        y = pr.data['train']['adj_err'][1:50]
        plt.plot(x, y)
    plt.show()

if __name__ == "__main__":
    exp_names = {'exp_1': 'AE (noise)', 'exp_2_evae': 'EVAE', 'exp_3_vae': 'AE'}
    plots_accuracies(exp_names)
