from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
import utils
from ae_datasets import d_selector, Dataloader
import models
import losess
import eval


parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--exp_name', type=str, default='exp_1', metavar='N',
                    help='experiment_name')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--dataset', type=str, default='community_ours', metavar='N',
                    help='community_ours | community_overfit | erdosrenyinodes_0.25_none | erdosrenyinodes_0.25_overfit')
parser.add_argument('--no-cuda', action='store_true', default=True,
                    help='we  did not use cuda in this experiment')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log_interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--test_interval', type=int, default=2, metavar='N',
                    help='how many epochs to wait before logging test')
parser.add_argument('--generate-interval', type=int, default=100, metavar='N',
                    help='how many epochs to wait before logging test')
parser.add_argument('--outf', type=str, default='outputs_ae', metavar='N',
                    help='folder to output vae')
parser.add_argument('--plots', type=int, default=0, metavar='N',
                    help='Plot images of the graphs & adjacency matrices')
parser.add_argument('--lr', type=float, default=1e-4, metavar='N',
                    help='learning rate')
parser.add_argument('--nf', type=int, default=64, metavar='N',
                    help='learning rate')
parser.add_argument('--emb_nf', type=int, default=8, metavar='N',
                    help='learning rate')
parser.add_argument('--K', type=int, default=8, metavar='N',
                    help='learning rate')
parser.add_argument('--model', type=str, default='ae_egnn', metavar='N',
                    help='available models: ae | ae_rf | ae_egnn | baseline')
parser.add_argument('--attention', type=int, default=0, metavar='N',
                    help='attention in the ae model')
parser.add_argument('--noise_dim', type=int, default=0, metavar='N',
                    help='break the symmetry applying noise at the input of the AE')
parser.add_argument('--n_layers', type=int, default=4, metavar='N',
                    help='number of layers for the autoencoder')
parser.add_argument('--reg', type=float, default=1e-3, metavar='N',
                    help='regularizer for the equivariant autoencoder')
parser.add_argument('--clamp', type=int, default=1, metavar='N',
                    help='clamp the output of the coords function if get too large (safe mechanism, it is not activated in practice)')
parser.add_argument('--weight_decay', type=float, default=1e-16, metavar='N',
                    help='clamp the output of the coords function if get too large')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
print(args)

print(args)
torch.manual_seed(args.seed)
device = torch.device("cuda" if args.cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

utils.create_folders(args)

#
dataset = d_selector.retrieve_dataset(args.dataset, partition="train", directed=True)
train_loader = Dataloader(dataset, batch_size=1)
dataset = d_selector.retrieve_dataset(args.dataset, partition="val", directed=True)
val_loader = Dataloader(dataset, batch_size=1, shuffle=False)
dataset = d_selector.retrieve_dataset(args.dataset, partition="test", directed=True)
test_loader = Dataloader(dataset, batch_size=1, shuffle=False)

if args.model == 'ae':
    model = models.AE(hidden_nf=args.nf, embedding_nf=args.emb_nf, noise_dim=args.noise_dim, act_fn=nn.SiLU(),
                      learnable_dec=1, device=device, attention=args.attention, n_layers=args.n_layers)
elif args.model == 'ae_rf':
    model = models.AE_rf(embedding_nf=args.K, nf=args.nf, device=device, n_layers=args.n_layers, reg=args.reg, act_fn=nn.SiLU(), clamp=args.clamp)
elif args.model == 'ae_egnn':
    model = models.AE_EGNN(hidden_nf=args.nf, K=args.K, act_fn=nn.SiLU(), device=device, n_layers=args.n_layers, reg=args.reg, clamp=args.clamp)
elif args.model == 'baseline':
    model = models.Baseline(device=device)
else:
    raise Exception('Wrong model %s' % args.model)

print(model)

optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

pr = eval.ProgressReporter(path=args.outf + '/' + args.exp_name, file_name='/output.json')


def train(epoch, loader):
    lr_scheduler.step(epoch)
    model.train()
    res = {'epoch': epoch, 'loss': 0, 'bce': 0, 'kl': 0, 'kl_coords': 0, 'adj_err': 0, 'coord_reg': 0, 'counter': 0, 'wrong_edges': 0, 'gt_edges': 0, 'possible_edges': 0}
    magnitudes = {'value':0, 'counter':0}
    for batch_idx, data in enumerate(loader):
        graph = data[0]

        nodes, edges, edge_attr, adj_gt = graph.get_dense_graph(store=True, loops=False)
        nodes, edges, edge_attr, adj_gt = nodes.to(device), [edges[0].to(device), edges[1].to(device)], edge_attr.to(device), adj_gt.to(device).detach()
        n_nodes = nodes.size(0)
        optimizer.zero_grad()

        adj_pred, z = model(nodes, edges, edge_attr)
        bce, kl = losess.vae_loss(adj_pred, adj_gt, None, None)
        kl_coords = torch.zeros(1)
        loss = bce

        loss = loss # normalize loss by the number of nodes

        loss.backward()
        optimizer.step()

        res['loss'] += loss.item()
        res['bce'] += bce.item()
        res['kl'] += kl.item()
        res['kl_coords'] += kl_coords.item()
        wrong_edges, adj_err = eval.adjacency_error(adj_pred, adj_gt)

        res['adj_err'] += adj_err
        res['counter'] += 1
        res['wrong_edges'] += wrong_edges
        res['gt_edges'] += torch.sum(adj_gt).item()
        res['possible_edges'] += n_nodes ** 2 - n_nodes
        if batch_idx % args.log_interval == 0:
            print('===> Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))
        magnitudes['value'] += torch.mean(torch.abs(z))
        magnitudes['counter'] += 1
    error = res['wrong_edges'] / res['possible_edges']
    print('Train avg bce: %.4f \t KL %.4f \t KL_coords %.4f \tAdj_err %.4f \nWrong edges %d \t gt edges %d \t Possible edges %d \t Error %.4f' % (res['bce'] / res['counter'], res['kl'] / res['counter'], res['kl_coords'] / res['counter'], res['adj_err'] / res['counter'], res['wrong_edges'], res['gt_edges'], res['possible_edges'], error))


def test(epoch, loader):
    model.eval()
    res = {'epoch': epoch, 'loss': 0, 'bce': 0, 'kl': 0, 'kl_coords': 0, 'adj_err': 0, 'counter': 0, 'wrong_edges': 0, 'gt_edges': 0, 'possible_edges': 0, 'tp': 0, 'fp': 0, 'fn': 0}
    with torch.no_grad():
        for idx, data in enumerate(loader):
            graph = data[0]
            n_nodes = graph.get_num_nodes()
            nodes, edges, edge_attr, adj_gt = graph.get_dense_graph(store=True, loops=False)
            nodes, edges, edge_attr, adj_gt = nodes.to(device), [edges[0].to(device), edges[1].to(device)], edge_attr.to(device), adj_gt.to(device)

            adj_pred, mu = model(nodes, edges, edge_attr)
            bce, kl = losess.vae_loss(adj_pred, adj_gt, None, None)
            loss = bce

            res['loss'] += loss.item()
            res['bce'] += bce.item()
            res['kl'] += kl.item()
            tp, fp, fn = eval.tp_fp_fn(adj_pred, adj_gt)
            res['tp'] += tp
            res['fp'] += fp
            res['fn'] += fn
            wrong_edges, adj_err = eval.adjacency_error(adj_pred, adj_gt)
            res['adj_err'] += adj_err
            res['counter'] += 1
            res['wrong_edges'] += wrong_edges
            res['gt_edges'] += torch.sum(adj_gt).item()
            res['possible_edges'] += n_nodes ** 2 - n_nodes

    res = utils.normalize_res(res, keys=['loss', 'bce', 'kl', 'kl_coords', 'adj_err'])
    error = res['wrong_edges']/ res['possible_edges']
    f1_score = 1.0*res['tp'] / (res['tp'] + 0.5*(res['fp'] + res['fn']))
    print('Test on %s \t \t \t \t loss: %.4f \t  bce: %.4f \t  kl: %.4f \t  kl_coords: %.4f \t Adj_err %.4f \nWrong edges %d \t gt edges %d \t Possible edges %d \t Error %.4f \t TP: %d \t FP: %d \t FN: %d \t F1-score: %.4f' % (loader.dataset.partition,
                                                                                    res['loss'],
                                                                                    res['bce'],
                                                                                    res['kl'],
                                                                                    res['kl_coords'],
                                                                                    res['adj_err'],
                                                                                    res['wrong_edges'],
                                                                                    res['gt_edges'],
                                                                                    res['possible_edges'],
                                                                                     error, res['tp'], res['fp'], res['fn'], f1_score))
    pr.add_epoch(res, loader.dataset.partition)
    return res


if __name__ == "__main__":
    best_bce_val = 1e8
    best_res_test = None
    best_epoch = 0
    for epoch in range(0, args.epochs):
        train(epoch, train_loader)
        if epoch % args.test_interval == 0:
            res_train = test(epoch, train_loader)
            res_val = test(epoch, val_loader)
            res_test = test(epoch, test_loader)

            if res_val['bce'] < best_bce_val:
                best_bce_val = res_val['bce']
                best_res_test = res_test
                best_epoch = epoch
            print("###############\n### Best result is: bce: %.4f, wrong_edges %d, error: %.4f, epoch %d" % (best_res_test['bce'],
                                                                                     best_res_test['wrong_edges'],
                                                                                     best_res_test['wrong_edges']/best_res_test['possible_edges'],
                                                                                     best_epoch))
            print("###############")


