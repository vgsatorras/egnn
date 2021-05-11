from torch.nn import functional as F
import torch


def vae_loss(adj_rec, adj_gt, mu, logvar, reduce='sum'):
    # Reconstruction + KL divergence losses summed over all elements and batch
    BCE = adj_bce(adj_rec, adj_gt, reduce)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    if mu is None:
        KLD = torch.zeros(1)
    else:
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE, KLD


def adj_bce(pred, gt, reduce='mean', weight=None):
    return F.binary_cross_entropy(pred.view(-1, 1), gt.view(-1, 1), reduction=reduce, weight=weight)

