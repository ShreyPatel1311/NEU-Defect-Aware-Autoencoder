"""Reconstruction targets, masking strategies and the contrastive (baseline) loss."""
import math
import torch, torch.nn.functional as F
from .data import IMG, N, P
from .models import patchify

_SOBEL = torch.tensor([[-1., 0, 1], [-2, 0, 2], [-1, 0, 1]])


def pixel_targets(x):
    """Per-patch standardised pixels (standard MAE 'norm-pix' target). (B,1,224,224) -> (B,196,256)."""
    t = patchify(x.float())
    return (t - t.mean(-1, keepdim=True)) / (t.var(-1, keepdim=True) + 1e-6).sqrt()


def hog_targets(x, cell=8):
    """Per-patch HOG descriptor. (B,1,224,224) -> (B,196,9*(16/cell)^2).
    Sobel gradients -> 9 unsigned orientation bins weighted by magnitude -> averaged over cell x cell pixels ->
    concatenated over the (16/cell)^2 cells of each patch -> L2-normalised per patch (rescaled by sqrt(dim))."""
    x = x.float(); k = _SOBEL.to(x.device)
    gx = F.conv2d(x, k.view(1, 1, 3, 3), padding=1); gy = F.conv2d(x, k.t().reshape(1, 1, 3, 3), padding=1)
    mag = (gx ** 2 + gy ** 2).sqrt(); ang = torch.atan2(gy, gx) % math.pi
    b = (ang / math.pi * 9).long().clamp(max=8)
    h = torch.zeros(x.shape[0], 9, IMG, IMG, device=x.device).scatter_(1, b, mag)
    k = P // cell; d = 9 * k * k
    h = F.avg_pool2d(h, cell)
    h = h.reshape(-1, 9, 14, k, 14, k).permute(0, 2, 4, 3, 5, 1).reshape(-1, N, d)
    return F.normalize(h, dim=-1) * d ** 0.5


def random_masking(B, ratio, device):
    """Uniform random masking (He et al.). Returns ids_keep (B,k), ids_restore (B,N), mask (B,N) True = masked."""
    noise = torch.rand(B, N, device=device)
    ids_shuffle = noise.argsort(1); ids_restore = ids_shuffle.argsort(1)
    keep = N - int(ratio * N)
    ids_keep = ids_shuffle[:, :keep]
    mask = torch.ones(B, N, dtype=torch.bool, device=device); mask.scatter_(1, ids_keep, False)
    return ids_keep, ids_restore, mask


def energy_masking(x, ratio, beta=1.0):
    """Defect-aware masking (label-free). Patch 'texture energy' e = pixel std; patches are ranked by
    beta * zscore(e) + Gumbel noise and the top `ratio` fraction is masked, so high-energy (defect-like) patches are
    hidden more often while the mask stays stochastic. beta = 0 recovers uniform random masking."""
    B = x.shape[0]
    e = patchify(x.float()).std(-1)
    z = (e - e.mean(1, keepdim=True)) / (e.std(1, keepdim=True) + 1e-6)
    g = -torch.log(-torch.log(torch.rand(B, N, device=x.device).clamp(1e-9, 1 - 1e-9)))
    ids_shuffle = (beta * z + g).argsort(1)
    ids_restore = ids_shuffle.argsort(1); keep = N - int(ratio * N)
    ids_keep = ids_shuffle[:, :keep]
    mask = torch.ones(B, N, dtype=torch.bool, device=x.device); mask.scatter_(1, ids_keep, False)
    return ids_keep, ids_restore, mask


def nt_xent(z, tau=0.2):
    """SimCLR loss for 2B projections ordered [view1 batch; view2 batch]."""
    zz = F.normalize(z.float(), dim=1)
    sim = zz @ zz.t() / tau
    sim.fill_diagonal_(-1e4)
    B = z.shape[0] // 2
    tgt = torch.cat([torch.arange(B, 2 * B), torch.arange(0, B)]).to(z.device)
    return F.cross_entropy(sim, tgt)
