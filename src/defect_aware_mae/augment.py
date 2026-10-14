"""Label-free GPU augmentation."""
import random
import cv2, numpy as np, torch, torch.nn.functional as F


def augment(x, strong=False):
    """x: (B,1,H,W) in [0,1] on GPU.
    weak  : flips, 90-degree rotations, brightness/contrast jitter, Gaussian noise
    strong: additionally random-resized-crop (scale 0.35-1), stronger jitter, Gaussian blur (p=0.3)"""
    B = x.shape[0]
    if strong:
        scale = torch.empty(B, device=x.device).uniform_(0.35, 1.0).sqrt()
        tx = (torch.rand(B, device=x.device) * 2 - 1) * (1 - scale)
        ty = (torch.rand(B, device=x.device) * 2 - 1) * (1 - scale)
        th = torch.zeros(B, 2, 3, device=x.device)
        th[:, 0, 0] = scale; th[:, 1, 1] = scale; th[:, 0, 2] = tx; th[:, 1, 2] = ty
        x = F.grid_sample(x, F.affine_grid(th, x.shape, align_corners=False), align_corners=False)
    f = torch.rand(B, device=x.device) < 0.5; x[f] = x[f].flip(-1)
    f = torch.rand(B, device=x.device) < 0.5; x[f] = x[f].flip(-2)
    k = torch.randint(0, 4, (B,), device=x.device)
    x = torch.stack([torch.rot90(xi, int(ki), (-2, -1)) for xi, ki in zip(x, k)])
    s = 0.4 if strong else 0.2
    b = 1 + (torch.rand(B, 1, 1, 1, device=x.device) * 2 - 1) * s
    c = 1 + (torch.rand(B, 1, 1, 1, device=x.device) * 2 - 1) * s
    m = x.mean((2, 3), keepdim=True)
    x = ((x - m) * c + m) * b
    if strong:
        bl = torch.rand(B, device=x.device) < 0.3
        if bl.any():
            x[bl] = torch.from_numpy(np.stack([cv2.GaussianBlur(i[0].cpu().numpy(), (0, 0), random.uniform(0.3, 1.5))
                                               for i in x[bl]]))[:, None].to(x.device)
    x = x + torch.randn_like(x) * 0.02
    return x.clamp(0, 1)
