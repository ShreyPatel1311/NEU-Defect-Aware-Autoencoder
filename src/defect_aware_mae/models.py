"""ViT encoder, lightweight MAE decoder, and frozen-feature extraction."""
import torch, torch.nn as nn, torch.nn.functional as F
from .data import IMG, N, P


class PatchEmbed(nn.Module):
    def __init__(self, d=768):
        super().__init__(); self.proj = nn.Conv2d(1, d, P, P)

    def forward(self, x): return self.proj(x).flatten(2).transpose(1, 2)


class Encoder(nn.Module):
    """ViT: 224x224 grayscale -> 196 patches of 16x16, 6 pre-norm layers, 12 heads, 768-d."""
    def __init__(self, d=768, depth=6, heads=12):
        super().__init__()
        self.patch_embed = PatchEmbed(d)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d))
        self.pos_embed = nn.Parameter(torch.zeros(1, N + 1, d))
        nn.init.trunc_normal_(self.pos_embed, std=.02); nn.init.trunc_normal_(self.cls_token, std=.02)
        layer = nn.TransformerEncoderLayer(d, heads, d * 4, dropout=0.0, batch_first=True, norm_first=True,
                                           activation="gelu")
        self.blocks = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x, ids_keep=None, return_layers=False):
        """ids_keep: (B,k) indices of visible patches (None = all). Positional embeddings are added before
        selection, so kept patches retain their true position."""
        x = self.patch_embed(x) + self.pos_embed[:, 1:]
        if ids_keep is not None:
            x = torch.gather(x, 1, ids_keep[..., None].expand(-1, -1, x.shape[-1]))
        cls = (self.cls_token + self.pos_embed[:, :1]).expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], 1)
        outs = []
        for blk in self.blocks.layers:
            x = blk(x); outs.append(x)
        x = self.norm(x)
        return (x, outs) if return_layers else x


def readout(enc, x, mode="mid"):
    """Image-level feature from a full (unmasked) image. Differentiable.
    cls : class token after the final norm
    mean: mean of patch tokens after the final norm
    mid : mean-pooled patch tokens of blocks 4-6, each instance-normalised, then averaged"""
    z, outs = enc(x, return_layers=True)
    if mode == "cls": return z[:, 0]
    if mode == "mean": return z[:, 1:].mean(1)
    if mode == "mid":
        return torch.stack([F.layer_norm(o[:, 1:].float(), (o.shape[-1],)).mean(1) for o in outs[3:]]).mean(0)
    raise ValueError(mode)


class Decoder(nn.Module):
    """MAE decoder: 256-d, 8 heads. `out_dim` is the per-patch target size (256 pixels, 36 for HOG, ...)."""
    def __init__(self, d=768, dd=256, depth=4, heads=8, out_dim=P * P):
        super().__init__()
        self.embed = nn.Linear(d, dd)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dd))
        self.pos_embed = nn.Parameter(torch.zeros(1, N + 1, dd))
        nn.init.trunc_normal_(self.pos_embed, std=.02); nn.init.normal_(self.mask_token, std=.02)
        layer = nn.TransformerEncoderLayer(dd, heads, dd * 4, dropout=0.0, batch_first=True, norm_first=True,
                                           activation="gelu")
        self.blocks = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dd)
        self.pred = nn.Linear(dd, P * P)
        self.pred = nn.Linear(dd, out_dim)   # deliberately re-created: keeps the RNG draw order of the released runs

    def forward(self, z, ids_restore):
        z = self.embed(z)
        B, L, D = z.shape
        mt = self.mask_token.expand(B, N + 1 - L, -1)
        x = torch.cat([z[:, 1:], mt], 1)
        x = torch.gather(x, 1, ids_restore[..., None].expand(-1, -1, D))
        x = torch.cat([z[:, :1], x], 1) + self.pos_embed
        return self.pred(self.norm(self.blocks(x)))[:, 1:]


def patchify(x):
    B = x.shape[0]
    return x.reshape(B, 1, 14, P, 14, P).permute(0, 2, 4, 1, 3, 5).reshape(B, N, P * P)


def unpatchify(p):
    B = p.shape[0]
    return p.reshape(B, 14, 14, P, P).permute(0, 1, 3, 2, 4).reshape(B, 1, IMG, IMG)


def load_encoder(path, device="cuda"):
    """Loads the 'encoder' state dict from a checkpoint file (or a bare encoder state dict)."""
    enc = Encoder().to(device)
    s = torch.load(path, map_location=device, weights_only=True)
    if "encoder" in s: s = s["encoder"]
    enc.load_state_dict(s); enc.eval()
    return enc


@torch.no_grad()
def embed(enc, x_uint8, mode="mid", bs=64, device=None):
    """Frozen features (n, 768) for uint8 images (n,224,224). `mode` as in `readout`."""
    device = device or next(enc.parameters()).device
    enc.eval(); out = []
    for i in range(0, len(x_uint8), bs):
        xb = torch.from_numpy(x_uint8[i:i + bs]).float().div(255)[:, None].to(device)
        with torch.autocast(torch.device(device).type, dtype=torch.float16, enabled=torch.device(device).type == "cuda"):
            out.append(readout(enc, xb, mode).float().cpu())
    return torch.cat(out).numpy()
