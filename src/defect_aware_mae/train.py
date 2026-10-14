"""Self-supervised pretraining on the NEU-DET *train* split (1,440 images; no labels, no boxes).

method = mae     masked autoencoder. --target {pix,hog,hogms,teacher,hogteacher}  --masking {random,energy}
method = simclr  contrastive baseline (NT-Xent) on the identical encoder, optimiser and schedule

The released Defect-aware MAE is:  --target hog --masking energy --strong 1 --dec_depth 2 --lr 1.5e-4 --epochs 1600
The validation split is never read here."""
import argparse, copy, json, math, os, time
import torch, torch.nn as nn, torch.nn.functional as F
from .augment import augment
from .data import N, P, load_split, repo_root, set_seed
from .models import Decoder, Encoder
from .objectives import energy_masking, hog_targets, nt_xent, pixel_targets, random_masking

OUT_DIM = {"pix": P * P, "hog": 36, "teacher": 768, "hogms": 189, "hogteacher": 36 + 768}


def get_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", default="mae", choices=["mae", "simclr"])
    ap.add_argument("--target", default="pix", choices=list(OUT_DIM))
    ap.add_argument("--masking", default="random", choices=["random", "energy"])
    ap.add_argument("--beta", type=float, default=1.0, help="defect-aware masking strength")
    ap.add_argument("--epochs", type=int, default=800)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-4, help="absolute peak learning rate")
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--mask", type=float, default=0.75, help="mask ratio")
    ap.add_argument("--strong", type=int, default=0, help="1 = strong augmentation for MAE views")
    ap.add_argument("--ema", type=float, default=0.996, help="teacher momentum (teacher targets only)")
    ap.add_argument("--dec_depth", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", required=True, help="run name; outputs go to <out>/<tag>/")
    ap.add_argument("--out", default=None, help="output root (default: <repo>/runs)")
    ap.add_argument("--init_from", default=None, help="checkpoint to initialise encoder (and decoder, if shapes match) from")
    ap.add_argument("--n", type=int, default=0, help="debug: use only the first n training images")
    ap.add_argument("--config", default=None, help="JSON file of arguments (command-line flags override it)")
    return ap


def parse_args(argv=None):
    ap = get_parser(); a = ap.parse_args(argv)
    if a.config:
        ap.set_defaults(**json.load(open(a.config))); a = ap.parse_args(argv)
    return a


def train(a, log=print):
    out = os.path.join(a.out or repo_root() / "runs", a.tag); os.makedirs(out, exist_ok=True)
    json.dump({k: v for k, v in vars(a).items() if k not in ("out", "config", "init_from") or (k == "init_from" and v)}, open(f"{out}/args.json", "w"), indent=1)
    set_seed(a.seed); dev = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True

    X, _, _ = load_split("train")                       # labels and boxes discarded
    if a.n: X = X[:a.n]
    X = torch.from_numpy(X).float().div(255)[:, None]
    enc = Encoder().to(dev); params = list(enc.parameters())
    dec = head = t_enc = None
    if a.method == "mae":
        dec = Decoder(depth=a.dec_depth, out_dim=OUT_DIM[a.target]).to(dev)
        params += list(dec.parameters())
        if a.target in ("teacher", "hogteacher"):
            t_enc = copy.deepcopy(enc)
            for p in t_enc.parameters(): p.requires_grad = False
    else:
        head = nn.Sequential(nn.Linear(768, 1024), nn.GELU(), nn.Linear(1024, 256), nn.GELU(), nn.Linear(256, 128)).to(dev)
        params += list(head.parameters())

    if a.init_from:                                   # optional warm start, e.g. from the released weights
        ck = torch.load(a.init_from, map_location=dev, weights_only=True)
        enc.load_state_dict(ck["encoder"])
        if dec is not None and "decoder" in ck:
            try: dec.load_state_dict(ck["decoder"])
            except RuntimeError: log("init_from: decoder shapes differ, decoder left at random initialisation")
        if t_enc is not None: t_enc.load_state_dict(enc.state_dict())

    decay = [p for p in params if p.ndim >= 2]; nodecay = [p for p in params if p.ndim < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": a.wd}, {"params": nodecay, "weight_decay": 0}],
                            lr=a.lr, betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler()
    spe = len(X) // a.bs; total = spe * a.epochs; warm = spe * max(1, min(40, a.epochs // 10))

    def lr_at(s):  # linear warm-up, cosine decay to zero
        return a.lr * s / warm if s < warm else a.lr * 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm)))

    @torch.no_grad()
    def teacher_targets(x):
        t_enc.eval(); _, outs = t_enc(x, return_layers=True)
        return torch.stack([F.layer_norm(o.float(), (768,)) for o in outs[-3:]]).mean(0)[:, 1:]

    def mae_loss(x):
        if a.masking == "random": ids_keep, ids_restore, mask = random_masking(x.shape[0], a.mask, dev)
        else: ids_keep, ids_restore, mask = energy_masking(x, a.mask, a.beta)
        pred = dec(enc(x, ids_keep), ids_restore).float()
        if a.target == "hogteacher":
            tt = teacher_targets(x)
            l = ((pred[..., :36] - hog_targets(x)) ** 2).mean(-1) + ((pred[..., 36:] - tt) ** 2).mean(-1)
        else:
            if a.target == "pix": t = pixel_targets(x)
            elif a.target == "hog": t = hog_targets(x)
            elif a.target == "hogms": t = torch.cat([hog_targets(x, c) for c in (4, 8, 16)], -1)
            else: t = teacher_targets(x)
            l = ((pred - t) ** 2).mean(-1)
        return (l * mask).sum() / mask.sum()            # loss on masked patches only

    hist = []; step = 0; t0 = time.time()
    for ep in range(a.epochs):
        enc.train(); perm = torch.randperm(len(X)); agg = {}
        for i in range(spe):
            for g in opt.param_groups: g["lr"] = lr_at(step)
            xb = X[perm[i * a.bs:(i + 1) * a.bs]].to(dev, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                if a.method == "mae":
                    loss = mae_loss(augment(xb, bool(a.strong))); key = "rec"
                else:
                    v = torch.cat([augment(xb.clone(), True), augment(xb.clone(), True)])
                    loss = nt_xent(head(enc(v)[:, 0])); key = "nce"
            if not torch.isfinite(loss): raise RuntimeError(f"non-finite loss at epoch {ep + 1}")
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(params, 3.0)
            scaler.step(opt); scaler.update(); step += 1
            if t_enc is not None:
                m = 1 - (1 - a.ema) * (math.cos(math.pi * step / total) + 1) / 2
                with torch.no_grad():
                    for ps, pt in zip(enc.parameters(), t_enc.parameters()): pt.mul_(m).add_(ps.detach(), alpha=1 - m)
            agg[key] = agg.get(key, 0) + loss.item() / spe
        agg |= {"ep": ep + 1, "min": round((time.time() - t0) / 60, 2), "lr": lr_at(step)}
        hist.append(agg); log(json.dumps(agg))
        if (ep + 1) % 100 == 0 or ep + 1 == a.epochs:
            ck = {"encoder": enc.state_dict(), "epoch": ep + 1}
            if dec is not None: ck["decoder"] = dec.state_dict()
            torch.save(ck, f"{out}/ckpt.pth"); json.dump(hist, open(f"{out}/history.json", "w"))
    log(f"done {os.path.relpath(out)}")
    return out, hist


def main(argv=None):
    train(parse_args(argv), log=lambda s: print(s, flush=True))


if __name__ == "__main__":
    main()
