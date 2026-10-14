# Defect-aware MAE

Label-free representation learning for steel-surface defect clustering. A Vision Transformer encoder is pretrained
with a masked autoencoder adapted to texture defects, and its frozen features are clustered into defect categories.

Released weights: <https://huggingface.co/Godseye1311/defect-aware-MAE>

## Method

| Component | Standard MAE | Defect-aware MAE |
|---|---|---|
| Reconstruction target | per-patch normalised pixels | per-patch HOG descriptor (9 orientation bins × 4 cells = 36-d, L2-normalised) |
| Masking (75%) | uniform random | defect-aware: patches with high local texture energy (pixel std) are masked more often (Gumbel-top-k, β = 1) |
| Augmentation | flips, 90° rotations, jitter, noise | plus random-resized-crop and blur |
| Decoder | 4 layers | 2 layers (256-d, 8 heads) |
| Image feature | CLS token | mean-pooled patch tokens of blocks 4–6, each instance-normalised, averaged (`mid`) |

Encoder: 224×224 grayscale → 196 patches of 16×16 → 6 pre-norm Transformer layers, 12 heads, 768-d.
Optimiser: AdamW (β = 0.9, 0.95), peak lr 1.5e-4, weight decay 0.05, batch 64, 40-epoch warm-up, cosine decay,
1,600 epochs, mixed precision. The contrastive baseline (SimCLR, NT-Xent, τ = 0.2) uses the same encoder,
optimiser and schedule for 800 epochs.

## Results

NEU-DET, 360 held-out validation images that no encoder saw during pretraining. Features are standardised,
projected to 2-D with UMAP and grouped by Ward agglomerative clustering (K = 6); accuracy is Hungarian-matched.
Mean ± std over 5 seeds.

| Model | Accuracy (%) | NMI | ARI | Silhouette |
|---|---|---|---|---|
| SimCLR baseline | 97.8 ± 1.3 | 0.955 ± 0.021 | 0.951 ± 0.027 | 0.736 ± 0.019 |
| **Defect-aware MAE** | **98.6 ± 0.2** | **0.968 ± 0.003** | **0.967 ± 0.004** | **0.838 ± 0.019** |

## Repository layout

```
src/defect_aware_mae/   installable package
  data.py               NEU-DET loading, CLAHE preprocessing, checksums
  augment.py            GPU augmentation (weak / strong)
  models.py             ViT encoder, MAE decoder, readouts, feature extraction
  objectives.py         HOG / pixel targets, random and defect-aware masking, NT-Xent
  train.py              MAE and SimCLR pretraining
  evaluate.py           held-out clustering protocol, robustness sweep, probes
  gradcam.py            Grad-CAM / Grad-CAM++ with localisation and faithfulness scores
  report.py             seed aggregation, tables, significance tests
  hub.py                download of released weights
scripts/                prepare_data.py, train.py, evaluate.py, gradcam.py, aggregate.py
configs/                settings of the released models; groups.json lists the stored runs of each model
notebooks/              01 reproduce results · 02 Grad-CAM analysis · 03 train from scratch (all executed)
results/runs/<tag>/     eval.json, args.json, history.json of each run (5 seeds per model + released checkpoints)
results/gradcam/        full Grad-CAM metrics for four models
figures/                figures produced by the notebooks
tests/                  fast unit tests (no data or GPU needed)
```

## Reproducing the results

### 1. Install

```bash
git clone <this repository> && cd defect-aware-MAE
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-lock.txt   # exact versions of every package (Python 3.12, CUDA 12.1 wheels)
pip install -e .
pytest -q                              # optional: 3 fast tests
```

`requirements.txt` lists the direct dependencies only. The cloud runs
behind the stored results used PyTorch 2.4.0 on Python 3.11; the notebooks were executed with `requirements-lock.txt`.

### 2. Data

Download the NEU-DET archive that contains `NEU-DET/train` and `NEU-DET/validation`, then:

```bash
python scripts/prepare_data.py --archive /path/to/archive.zip
```

This unpacks to `data/NEU-DET/`, builds `data/cache/{train,validation}.npz` (CLAHE with clip limit 2.0 and 8×8
tiles, resize to 224×224) and prints an md5 of each split. `matches reference` means the inputs are byte-identical
to the ones behind the released results. Split: 1,440 train / 360 validation images (240 / 60 per class).
Set `DAMAE_DATA=/some/dir` to keep the data elsewhere.

### 3. Check the released weights (minutes, no training)

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/01_reproduce_results.ipynb
```

or from the command line:

```bash
python -c "from defect_aware_mae import hub; print(hub.download('defect_aware_mae')[0])"
python scripts/evaluate.py weights/defect_aware_mae_seed0.pth --modes mid
python scripts/evaluate.py weights/simclr_baseline_seed0.pth --modes cls
```

Expected for the released seed-0 checkpoints (stored when they were trained, RTX A4000):

| Checkpoint | Readout | Accuracy | NMI | Silhouette |
|---|---|---|---|---|
| `defect_aware_mae_seed0.pth` | `mid` | 98.33 | 0.963 | 0.827 |
| `simclr_baseline_seed0.pth` | `cls` | 98.33 | 0.955 | 0.738 |

### 4. Retrain

```bash
python scripts/train.py --config configs/defect_aware_mae.json --seed 0 --tag damae_s0   # ~50 min on an RTX A4000
python scripts/evaluate.py runs/damae_s0/ckpt.pth --out runs/damae_s0/eval.json
python scripts/gradcam.py  runs/damae_s0/ckpt.pth damae_s0 --mode mid

python scripts/train.py --config configs/simclr.json --seed 0 --tag simclr_s0            # ~90 min
python scripts/evaluate.py runs/simclr_s0/ckpt.pth --out runs/simclr_s0/eval.json
```

Use seeds 0–4 for the 5-seed tables. Any stored run can be repeated from its `results/runs/<tag>/args.json`:

```bash
python scripts/train.py --config results/runs/r5_dec2_ep1600_s3/args.json --tag rerun
```

`--init_from <checkpoint>` starts from existing weights (see `notebooks/03_train_from_scratch.ipynb`).

### 5. Rebuild the tables

```bash
python scripts/aggregate.py                      # results table, each model with its own readout
python scripts/aggregate.py --readout cls        # same runs, CLS readout
```

## License

Code and released weights: MIT (see `LICENSE`). The NEU-DET dataset belongs to its original authors and is not
covered by this license:

> K. Song and Y. Yan, "A noise robust method based on completed local binary patterns for hot-rolled steel strip
> surface defects," *Applied Surface Science*, vol. 285, pp. 858–864, 2013.

> X. Lv, F. Duan, J. Jiang, X. Fu, and L. Gan, "Deep metallic surface defect detection: The new benchmark and
> detection network," *Sensors*, vol. 20, no. 6, p. 1562, 2020.
