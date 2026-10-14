# Defect-aware MAE

Label-free representation learning for steel-surface defect clustering. A Vision Transformer encoder is pretrained
with a masked autoencoder adapted to texture defects, and its frozen features are clustered into defect categories.
Pretraining uses **no class labels, no bounding boxes and no contrastive loss**.

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

NEU-DET, 360 held-out validation images that no encoder saw during pretraining. Mean ± std over 5 seeds.
Clustering: standardised features → UMAP 2-D → Ward agglomerative clustering (K = 6) → Hungarian-matched accuracy.

| Metric | Defect-aware MAE | SimCLR baseline | Significant? (Welch t-test) |
|---|---|---|---|
| Accuracy (UMAP + Ward) | **98.6 ± 0.2** | 97.8 ± 1.3 | no (p = 0.29) |
| NMI / ARI | **0.968 / 0.967** | 0.955 / 0.951 | no |
| Silhouette, UMAP 2-D | **0.838 ± 0.019** | 0.736 ± 0.019 | **yes** |
| Silhouette, PCA-50 | **0.503** | 0.197 | **yes** |
| Accuracy, mean over 12 UMAP settings × 10 seeds | 97.4 ± 0.6 | 97.6 ± 0.4 | no |
| Accuracy, PCA-50 + k-means | 95.4 ± 1.0 | 95.4 ± 1.1 | no |
| Accuracy, raw 768-d + k-means | 96.0 ± 0.3 | 95.8 ± 1.2 | no |
| Accuracy, fully inductive (fit on train, assign held-out) | 89.4 ± 4.2 | **95.0 ± 1.1** | borderline (p = 0.055) |
| Linear probe (uses labels) | **99.5 ± 0.1** | 98.8 ± 0.2 | **yes** (p = 0.001) |
| 20-NN probe (uses labels) | 96.3 ± 0.5 | **97.3 ± 0.4** | **yes** (p = 0.019) |

References on the same held-out protocol: raw pixels 51.7%, randomly initialised encoder 51.1%, and an earlier
model trained jointly with a box-supervised detection branch 86.7% (silhouette 0.71).

In short: accuracy on par with a tuned contrastive method, with clearly tighter and more stable clusters.
All tables, ablations and significance tests are in `notebooks/01_reproduce_results.ipynb`.

### Limitations

- **Model selection used the held-out split.** About 70 design variants were compared on the same 360 images that
  produce the reported numbers, with no separate development split, so the numbers are somewhat optimistic.
- **The baseline received less tuning** than the MAE (a learning-rate sweep, 800 epochs vs 1,600).
- **SimCLR is better in the fully inductive setting.** The 1-layer-decoder MAE variant closes that gap (95.3 ± 0.3)
  at slightly lower accuracy (98.0 ± 0.5).
- **One small dataset**; the number of clusters (K = 6) is given.
- **The readout matters.** The MAE's CLS token is a weak feature; results rely on the `mid` readout.
- **Explanations localise defects only modestly** (Grad-CAM energy inside annotated boxes is 1.15× a uniform map;
  random-init control 0.50×; Grad-CAM++ is not better). See `notebooks/02_gradcam_analysis.ipynb`.

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
configs/                settings of the released models; groups.json maps experiments to stored runs
notebooks/              01 reproduce results · 02 Grad-CAM analysis · 03 train from scratch (all executed)
results/runs/<tag>/     eval.json, args.json, history.json for each of the 74 training runs
results/gradcam/        full Grad-CAM metrics for four models
results/pilot/          pilot study and reference baselines (raw pixels, random init, earlier v1 model)
results/provenance/     cloud run log: GPU model, instance and timing of every training run
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

A CUDA GPU is needed for training. Evaluation and the notebooks also run on CPU, slowly.
`requirements.txt` lists the direct dependencies only, if you prefer to resolve versions yourself. The cloud runs
behind the stored results used PyTorch 2.4.0 on Python 3.11. The saved notebook outputs were produced with
PyTorch 2.9.1; the evaluation of the released weights was re-checked with `requirements-lock.txt` (PyTorch 2.4.0)
and gives the same accuracy (98.33%).

### 2. Data

NEU-DET (Song & Yan, Northeastern University; 1,800 images, 6 classes) is not redistributed here. Download the
NEU-DET archive that contains `NEU-DET/train` and `NEU-DET/validation` (for example the "NEU-DET" surface-defect
dataset on Kaggle), then:

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
python scripts/aggregate.py                      # all experiment groups, each with its own readout
python scripts/aggregate.py --readout cls        # same runs, CLS readout
```

### What to expect when reproducing

- **Same seed, same GPU model:** identical numbers. The seed-0 MAE was trained three times on RTX A4000 machines and
  gave 98.33% accuracy and silhouette 0.827 each time.
- **Same seed, different GPU model:** slightly different runs. SimCLR seed 0 gave 99.17% on an RTX 4070S and 98.33% on
  an RTX A4000. Compare against the 5-seed mean ± std, not a single number.
- **Evaluating released weights on another GPU:** half-precision features differ in the last digits and UMAP
  amplifies this, so accuracies can move by a few tenths of a point.
- Runs whose tag starts with `r1`–`r6` or `f_` were produced by an earlier layout of this code. The trainer here
  was verified to produce bit-identical losses to it for the released MAE and SimCLR settings.

## Provenance of stored runs

`configs/groups.json` lists which runs form each table row. Tags encode the round in which a run was made
(`r1` recipe sweep and first targets, `r2`–`r3` masking / augmentation / teacher targets, `r4`–`r6` decoder depth,
mask ratio and training length, `f_` extra seeds, `final_` the released checkpoints). `results/pilot/` holds the
first 200-epoch study, which used a learning rate that was too low for the MAE and is kept only as a record.

## License

Code and released weights: MIT (see `LICENSE`). The NEU-DET dataset belongs to its original authors and is not
covered by this license:

> K. Song and Y. Yan, "A noise robust method based on completed local binary patterns for hot-rolled steel strip
> surface defects," *Applied Surface Science*, vol. 285, pp. 858–864, 2013.
