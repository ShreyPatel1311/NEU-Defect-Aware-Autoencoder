"""Download released weights from the Hugging Face Hub."""
from pathlib import Path
from .data import repo_root

REPO_ID = "Godseye1311/defect-aware-MAE"
WEIGHTS = {"defect_aware_mae": ("defect_aware_mae_seed0.pth", "mid"),   # (file, recommended readout)
           "simclr": ("simclr_baseline_seed0.pth", "cls")}


def download(name="defect_aware_mae", local_dir=None):
    """Returns (path to checkpoint, recommended readout). Files are cached under <repo>/weights/."""
    from huggingface_hub import hf_hub_download
    fn, mode = WEIGHTS[name]
    local_dir = Path(local_dir or repo_root() / "weights")
    return hf_hub_download(REPO_ID, fn, local_dir=local_dir), mode
