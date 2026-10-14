"""Fast checks that need neither the dataset nor a GPU."""
import numpy as np, torch
from defect_aware_mae.data import N
from defect_aware_mae.evaluate import hungarian_accuracy
from defect_aware_mae.models import Decoder, Encoder, readout
from defect_aware_mae.objectives import energy_masking, hog_targets, nt_xent, pixel_targets, random_masking


def test_shapes_and_masking():
    torch.manual_seed(0); x = torch.rand(2, 1, 224, 224)
    assert hog_targets(x).shape == (2, N, 36) and pixel_targets(x).shape == (2, N, 256)
    for ids_keep, ids_restore, mask in (random_masking(2, 0.75, "cpu"), energy_masking(x, 0.75)):
        assert ids_keep.shape == (2, 49) and mask.sum(1).tolist() == [147, 147]
    enc, dec = Encoder(), Decoder(depth=2, out_dim=36)
    ids_keep, ids_restore, _ = energy_masking(x, 0.75)
    assert dec(enc(x, ids_keep), ids_restore).shape == (2, N, 36)
    for m in ("cls", "mean", "mid"): assert readout(enc, x, m).shape == (2, 768)
    assert torch.isfinite(nt_xent(torch.randn(8, 128)))


def test_energy_masking_prefers_textured_patches():
    torch.manual_seed(0); x = torch.full((64, 1, 224, 224), 0.5)
    x[:, :, :112] += 0.3 * torch.randn(64, 1, 112, 224)          # top half textured, bottom half flat
    _, _, mask = energy_masking(x.clamp(0, 1), 0.5, beta=2.0)
    m = mask.float().reshape(64, 14, 14)
    assert m[:, :7].mean() > 0.8 > 0.2 > m[:, 7:].mean()


def test_hungarian_accuracy():
    y = np.array([0, 0, 1, 1, 2, 2]); assert hungarian_accuracy(y, np.array([2, 2, 0, 0, 1, 1])) == 1.0
    assert abs(hungarian_accuracy(y, np.array([2, 2, 0, 1, 1, 1])) - 5 / 6) < 1e-9
