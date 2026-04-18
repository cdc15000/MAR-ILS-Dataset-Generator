#!/usr/bin/env python3
"""
reference_dlsc_imar.py
======================
Deep Learning Sinogram Completion iMAR (DLSC-iMAR) — per-realization
Deep Image Prior for metal-trace sinogram inpainting.
ASTM WKXXXXX v1.0.0 — Tier-aware.

Motivation
----------
Classical sinogram-domain MAR (iMAR, ASD-iMAR) replaces metal-trace rays
with values derived from analytic priors or local diffusion.  In high-
blockage tiers (T2_SB: 40%, 1.5 mm gap) the lesion signal is lost because
neither the smooth prior nor local PDE propagation can reconstruct the
missing Radon-space projections.

Deep Image Prior (DIP) uses a *randomly initialised* convolutional network
as an implicit structural prior.  The network is optimised per-realization
to reproduce the clean rays of the measured sinogram.  Its multi-scale
convolutional architecture (U-Net with skip connections) forces globally
coherent interpolation into the metal trace — a fundamentally stronger
prior than local PDE diffusion.

Algorithm (per realization)
---------------------------
  1. Metal mask + weight matrix       : Same as iMAR.
  2. Fixed random input z             : z ~ N(0, 0.1²), shape (1, 8, 384, 512).
  3. DIP optimisation loop (N iter)   :
       a. z_noisy = z + (1/30)·ε, ε ~ N(0,1)  (input noise regularisation)
       b. sino_pred = f_θ(z_noisy)
       c. L = MSE(sino_pred[clean], sino_meas[clean])
            + λ_c · Consistency(sino_pred[metal])
       d. Adam step on θ; cosine LR decay.
  4. Extract best sinogram            : sino_dlsc = best f_θ*(z) by clean MSE.
  5. Enforce clean rays               : sino_dlsc[clean] = sino_meas[clean].
  6. SG continuity enforcement        : Same as iMAR.
  7. FBP + HU calibration + metal     : Same as iMAR.

Sinogram Consistency Loss
-------------------------
  L_consistency = mean(ReLU(-pred[metal])²)        — non-negativity
                + mean(ReLU(pred[metal] - ub)²)    — upper bound
  where ub = 1.5 × max(sino_meas[clean]) (normalised), preventing
  the network from generating physically impossible attenuation values.

Device: torch.device("mps") (Apple Silicon).
Processing: sequential (one MPS training loop per realization).

Outputs
-------
  <input_dir>/dlsc_imar_recon/LP/realization_NNN/slice_0129.dcm  (N=1..40)
  <input_dir>/dlsc_imar_recon/LA/realization_NNN/slice_0129.dcm  (N=1..40)
"""

import argparse
import copy
import os
import time
import warnings
from pathlib import Path

import h5py
import numpy as np
import pydicom
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError as e:
    raise ImportError(
        "reference_dlsc_imar.py requires PyTorch (optional dependency, "
        "not listed in requirements.txt). Install with: pip install torch"
    ) from e
from pydicom.uid import generate_uid
from scipy.ndimage import gaussian_filter
from scipy.signal import savgol_filter
from skimage.transform import iradon, radon

from tier_config import (
    BACKGROUND_HU,
    TIER_REGISTRY,
    TierConfig,
)

# ---------------------------------------------------------------------------
# Normative constants (tier-independent)
# ---------------------------------------------------------------------------
LESION_SLICE_INDEX = 128
N_ANGLES           = 360
N_DETECTORS        = 512
THETA_DEG          = np.linspace(0, 180, N_ANGLES, endpoint=False)
METAL_HU_THRESH    = 1500
METAL_HU           = 3000.0
N_REALIZATIONS     = 40

MU_WATER     = 0.2059
VOXEL_CM     = 0.05
SCATTER_FRAC = 0.05

SG_WINDOW = 9
SG_ORDER  = 3

# ---------------------------------------------------------------------------
# DIP configuration
# ---------------------------------------------------------------------------
PAD_H              = 384       # pad 360 → 384 (divisible by 8 for 3 pool layers)
DIP_IN_CHANNELS    = 8         # random noise input channels
DIP_BASE_CHANNELS  = 32        # U-Net base width (→ 32, 64, 128)
DEFAULT_N_ITER     = 500       # DIP optimisation iterations per realization
DEFAULT_LR         = 0.01      # Adam learning rate
DEFAULT_LAM_C      = 0.1       # consistency loss weight
INPUT_NOISE_STD    = 1.0/30.0  # input noise regularisation σ
BEST_AFTER         = 100       # only track best output after this iteration

DEVICE = torch.device("mps")


# ---------------------------------------------------------------------------
# U-Net architecture
# ---------------------------------------------------------------------------

class _DoubleConv(nn.Module):
    """Conv3×3 → BN → LeakyReLU → Conv3×3 → BN → LeakyReLU."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class SinoUNet(nn.Module):
    """
    Lightweight 3-level U-Net for sinogram inpainting.

    Input : (B, 8,   384, 512) — fixed random noise
    Output: (B, 1,   384, 512) — predicted sinogram (normalised)

    Encoder: 32 → 64 → 128;  Bottleneck: 128;  Decoder: 128 → 64 → 32 → 1
    ~900 K parameters.
    """
    def __init__(self, in_ch: int = DIP_IN_CHANNELS,
                 base: int = DIP_BASE_CHANNELS):
        super().__init__()
        self.enc1 = _DoubleConv(in_ch, base)
        self.enc2 = _DoubleConv(base, base * 2)
        self.enc3 = _DoubleConv(base * 2, base * 4)
        self.bottleneck = _DoubleConv(base * 4, base * 4)

        self.up3  = nn.ConvTranspose2d(base * 4, base * 4, 2, stride=2)
        self.dec3 = _DoubleConv(base * 8, base * 2)
        self.up2  = nn.ConvTranspose2d(base * 2, base * 2, 2, stride=2)
        self.dec2 = _DoubleConv(base * 4, base)
        self.up1  = nn.ConvTranspose2d(base, base, 2, stride=2)
        self.dec1 = _DoubleConv(base * 2, base)

        self.out_conv = nn.Conv2d(base, 1, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b  = self.bottleneck(self.pool(e3))

        d3 = self.dec3(torch.cat([self.up3(b), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))

        return self.out_conv(d1)


# ---------------------------------------------------------------------------
# Deep Image Prior inpainting
# ---------------------------------------------------------------------------

def dip_inpaint(
    sino_meas:   np.ndarray,   # (360, 512)
    clean_mask:  np.ndarray,   # (360, 512) bool
    n_iter:      int   = DEFAULT_N_ITER,
    lr:          float = DEFAULT_LR,
    lam_c:       float = DEFAULT_LAM_C,
    verbose:     bool  = False,
) -> np.ndarray:
    """
    Run Deep Image Prior on one sinogram.
    Returns inpainted sinogram (360, 512) in original neper scale.
    """
    metal_mask = ~clean_mask

    # ── Normalise to [0, ~1] ────────────────────────────────────────────
    sino_max = float(np.max(np.abs(sino_meas[clean_mask]))) + 1e-8
    sino_norm = sino_meas / sino_max

    # ── Pad angles 360 → 384 ───────────────────────────────────────────
    pad_top = (PAD_H - N_ANGLES) // 2        # 12
    pad_bot = PAD_H - N_ANGLES - pad_top     # 12

    sino_pad  = np.pad(sino_norm, ((pad_top, pad_bot), (0, 0)), mode="edge")
    clean_pad = np.pad(clean_mask, ((pad_top, pad_bot), (0, 0)),
                       mode="constant", constant_values=True)
    metal_pad = ~clean_pad

    # ── Tensors on MPS ──────────────────────────────────────────────────
    target_t = torch.from_numpy(sino_pad).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
    cmask_t  = torch.from_numpy(clean_pad.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(DEVICE)
    mmask_t  = torch.from_numpy(metal_pad.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(DEVICE)

    # ── Fixed noise input ───────────────────────────────────────────────
    z = torch.randn(1, DIP_IN_CHANNELS, PAD_H, N_DETECTORS, device=DEVICE) * 0.1

    # ── Network + optimiser ─────────────────────────────────────────────
    net = SinoUNet().to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_iter)

    best_loss = float("inf")
    best_out  = None

    for it in range(n_iter):
        # Input noise regularisation (prevents overfitting)
        z_in = z + INPUT_NOISE_STD * torch.randn_like(z)

        pred = net(z_in)

        # ── Clean-ray reconstruction loss ───────────────────────────────
        loss_clean = F.mse_loss(pred * cmask_t, target_t * cmask_t)

        # ── Sinogram consistency loss on metal trace ────────────────────
        pred_metal = pred * mmask_t
        loss_neg  = torch.mean(F.relu(-pred_metal) ** 2)
        loss_high = torch.mean(F.relu(pred_metal - 1.5) ** 2)
        loss_cons = loss_neg + loss_high

        loss = loss_clean + lam_c * loss_cons

        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()

        # Track best (after warm-up)
        with torch.no_grad():
            if it >= BEST_AFTER and loss_clean.item() < best_loss:
                best_loss = loss_clean.item()
                best_out = pred.detach().clone()

        if verbose and (it + 1) % 100 == 0:
            print(f"    iter {it+1:4d}  L_clean={loss_clean.item():.6f}  "
                  f"L_cons={loss_cons.item():.6f}")

    # ── Extract result ──────────────────────────────────────────────────
    with torch.no_grad():
        if best_out is None:
            best_out = net(z)
        result_pad = best_out.squeeze().cpu().numpy()

    # Unpad + denormalise
    result = result_pad[pad_top:pad_top + N_ANGLES, :] * sino_max

    # Hard-pin clean rays to measured values (Dirichlet)
    result[clean_mask] = sino_meas[clean_mask]

    return result


# ---------------------------------------------------------------------------
# Core routines
# ---------------------------------------------------------------------------

def _fbp(sino: np.ndarray) -> np.ndarray:
    return iradon(sino.T, theta=THETA_DEG, filter_name="ramp", circle=True)


def _fwd(img: np.ndarray) -> np.ndarray:
    return radon(img, theta=THETA_DEG, circle=True).T


# ---------------------------------------------------------------------------
# DLSC-iMAR core (single 2D slice)
# ---------------------------------------------------------------------------

def dlsc_imar_slice(
    sino_meas: np.ndarray,
    ref_hu:    np.ndarray,
    tier:      TierConfig,
    n_iter:    int   = DEFAULT_N_ITER,
    lr:        float = DEFAULT_LR,
    verbose:   bool  = False,
) -> np.ndarray:
    """
    Parameters
    ----------
    sino_meas : (360, 512) float64  measured line integrals (neper)
    ref_hu    : (512, 512) float64  noMAR FBP image (DICOM HU)
    tier      : TierConfig          tier geometry
    n_iter    : int                 DIP iterations
    lr        : float               Adam learning rate
    verbose   : bool                print per-100-iter loss

    Returns
    -------
    img_dlsc : (512, 512) float64  DLSC-iMAR-corrected image in HU
    """
    # ── 1. Metal mask and weight matrix ──────────────────────────────────
    metal_mask     = ref_hu > METAL_HU_THRESH
    metal_sino     = _fwd(metal_mask.astype(float))
    per_angle_peak = metal_sino.max(axis=1, keepdims=True)
    W = (metal_sino < 0.05 * np.maximum(per_angle_peak, 1e-6)).astype(float)
    metal_trace = W < 0.5
    clean_mask  = ~metal_trace

    # ── 2. DIP sinogram inpainting ──────────────────────────────────────
    sino_dlsc = dip_inpaint(
        sino_meas, clean_mask, n_iter=n_iter, lr=lr, verbose=verbose,
    )

    # ── 3. Savitzky-Golay continuity enforcement ────────────────────────
    sino_smooth = sino_dlsc.copy()
    hw = SG_WINDOW // 2 + 2
    for a in range(sino_smooth.shape[0]):
        metal_cols = np.where(metal_trace[a])[0]
        if metal_cols.size == 0:
            continue
        for boundary in (int(metal_cols.min()), int(metal_cols.max())):
            lo = max(0, boundary - hw)
            hi = min(sino_smooth.shape[1], boundary + hw + 1)
            seg = sino_smooth[a, lo:hi].copy()
            n = len(seg)
            if n < SG_WINDOW:
                continue
            wl = SG_WINDOW if SG_WINDOW <= n else (n if n % 2 == 1 else n - 1)
            if wl > SG_ORDER:
                sino_smooth[a, lo:hi] = savgol_filter(seg, wl, SG_ORDER)

    # ── 4. Final FBP ────────────────────────────────────────────────────
    x = _fbp(sino_smooth)

    # ── 5. HU calibration ───────────────────────────────────────────────
    x_hu = (x / (MU_WATER * VOXEL_CM) - 1.0) * 1000.0
    _rows, _cols = np.mgrid[0:512, 0:512]
    body_ellipse = (
        (_cols - 256.0) ** 2 / float(tier.body_semi_x_vox ** 2) +
        (_rows - 256.0) ** 2 / float(tier.body_semi_y_vox ** 2)
    ) < 1.0
    body_mask = body_ellipse & ~metal_mask
    body_mean = float(np.mean(x_hu[body_mask]))
    x_hu = x_hu + (BACKGROUND_HU - body_mean)

    # ── 6. Restore metal voxels ─────────────────────────────────────────
    x_hu[metal_mask] = METAL_HU

    return x_hu


# ---------------------------------------------------------------------------
# DICOM I/O helpers
# ---------------------------------------------------------------------------

def _load_noMAR_dicom(dcm_path: Path) -> tuple:
    dcm = pydicom.dcmread(str(dcm_path))
    hu  = dcm.pixel_array.astype(float) * float(dcm.RescaleSlope) + float(dcm.RescaleIntercept)
    return hu, dcm


def _save_dlsc_dicom(img_hu: np.ndarray, template_dcm, out_path: Path,
                     n_iter: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dcm = copy.deepcopy(template_dcm)

    slope     = float(dcm.RescaleSlope)
    intercept = float(dcm.RescaleIntercept)
    pixel_arr = np.clip(
        np.round((img_hu - intercept) / slope), -32768, 32767
    ).astype(np.int16)
    dcm.PixelData = pixel_arr.tobytes()

    dcm.SeriesInstanceUID = generate_uid()
    dcm.SOPInstanceUID    = generate_uid()
    dcm.SeriesDescription = f"DLSC-iMAR (DIP N={n_iter})"

    dcm.save_as(str(out_path))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    warnings.warn(
        "This algorithm uses parallel-beam geometry (Rev 03). "
        "It is NOT compatible with v7.0.0 fan-beam datasets.",
        DeprecationWarning, stacklevel=2,
    )
    ap = argparse.ArgumentParser(
        description="DLSC-iMAR benchmark — Deep Image Prior sinogram completion")
    ap.add_argument("--input-dir", default="./astm_reference_dataset",
                    help="Root of the tier dataset")
    ap.add_argument("--tier", default="T1_AB",
                    choices=list(TIER_REGISTRY.keys()),
                    help="Imaging scenario tier (default: T1_AB)")
    ap.add_argument("--n-iter", type=int, default=DEFAULT_N_ITER,
                    help=f"DIP iterations per realization (default: {DEFAULT_N_ITER})")
    ap.add_argument("--lr", type=float, default=DEFAULT_LR,
                    help=f"Adam learning rate (default: {DEFAULT_LR})")
    ap.add_argument("--verbose", action="store_true",
                    help="Print per-100-iter loss for each realization")
    args = ap.parse_args()

    tier        = TIER_REGISTRY[args.tier]
    dataset_dir = Path(args.input_dir).resolve()
    output_dir  = dataset_dir / "dlsc_imar_recon"

    print(f"Tier       : {tier.tier_id}  —  {tier.description}")
    print(f"Body       : {tier.body_semi_x_vox}×{tier.body_semi_y_vox} vox  "
          f"({'circle' if tier.is_circular_body else 'ellipse'})")
    print(f"Metal      : {tier.metal_material}  μ={tier.metal_mu_cm} cm⁻¹  "
          f"r={tier.metal_radius_vox} vox  blockage={tier.blockage_frac*100:.1f}%")
    print(f"Lesion     : {tier.lesion_semi_major_vox}×{tier.lesion_semi_minor_vox} vox  "
          f"gap={tier.gap_vox} vox ({tier.gap_mm:.1f} mm)")
    print(f"DIP        : N={args.n_iter}  lr={args.lr}  λ_c={DEFAULT_LAM_C}  "
          f"device={DEVICE}")
    print(f"Dataset    : {dataset_dir}")
    print(f"Output     : {output_dir}")
    print(f"Processing : sequential (one MPS training loop per realization)")
    print()

    total = 2 * N_REALIZATIONS
    done  = 0
    t0    = time.time()

    for cond in ("LP", "LA"):
        for r in range(1, N_REALIZATIONS + 1):
            tag     = f"realization_{r:03d}"
            h5_path = dataset_dir / "sinograms" / cond / f"{tag}.h5"
            dcm_src = dataset_dir / "noMAR_recon" / cond / tag / "slice_0129.dcm"
            dcm_dst = output_dir  / cond / tag / "slice_0129.dcm"

            if dcm_dst.exists():
                done += 1
                print(f"  [{done:3d}/{total}] {cond}/{tag} skip")
                continue

            with h5py.File(str(h5_path), "r") as f:
                sino = f["line_integrals"][LESION_SLICE_INDEX].astype(np.float64)

            ref_hu, template_dcm = _load_noMAR_dicom(dcm_src)

            t_start = time.time()
            img_dlsc = dlsc_imar_slice(
                sino, ref_hu, tier,
                n_iter=args.n_iter, lr=args.lr, verbose=args.verbose,
            )
            dt = time.time() - t_start

            _save_dlsc_dicom(img_dlsc, template_dcm, dcm_dst, args.n_iter)
            done += 1

            elapsed = time.time() - t0
            eta = elapsed / done * (total - done) if done > 0 else 0
            print(f"  [{done:3d}/{total}] {cond}/{tag} done  "
                  f"({dt:.1f}s)  ETA {eta:.0f}s", flush=True)

    elapsed_total = time.time() - t0
    print(f"\nDLSC-iMAR complete in {elapsed_total:.0f}s → {output_dir}")
    print("Next step:")
    print(f"  python run_cho_analysis_v6_0.py \\")
    print(f"      --dataset-dir {dataset_dir} \\")
    print(f"      --mar-output-dir {output_dir} \\")
    print(f"      --tier {args.tier} \\")
    print(f"      --internal-noise-sigma 15 \\")
    print(f"      --results-file dlsc_imar_{args.tier.lower()}_results.json")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
