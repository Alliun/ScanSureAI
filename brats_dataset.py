"""
ScanSure AI v2 — brats_dataset.py
BraTS 2020 / 2023 dataset loader.

BraTS 2020 folder structure (MICCAI_BraTS2020_TrainingData):
    MICCAI_BraTS2020_TrainingData/
    └── BraTS20_Training_001/
        ├── BraTS20_Training_001_flair.nii.gz   ← used for training
        ├── BraTS20_Training_001_t1.nii.gz
        ├── BraTS20_Training_001_t1ce.nii.gz
        ├── BraTS20_Training_001_t2.nii.gz
        └── BraTS20_Training_001_seg.nii.gz     ← ground truth mask
"""

import numpy as np
import torch
from torch.utils.data import Dataset
import nibabel as nib
from pathlib import Path
from PIL import Image


def load_nifti(path: str) -> np.ndarray:
    return nib.load(path).get_fdata().astype(np.float32)


def normalize_volume(volume: np.ndarray) -> np.ndarray:
    p1  = np.percentile(volume, 1)
    p99 = np.percentile(volume, 99)
    volume = np.clip(volume, p1, p99)
    return (volume - p1) / (p99 - p1 + 1e-8)


def binarize_mask(mask: np.ndarray) -> np.ndarray:
    """Any tumor label > 0 becomes 1 (binary lesion mask)."""
    return (mask > 0).astype(np.float32)


def find_flair_and_seg(patient_dir: Path):
    """
    Auto-detect FLAIR and seg files.
    BraTS 2020: *_flair.nii.gz and *_seg.nii.gz
    BraTS 2023: *-t2f.nii.gz  and *-seg.nii.gz
    """
    flair_files = (list(patient_dir.glob("*_flair.nii.gz")) or
                   list(patient_dir.glob("*_flair.nii"))     or
                   list(patient_dir.glob("*-t2f.nii.gz"))    or
                   list(patient_dir.glob("*-t2f.nii")))

    seg_files   = (list(patient_dir.glob("*_seg.nii.gz")) or
                   list(patient_dir.glob("*_seg.nii"))     or
                   list(patient_dir.glob("*-seg.nii.gz"))  or
                   list(patient_dir.glob("*-seg.nii")))

    if not flair_files or not seg_files:
        return None, None

    return str(flair_files[0]), str(seg_files[0])


def get_valid_slices(mask_volume: np.ndarray, min_lesion_ratio: float = 0.001):
    """Return axial slice indices that contain enough lesion pixels."""
    valid = []
    for i in range(mask_volume.shape[2]):
        if mask_volume[:, :, i].mean() >= min_lesion_ratio:
            valid.append(i)
    return valid


class BraTSDataset(Dataset):
    """
    PyTorch Dataset for BraTS 2020/2023 2D axial slice segmentation.

    Args:
        root_dir        : Path to MICCAI_BraTS2020_TrainingData folder
        img_size        : Resize slices to this square size
        min_lesion_ratio: Skip slices with less lesion than this fraction
        max_patients    : Only load this many patients (for quick tests)
    """

    def __init__(self, root_dir, img_size=256, min_lesion_ratio=0.001, max_patients=None):
        self.img_size         = img_size
        self.min_lesion_ratio = min_lesion_ratio
        self.slices           = []

        root         = Path(root_dir)
        patient_dirs = sorted([d for d in root.iterdir() if d.is_dir()])
        if max_patients:
            patient_dirs = patient_dirs[:max_patients]

        print(f"[BraTS] Scanning {len(patient_dirs)} patients...")
        skipped = 0

        for patient_dir in patient_dirs:
            flair_path, seg_path = find_flair_and_seg(patient_dir)
            if not flair_path:
                print(f"  ⚠  Skipping {patient_dir.name} — files not found")
                skipped += 1
                continue

            # Binary mask used only to find valid slices
            mask_vol = binarize_mask(load_nifti(seg_path))
            for s in get_valid_slices(mask_vol, min_lesion_ratio):
                self.slices.append((flair_path, seg_path, s))

        print(f"[BraTS] ✅  Total valid slices: {len(self.slices)}")
        if skipped:
            print(f"[BraTS] ⚠   Skipped: {skipped} patients")

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        flair_path, seg_path, slice_idx = self.slices[idx]

        # Load and normalize FLAIR volume
        flair_vol = normalize_volume(load_nifti(flair_path))
        # Load and binarize mask volume
        seg_vol   = binarize_mask(load_nifti(seg_path))

        # Extract 2D axial slices
        image_2d = flair_vol[:, :, slice_idx]
        mask_2d  = seg_vol[:, :, slice_idx]

        # Resize — BILINEAR for image, NEAREST for mask (preserve binary values)
        image_2d = self._resize(image_2d, Image.BILINEAR)
        mask_2d  = self._resize(mask_2d,  Image.NEAREST)

        # Ensure mask is strictly binary after resize
        mask_2d = (mask_2d > 0.5).astype(np.float32)

        # Add channel dim → [1, H, W]
        return (torch.from_numpy(image_2d).unsqueeze(0),
                torch.from_numpy(mask_2d).unsqueeze(0))

    def _resize(self, arr: np.ndarray, resample) -> np.ndarray:
        """Resize 2D numpy array to (img_size, img_size)."""
        pil = Image.fromarray(arr).resize((self.img_size, self.img_size), resample)
        return np.array(pil, dtype=np.float32)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python brats_dataset.py C:\\path\\to\\MICCAI_BraTS2020_TrainingData")
        sys.exit(1)

    ds = BraTSDataset(root_dir=sys.argv[1], max_patients=3)
    if len(ds) == 0:
        print("No valid slices found. Check your folder structure.")
    else:
        img, mask = ds[0]
        print(f"Image : {img.shape}  range [{img.min():.3f}, {img.max():.3f}]")
        print(f"Mask  : {mask.shape} unique {mask.unique().tolist()}")
        print(f"Lesion pixels: {mask.mean()*100:.2f}%")
