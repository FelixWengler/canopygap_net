from pathlib import Path
import torch
from torch.utils.data import Dataset
import rasterio
import numpy as np

class S2S1DSMTileFolderDataset(Dataset):
    """
    Dataset that reads tiled Sentinel-2 + Sentinel-1 + CG Fraction chips from folders.

    Expected structure:
      root/
        S2_0608/
          x0085_y0059_2018.tif        (C=10, H=256, W=256)
        S1_0608/
          x0085_y0059_2018.tif        (C=2,  H=128, W=128)  [VH,VV]
        CG/
          x0085_y0059_2018.tif        (C=1,  H=256, W=256)
    """

    def __init__(
        self,
        root_dir,
        s2_subdir="S2",
        s1_subdir="S1",
        dsm_subdir="grfra",
        s2_divisor=10000.0,
        s2_clamp01=True,
        s1_nodata=-32768.0,
        s1_use_log1p=True,
        transforms=None,
    ):
        self.root = Path(root_dir)
        self.s2_dir = self.root / s2_subdir
        self.s1_dir = self.root / s1_subdir
        self.dsm_dir = self.root / dsm_subdir

        if not self.s2_dir.exists():
            raise FileNotFoundError(f"Missing S2 directory: {self.s2_dir}")
        if not self.s1_dir.exists():
            raise FileNotFoundError(f"Missing S1 directory: {self.s1_dir}")
        if not self.dsm_dir.exists():
            raise FileNotFoundError(f"Missing DSM directory: {self.dsm_dir}")

        self.s2_divisor = float(s2_divisor)
        self.s2_clamp01 = bool(s2_clamp01)
        self.s1_nodata = float(s1_nodata)
        self.s1_use_log1p = bool(s1_use_log1p)
        self.transforms = transforms

        # build list of triplets: require S2, S1, CG all exist with same name
        self.files = sorted([
            f for f in self.s2_dir.glob("*.tif")
            if (self.s1_dir / f.name).exists() and (self.dsm_dir / f.name).exists()
        ])

        if len(self.files) == 0:
            raise RuntimeError(f"No paired S2/S1/BDOM tiles found in {root_dir}")

    def __len__(self):
        return len(self.files)

    @staticmethod
    def _read(path: Path) -> np.ndarray:
        with rasterio.open(path) as src:
            arr = src.read()  # (C,H,W) or (1,H,W)
        return arr.astype(np.float32)

    def __getitem__(self, idx):
        s2_path = self.files[idx]
        s1_path = self.s1_dir / s2_path.name
        dsm_path = self.dsm_dir / s2_path.name

        # ---- S2 (10m) ----
        s2 = torch.from_numpy(self._read(s2_path)).float()   # (C,256,256)
        s2 = s2 / self.s2_divisor
        if self.s2_clamp01:
            s2 = torch.clamp(s2, 0.0, 1.0)

        # ---- S1 (20m) ----
        s1 = torch.from_numpy(self._read(s1_path)).float()   # (2,128,128) [VH,VV]
        # nodata handling
        if self.s1_nodata is not None:
            s1 = torch.where(s1 == self.s1_nodata, torch.zeros_like(s1), s1)

        # range compression (recommended for linear-scale SAR)
        if self.s1_use_log1p:
            s1 = torch.log1p(torch.clamp(s1, min=0.0))

        # Optional: add log-ratio channel 
  
        vh = s1[0:1]
        vv = s1[1:2]
        ratio = vh - vv  # in log space: log(VH/VV)
        s1 = torch.cat([s1, ratio], dim=0)  # now (3,128,128)

        # ---- LABEL ----
        dsm = torch.from_numpy(self._read(dsm_path)).float()

        if dsm.ndim == 2:
            dsm = dsm[None, ...]

        dsm = dsm / 100.0          # convert percent → fraction
        dsm = torch.clamp(dsm, 0.0, 1.0)

        sample = {"s2": s2, "s1": s1, "label": dsm}

        if self.transforms:
            sample = self.transforms(sample)

        return sample