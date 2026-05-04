"""BaseAdapter ABC for baseline reproductions.

Contract: every baseline must produce a [T_out, J_b, 13] numpy array
matching the target skeleton's joint ordering (cond_dict[skel_b]['joints_names']).

Channel layout (13-dim):
  0:3   root-relative joint position
  3:9   6D rotation
  9:12  velocity
  12    binary foot contact
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict
import numpy as np


class BaseAdapter(ABC):
    """Abstract base class for baseline adapters."""

    name: str = "abstract"

    def __init__(self, cond_dict, contact_groups=None):
        self.cond_dict = cond_dict
        self.contact_groups = contact_groups

    @abstractmethod
    def prepare(self, pair: Dict[str, Any]) -> Any:
        """Convert a pair entry to method-specific input.

        Args:
          pair: dict with keys skel_a, skel_b, file_a, file_b, action, pair_id

        Returns:
          Method-specific input (passed to run())
        """
        pass

    @abstractmethod
    def run(self, inputs: Any) -> np.ndarray:
        """Execute the baseline on inputs.

        Returns:
          [T_out, J_b, 13] motion on the target skeleton
        """
        pass

    def postprocess(self, output: np.ndarray, pair: Dict[str, Any]) -> np.ndarray:
        """Optional post-processing. Default: return as-is."""
        return output

    def save_output(self, motion: np.ndarray, pair_id: int, out_dir: Path) -> Path:
        """Save output following the unified contract."""
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f'pair_{pair_id:04d}.npy'
        np.save(out_path, motion.astype(np.float32))
        return out_path
