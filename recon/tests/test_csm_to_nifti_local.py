"""Tests for lightweight BART CSM RSS conversion."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

from recon.bart.bart_utils.bart_io import write_cfl
from recon.bart.csm_to_nifti_local import compute_csm_rss, save_logical_csm_rss


class LocalCsmToNiftiTests(unittest.TestCase):
    """Validate bounded-memory RSS conversion and synthetic NIfTI geometry."""

    def test_converter_preserves_logical_order_and_synthetic_spacing(self) -> None:
        """Local conversion should collapse only coil and preserve GRE axes."""
        maps = np.zeros((3, 2, 4, 2), dtype=np.complex64)
        maps[..., 0] = 1.0 + 2.0j
        maps[..., 1] = 3.0 + 4.0j
        expected = np.sqrt(np.sum(np.abs(maps) ** 2, axis=3)).astype(np.float32)

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            base = root / "coil_sens"
            output = root / "coil_sens_rss.nii.gz"
            write_cfl(base, maps)
            saved_path = save_logical_csm_rss(
                csm=base,
                output=output,
                voxel_size_mm=(0.7, 0.8, 0.9),
                partition_chunk=2,
            )
            image = nib.load(saved_path)
            actual = np.asarray(image.dataobj)

        self.assertEqual(actual.shape, (3, 2, 4))
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(image.header.get_zooms()[:3], (0.7, 0.8, 0.9))

    def test_converter_rejects_truncated_cfl(self) -> None:
        """A truncated CSM must not be accepted as a complete map volume."""
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder) / "coil_sens"
            write_cfl(base, np.ones((3, 2, 4, 2), dtype=np.complex64))
            with base.with_suffix(".cfl").open("r+b") as stream:
                stream.truncate(stream.seek(0, 2) - np.dtype(np.complex64).itemsize)
            with self.assertRaisesRegex(ValueError, "CFL size mismatch"):
                compute_csm_rss(base)


if __name__ == "__main__":
    unittest.main()
