from __future__ import annotations

import unittest

import numpy as np
import torch

from recon.utils.coil_compression_kspace import remove_readout_oversampling_kspace


def _centered_fft(array: np.ndarray, axis: int = 0) -> np.ndarray:
    """Return the orthonormal centered FFT of an image array.

    Args:
        array: Complex image array.
        axis: Transform axis.

    Returns:
        Centered complex k-space.
    """

    return np.fft.fftshift(
        np.fft.fft(
            np.fft.ifftshift(array, axes=(axis,)),
            axis=axis,
            norm="ortho",
        ),
        axes=(axis,),
    )


def _centered_ifft(array: np.ndarray, axis: int = 0) -> np.ndarray:
    """Return the orthonormal centered inverse FFT of a k-space array.

    Args:
        array: Complex centered k-space.
        axis: Transform axis.

    Returns:
        Centered complex image.
    """

    return np.fft.fftshift(
        np.fft.ifft(
            np.fft.ifftshift(array, axes=(axis,)),
            axis=axis,
            norm="ortho",
        ),
        axes=(axis,),
    )


def _embed_center(array: np.ndarray, oversampling_factor: int) -> np.ndarray:
    """Embed a logical-FOV image in a larger readout FOV.

    Args:
        array: Logical-FOV image with readout first.
        oversampling_factor: Integer target readout expansion.

    Returns:
        Zero-padded oversampled-FOV image.
    """

    shape = list(array.shape)
    shape[0] *= oversampling_factor
    embedded = np.zeros(shape, dtype=array.dtype)
    start = shape[0] // 2 - array.shape[0] // 2
    embedded[start : start + array.shape[0]] = array
    return embedded


class ReadoutOversamplingTests(unittest.TestCase):
    """Verify alias-free readout de-oversampling for GRE coil calibration."""

    def test_numpy_recovers_centered_logical_fov(self) -> None:
        """Image-domain cropping should exactly recover logical k-space."""

        rng = np.random.default_rng(7)
        image = (
            rng.standard_normal((8, 3, 2))
            + 1j * rng.standard_normal((8, 3, 2))
        ).astype(np.complex64)
        raw_kspace = _centered_fft(_embed_center(image, 4)).astype(np.complex64)

        actual = remove_readout_oversampling_kspace(raw_kspace, 4)
        expected = _centered_fft(image).astype(np.complex64)

        self.assertEqual(actual.shape, (8, 3, 2))
        self.assertEqual(actual.dtype, np.complex64)
        np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)

    def test_torch_preserves_device_dtype_and_values(self) -> None:
        """The PyTorch path should match NumPy without host conversion."""

        rng = np.random.default_rng(11)
        image = (
            rng.standard_normal((6, 2, 3))
            + 1j * rng.standard_normal((6, 2, 3))
        ).astype(np.complex64)
        raw_kspace = _centered_fft(_embed_center(image, 2)).astype(np.complex64)
        tensor = torch.from_numpy(raw_kspace)

        actual = remove_readout_oversampling_kspace(tensor, 2)
        expected = _centered_fft(image).astype(np.complex64)

        self.assertEqual(actual.shape, (6, 2, 3))
        self.assertEqual(actual.dtype, tensor.dtype)
        self.assertEqual(actual.device, tensor.device)
        np.testing.assert_allclose(actual.numpy(), expected, rtol=2e-6, atol=2e-6)

    def test_outside_fov_signal_is_cropped_instead_of_aliased(self) -> None:
        """Shoulder-like signal outside the nominal FOV must not wrap inward."""

        logical_image = np.zeros((8, 1), dtype=np.complex64)
        logical_image[2:6] = 1.0
        oversampled_image = _embed_center(logical_image, 4)
        oversampled_image[2:5] = 20.0
        raw_kspace = _centered_fft(oversampled_image).astype(np.complex64)

        corrected = remove_readout_oversampling_kspace(raw_kspace, 4)
        corrected_image = _centered_ifft(corrected)
        strided_image = _centered_ifft(raw_kspace[::4])

        np.testing.assert_allclose(corrected_image, logical_image, rtol=2e-6, atol=2e-6)
        self.assertGreater(
            np.linalg.norm(strided_image - logical_image),
            0.5 * np.linalg.norm(logical_image),
        )

    def test_supports_nonleading_readout_axis(self) -> None:
        """The selected readout axis should be processed independently."""

        image = np.arange(12, dtype=np.float32).reshape(2, 6).astype(np.complex64)
        oversampled = np.zeros((2, 12), dtype=np.complex64)
        oversampled[:, 3:9] = image
        raw_kspace = _centered_fft(oversampled, axis=1).astype(np.complex64)

        actual = remove_readout_oversampling_kspace(raw_kspace, 2, axis=1)

        np.testing.assert_allclose(
            actual,
            _centered_fft(image, axis=1),
            rtol=2e-6,
            atol=2e-6,
        )

    def test_rejects_invalid_inputs(self) -> None:
        """Invalid type, geometry, values, and axes should fail explicitly."""

        valid = np.ones((8, 2), dtype=np.complex64)
        invalid_cases = (
            (valid, 0, 0),
            (valid, 2.0, 0),
            (valid, 3, 0),
            (valid, 2, 2),
            (valid, 2, 0.0),
            (valid.real, 2, 0),
        )
        for array, factor, axis in invalid_cases:
            with self.subTest(factor=factor, axis=axis):
                with self.assertRaises(ValueError):
                    remove_readout_oversampling_kspace(array, factor, axis=axis)

        nonfinite = valid.copy()
        nonfinite[0, 0] = np.nan + 0j
        with self.assertRaisesRegex(ValueError, "non-finite"):
            remove_readout_oversampling_kspace(nonfinite, 2)
        with self.assertRaises(TypeError):
            remove_readout_oversampling_kspace([[1j]], 1)


if __name__ == "__main__":
    unittest.main()
