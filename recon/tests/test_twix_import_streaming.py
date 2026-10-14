import unittest
from unittest import mock

import numpy as np

from recon.utils import twix_import


DATA_DIMS = [
    "Col",
    "Cha",
    "Lin",
    "Par",
    "Sli",
    "Ave",
    "Phs",
    "Eco",
    "Rep",
    "Set",
    "Seg",
    "Ida",
    "Idb",
    "Idc",
    "Idd",
    "Ide",
]


class FakeStream:
    """Provide mapVBVD-like explicit selection over a dense test array."""

    def __init__(self, values):
        self.values = np.asarray(values, dtype=np.complex64)
        self.dataDims = list(DATA_DIMS)
        self.fullSize = np.asarray(self.values.shape)
        self.squeeze = False
        self.flagRemoveOS = True
        self.selections = []

    def __getitem__(self, selection):
        self.selections.append(selection)
        rank_preserving = tuple(
            slice(index, index + 1) if isinstance(index, int) else index
            for index in selection
        )
        return self.values[rank_preserving]


class FakeTwix:
    """Hold fake IMAGE and REFSCAN streams."""

    def __init__(self, *, image=None, refscan=None):
        self.image = image
        self.refscan = refscan


class StreamingTwixImportTests(unittest.TestCase):
    def test_load_ref_set_selects_before_materialization(self):
        shape = [2, 3, 4, 5, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1]
        values = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
        stream = FakeStream(values)
        with mock.patch.object(
            twix_import.mapvbvd,
            "mapVBVD",
            return_value=FakeTwix(refscan=stream),
        ):
            result = twix_import.load_ref_set(
                "input.dat", 1, lin_count=2, par_count=3
            )

        expected = values[:, :, :2, :3, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
        expected = np.transpose(expected, (0, 2, 3, 1))
        np.testing.assert_allclose(result.numpy(), expected)
        self.assertEqual((2, 2, 3, 3), tuple(result.shape))
        self.assertEqual(slice(0, 2), stream.selections[0][2])
        self.assertEqual(slice(0, 3), stream.selections[0][3])
        self.assertEqual(1, stream.selections[0][9])

    def test_image_is_compressed_per_partition_chunk(self):
        shape = [2, 3, 2, 3, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1]
        values = (
            np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
            + 1j
        ).astype(np.complex64)
        stream = FakeStream(values)
        basis = np.asarray(
            [[1, 0], [0, 1], [0.5, -0.25j]], dtype=np.complex64
        )
        with mock.patch.object(
            twix_import.mapvbvd,
            "mapVBVD",
            return_value=FakeTwix(image=stream),
        ):
            result, acquired_shape, ncoil = (
                twix_import.load_img_coil_compressed(
                    "input.dat",
                    basis,
                    output_shape=(2, 3, 4, 2),
                    par_chunk=2,
                )
            )

        physical = values[:, :, :, :, 0, 0, 0, :, 0, 0, 0, 0, 0, 0, 0, 0]
        physical = np.transpose(physical, (0, 2, 3, 4, 1))
        expected = np.zeros((2, 3, 4, 2, 2), dtype=np.complex64)
        expected[:, :2, :3] = physical @ basis
        np.testing.assert_allclose(result.numpy(), expected)
        self.assertEqual((2, 2, 3, 2, 3), acquired_shape)
        self.assertEqual(3, ncoil)
        self.assertEqual(4, len(stream.selections))


if __name__ == "__main__":
    unittest.main()
