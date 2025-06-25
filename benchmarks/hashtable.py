import string

import h5py
import numpy as np

from versioned_hdf5.hashtable import Hashtable

from .common import Benchmark


class TimeHashtable(Benchmark):
    params = [
        # Note: 'T' requires numpy >=2.0, h5py >=3.14, versioned_hdf5 >=2.1
        ["f", "S", "O", "T"],
        [1, 4, 8, 64, 256],
    ]
    param_names = ["dtype", "nbytes"]

    CHUNKS = (100, 100)

    def setup(self, dtype, nbytes):
        super().setup()
        if dtype == "f" and nbytes not in (4, 8):
            raise NotImplementedError()
        if dtype in ("f", "S"):
            dtype = f"{dtype}{nbytes}"
        if dtype == "O":  # object st
            dtype = h5py.string_dtype()
        if dtype == "T" and np.__version__.startswith("1."):
            # Note: versioned_hdf5 <2.1 will return bugous metrics
            raise NotImplementedError("Requires numpy >=2.0")

        # Not needed to benchmark hash() but required to initialize the
        # Hashtable instance.
        with self.vfile.stage_version("init") as sv:
            # Create an initial empty hashtable
            sv.create_dataset(
                "x",
                shape=self.CHUNKS,
                chunks=self.CHUNKS,
                dtype=dtype,
                fillvalue=0 if dtype in ("f4", "f8") else "",
            )
        self.hashtable = Hashtable(self.file, "x")

        if dtype in ("f4", "f8"):
            self.chunk = self.rng.standard_normal(self.CHUNKS).astype(dtype)
        else:
            self.chunk = self.rand_strings(self.CHUNKS, 0, nbytes, dtype)

    def time_hash(self, dtype, nbytes):
        self.hashtable.hash(self.chunk)
