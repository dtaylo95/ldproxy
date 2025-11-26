#!/usr/bin/env python3

"""
(C) 2025 Dylan Taylor

I/O functions for various genotype file formats.
"""

from typing import Iterator, Any
from pathlib import Path

import pgenlib as pg
import numpy as np
import pandas as pd
import sys

__version__ = "0.0.1"




#==============================================================================#
#                                                                              #
#                             Simple Logging Class                             #
#                                                                              #
#==============================================================================#

class SimpleLogger:
    
    def __init__(self, logpath: str | Path = None, verbose: bool = True):
        self.console = sys.stderr
        self.verbose = verbose
        self.logfile = open(logpath, 'w') if logpath else None

    def log(self, message: str, end: str = '\n') -> None:
        if self.verbose:
            self.console.write(message + end)
        if self.logfile:
            self.logfile.write(message + end)
            self.logfile.flush()

    def close(self) -> None:
        if self.logfile:
            self.logfile.close()
            self.logfile = None        

    def __enter__(self) -> "SimpleLogger":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()



#==============================================================================#
#                                                                              #
#                             Genotype Chunk Class                             #
#                                                                              #
#==============================================================================#

class GenotypeChunk:

    def __init__(self,
                 variant_info_df: pd.DataFrame,
                 genotypes: np.ndarray):
        if not isinstance(variant_info_df, pd.DataFrame):
            raise TypeError("variant_info_df must be a pandas DataFrame")

        if not isinstance(genotypes, np.ndarray):
            raise TypeError("genotypes must be a NumPy ndarray")

        if genotypes.ndim != 2:
            raise ValueError("genotypes must be a 2D NumPy array")

        if len(variant_info_df) != genotypes.shape[0]:
            raise ValueError(
                f"variant_info rows ({len(variant_info_df)}) must match "
                f"genotype variant count ({genotypes.shape[0]})"
            )

        self.variant_info = variant_info_df.reset_index(drop=True)
        self.genotypes = genotypes

    def __len__(self) -> int:
        return self.genotypes.shape[0]




#==============================================================================#
#                                                                              #
#                              PGEN Reader/Writer                              #
#                                                                              #
#==============================================================================#

#======================================#
#             PGEN Reader              #
#======================================#

class PGENReader():

    def __init__(self, path: str | Path):
        self.path: Path = Path(path)
        self.pgen: pg.PgenReader = pg.PgenReader(str(self.path.with_suffix('.pgen')).encode())
        self._n_samples: int = self.pgen.get_raw_sample_ct()
        self._n_variants: int = self.pgen.get_variant_ct()

    def close(self) -> None:
        if self.pgen is not None:
            self.pgen.close()
            self.pgen = None        

    def __enter__(self) -> "PGENReader":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
        return False
    
    @property
    def n_samples(self) -> int:
        return self._n_samples

    @property
    def n_variants(self) -> int:
        return self._n_variants

    def iter_chunks(self, chunk_size: int) -> Iterator[np.ndarray]:
        if self.pgen is None:
            raise RuntimeError("Reader is closed.")
        for start in range(0, self._n_variants, chunk_size):
            end = min(start + chunk_size, self._n_variants)
            act_chunk_size = end - start
            G = np.empty((act_chunk_size, self._n_samples), dtype=np.int8)
            self.pgen.read_range(start, end, G)
            yield G


#======================================#
#             PGEN Writer              #
#======================================#

class PGENWriter():

    def __init__(self, path: str | Path, n_samples: int, n_variants: int):
        self.path: Path = Path(path)
        self.pgen: pg.PgenWriter = pg.PgenWriter(
            str(self.path.with_suffix('.pgen')).encode(),
            n_samples,
            n_variants,
            nonref_flags = False,
            dosage_present = True
        )

    def close(self) -> None:
        if self.pgen is not None:
            self.pgen.close()
            self.pgen = None        

    def __enter__(self) -> "PGENReader":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
        return False
    
    def write_chunk(self, G_chunk):
        for m in range(G_chunk.shape[0]):
            self.pgen.append_dosages(G_chunk[m, :])