#!/usr/bin/env python3

"""
(C) 2025 Dylan Taylor

I/O functions for various genotype file formats.
"""

from abc import ABC, abstractmethod
from typing import Iterator, Any
from pathlib import Path

from cyvcf2 import VCF, Writer
import numpy as np
import pandas as pd
from datetime import datetime
import time

__version__ = "0.0.1"


#==============================================================================#
#                                                                              #
#                             Genotype Chunk Class                             #
#                                                                              #
#==============================================================================#

class GenotypeChunk:
    """
    A container for a contiguous block of genotype data.

    Each chunk represents `N` variants across `M` samples, consisting of:

    - `variant_info`: a pandas DataFrame with N rows
    - `genotypes`: a 2D NumPy array of shape (N, M)

    This object is typically produced by `GenotypeReader.iter_chunks()`.

    Parameters
    ----------
    variant_info_df : pd.DataFrame
        Variant-level metadata for the chunk. Must have one row per variant.
        Required columns depend on file format, but typically include
        ``CHROM``, ``POS``, ``ID``, ``REF``, ``ALT``.
    genotypes : np.ndarray, shape (N, M)
        Genotype values for N variants and M samples. Must be a 2D NumPy array.

    Raises
    ------
    TypeError
        If `variant_info_df` is not a pandas DataFrame or `genotypes` is
        not a NumPy array.
    ValueError
        If the number of variants does not match between metadata and
        genotype matrix.
    """

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
        """Number of variants in the chunk."""
        return self.genotypes.shape[0]




#==============================================================================#
#                                                                              #
#                              Base Reader/Writer                              #
#                                                                              #
#==============================================================================#

#======================================#
#             Base Reader              #
#======================================#

class GenotypeReader(ABC):
    """
    Abstract base class for reading genotype data from different file formats
    (e.g., VCF, BGEN, PLINK PGEN).

    Subclasses must implement a context manager interface as well as a
    chunked genotype iterator (`iter_chunks`). Typical usage:

        with VCFReader("data.vcf.gz") as reader:
            print(reader.n_samples, reader.n_variants)
            for chunk in reader.iter_chunks(1024):
                process(chunk)

    Attributes
    ----------
    path : Path
        Path to the underlying genotype file.
    n_samples : int | None
        Number of samples in the dataset. Set by subclasses in `__enter__`.
    n_variants : int | None
        Number of variants in the dataset. Set by subclasses in `__enter__`.
    """

    def __init__(self, path: str | Path):
        """
        Parameters
        ----------
        path : Path
            Path to the genotype file that this reader will operate on.
        """
        self.path = Path(path)
        self.n_samples: int | None = None
        self.n_variants: int | None = None

    # ---- Context manager API -------------------------------------------------

    @abstractmethod
    def __enter__(self) -> "GenotypeReader":
        """
        Open file handles and initialize sample/variant metadata.

        Returns
        -------
        GenotypeReader
            The initialized reader object.
        """
        raise NotImplementedError

    @abstractmethod
    def __exit__(self,
                 exc_type: type[BaseException] | None,
                 exc: BaseException | None,
                 tb: Any) -> None:
        """
        Close any underlying resources associated with the reader.

        Parameters
        ----------
        exc_type : type | None
            The exception type, if an exception occurred.
        exc : BaseException | None
            The exception instance, if an exception occurred.
        tb : traceback | None
            The traceback, if an exception occurred.
        """
        raise NotImplementedError

    # ---- Iteration API -------------------------------------------------------

    @abstractmethod
    def iter_chunks(self, chunk_size: int) -> Iterator[GenotypeChunk]:
        """
        Iterate over genotype data in fixed-size chunks.

        Parameters
        ----------
        chunk_size : int
            Number of variants per chunk.

        Yields
        ------
        Any
            A chunk of genotype data. The exact type and structure are
            defined by the subclass (e.g., NumPy arrays, dictionaries,
            pandas DataFrames, or custom objects).

        Notes
        -----
        This method should not load the entire dataset into memory.
        Implementations should read only `chunk_size` variants at a time.
        """
        raise NotImplementedError


#======================================#
#             Base Writer              #
#======================================#

class GenotypeWriter(ABC):
    """
    Abstract base class for writing genotype data to various file formats
    (e.g., VCF, PLINK PGEN, BGEN).

    Subclasses must implement a context manager interface and the
    `write_chunk` method for writing GenotypeChunk objects.

    Typical usage:

        with VCFWriter("output.vcf.gz") as writer:
            for chunk in reader.iter_chunks(1024):
                writer.write_chunk(chunk)

    Attributes
    ----------
    path : Path
        Path to the output file.
    """

    def __init__(self, out_prefix: str | Path):
        """
        Initialize a GenotypeWriter.

        Parameters
        ----------
        out_prefix : str | Path
            Prefix of genotype file with path.
        """
        self.out_prefix = Path(out_prefix)

    # ---- Context manager API -------------------------------------------------
    
    @abstractmethod
    def __enter__(self) -> "GenotypeWriter":
        """
        Open the output file and initialize any required resources.

        Returns
        -------
        GenotypeWriter
            The initialized writer object.
        """
        raise NotImplementedError

    @abstractmethod
    def __exit__(self,
                 exc_type: type[BaseException] | None,
                 exc_value: BaseException | None,
                 traceback: Any) -> None:
        """
        Close the output file and release any resources.

        Parameters
        ----------
        exc_type : type | None
            The exception type if an exception occurred.
        exc_value : BaseException | None
            The exception instance if an exception occurred.
        traceback : traceback | None
            The traceback if an exception occurred.
        """
        raise NotImplementedError

    # ---- Write API ----------------------------------------------------------

    @abstractmethod
    def write_chunk(self, chunk: GenotypeChunk) -> None:
        """
        Write a single chunk of genotype data to the output file.

        Parameters
        ----------
        chunk : GenotypeChunk
            A GenotypeChunk containing variant metadata and genotype
            matrix to be written.

        Notes
        -----
        Implementations should handle writing efficiently in chunks
        without loading the entire dataset into memory.
        """
        raise NotImplementedError




#==============================================================================#
#                                                                              #
#                              VCF Reader/Writer                               #
#                                                                              #
#==============================================================================#

#======================================#
#              VCF Reader              #
#======================================#

class VCFReader(GenotypeReader):
    """
    Reader for VCF files using cyvcf2.

    Provides a context manager interface and iterates over variants
    in chunks as GenotypeChunk objects.

    Parameters
    ----------
    path : Path | str
        Path to the VCF file. Can be uncompressed or bgzipped.

    Attributes
    ----------
    n_samples : int | None
        Number of samples in the VCF. Set when entering the context.
    n_variants : int | None
        Number of variants in the VCF. Optional; can be set if known.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.vcf: VCF | None = None
        self.n_samples: int | None = None
        self.n_variants: int | None = None

    def __enter__(self) -> "VCFReader":
        """Open the VCF file, initialize sample metadata, and count variants."""
        self.vcf = VCF(str(self.path), gts012=True)
        self.n_samples = len(self.vcf.samples)
        self.n_variants = sum(1 for _ in self.vcf)
        # Re-open iterator so self.vcf can still be used
        self.vcf = VCF(str(self.path), gts012=True)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Close the VCF file."""
        if self.vcf is not None:
            self.vcf.close()
        self.vcf = None
        return False  # propagate exceptions

    def _process_variant(self, variant) -> np.ndarray:
        """
        Convert a cyvcf2 variant into a genotype array.

        Uses the DS field if available; otherwise, falls back to gt_types.
        Missing genotypes are represented as NaN.

        Parameters
        ----------
        variant : cyvcf2.Variant
            A single variant object from cyvcf2.

        Returns
        -------
        np.ndarray, shape (n_samples,)
            Genotype values as float32.
        """
        if "DS" in variant.FORMAT:
            arr = variant.format("DS").astype(np.float32).squeeze()
        else:
            arr = variant.gt_types.astype(np.float32)
            arr[arr == 3] = np.nan  # 3 represents missing in gt_types
        return arr

    def iter_chunks(self, chunk_size: int) -> Iterator[GenotypeChunk]:
        """
        Iterate over the VCF in fixed-size variant chunks.

        Parameters
        ----------
        chunk_size : int
            Number of variants per GenotypeChunk.

        Yields
        ------
        GenotypeChunk
            A chunk containing variant metadata and genotype matrix.
        """
        if self.vcf is None:
            raise RuntimeError("VCFReader must be used as a context manager")

        rows, genos = [], []

        for variant in self.vcf:
            rows.append({
                "CHROM": variant.CHROM,
                "POS": variant.POS,
                "ID": variant.ID or ".",
                "REF": variant.REF,
                "ALT": list(variant.ALT),
                "CM": np.nan  # VCF does not include centimorgan
            })
            genos.append(self._process_variant(variant))

            if len(genos) == chunk_size:
                df = pd.DataFrame(rows)
                yield GenotypeChunk(df, np.stack(genos, dtype=np.float32))
                rows, genos = [], []

        # yield any remaining variants
        if genos:
            df = pd.DataFrame(rows)
            yield GenotypeChunk(df, np.stack(genos, dtype=np.float32))


#======================================#
#              VCF Writer              #
#======================================#

class VCFWriter(GenotypeWriter):
    """
    Write genotype data to a VCFv4.2 file in chunks.

    This class is intended to be used as a context manager:

        with VCFWriter("out_prefix", n_samples) as writer:
            for chunk in reader.iter_chunks(1024):
                writer.write_chunk(chunk)

    The actual file written will be "<prefix>.vcf".

    Attributes
    ----------
    n_samples : int
        Number of samples in the dataset.
    samples : list[str]
        Generated sample names.
    out : TextIO | None
        File handle for the output VCF file.
    """

    def __init__(self, out_prefix: str | Path, n_samples: int):
        """
        Initialize the VCFWriter.

        Parameters
        ----------
        out_prefix : str | Path
            Prefix for the output VCF file. The actual file will be
            "<prefix>.vcf".
        n_samples : int
            Number of samples in the dataset.
        """
        out_prefix = Path(out_prefix)
        self.out_path = out_prefix.with_suffix(".vcf")  # append .vcf
        self.n_samples = n_samples
        self.samples: list[str] | None = None
        self.out: Any = None

    def __enter__(self) -> "VCFWriter":
        """Open the output file, generate sample names, and write the VCF header."""
        self.out = open(self.out_path, "w")

        # Generate sample names
        nd = len(str(self.n_samples))
        self.samples = [f"PS{i:0{nd}d}" for i in range(1, self.n_samples + 1)]

        # Minimal header
        header_lines = [
            "##fileformat=VCFv4.2",
            f"##fileDate={pd.Timestamp.now().strftime('%Y%m%d')}",
            "##source=LDPROXYv0.0.1",
            '##FORMAT=<ID=DS,Number=1,Type=Float,Description="Dosage">'
        ]
        self.out.write("\n".join(header_lines) + "\n")

        # Column header line
        self.out.write(
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t"
            + "\t".join(self.samples) + "\n"
        )

        return self

    def __exit__(self, exc_type: type[BaseException] | None,
                 exc_value: BaseException | None,
                 traceback: Any) -> None:
        """Close the VCF file."""
        if self.out is not None:
            self.out.close()
        self.out = None
        # propagate exceptions
        return None

    def _chunk_to_vcf_df(self, chunk: GenotypeChunk) -> pd.DataFrame:
        """
        Convert a GenotypeChunk to a pandas DataFrame in VCF format.

        Parameters
        ----------
        chunk : GenotypeChunk
            The chunk to convert.

        Returns
        -------
        pd.DataFrame
            DataFrame ready to write to a VCF file.
        """
        n_variants, n_samples = chunk.genotypes.shape
        if self.samples is None or len(self.samples) != n_samples:
            raise ValueError("Sample count mismatch")

        vi = chunk.variant_info

        df = pd.DataFrame({
            "CHROM": vi["CHROM"],
            "POS": vi["POS"],
            "ID": vi["ID"].fillna(".").replace("", "."),
            "REF": vi["REF"],
            "ALT": vi["ALT"].apply(lambda x: ",".join(x) if isinstance(x, (list, tuple)) else x),
            "QUAL": ".",
            "FILTER": ".",
            "INFO": ".",
            "FORMAT": "DS"
        })

        geno_df = pd.DataFrame(chunk.genotypes, columns=self.samples)
        df = pd.concat([df, geno_df], axis=1)
        return df.fillna(".")

    def write_chunk(self, chunk: GenotypeChunk) -> None:
        """
        Write a GenotypeChunk to the output VCF file.

        Parameters
        ----------
        chunk : GenotypeChunk
            Chunk containing variant info and genotype matrix.
        """
        if self.out is None:
            raise RuntimeError("VCFWriter must be used as a context manager")
        df = self._chunk_to_vcf_df(chunk)
        df.to_csv(
            self.out,
            sep="\t",
            index=False,
            header=False,
            na_rep="."
        )
