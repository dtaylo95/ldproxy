#!/usr/bin/env python3

"""
(C) 2025 Dylan Taylor

I/O functions for various genotype file formats.
"""

from cyvcf2 import VCF, Writer
import numpy as np
import pandas as pd
from datetime import datetime

__version__ = "0.0.1"

#======================================#
#             Base Classes             #
#======================================#

class GenotypeChunk:
    """
    Container for a chunk of genotypes and associated variant metadata.
    """

    def __init__(self, variant_info, genotypes):
        self.variant_info = variant_info      # list of dicts
        self.genotypes = genotypes            # np.ndarray (n_variants, n_samples)

        # Validation
        if len(self.variant_info) != self.genotypes.shape[0]:
            raise ValueError(
                f"variant_info length ({len(self.variant_info)}) "
                f"must match number of variants in genotypes ({self.genotypes.shape[0]})"
            )

    def __iter__(self):
        """
        Iterate over (variant_info, genotype_row) pairs.
        """
        return zip(self.variant_info, self.genotypes)


#======================================#
#               VCF I/O                #
#======================================#

class VCFReader:
    def __init__(self, path):
        self.vcf_path = path
        self.vcf = None
        tmp = VCF(path)
        self.n_samples = len(tmp.samples)
        tmp.close()

    def __enter__(self):
        self.vcf = VCF(self.vcf_path, gts012=True)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.vcf is not None:
            self.vcf.close()
        return False

    def _process_variant(self, variant):
        if "DS" in variant.FORMAT:
            arr = variant.format("DS").astype(np.float32).squeeze()
        else:
            arr = variant.gt_types.astype(np.float32)
            arr[arr==3] = np.nan  # Set missing genotypes to NaN
        return arr

    def iter_chunks(self, chunk_size):
        """Yield GenotypeChunk objects of up to chunk_size variants."""
        if self.vcf is None:
            raise RuntimeError("VCFReader must be used within a context or explicitly opened")

        infos, genos = [], []
        for variant in self.vcf:
            info = {
                "chrom": variant.CHROM,
                "pos": variant.POS,
                "id": variant.ID or ".",
                "ref": variant.REF,
                "alt": variant.ALT,
            }
            infos.append(info)
            genos.append(self._process_variant(variant))

            if len(genos) == chunk_size:
                yield GenotypeChunk(infos, np.stack(genos))
                infos, genos = [], []

        if genos:
            yield GenotypeChunk(infos, np.stack(genos))


class VCFWriter:
    """
    Simple VCF writer using plain Python I/O.
    Supports DS-format writing and context manager protocol.
    """

    def __init__(self, path, n_samples):
        self.out_path = path
        self.n_samples = n_samples
        self.out_fs = None

        # Generate sample names
        nd = len(str(self.n_samples))
        self.samples = [f"PS{i:0{nd}d}" for i in range(1, self.n_samples + 1)]

    def __enter__(self):
        self.out_fs = open(self.out_path, "w")

        # Basic VCF header
        header_lines = [
            '##fileformat=VCFv4.2',
            '##fileDate=' + datetime.now().strftime("%Y%m%d"),
            '##source=LDPROXYv' + __version__,
            '##FORMAT=<ID=DS,Number=1,Type=Float,Description="Dosage">'
        ]

        self.out_fs.write("\n".join(header_lines) + "\n")
        self.out_fs.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t" + "\t".join(self.samples) + "\n")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.out_fs is not None:
            self.out_fs.close()
        return False

    def write_variant(self, info, ds):
        ds_str = "\t".join(str(x) if not np.isnan(x) else "." for x in ds)
        line = (
            f"{info['chrom']}\t{info['pos']}\t{info['id']}\t{info['ref']}\t"
            f"{','.join(info['alt'])}\t.\t.\t.\tDS\t{ds_str}\n"
        )
        self.out_fs.write(line)

    def write_chunk(self, chunk):
        if isinstance(chunk.variant_info, pd.DataFrame):
            for i, row in chunk.variant_info.iterrows():
                ds = chunk.genotypes[i]
                info = row.to_dict()
                self.write_variant(info, ds)
        else:
            for info, ds in chunk:
                self.write_variant(info, ds)

