#!/usr/bin/env python3

"""
(C) 2025 Dylan Taylor

ldproxy is a command-line tool for generating privacy-preserving pseudo-
genotypes that faithfully reproduce real-data LD structure.

This allows sharing of LD information for applications like fine-mapping and
colocalization without exposing raw genotype data.
"""

from argparse import ArgumentParser
from pathlib import Path
import logging
from geno_io import SimpleLogger, PGENReader, PFILEWriter
from transform import random_orthonormal_matrix, apply_random_projection, scale_to_dosage
import numpy as np
import math

def main():

    #======================================#
    #           Argument Parsing           #
    #======================================#

    parser = ArgumentParser(description="ldproxy: Privacy-preserving LD proxy pseudo-genotype generator")
    
    input_group = parser.add_argument_group(title="INPUT")
    mut_input_group = input_group.add_mutually_exclusive_group(required=True)
    mut_input_group.add_argument('--vcf', type=Path, help="Input VCF file")
    mut_input_group.add_argument('--bfile', type=Path, help="Input PLINK binary file prefix")
    mut_input_group.add_argument('--pfile', type=Path, help="Input PLINK2 binary file prefix")

    output_group = parser.add_argument_group(title="OUTPUT")
    output_group.add_argument('--out', type=Path, required=True, help="Output file prefix for generated pseudo-genotypes")
    mut_output_group = output_group.add_mutually_exclusive_group(required=True)
    mut_output_group.add_argument('--make-vcf', action='store_true', help="Output in VCF format")
    mut_output_group.add_argument('--make-bed', action='store_true', help="Output in PLINK binary format")
    mut_output_group.add_argument('--make-pgen', action='store_true', help="Output in PLINK PGEN format")

    param_group = parser.add_argument_group(title="PARAMETERS")
    param_group.add_argument('--K', type=int, default=1_000, help="Number of pseudo-individuals to generate (default: 1000)")
    param_group.add_argument('--chunk_size', type=int, default=10_000, help="Number of variants to process at once (default: 10,000)")
    param_group.add_argument('--seed', type=int, default=None, help="Random seed for reproducibility")

    args = parser.parse_args()


    #======================================#
    #            Set Up Logging            #
    #======================================#

    # Create log file
    log_file = args.out.with_suffix('.log')
    log_file.parent.mkdir(parents=True, exist_ok=True)

    # Set up logger
    logger = SimpleLogger(log_file, verbose=True)


    #======================================#
    #      Log Command Line Arguments      #
    #======================================#

    # Get non-defaults
    opts = vars(args)
    non_defaults = [k for k, v in opts.items() if v != parser.get_default(k)]

    logger.log(f"Logging to {log_file}")
    logger.log("Options in effect:")
    for k in non_defaults:
        v = opts[k]
        # If it’s a boolean flag and True, just print the flag
        if isinstance(v, bool) and v:
            logger.log(f"  --{k}")
        else:
            logger.log(f"  --{k} {v}")
    logger.log("")


    #======================================#
    #             Get IO Types             #
    #======================================#

    # Input file type
    in_type = None
    for action in mut_input_group._group_actions:
        if getattr(args, action.dest):
            in_type = action.dest
            break

    # Output file type
    out_type = None
    for action in mut_output_group._group_actions:
        if getattr(args, action.dest):
            out_type = args.out_type = action.dest.replace("make_", "")
            break


    #======================================#
    #        Set Up Genotype Reader        #
    #======================================#

    geno_reader = PGENReader(args.pfile.with_suffix('.pgen'))

    N = geno_reader.n_samples
    M = geno_reader.n_variants

    logger.log(f"{N} samples loaded from {args.pfile.with_suffix('.psam')}.")
    logger.log(f"{M} variants loaded from {args.pfile.with_suffix('.pvar')}.")


    #======================================#
    #      Generate Projection Matrix      #
    #======================================#

    K = args.K

    # Check for appropriate pseudo-samples selection
    if K > N:
        raise ValueError(f"The number of requested pseudo-samples ({K}) cannot exceed the number of real samples ({N}). Try again with a smaller `--K`.")

    # Generate the matrix
    logger.log(f"Generating {N}x{K} projection matrix... ", end='')
    Q = random_orthonormal_matrix(N, K, args.seed)
    logger.log("Done.")


    #======================================#
    #            Process Chunks            #
    #======================================#

    chunk_size = args.chunk_size
    n_chunks = math.ceil(M / chunk_size)
    
    geno_writer = PFILEWriter(args.out, N, M)

    for G_chunk in geno_reader.iter_chunks(chunk_size):
        P_chunk = apply_random_projection(G_chunk, Q)
        D_chunk = scale_to_dosage(P_chunk)
        geno_writer.write_pgen_chunk(D_chunk)

    geno_writer.write_psam()
    geno_writer.copy_pvar(args.pfile.with_suffix('.pvar'))
    
    geno_reader.close()
    geno_writer.close()
    logger.close()

if __name__ == "__main__":
    main()
