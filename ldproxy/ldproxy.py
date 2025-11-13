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
from geno_io import VCFReader, VCFWriter

def main():

    #======================================#
    #           Argument Parsing           #
    #======================================#

    parser = ArgumentParser(description="ldproxy: Privacy-preserving LD proxy genotype generator")
    
    input_group = parser.add_argument_group(title="INPUT")
    mut_input_group = input_group.add_mutually_exclusive_group(required=True)
    mut_input_group.add_argument('--vcf', type=Path, help="Input VCF file")
    mut_input_group.add_argument('--bfile', type=Path, help="Input PLINK binary file prefix")
    mut_input_group.add_argument('--pgen', type=Path, help="Input PLINK PGEN file prefix")

    output_group = parser.add_argument_group(title="OUTPUT")
    output_group.add_argument('--out', type=Path, required=True, help="Output file prefix for generated pseudo-genotypes")
    mut_output_group = output_group.add_mutually_exclusive_group(required=True)
    mut_output_group.add_argument('--make-vcf', action='store_true', help="Output in VCF format")
    mut_output_group.add_argument('--make-bed', action='store_true', help="Output in PLINK binary format")
    mut_output_group.add_argument('--make-pgen', action='store_true', help="Output in PLINK PGEN format")

    param_group = parser.add_argument_group(title="PARAMETERS")
    param_group.add_argument('--N', type=int, default=1000, help="Number of pseudo-individuals to generate (default: 1000)")
    param_group.add_argument('--seed', type=int, default=None, help="Random seed for reproducibility")

    args = parser.parse_args()


    #======================================#
    #            Set Up Logging            #
    #======================================#

    # Create log file
    log_file = args.out.with_suffix('.log')
    log_file.parent.mkdir(parents=True, exist_ok=True)

    # Set up logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",  # <- no timestamp or [INFO]
        handlers=[
            logging.FileHandler(log_file, mode='w'),
            logging.StreamHandler()
        ]
    )

    logger = logging.getLogger(__name__)


    #======================================#
    #      Log Command Line Arguments      #
    #======================================#

    # Get non-defaults
    opts = vars(args)
    non_defaults = [k for k, v in opts.items() if v != parser.get_default(k)]

    logger.info(f"Logging to {log_file}\n")
    logger.info("Command line options in effect:")
    for k in non_defaults:
        v = opts[k]
        # If it’s a boolean flag and True, just print the flag
        if isinstance(v, bool) and v:
            logger.info(f"  --{k}")
        else:
            logger.info(f"  --{k} {v}")


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

    Reader = VCFReader
    Writer = VCFWriter

    with Reader(args.vcf) as reader, Writer(args.out.with_suffix('.vcf'), args.N) as writer:
        for g_chunk in reader.iter_chunks(chunk_size=1000):
            writer.write_chunk(g_chunk)

if __name__ == "__main__":
    main()
