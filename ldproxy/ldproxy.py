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


if __name__ == "__main__":
    main()
