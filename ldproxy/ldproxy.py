#!/usr/bin/env python3

"""
(C) 2025 Dylan Taylor

ldproxy is a command-line tool for generating privacy-preserving pseudo-
genotypes that faithfully reproduce real-data LD structure.

This allows sharing of LD information for applications like fine-mapping and
colocalization without exposing raw genotype data.
"""

from argparse import ArgumentParser
from logging import Logger


def main():

    # Parse command-line arguments
    parser = ArgumentParser(description="ldproxy: Privacy-preserving LD proxy genotype generator")
    
    input_group = parser.add_argument_group(title="Input Options", description="Specify the input genotype file")
    mut_input_group = input_group.add_mutually_exclusive_group(required=True)
    mut_input_group.add_argument('--vcf', type=str, help="Input VCF file")
    mut_input_group.add_argument('--bfile', type=str, help="Input PLINK binary file prefix")
    mut_input_group.add_argument('--pgen', type=str, help="Input PLINK PGEN file prefix")

    args = parser.parse_args()


if __name__ == "__main__":
    main()
