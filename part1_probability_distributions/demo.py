"""Live demo for the presentation.

    python demo.py 70            # height in inches
    python demo.py 178 --unit cm # height in centimetres
    python demo.py 70 --table    # also print the tracking table
"""
import argparse

import pandas as pd

from em_gmm import fit_em, load_heights, print_classification

parser = argparse.ArgumentParser()
parser.add_argument("height", type=float)
parser.add_argument("--unit", choices=["in", "cm"], default="in")
parser.add_argument("--table", action="store_true")
args = parser.parse_args()

x, _ = load_heights()
result = fit_em(x)
if args.table:
    pd.set_option("display.float_format", "{:.4f}".format)
    print(result.table(3), "\n")
print_classification(args.height, result.params, args.unit)
