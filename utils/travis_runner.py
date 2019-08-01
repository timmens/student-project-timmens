#!/usr/bin/env python
"""This script manages all tasks for the TRAVIS build server."""
import subprocess as sp
from glob import glob

if __name__ == "__main__":

    subdirectories = glob("*/")
    donotcheckdir = ["causal_tree/", "ARCHIVE/"]
    donotchecknb = []

    for subdir in subdirectories:
        if subdir in donotcheckdir:
            continue
        for notebook in glob(subdir + "*.ipynb"):
            if notebook.split("/", 1)[1] in donotchecknb:
                continue
            cmd1 = "jupyter nbconvert --execute "
            cmd2 = "{}  --ExecutePreprocessor.timeout=-1".format(notebook)
            cmd = cmd1 + cmd2
            sp.check_call(cmd, shell=True)
