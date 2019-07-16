#!/usr/bin/env python
"""This script manages all tasks for the TRAVIS build server."""
import subprocess as sp
import glob

if __name__ == "__main__":
    subdirectories = ["simpsons_paradox"]

    for subdir in subdirectories:
        for notebook in glob.glob(subdir + "/*.ipynb"):
            cmd1 = "jupyter nbconvert --execute "
            cmd2 = "{}  --ExecutePreprocessor.timeout=-1".format(notebook)
            cmd = cmd1 + cmd2
            sp.check_call(cmd, shell=True)
