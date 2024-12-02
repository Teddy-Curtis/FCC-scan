import numpy as np
import awkward as ak 
import pandas as pd 
import time, copy
import sys
import argparse
import json
import glob
from tqdm import tqdm
import uproot
import os
from fcc_study.post.SignalRegion.interpolateSignal import doInterpolation
from fcc_study.utils.submit_utils import submitToBatch
import subprocess

def parse_arguments():

    parser = argparse.ArgumentParser(
        description="Masses to evaulate and interpolate pNN at.")
    
    parser.add_argument(
        "--train_direc",
        required=True,
        default=None,
        type=str,
        help="Directory that contains the model and the training information.")
    
    parser.add_argument(
        "--output_direc",
        required=False,
        default=None,
        type=str,
        help="Directory to save the outputs, will fall back to direc if not specified.")
    
    parser.add_argument(
        "--combine_name",
        required=False,
        default=None,
        type=str,
        help="Name for the Combine directory.")
    
    parser.add_argument(
        "--interp",
        required=False,
        default=False,
        action="store_true",
        help="To interpolate.")
    

    parser.add_argument(
        "--evaluate",
        required=False,
        default=False,
        action="store_true",
        help="To Evaluate the backgrounds.")
    
    parser.add_argument(
        "--runCombine",
        required=False,
        default=False,
        action="store_true",
        help="Whether to run combine.")
    

    return parser.parse_args()


parser = parse_arguments()
train_direc = parser.train_direc
output_direc_input = parser.output_direc
combine_name = parser.combine_name

if output_direc_input is not None:
    output_direc = output_direc_input
else:
    output_direc = f"{train_direc}/combine_{combine_name}"

os.makedirs(output_direc, exist_ok=True)


# Now I want to define the masses that I want to run over
# does output already exist, if so read in the masses
if os.path.exists(f"{output_direc}/mass_scan.txt"):
    print(f"Loading previous mass scan from {output_direc}/mass_scan.txt")
    mass_scan = np.loadtxt(f"{output_direc}/mass_scan.txt")
else:
    # get name of all the signal files and then get the masses
    sig_files = sorted(glob.glob(f"{train_direc}/data/test/awkward/mH*.parquet"))
    mass_scan = []

    for file in sig_files:
        mH = file.split("mH")[1].split("_")[0]
        mA = file.split("mA")[1].split("_")[0]

        mass_scan.append([int(mH), int(mA)])

    mass_scan = np.array(mass_scan)


    # Now save the mass scan
    np.savetxt(f"{output_direc}/mass_scan.txt", mass_scan)





if parser.interp:
    # Do the interpolation 
    doInterpolation(train_direc, output_direc)


if parser.evaluate:
    # Get the unique mHs and submit for each
    mHs = np.unique(mass_scan[:, 0])

    for mH in mHs:

        print(f"Submitting for mH: {mH}")
        name = f"evaluateBackgrounds_mH{mH}"
        save_location = f"{output_direc}/batch/exec_files/{name}.sh"
        exec_info = {
            "SCRIPT" : "/vols/cms/emc21/FCC/FCC-Study/fcc_study/post/evaluateBackgroundsGrid.py",
            "ARGS" : f"--train_direc {train_direc} --output_direc {output_direc} --mH {mH} --ecom {int(ecom)}",
            "save_location" : save_location
        }

        logOutErr_file = f"{output_direc}/batch/log_files/{name}"
        sub_info = {
            "EXECUTABLEFILE" : save_location,
            "LOGFILE" : f"{logOutErr_file}.log",
            "ERRORFILE" : f"{logOutErr_file}.err",
            "OUTPUTFILE" : f"{logOutErr_file}.out",
            "NAME" : name,
            "RUNTIME" : "10799"
        }

        submitToBatch(exec_info, sub_info)


