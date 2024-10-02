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
from fcc_study.post.interpolateSignal import doInterpolation
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
        "--grid_space",
        required=False,
        default=None,
        type=float,
        help="Directory to save the outputs, will fall back to direc if not specified.")
    
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
grid_space = parser.grid_space

if output_direc_input is not None:
    output_direc = output_direc_input
else:
    output_direc = f"{train_direc}/combine_grid_space_{grid_space}"

os.makedirs(output_direc, exist_ok=True)

ecom = np.loadtxt(f"{train_direc}/ecom.txt")

# Now I want to define the masses that I want to run over
# does output already exist, if so read in the masses
if os.path.exists(f"{output_direc}/mass_scan.txt"):
    print(f"Loading previous mass scan from {output_direc}/mass_scan.txt")
    mass_scan = np.loadtxt(f"{output_direc}/mass_scan.txt")
else:
    # get name of all the signal files and then get the masses
    signal_files = glob.glob(f"{train_direc}/data/test/awkward/mH*.parquet")
    mass_list = []
    for f in signal_files:
        file_name = f.split("/")[-1]
        mH = int(file_name.split("mH")[1].split("_")[0])
        mA = int(file_name.split("mA")[1].split(".")[0])
        # If the sum of the masses is greater than the ecom then skip this point
        if mH + mA >= ecom:
            continue

        mass_list.append([mH, mA])

    masses = np.array(mass_list)
    
    # Now I want to define the masses in the scan
    mHs = np.unique(masses[:, 0])
    # Now get the min and max difference between the mHs and mAs
    diffs = masses[masses[:,0] == mHs[0]][:, 1] - masses[masses[:,0] == mHs[0]][:, 0]
    min_diff = np.min(diffs)
    max_diff = np.max(diffs)

    mass_scan = []
    for mH in np.arange(np.min(mHs), np.max(mHs) + grid_space, grid_space):
        for diff in np.arange(min_diff, max_diff + grid_space, grid_space):
            mA = mH + diff
            if mH + mA >= ecom:
                continue
            mass_scan.append([mH, mA])


    # Now save the mass scan
    np.savetxt(f"{output_direc}/mass_scan.txt", mass_scan)





if parser.interp:
    # Do the interpolation 
    doInterpolation(train_direc, output_direc, ecom)


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


# Now I want to to submit the combine jobs to the batch 

submit_template = """
executable = EXECUTABLEFILE
output = OUTPUTFILE
error = ERRORFILE
log = LOGFILE

arguments = ARGS

JobBatchName = NAME
+MaxRuntime = RUNTIME
+OnExitHold   = ExitStatus != 0
getenv        = True
queue 1
"""

def applyReplacements(file_string, replacements):
    for key, val in replacements.items():
        file_string = file_string.replace(key, val)
    return file_string




if parser.runCombine:
    os.makedirs(f"{output_direc}/batch/runCombine/exec_files", exist_ok=True)
    # And for the log files
    os.makedirs(f"{output_direc}/batch/runCombine/log_files", exist_ok=True)

    # Loop over every mH, and submit for all mAs for that mH

    for mH, mA in mass_scan:

        datacard_files = ['/vols/cms/emc21/FCC/FCC-Study/fcc_study/post/EE_datacard.txt']

        # Copy all the datacards to the right place
        for datacard in datacard_files:
            os.system(f"cp {datacard} {output_direc}/combine/mH{mH}_mA{mA}/.")

        if not mH % 5:
            datacard_files = ['/vols/cms/emc21/FCC/FCC-Study/fcc_study/post/EE_datacard_notIntMH.txt',
                            '/vols/cms/emc21/FCC/FCC-Study/fcc_study/post/MuMu_datacard_notIntMH.txt']
        else:
            datacard_files = ['/vols/cms/emc21/FCC/FCC-Study/fcc_study/post/EE_datacard.txt',
                                '/vols/cms/emc21/FCC/FCC-Study/fcc_study/post/MuMu_datacard.txt']
        
            #datacard_mumu = '/vols/cms/emc21/FCC/FCC-Study/fcc_study/post/MuMu_datacard.txt'
        for d_file in datacard_files:
            new_name = d_file.split("/")[-1].replace("_notIntMH", "")
            os.system(f"cp {d_file} {output_direc}/combine/mH{mH}_mA{mA}/{new_name}")



        print(f"Submitting for mH={mH}, mA={mA}")

        logOutErr_file = f"{output_direc}/batch/runCombine/log_files/run_combine_mH{mH}_mA{mA}"
        sub_info = {
            "EXECUTABLEFILE" : "/vols/cms/emc21/FCC/FCC-Study/fcc_study/post/runCombine.sh",
            "LOGFILE" : f"{logOutErr_file}.log",
            "ERRORFILE" : f"{logOutErr_file}.err",
            "OUTPUTFILE" : f"{logOutErr_file}.out",
            "ARGS" : f"{output_direc}/combine/mH{mH}_mA{mA} {mH} {mA}", 
            "NAME" : f"run_combine_mH{mH}_mA{mA}",
            "RUNTIME" : "3600"
        }

        # First write the executable file
        file = copy.deepcopy(submit_template)

        file = applyReplacements(file, sub_info)

        # # Now save in the sub location directory
        sub_file = f"{output_direc}/batch/runCombine/exec_files/run_combine_mH{mH}_mA{mA}.txt"
        with open(sub_file, "w") as f:
            f.write(file)

        


        
        # Now submit
        cmd = f"condor_submit {sub_file}"
        status, out = subprocess.getstatusoutput(cmd)
        print(out)


