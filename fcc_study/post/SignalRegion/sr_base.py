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

    parser.add_argument(
        "--signalEvalNoInterp",
        required=False,
        default=False,
        action="store_true",
        help="Whether to get the signal histograms, but without any interpolation.")

    parser.add_argument(
        "--backgroundEvalNoInterp",
        required=False,
        default=False,
        action="store_true",
        help="Whether to get the background histograms, but without any interpolation.")
    
    parser.add_argument(
        "--diff",
        required=False,
        default=0.0,
        type=float,
        help="Mass splittings for the mass scan."
    )

    parser.add_argument(
        "--ecom",
        required=False,
        default=0.0,
        type=float,
        help="Centre of mass energy."
    )
    

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


    if parser.diff != 0.0:
        diff = parser.diff
        ecom = parser.ecom
        # Add in additional masses
        new_masses = []

        mH_diff = 2.5
        mA_diff = 2

        mH_min = np.min(np.array(mass_scan)[:,0])
        mH_max = np.max(np.array(mass_scan)[:,0])
        mHs_add = np.arange(mH_min, mH_max+mH_diff, mH_diff)
        mHs_add = [mH for mH in mHs_add if mH not in np.array(mass_scan)[:,0]]

        masses_to_add = []
        for mH in mHs_add:
            mA_max = ecom - mH
            mAs_add = np.arange(mH+10, mA_max+mA_diff, mA_diff)
            for mA_ in mAs_add:
                if [mH, mA_] not in mass_scan:
                    masses_to_add.append([mH, mA_])

        mass_scan = mass_scan + masses_to_add

    mass_scan = np.array(mass_scan)

    # Now save the mass scan
    np.savetxt(f"{output_direc}/mass_scan.txt", mass_scan)





if parser.interp:
    # Do the interpolation 
    doInterpolation(train_direc, output_direc)


executable_template = """
#!/bin/bash

source /vols/grid/cms/setup.sh
voms-proxy-init --rfc --voms cms --valid 192:00

eval "$(micromamba shell hook --shell bash)"
micromamba activate FCC-forAMstudent3

cd /vols/cms/emc21/FCC/FCC-Study/fcc_study

python SCRIPT ARGS
"""


submit_template = """
executable = EXECUTABLEFILE
output = OUTPUTFILE
error = ERRORFILE
log = LOGFILE
JobBatchName = NAME
+MaxRuntime = 10799
+OnExitHold   = ExitStatus != 0
getenv        = True
queue 1
"""


def applyReplacements(file_string, replacements):
    for key, val in replacements.items():
        file_string = file_string.replace(key, val)
    return file_string


def submitToBatch(exec_info, sub_info):
    os.makedirs(os.path.dirname(exec_info['save_location']), exist_ok=True)
    os.makedirs(os.path.dirname(sub_info['LOGFILE']), exist_ok=True)
    # First write the executable file
    file = copy.deepcopy(executable_template)

    file = applyReplacements(file, exec_info)
    
    # Now save in the output directory
    with open(exec_info['save_location'], "w") as f:
        f.write(file)

    # Give it the correct permissions
    cmd = f"chmod +x {exec_info['save_location']}"
    status, out = subprocess.getstatusoutput(cmd)


    # Make the submit script
    # Open the template, change and then submit
    file = copy.deepcopy(submit_template)

    file = applyReplacements(file, sub_info)

    # Now write submit
    with open("submit.sh", "w") as f:
        f.write(file)

    cmd = "condor_submit submit.sh"
    status, out = subprocess.getstatusoutput(cmd)
    print(out)


if parser.signalEvalNoInterp:
    name = f"get_sig_hists_{combine_name}"
    save_location = f"{output_direc}/batch/get_sig_hists/exec_files/{name}.sh"
    exec_info = {
        "SCRIPT" : "/vols/cms/emc21/FCC/FCC-Study/fcc_study/post/SignalRegion/makeSigHistsFromAlreadyEvaluated.py",
        "ARGS" : f"--train_direc {train_direc} --output_direc {output_direc}",
        "save_location" : save_location
    }

    logOutErr_file = f"{output_direc}/batch/get_sig_hists/log_files/{name}"
    sub_info = {
        "EXECUTABLEFILE" : save_location,
        "LOGFILE" : f"{logOutErr_file}.log",
        "ERRORFILE" : f"{logOutErr_file}.err",
        "OUTPUTFILE" : f"{logOutErr_file}.out",
        "NAME" : name,
        "RUNTIME" : "10799"
    }

    submitToBatch(exec_info, sub_info)


if parser.backgroundEvalNoInterp:
    name = f"get_bkg_hists_{combine_name}"
    save_location = f"{output_direc}/batch/get_bkg_hists/exec_files/{name}.sh"
    exec_info = {
        "SCRIPT" : "/vols/cms/emc21/FCC/FCC-Study/fcc_study/post/SignalRegion/makeBkgHistsFromAlreadyEvaluated.py",
        "ARGS" : f"--train_direc {train_direc} --output_direc {output_direc}",
        "save_location" : save_location
    }

    logOutErr_file = f"{output_direc}/batch/get_bkg_hists/log_files/{name}"
    sub_info = {
        "EXECUTABLEFILE" : save_location,
        "LOGFILE" : f"{logOutErr_file}.log",
        "ERRORFILE" : f"{logOutErr_file}.err",
        "OUTPUTFILE" : f"{logOutErr_file}.out",
        "NAME" : name,
        "RUNTIME" : "10799"
    }

    submitToBatch(exec_info, sub_info)




executable_template = """
#!/bin/bash

source /vols/grid/cms/setup.sh
voms-proxy-init --rfc --voms cms --valid 192:00

eval "$(micromamba shell hook --shell bash)"
micromamba activate FCC-forAMstudent3

cd /vols/cms/emc21/FCC/FCC-Study/fcc_study

python SCRIPT ARGS
"""


submit_template = """
executable = EXECUTABLEFILE
output = OUTPUTFILE
error = ERRORFILE
log = LOGFILE
JobBatchName = NAME
request_gpus = 1
+MaxRuntime = 259200
+OnExitHold   = ExitStatus != 0
getenv        = True
queue 1
"""


def applyReplacements(file_string, replacements):
    for key, val in replacements.items():
        file_string = file_string.replace(key, val)
    return file_string


def submitToBatch(exec_info, sub_info):
    os.makedirs(os.path.dirname(exec_info['save_location']), exist_ok=True)
    os.makedirs(os.path.dirname(sub_info['LOGFILE']), exist_ok=True)
    # First write the executable file
    file = copy.deepcopy(executable_template)

    file = applyReplacements(file, exec_info)
    
    # Now save in the output directory
    with open(exec_info['save_location'], "w") as f:
        f.write(file)

    # Give it the correct permissions
    cmd = f"chmod +x {exec_info['save_location']}"
    status, out = subprocess.getstatusoutput(cmd)


    # Make the submit script
    # Open the template, change and then submit
    file = copy.deepcopy(submit_template)

    file = applyReplacements(file, sub_info)

    # Now write submit
    with open("submit.sh", "w") as f:
        f.write(file)

    cmd = "condor_submit submit.sh"
    status, out = subprocess.getstatusoutput(cmd)
    print(out)


if parser.evaluate:

    name = f"get_bkg_hists_{combine_name}"
    save_location = f"{output_direc}/batch/get_bkg_hists/exec_files/{name}.sh"
    exec_info = {
        "SCRIPT" : "/vols/cms/emc21/FCC/FCC-Study/fcc_study/post/SignalRegion/evaluateBackgroundsGrid.py",
        "ARGS" : f"--train_direc {train_direc} --output_direc {output_direc}",
        "save_location" : save_location
    }

    logOutErr_file = f"{output_direc}/batch/get_bkg_hists/log_files/{name}"
    sub_info = {
        "EXECUTABLEFILE" : save_location,
        "LOGFILE" : f"{logOutErr_file}.log",
        "ERRORFILE" : f"{logOutErr_file}.err",
        "OUTPUTFILE" : f"{logOutErr_file}.out",
        "NAME" : name,
        "RUNTIME" : "36000"
    }

    submitToBatch(exec_info, sub_info)


