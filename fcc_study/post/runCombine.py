"""
This is the base script that defines our signal region. This is where we define 
the binning and and the parameter grid.
This calls to the two other scripts that 1) interpolate the signal and the 
grid of points and 2) evaluates the background MC at the grid of points. 
"""
import subprocess
import numpy as np 
import os
import argparse, sys
import copy



def parse_arguments():

    parser = argparse.ArgumentParser(
        description="Masses to evaulate and interpolate pNN at.")
    
    parser.add_argument(
        "--combine_direc_base",
        required=True,
        default=None,
        type=str,
        help="Directory that contains the combine data in.")
    

    
    parser = parser.parse_args()
    
    parser_kwargs = parser._get_kwargs()
    for arg, val in parser_kwargs:
        print(f"{arg} : {val}")

    return parser


parser = parse_arguments()
combine_direc_base = parser.combine_direc_base

# Now I need to get all of the mass points
masses = np.loadtxt("/vols/cms/emc21/FCC/FCC-Study/runs/e240_full_run/run25/masses.txt")

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



os.makedirs(f"{combine_direc_base}/batch/exec_files", exist_ok=True)
# And for the log files
os.makedirs(f"{combine_direc_base}/batch/log_files", exist_ok=True)

# Loop over every mH, and submit for all mAs for that mH
for mH, mA in masses:

    mH = int(mH)
    mA = int(mA)

    print(f"Submitting for mH={mH}, mA={mA}")

    logOutErr_file = f"{combine_direc_base}/batch/log_files/run_combine_mH{mH}_mA{mA}"
    sub_info = {
        "EXECUTABLEFILE" : "/vols/cms/emc21/FCC/FCC-Study/fcc_study/post/runCombine.sh",
        "LOGFILE" : f"{logOutErr_file}.log",
        "ERRORFILE" : f"{logOutErr_file}.err",
        "OUTPUTFILE" : f"{logOutErr_file}.out",
        "ARGS" : f"{combine_direc_base}/mH{mH}_mA{mA} {mH} {mA}", 
        "NAME" : f"run_combine_mH{mH}_mA{mA}",
        "RUNTIME" : "3600"
    }

    # First write the executable file
    file = copy.deepcopy(submit_template)

    file = applyReplacements(file, sub_info)

    # # Now save in the sub location directory
    sub_file = f"{combine_direc_base}/batch/exec_files/run_combine_mH{mH}_mA{mA}.txt"
    with open(sub_file, "w") as f:
        f.write(file)
    
    # Now submit
    cmd = f"condor_submit {sub_file}"
    status, out = subprocess.getstatusoutput(cmd)
    print(out)


