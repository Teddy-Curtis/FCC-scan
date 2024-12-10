import numpy as np
import copy
import argparse
import os, sys
import subprocess

def parse_arguments():

    parser = argparse.ArgumentParser(
        description="Masses to evaulate and interpolate pNN at.")
    

    parser.add_argument(
        "--output_direc",
        required=False,
        default=None,
        type=str,
        help="Directory to save the outputs, will fall back to direc if not specified.")
    
    
    parser.add_argument(
        "--runCombine",
        required=False,
        default=False,
        action="store_true",
        help="Whether to run combine.")
    

    return parser.parse_args()


parser = parse_arguments()
output_direc = parser.output_direc

print(f"Loading previous mass scan from {output_direc}/mass_scan.txt")
mass_scan = np.loadtxt(f"{output_direc}/mass_scan.txt")




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

        # datacard_files = ['/vols/cms/emc21/FCC/FCC-Study/fcc_study/post/EE_datacard.txt']

        # # Copy all the datacards to the right place
        # for datacard in datacard_files:
        #     os.system(f"cp {datacard} {output_direc}/combine/mH{mH}_mA{mA}/.")

        if not mH % 5:
            print(f"Using the notIntMH datacards for mH={mH}, mA={mA}")
            datacard_files = ['/vols/cms/emc21/FCC/FCC-Study/fcc_study/post/combine/EE_datacard_notIntMH.txt',
                            '/vols/cms/emc21/FCC/FCC-Study/fcc_study/post/combine/MuMu_datacard_notIntMH.txt']
        else:
            datacard_files = ['/vols/cms/emc21/FCC/FCC-Study/fcc_study/post/combine/EE_datacard.txt',
                                '/vols/cms/emc21/FCC/FCC-Study/fcc_study/post/combine/MuMu_datacard.txt']

        for d_file in datacard_files:
            new_name = d_file.split("/")[-1].replace("_notIntMH", "")
            os.system(f"cp {d_file} {output_direc}/combine/mH{mH}_mA{mA}/{new_name}")



        print(f"Submitting for mH={mH}, mA={mA}")

        logOutErr_file = f"{output_direc}/batch/runCombine/log_files/run_combine_mH{mH}_mA{mA}"
        sub_info = {
            "EXECUTABLEFILE" : "/vols/cms/emc21/FCC/FCC-Study/fcc_study/post/combine/runCombine.sh",
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
