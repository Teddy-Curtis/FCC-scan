import copy, subprocess, os

executable_template = """
#!/bin/bash

source /vols/grid/cms/setup.sh
voms-proxy-init --rfc --voms cms --valid 192:00

eval "$(micromamba shell hook --shell bash)"
micromamba activate FCC-forAMstudent2

cd /vols/cms/emc21/FCC/FCC-Study/fcc_study

python SCRIPT ARGS
"""


submit_template = """
executable = EXECUTABLEFILE
output = OUTPUTFILE
error = ERRORFILE
log = LOGFILE
RequestMemory = 8G
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