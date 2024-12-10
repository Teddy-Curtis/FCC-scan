executable = execute_main_scenario.sh
output = runs/run_ecom$(ecom)_scenario$(scenario).out
error = runs/run_ecom$(ecom)_scenario$(scenario).err
log = runs/run_ecom$(ecom)_scenario$(scenario).log

file = /vols/cms/emc21/FCC/FCC-Study/main_e$(ecom)_scenarios.py

arguments = $(file) $(scenario)

request_gpus = 1
JobBatchName = train_pNN_e$(ecom)_scenarios
+MaxRuntime = 259200
+OnExitHold   = ExitStatus != 0
getenv        = True

queue ecom,scenario from (
	365,2
	365,3
)
