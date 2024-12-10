executable = execute_evaluate_data_scenario.sh
output = runs/evaluate_main_scenario$(scenario).out
error = runs/evaluate_main_scenario$(scenario).err
log = runs/evaluate_main_scenario$(scenario).log


arguments = $(scenario)

request_gpus = 1
JobBatchName = evaluate_scenarios
+MaxRuntime = 259200
+OnExitHold   = ExitStatus != 0
getenv        = True

queue scenario from (
	1
    2
    3
)
