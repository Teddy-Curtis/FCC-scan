import glob
import awkward as ak
import numpy as np
import json
from fcc_study.pNN.training.train import getRunLoc
from fcc_study.pNN.training.preprocessing_datasetClasses import getDataAwkward, consistentTrainTestSplit
from fcc_study.pNN.training.preprocessing_datasetClasses import normaliseWeights, scaleFeatures, CustomDataset, combineInChunks, applyScaler, applyInverseScaler
from fcc_study.pNN.training.train import trainNN
import copy, uproot, os
import matplotlib.pyplot as plt
import mplhep as hep
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

######################## Define Hyperparams and Model #########################
base_run_dir = "runs/e365_full_run_fixed"
run_loc = getRunLoc(base_run_dir)


# Open up the signal and background sample information
with open(f"Data/stage2_all/backgrounds.json", "r") as f:
    backgrounds = json.load(f)
with open(f"Data/stage2_all/signals.json", "r") as f:
    signals = json.load(f)

samples = {
    "backgrounds" : backgrounds,
    "signal" : signals,
    "Luminosity" : 5000000,
    "test_size" : 0.3, # e.g. 0.2 means 20% of data used for test set
    "val_size" : 0.2 # e.g. 0.2 means 20% of data used for validation set
    }
# Save the samples used for the run
with open(f"{run_loc}/samples.json", "w") as f: 
    json.dump(samples, f, indent=4)


branches = ['Zcand_m',
 'Zcand_pt',
 'Zcand_pz',
 'Zcand_p',
 'Zcand_povere',
 'Zcand_e',
 'Zcand_costheta',
 'Zcand_recoil_m',
 'lep1_pt',
 'lep1_eta',
 'lep1_e',
 'lep1_charge',
 'lep2_pt',
 'lep2_eta',
 'lep2_e',
 'lep2_charge',
 'lep_chargeprod',
 'cosDphiLep',
 'cosThetaStar',
 'cosThetaR',
 'n_jets',
 'MET_e',
 'MET_pt',
 'MET_eta',
 'MET_phi',
 'n_muons',
 'n_electrons']

# Save the branches used for the run
with open(f"{run_loc}/branches.json", "w") as f: 
    json.dump(branches, f, indent=4)


params = {
        'hyperparams' : { 
            'epochs' : 40,                           #! Change 
            'early_stop' : 20,
            'batch_size': 2000,
            'optimizer' : 'Adam', 
            'optimizer_params' : {
                'lr': 0.00001
            },
            'criterion' : 'WeightedBCEWithLogitsLoss',
            'criterion_params' : {
            },
            'scheduler' : 'ReduceLROnPlateau', 
            'scheduler_params' : {
                "patience" : 20, 
                'factor' : 0.5,
                'verbose' : True,
                'eps' : 1e-7 # No point in going smaller than this
            },
            "scheduler_requires_loss" : True
        },
        'model' : 'MLPRelu',
        'model_params' : {
            'input_features' : len(branches) + 2,
            'fc_params' : [(0.0, 250), (0.2, 250), (0.2, 250), (0.2, 250)],
            'output_classes' : 1,
            'num_masses' : 2,
        }
    }

print("All parameters: \n", json.dumps(params, indent=4))
# Save all the parameters to a json file
with open(f"{run_loc}/params.json", "w") as f: 
    json.dump(params, f, indent=4)


# train_files_sig = sorted(glob.glob(f"/vols/cms/emc21/FCC/FCC-Study/Data/stage2_all/awkward_files/train/*h2h2ll.parquet"))
# train_files_bkg = glob.glob(f"/vols/cms/emc21/FCC/FCC-Study/Data/stage2_all/awkward_files/train/*.parquet")
# train_files_bkg = [file for file in train_files_bkg if "h2h2" not in file]
# train_files = train_files_sig + train_files_bkg
train_files = glob.glob(f"/vols/cms/emc21/FCC/FCC-Study/Data/stage2_all/awkward_files/train/*.parquet")
train_data = []
for file in train_files:
    train_data.append(ak.from_parquet(file))
train_data = combineInChunks(train_data)

# val_files_sig = sorted(glob.glob(f"/vols/cms/emc21/FCC/FCC-Study/Data/stage2_all/awkward_files/val/*h2h2ll.parquet"))
# val_files_bkg = glob.glob(f"/vols/cms/emc21/FCC/FCC-Study/Data/stage2_all/awkward_files/val/*.parquet")
# val_files_bkg = [file for file in val_files_bkg if "h2h2" not in file]
# val_files = val_files_sig + val_files_bkg
val_files = glob.glob(f"/vols/cms/emc21/FCC/FCC-Study/Data/stage2_all/awkward_files/val/*.parquet")
val_data = []
for file in val_files:
    val_data.append(ak.from_parquet(file))
val_data = combineInChunks(val_data)

bkg = train_data[train_data['class'] == 0]
bkg_procs  = np.unique(list(bkg.process))
print("Background processes: ")
for  proc in bkg_procs:
    print(proc)

# Replace any nans with 0
for br in branches:
    train_data[br] = ak.nan_to_num(train_data[br], 0)
    val_data[br] = ak.nan_to_num(val_data[br], 0)

# Now normalise weights so that the signal points sum to the same weight
# e.g. each BP sum to 1, then reweight backgrounds so that 
# sum(backgrounds) = sum(signal)
train_data = normaliseWeights(train_data)
val_data = normaliseWeights(val_data)

# Now scale the features so that intput features have mean 0 and std 1
# and that the masses are scaled to be between 0 and 1
train_data, val_data, feat_scaler, mass_scaler = scaleFeatures(train_data, 
                                                                val_data, 
                                                                branches,
                                                                run_loc)


# # Now put these both into helper classes for training 
train_dataset = CustomDataset(train_data, branches, feat_scaler, mass_scaler)
val_dataset = CustomDataset(val_data, branches, feat_scaler, mass_scaler)
train_dataset.shuffleMasses()
val_dataset.shuffleMasses()


######################### Training #################################

trainer = trainNN(params, branches, run_loc)


trainer.trainModel(train_dataset, val_dataset)



######################### Evaluation #################################
def evaluateModelOnData(
    data, branches, masses, feat_scaler, mass_scaler, trainer
):
    
    # Add the weights to the test data
    data['weight'] = copy.deepcopy(data['weight_nominal'])

    # Now scale the features
    data = applyScaler(data, feat_scaler, branches)
    data = applyScaler(data, mass_scaler, ["mH", "mA"])
    dataset = CustomDataset(data, branches, feat_scaler, mass_scaler)
    #dataset.shuffleMasses()

    data = trainer.getProbsForEachMass(dataset, masses)

    return data

def saveSamples(evs, run_loc, scaler, features, run_name = "train"):
    print(f"Saving samples for {run_name}")

    # Find the unique processes, and loop over them
    unique_procs = np.unique(evs['process'])
    print(unique_procs)
    for proc in unique_procs:
        print(proc)
        # Get the proc data then loop over specific proc and save
        proc_data = evs[evs['process'] == proc]


        scaled_data = applyInverseScaler(proc_data, scaler, features)
        scaled_data = copy.deepcopy(scaled_data)
        scaled_data = ak.values_astype(scaled_data, "float32")


        # Save the data
        for file_type in ['root', 'awkward']:
            os.makedirs(f"{run_loc}/data/{run_name}/{file_type}", exist_ok=True)
        
        ak.to_parquet(scaled_data, f"{run_loc}/data/{run_name}/awkward/{proc}.parquet")
        df = ak.to_dataframe(scaled_data)
        #df.to_csv(f"{run_loc}/data/{run_name}/awkward/{proc}.parquet")

        with uproot.recreate(f"{run_loc}/data/{run_name}/root/{proc}.root") as file:
            file["Events"] = df

        print("Saved!")

def evaluateAllData(run_name, all_masses):
    # files_sig = glob.glob(f"/vols/cms/emc21/FCC/FCC-Study/Data/stage2_all/awkward_files/{run_name}/*h2h2*.parquet")
    # files_bkg = glob.glob(f"/vols/cms/emc21/FCC/FCC-Study/Data/stage2_all/awkward_files/{run_name}/*.parquet")
    # files_bkg = [file for file in files_bkg if "h2h2" not in file]
    # files = files_sig + files_bkg
    files = glob.glob(f"/vols/cms/emc21/FCC/FCC-Study/Data/stage2_all/awkward_files/{run_name}/*.parquet")
    data = []
    for file in files:
        data.append(ak.from_parquet(file))
    data = combineInChunks(data)

    # Replace any nans with 0
    for br in branches:
        data[br] = ak.nan_to_num(data[br], 0)


    print("all_masses: ", all_masses)

    # Now do this in parts: for backgrounds can combine all then evaluate model 
    # Then save separately, but for signal I don't want to evaluate the signal 
    # on other signal point masses.
    bkg_train = data[data['class'] == 0]
    bkg_train = evaluateModelOnData(bkg_train, branches, all_masses, feat_scaler, mass_scaler, trainer)

    saveSamples(bkg_train, run_loc, feat_scaler, branches, run_name = run_name)


    # Get all the pnn_output branches
    pnn_output_branches = [f for f in ak.fields(bkg_train) if "pnn_output" in f]


    # Now I need to loop over all the signal points and evaluate the model on them
    sig_train = data[data['class'] == 1]
    sig_procs = np.unique(list(sig_train.process))
    for sig_proc in sig_procs:
        print(f"Processing signal process: {sig_proc}")
        sig_data = copy.deepcopy(sig_train[sig_train['process'] == sig_proc])

        sig_data['weight'] = copy.deepcopy(sig_data['weight_nominal'])
        sig_data = applyScaler(sig_data, feat_scaler, branches)
        sig_data = applyScaler(sig_data, mass_scaler, ["mH", "mA"])

        sig_dataset = CustomDataset(sig_data, branches, feat_scaler, mass_scaler)

        masses = sig_dataset.unique_masses

        sig_data = trainer.getProbsForEachMass(sig_dataset, masses)

        # Now fill in the pnn_output branches
        for pnn_output_branch in pnn_output_branches:
            if pnn_output_branch not in ak.fields(sig_data):
                sig_data[pnn_output_branch] = np.ones_like(sig_data['Zcand_m']) * -1

        
        # Now save the data
        saveSamples(sig_data, run_loc, feat_scaler, branches, run_name = run_name)


unique_masses = train_dataset.unique_masses

evaluateAllData("train", unique_masses)
evaluateAllData("val", unique_masses)
evaluateAllData("test", unique_masses)



# train_files_sig = glob.glob(f"/vols/cms/emc21/FCC/FCC-Study/Data/stage2_all/awkward_files/train/*.parquet") #! Remove the mH80 bit!!!
# train_files_bkg = glob.glob(f"/vols/cms/emc21/FCC/FCC-Study/Data/stage2_all/awkward_files/train/*.parquet")
# train_files_bkg = [file for file in train_files_bkg if "h2h2" not in file]
# train_files = train_files_sig + train_files_bkg
# train_data = []
# for file in train_files:
#     train_data.append(ak.from_parquet(file))
# train_data = combineInChunks(train_data)


# # Now evaluate the model on the train and val data
# unique_masses = copy.copy(train_dataset.unique_masses)
# print("Unique masses: ", unique_masses)

# # Now do this in parts: for backgrounds can combine all then evaluate model 
# # Then save separately, but for signal I don't want to evaluate the signal 
# # on other signal point masses.
# bkg_train = train_data[train_data['class'] == 0]
# bkg_train = evaluateModelOnData(bkg_train, branches, unique_masses, feat_scaler, mass_scaler, trainer)

# saveSamples(bkg_train, run_loc, feat_scaler, branches, run_name = "train")

# # Get all the pnn_output branches
# pnn_output_branches = [f for f in ak.fields(bkg_train) if "pnn_output" in f]


# # Now I need to loop over all the signal points and evaluate the model on them
# sig_train = train_data[train_data['class'] == 1]
# sig_procs = np.unique(list(sig_train.process))
# for sig_proc in sig_procs:
#     print(f"Processing signal process: {sig_proc}")
#     sig_data = copy.deepcopy(sig_train[sig_train['process'] == sig_proc])

#     sig_data['weight'] = copy.deepcopy(sig_data['weight_nominal'])
#     sig_data = applyScaler(sig_data, feat_scaler, branches)
#     sig_data = applyScaler(sig_data, mass_scaler, ["mH", "mA"])

#     sig_dataset = CustomDataset(sig_data, branches, feat_scaler, mass_scaler)

#     masses = sig_dataset.unique_masses

#     sig_data = trainer.getProbsForEachMass(sig_dataset, samples, masses)

#     # Now fill in the pnn_output branches
#     for pnn_output_branch in pnn_output_branches:
#         if pnn_output_branch not in ak.fields(sig_data):
#             sig_data[pnn_output_branch] = np.ones_like(sig_data['Zcand_m']) * -1

    
#     # Now save the data
#     saveSamples(sig_data, run_loc, feat_scaler, branches, run_name = "train")


# # Now repeat for the validation and test data

# val_files_sig = glob.glob(f"/vols/cms/emc21/FCC/FCC-Study/Data/stage2_all/awkward_files/val/*.parquet") #! Remove the mH80 bit!!!
# val_files_bkg = glob.glob(f"/vols/cms/emc21/FCC/FCC-Study/Data/stage2_all/awkward_files/val/*.parquet")
# val_files_bkg = [file for file in val_files_bkg if "h2h2" not in file]
# val_files = val_files_sig + val_files_bkg
# val_data = []
# for file in val_files:
#     val_data.append(ak.from_parquet(file))
# val_data = combineInChunks(val_data)

# bkg_val = val_data[val_data['class'] == 0]
# bkg_val = evaluateModelOnData(bkg_val, branches, unique_masses, feat_scaler, mass_scaler, trainer)

# saveSamples(bkg_val, run_loc, feat_scaler, branches, run_name = "val")

# # Now I need to loop over all the signal points and evaluate the model on them
# sig_val = val_data[val_data['class'] == 1]
# sig_procs = np.unique(list(sig_val.process))
# for sig_proc in sig_procs:
#     print(f"Processing signal process: {sig_proc}")
#     sig_data = copy.deepcopy(sig_val[sig_val['process'] == sig_proc])

#     sig_data['weight'] = copy.deepcopy(sig_data['weight_nominal'])
#     sig_data = applyScaler(sig_data, feat_scaler, branches)
#     sig_data = applyScaler(sig_data, mass_scaler, ["mH", "mA"])

#     sig_dataset = CustomDataset(sig_data, branches, feat_scaler, mass_scaler)

#     masses = sig_dataset.unique_masses

#     sig_data = trainer.getProbsForEachMass(sig_dataset, samples, masses)

#     # Now fill in the pnn_output branches
#     for pnn_output_branch in pnn_output_branches:
#         if pnn_output_branch not in ak.fields(sig_data):
#             sig_data[pnn_output_branch] = np.ones_like(sig_data['Zcand_m']) * -1

    
#     # Now save the data
#     saveSamples(sig_data, run_loc, feat_scaler, branches, run_name = "val")


# # Now do the same for the test data
# test_files_sig = glob.glob(f"/vols/cms/emc21/FCC/FCC-Study/Data/stage2_all/awkward_files/test/*.parquet") #! Remove the mH80 bit!!!
# test_files_bkg = glob.glob(f"/vols/cms/emc21/FCC/FCC-Study/Data/stage2_all/awkward_files/test/*.parquet")
# test_files_bkg = [file for file in test_files_bkg if "h2h2" not in file]
# test_files = test_files_sig + test_files_bkg
# test_data = []
# for file in test_files:
#     test_data.append(ak.from_parquet(file))
# test_data = combineInChunks(test_data)

# bkg_test = test_data[test_data['class'] == 0]
# bkg_test = evaluateModelOnData(bkg_test, branches, unique_masses, feat_scaler, mass_scaler, trainer)

# saveSamples(bkg_test, run_loc, feat_scaler, branches, run_name = "test")

# # Now I need to loop over all the signal points and etestuate the model on them
# sig_test = test_data[test_data['class'] == 1]
# sig_procs = np.unique(list(sig_test.process))
# for sig_proc in sig_procs:
#     print(f"Processing signal process: {sig_proc}")
#     sig_data = copy.deepcopy(sig_test[sig_test['process'] == sig_proc])

#     sig_data['weight'] = copy.deepcopy(sig_data['weight_nominal'])
#     sig_data = applyScaler(sig_data, feat_scaler, branches)
#     sig_data = applyScaler(sig_data, mass_scaler, ["mH", "mA"])

#     sig_dataset = CustomDataset(sig_data, branches, feat_scaler, mass_scaler)

#     masses = sig_dataset.unique_masses

#     sig_data = trainer.getProbsForEachMass(sig_dataset, samples, masses)

#     # Now fill in the pnn_output branches
#     for pnn_output_branch in pnn_output_branches:
#         if pnn_output_branch not in ak.fields(sig_data):
#             sig_data[pnn_output_branch] = np.ones_like(sig_data['Zcand_m']) * -1

    
#     # Now save the data
#     saveSamples(sig_data, run_loc, feat_scaler, branches, run_name = "test")



# train_data = trainer.getProbsForEachMass(train_dataset, samples, unique_masses)
# val_data = trainer.getProbsForEachMass(val_dataset, samples, unique_masses)

# # Now save these
# saveSignalSamples(train_data, run_loc, feat_scaler, branches, run_name = "train")
# saveSignalSamples(val_data, run_loc, feat_scaler, branches, run_name = "val")
# saveBackgroundSamples(train_data, run_loc, feat_scaler, branches, run_name = "train")
# saveBackgroundSamples(val_data, run_loc, feat_scaler, branches, run_name = "val")
# os.makedirs(f"{run_loc}/data/train", exist_ok=True)
# os.makedirs(f"{run_loc}/data/val", exist_ok=True)
# ak.to_parquet(copy.deepcopy(train_data), f"{run_loc}/train_data.parquet")
# ak.to_parquet(copy.deepcopy(val_data), f"{run_loc}/val_data.parquet")

# Now delete the train and val data, read in the test data and evaluate the 
# model on that
# del train_data
# del val_data

# test_files = glob.glob(f"/vols/cms/emc21/FCC/FCC-Study/Data/stage2_all/awkward_files/test/*.parquet")
# test_data = []
# for file in test_files:
#     test_data.append(ak.from_parquet(file))
# test_data = combineInChunks(test_data)

# # Add the weights to the test data
# test_data['weight'] = copy.deepcopy(test_data['weight_nominal'])

# # Now scale the features
# test_data = applyScaler(test_data, feat_scaler, branches)
# test_data = applyScaler(test_data, mass_scaler, ["mH", "mA"])
# test_dataset = CustomDataset(test_data, branches, feat_scaler, mass_scaler)
# test_dataset.shuffleMasses()

# test_data = trainer.getProbsForEachMass(test_dataset, samples, unique_masses)

# # Now save the test data
# saveSignalSamples(test_data, run_loc, feat_scaler, branches, run_name = "test")
# saveBackgroundSamples(test_data, run_loc, feat_scaler, branches, run_name = "test")

print("Done!")

# Now do some extra plotting
# Get list of all the signal names
# sig_procs = np.unique(list(val_data[val_data['class'] == 1].process))
# # Get list of all the background names
# bkg_procs = np.unique(list(val_data[val_data['class'] == 0].process))

# Define the bins for the histogram
# bins = np.linspace(0, 1, 50)
# # Loop over all the signal processes and plot signal versus background
# for sig_proc in sig_procs:
#     plt.close()
    
#     bkg_hists = []
#     bkg_weights = []
#     for bkg_proc in bkg_procs:
#         # Get each background process
#         bkg = val_data[val_data['process'] == bkg_proc]
#         # Get the histogram for it
#         bkg_hist = np.histogram(ak.flatten(bkg[f'pnn_output_bp{sig_proc}']), bins=bins, weights = bkg['weight'])[0]
#         # append to background list
#         bkg_hists.append(bkg_hist)
    
#     # Plot all backgrounds, stacked on top of each other
#     _ = hep.histplot(bkg_hists, bins=bins, histtype='fill', label=bkg_procs, stack=True)

#     # Now get the signal and plot that on top
#     signal = val_data[val_data['process'] == sig_proc]
#     _ = plt.hist(signal[f'pnn_output_bp{sig_proc}'], bins=bins, histtype='step', 
#                  label=sig_proc, weights = signal['weight'], color='black',
#                  linestyle='--')

#     plt.xlabel('PNN output')
#     plt.ylabel('Events')
#     plt.title(f'PNN output for {sig_proc} vs background')

#     plt.legend()
#     plt.yscale('log')
#     plt.savefig(f'pnn_output_bp{sig_proc}.png')
#     plt.show()
