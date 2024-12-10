import glob
import awkward as ak
import numpy as np
import json
from fcc_study.pNN.training.train import getRunLoc
from fcc_study.pNN.training.preprocessing_datasetClasses import getDataAwkward, consistentTrainTestSplit
from fcc_study.pNN.training.preprocessing_datasetClasses import normaliseWeightsFast, scaleFeatures, CustomDataset, combineInChunks, applyScaler, applyInverseScaler
from fcc_study.pNN.training.train import trainNN
import copy, uproot, os
import matplotlib.pyplot as plt
import mplhep as hep
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import argparse

def parse_arguments():

    parser = argparse.ArgumentParser(
        description="Masses to evaulate and interpolate pNN at.")
    
    parser.add_argument(
        "--scenario",
        required=True,
        default=None,
        type=str,
        help="Which scenario to run.")
    
    parser = parser.parse_args()

    parser_kwargs = parser._get_kwargs()
    for arg, val in parser_kwargs:
        print(f"{arg} : {val}")


    return parser

######################## Define Hyperparams and Model #########################
parser = parse_arguments()
scenario = parser.scenario

base_run_dir = f"runs/e365NewestData/scenario_{scenario}"
run_loc = getRunLoc(base_run_dir)

# Only read in 10 signal files for now
train_files = glob.glob(f"/vols/cms/emc21/FCC/FCC-Study/Data/NewestDataComplete/ecom365/scenario{scenario}/awkward_files/train/*.parquet")
train_data = []
for file in train_files:
    evs = ak.from_parquet(file)
    evs['normed_weight'] = evs['weight_nominal']
    if "_h2h2" in file:
        evs["normed_weight"] = evs['normed_weight'] / (np.sum(evs['normed_weight']))  
    train_data.append(evs)
train_data = combineInChunks(train_data)

val_files = glob.glob(f"/vols/cms/emc21/FCC/FCC-Study/Data/NewestDataComplete/ecom365/scenario{scenario}/awkward_files/val/*.parquet")
val_data = []
for file in val_files:
    evs = ak.from_parquet(file)
    evs['normed_weight'] = evs['weight_nominal']
    if "_h2h2" in file:
        evs["normed_weight"] = evs['normed_weight'] / (np.sum(evs['normed_weight']))  
    val_data.append(evs)
val_data = combineInChunks(val_data)



# Open up the signal and background sample information
with open(f"Data/NewestDataComplete/ecom365/backgrounds.json", "r") as f:
    backgrounds = json.load(f)
with open(f"Data/NewestDataComplete/ecom365/scenario{scenario}_sig_dict.json", "r") as f:
    signals = json.load(f)






samples = {
    "backgrounds" : backgrounds,
    "signal" : signals,
    "Luminosity" : 3_000_000,
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
            'epochs' : 100, 
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


bkg = train_data[train_data['class'] == 0]
bkg_procs  = np.unique(list(bkg.process))
print("Background processes: ")
for  proc in bkg_procs:
    print(proc)


def replaceNaNsWith0(data):
    for branch in branches:
        data[branch] = ak.nan_to_num(data[branch], nan=0)
    return data

train_data = replaceNaNsWith0(train_data)
val_data = replaceNaNsWith0(val_data)

# Now normalise weights so that the signal points sum to the same weight
# e.g. each BP sum to 1, then reweight backgrounds so that 
# sum(backgrounds) = sum(signal)
train_data = normaliseWeightsFast(train_data)
val_data = normaliseWeightsFast(val_data)

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

def evaluateAllData(run_name, all_masses, scenario):
    files = glob.glob(f"/vols/cms/emc21/FCC/FCC-Study/Data/NewestDataComplete/ecom365/scenario{scenario}/awkward_files/{run_name}/*.parquet")
    data = []
    for file in files:
        data.append(ak.from_parquet(file))
    data = combineInChunks(data)

    data = replaceNaNsWith0(data)


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

evaluateAllData("train", unique_masses, scenario)
evaluateAllData("val", unique_masses, scenario)
evaluateAllData("test", unique_masses, scenario)

print("Done!")
