
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
import importlib
from fcc_study.pNN.utils import convertToNumpy
import torch, pickle
from tqdm import tqdm
from torch.utils.data import DataLoader
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

parser = parse_arguments()
scenario = parser.scenario


directory = f"/vols/cms/emc21/FCC/FCC-Study/runs/e365NewestData/scenario_{scenario}/run1"
run_loc = directory

def getTrainingInfo(train_direc):
    with open(f"{train_direc}/samples.json", "r") as f:
        samples = json.load(f)

    with open(f"{train_direc}/branches.json", "r") as f:
        pnn_branches_full = json.load(f)

    with open(f"{train_direc}/params.json", "r") as f:
        params = json.load(f)

    scaler = pickle.load(open(f"{train_direc}/scaler.pkl", "rb"))
    mass_scaler = pickle.load(open(f"{train_direc}/mass_scaler.pkl", "rb"))


    return samples, pnn_branches_full, params, scaler, mass_scaler


def loadModel(params, train_direc):
    # First import the model and instantiate it
    model_module = importlib.import_module("fcc_study.pNN.training.model")
    # Initiate specific model
    model = getattr(
        model_module,
        params["model"],
    )(params["model_params"])
    print(model)
    # Get gpu if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")
    print(f"device type = {type(device)}")
    # Now load in the model state
    model.load_state_dict(torch.load(f"{train_direc}/model.pt", map_location=device))
    model.to(device)

    return model, device


# Load in the training info
samples, branches, params, feat_scaler, mass_scaler = getTrainingInfo(directory)

# Load in the model
model, device = loadModel(params, directory)

class Evaluator():
    def __init__(self, model, device):
        self.model = model
        self.device = device

    
    def getProbs(self, data):
        # Input the data, and get the probs out
        print("Finding NN output probabilities")
        probabilities = []
        loader = DataLoader(
            data,
            batch_size=10000,
            shuffle=False,
            pin_memory=True,
            num_workers=8,
        )
        self.model.eval()
        for x, y, masses, w, wc in tqdm(loader):
            # Transform the data
            x = x.to(self.device)
            y = y.to(self.device)
            masses = masses.to(self.device)
            w = w.to(self.device)

            out = self.model(x, masses)
            prob = torch.sigmoid(out)
            probabilities.append(prob.detach().cpu().numpy())

        # Now flatten
        probabilities = np.concatenate(probabilities, axis=0)
        return probabilities

    def getProbsForEachMass(self, dataset, unique_masses):
        # Find the probs for each sample, for each mass point
        # Think this will be easier as a pandas df
        probabilities = []
        # Loop over all masses
        for mass in unique_masses:
            print(f"Finding probabilities for mass = {mass}")
            # Set dataset mass info as the chosen mass
            dataset.setAllMasses(mass)
            # Now evaluate
            probs = self.getProbs(dataset)
            probabilities.append(probs)

        # probabilities = np.concatenate(probabilities, axis=-1)
        mass_scaler = dataset.mass_scaler  # Want to convert to normal masses
        unique_masses = mass_scaler.inverse_transform(unique_masses)
        print(f"unique_masses = ")
        print(unique_masses)
        # Convert e.g. [80, 100, 120] to string "mH80_mA100_mHch120"
        mass_strings = self.convertMassesToString(unique_masses)
        for m, probs in zip(mass_strings, probabilities):
            dataset.data[m] = ak.flatten(probs)

        return dataset.data

    def convertMassesToString(self, masses):
        # This finds which BP it it is. This works by finding the nearest
        # rather than the exact because sometimes the masses are not exactly
        # the same after converting with the mass_scaler (before pNN) and
        # then back

        masses_strings = []
        for mass in masses:
            ms, md = int(np.rint(mass[0])), int(np.rint(mass[1]))

            masses_strings.append(f"pnn_output_mH{ms}_mA{md}")

        print(f"Masses {masses} converted to {masses_strings}")
        return masses_strings

trainer = Evaluator(model, device)

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
    files = glob.glob(f"/vols/cms/emc21/FCC/FCC-Study/Data/NewestDataComplete/ecom365/scenario{scenario}/awkward_files/{run_name}/*.parquet")
    bkg_files = [file for file in files if "h2h2" not in file]
    sig_files = [file for file in files if "h2h2" in file]

    bkg_data = []
    for file in bkg_files:
        bkg_data.append(ak.from_parquet(file))
    bkg_data = combineInChunks(bkg_data)

    # Replace any nans with 0
    for br in branches:
        bkg_data[br] = ak.nan_to_num(bkg_data[br], 0)


    print("all_masses: ", all_masses)

    # Now do this in parts: for backgrounds can combine all then evaluate model 
    # Then save separately, but for signal I don't want to evaluate the signal 
    # on other signal point masses.
    bkg_data = evaluateModelOnData(bkg_data, branches, all_masses, feat_scaler, mass_scaler, trainer)

    saveSamples(bkg_data, run_loc, feat_scaler, branches, run_name = run_name)


    # Get all the pnn_output branches
    pnn_output_branches = [f for f in ak.fields(bkg_data) if "pnn_output" in f]

    # Now delete bkg_data and load in the signal data
    del bkg_data

    # Need to pair all of the signal files that have the same mass point 
    # in the name
    mass_points = []
    for f in sig_files:
        mass_point = f.split("/")[-1].split("_h2h2")[0]
        if mass_point not in mass_points:
            mass_points.append(mass_point)
    # Now get all the files that have the same mass point
    sig_file_dict = {}
    for mass_point in mass_points:
        sig_file_dict[mass_point] = [file for file in sig_files if mass_point in file]

    # Now evaluate the signal data
    for mass_point, mass_files in sig_file_dict.items():

        file_name = f"{run_loc}/data/{run_name}/awkward/{mass_point}.parquet"
        # Check if it exists, if it does then skip the evaluation
        if os.path.isfile(file_name):
            print(f"Skipping {mass_point} as already evaluated.")
            continue

        sig_train = []
        for file in mass_files:
            sig_train.append(ak.from_parquet(file))
        sig_train = combineInChunks(sig_train)

        # Replace any nans with 0
        for br in branches:
            sig_train[br] = ak.nan_to_num(sig_train[br], 0)

        # Now I need to loop over all the signal points and evaluate the model on them
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


# First I need to load in all of the signal data to get all of the unique masses
val_files = glob.glob(f"/vols/cms/emc21/FCC/FCC-Study/Data/NewestDataComplete/ecom365/scenario{scenario}/awkward_files/val/*h2h2ll*.parquet")
val_data = []
for file in val_files:
    val_data.append(ak.from_parquet(file))
val_data = combineInChunks(val_data)
masses = convertToNumpy(val_data, ['mH', 'mA'])
# Scale the masses
masses = mass_scaler.transform(masses)
unique_masses = copy.deepcopy(np.unique(masses, axis=0))

del val_data

evaluateAllData("test", unique_masses)
evaluateAllData("val", unique_masses)
evaluateAllData("train", unique_masses)