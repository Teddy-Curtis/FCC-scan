import argparse 
import json, copy, pickle, importlib, os, glob
import awkward as ak
import numpy as np
import uproot
import torch
from fcc_study.pNN.training.preprocessing_datasetClasses import CustomDataset, applyScaler, applyInverseScaler, combineInChunks
from tqdm import tqdm
from torch.utils.data import DataLoader
from fcc_study.pNN.utils import convertToNumpy

def parse_arguments():

    parser = argparse.ArgumentParser(description="Evaluate model on a dataset.")

    parser.add_argument(
        "--run_loc",
        required=True,
        default=None,
        type=str,
        help="Training directory where the model is.",
    )

    return parser.parse_args()




def getTrainingInfo(train_direc):
    with open(f"{train_direc}/samples.json", "r") as f:
        samples = json.load(f)

    with open(f"{train_direc}/branches.json", "r") as f:
        pnn_branches = json.load(f)

    with open(f"{train_direc}/params.json", "r") as f:
        params = json.load(f)

    feat_scaler = pickle.load(open(f"{train_direc}/scaler.pkl", "rb"))
    mass_scaler = pickle.load(open(f"{train_direc}/mass_scaler.pkl", "rb"))

    # # Load in the masses
    # masses = np.load(f"{train_direc}/masses.npy")

    # #! Needs to change
    # #masses = np.array([[80, 100], [80, 110], [80, 120], [80, 130], [80, 140], [80, 150], [80, 160], [80, 170], [80, 180]])
    # masses = np.float32(masses)

    # # Convert the masses for input into the model
    # masses = mass_scaler.transform(masses)

    return samples, pnn_branches, params, feat_scaler, mass_scaler


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


class EvaluateModel:
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def getProbs(self, data):
        # Input the data, and get the probs out
        print("Finding NN output probabilities")
        probabilities = []
        print("Making dataloader")
        loader = DataLoader(
            data,
            batch_size=10000,
            shuffle=False,
            pin_memory=True,
            num_workers=8,
        )
        print("Made dataloader")
        self.model.eval()
        print("Set model to eval")
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
            print("Setting masses")
            dataset.setAllMasses(mass)
            print("Set masses")
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
            ms, md = np.round(mass[0], 1), np.round(mass[1], 1)

            masses_strings.append(f"pnn_output_mH{ms}_mA{md}")

        print(f"Masses {masses} converted to {masses_strings}")
        return masses_strings


def evaluateModelOnData(
    data, branches, masses, feat_scaler, mass_scaler, trainer
):
    
    # Add the weights to the test data
    if "weight_nominal" in ak.fields(data):
        data['weight'] = copy.deepcopy(data['weight_nominal'])
    else:
        data['weight'] = np.ones_like(data['weight_nominal_scaled'])

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

def evaluateAllData(run_name, all_masses, run_loc, feat_scaler, mass_scaler, evaluator):
    # files_sig = glob.glob(f"/vols/cms/emc21/FCC/FCC-Study/Data/stage2/awkward_files/{run_name}/*h2h2*.parquet")
    # files_bkg = glob.glob(f"/vols/cms/emc21/FCC/FCC-Study/Data/stage2/awkward_files/{run_name}/*.parquet")
    # files_bkg = [file for file in files_bkg if "h2h2" not in file]
    # files = files_sig + files_bkg
    files = glob.glob(f"/vols/cms/emc21/FCC/FCC-Study/Data/stage2/awkward_files/{run_name}/*h2h2*.parquet")
    data = []
    for file in files:
        print(file)
        data.append(ak.from_parquet(file))
    data = combineInChunks(data)


    print("all_masses: ", all_masses)

    # # Now do this in parts: for backgrounds can combine all then evaluate model 
    # # Then save separately, but for signal I don't want to evaluate the signal 
    # # on other signal point masses.
    # bkg_train = data[data['class'] == 0]
    # bkg_train = evaluateModelOnData(bkg_train, branches, all_masses, feat_scaler, mass_scaler, evaluator)

    # saveSamples(bkg_train, run_loc, feat_scaler, branches, run_name = run_name) 

    bkg_train = ak.from_parquet("/vols/cms/emc21/FCC/FCC-Study/runs/e240_full_run_fixedLumis/run1/data/train/awkward/wzp6_ee_eeH_ecm240.parquet")
    # Get all the pnn_output branches
    pnn_output_branches = [f for f in ak.fields(bkg_train) if "pnn_output" in f]

    print(f"All pnn output branches = {pnn_output_branches}")


    # Now I need to loop over all the signal points and evaluate the model on them
    sig_train = data[data['class'] == 1]
    sig_procs = np.unique(list(sig_train.process))
    for sig_proc in sig_procs:
        print(f"Processing signal process: {sig_proc}")
        sig_data = sig_train[sig_train['process'] == sig_proc]

        sig_data['weight'] = copy.deepcopy(sig_data['weight_nominal'])
        sig_data = applyScaler(sig_data, feat_scaler, branches)
        sig_data = applyScaler(sig_data, mass_scaler, ["mH", "mA"])

        sig_dataset = CustomDataset(sig_data, branches, feat_scaler, mass_scaler)

        masses = sig_dataset.unique_masses

        sig_data = evaluator.getProbsForEachMass(sig_dataset, masses)

        # Now fill in the pnn_output branches
        for pnn_output_branch in pnn_output_branches:
            if pnn_output_branch not in ak.fields(sig_data):
                sig_data[pnn_output_branch] = np.ones_like(sig_data['Zcand_m']) * -1

        
        # Now save the data
        saveSamples(sig_data, run_loc, feat_scaler, branches, run_name = run_name)




if __name__ == "__main__":
    args = parse_arguments()
    run_loc = args.run_loc

    # First I need to load in all of the 
    samples, branches, params, feat_scaler, mass_scaler = getTrainingInfo(run_loc)
    model, device = loadModel(params, run_loc)

    evaluator = EvaluateModel(model, device)

    # Get the unique masses
    val_files = glob.glob(f"/vols/cms/emc21/FCC/FCC-Study/Data/stage2/awkward_files/val/*h2h2*.parquet")
    val_data = []
    for file in val_files:
        print(file)
        val_data.append(ak.from_parquet(file, columns=['mH', 'mA']))
    val_data = combineInChunks(val_data)

    masses = convertToNumpy(val_data, ['mH', 'mA'])
    unique_masses = np.unique(masses, axis=0)
    mass_scaler = pickle.load(open(f"/vols/cms/emc21/FCC/FCC-Study/runs/e240_full_run/run25/mass_scaler.pkl", "rb"))
    unique_masses = mass_scaler.transform(unique_masses)


    # Get the feature and mass scalers
    # evaluateAllData("train", unique_masses, run_loc, feat_scaler, mass_scaler, evaluator)
    # evaluateAllData("val", unique_masses, run_loc, feat_scaler, mass_scaler, evaluator)
    evaluateAllData("test", unique_masses, run_loc, feat_scaler, mass_scaler, evaluator)