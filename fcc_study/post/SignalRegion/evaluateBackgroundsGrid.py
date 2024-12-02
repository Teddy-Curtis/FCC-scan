import numpy as np
import awkward as ak 
import pandas as pd 
import time
import sys
import argparse
import json
import glob
from tqdm import tqdm
import uproot, pickle
import os
from fcc_study.post.evaluateData import getTrainingInfo, loadModel, EvaluateModel, convertToNumpy, evaluateModelOnData
from fcc_study.pNN.training.preprocessing_datasetClasses import applyInverseScaler

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
        "--mH",
        required=True,
        default=None,
        type=float,
        help="mH to interpolate for.")

    return parser.parse_args()


def combineChunk(event_list):
    chunk_size = 10
    num_chunks = len(event_list) // chunk_size

    if num_chunks == 0:
        evs = ak.merge_option_of_records(ak.concatenate(event_list))
        return evs

    events = []
    for i in range(num_chunks + 1):
        evs = event_list[i * chunk_size:(i + 1) * chunk_size]
        if len(evs) == 0:
            continue
        evs = ak.merge_option_of_records(ak.concatenate(evs))
        events.append(evs)
    
    return events

def combineInChunks(event_list):
    print("Inside combine in chunks")
    combined = False
    while not combined:
        event_list = combineChunk(event_list)
        if isinstance(event_list, ak.Array):
            print("Here")
            combined = True

    return event_list



parser = parse_arguments()
train_direc = parser.train_direc
output_direc = parser.output_direc
mH = parser.mH


# Load in the mass scaler
mass_scaler = pickle.load(open(f"{train_direc}/mass_scaler.pkl", "rb"))

# Load in the mass scan
mass_pairs = np.loadtxt(f"{output_direc}/mass_scan.txt")

# mass_pairs = mass_pairs[mass_pairs[:, 0] == mH]# [:2] #! Change!
print(f"Running over mass pairs: {mass_pairs}")

# I need to convert the mass pairs
mass_pairs_scaled  = mass_scaler.transform(mass_pairs)
mass_pairs_scaled = ak.values_astype(mass_pairs_scaled, "float32")

bins = np.linspace(0.9, 1, 25)

# Load the signal and weights
# branches = ['n_muons', 'n_electrons', 'pnn_output_*', 'weight_nominal_scaled', 'mH', 'mA']

bkg = []
#test_files = glob.glob(f"data/test/awkward/*.parquet")
files = glob.glob(f"{train_direc}/data/test/awkward/*.parquet")
files = [f for f in files if "mH" not in f]
for file in tqdm(files): 
    bkg.append(ak.from_parquet(file))

bkg = combineInChunks(bkg)
# bkg = rescaleWeightsToCorrectLumi(bkg, ecom)

bkg = ak.values_astype(bkg, "float32")


# First I need to load in all of the 
samples, branches, params, feat_scaler, mass_scaler = getTrainingInfo(train_direc)
model, device = loadModel(params, train_direc)

evaluator = EvaluateModel(model, device)



bkg = evaluateModelOnData(bkg, branches, mass_pairs_scaled, feat_scaler, mass_scaler, evaluator)

# Convert the features back to the unscaled features
bkg = applyInverseScaler(bkg, feat_scaler, branches)

for f in ak.fields(bkg):
    print(f)

# Now for each background process, and for each mass pair, I need to make the 
# histograms
# Now for each mass pair, I need to make the histograms
for mH, mA in mass_pairs:
        pnn_var = f"pnn_output_mH{mH}_mA{mA}"

        histograms = {}
        bkg_procs = np.unique(bkg.process)
        for bkg_proc in bkg_procs:
            for process in ['Electron', 'Muon']:

                histogram_dict = {}

                bkg_dilep = bkg[bkg[f'n_{process.lower()}s'] > 1 ]


                bkg_data = bkg_dilep[bkg_dilep['process'] == bkg_proc]
                
                hist, bins = np.histogram(bkg_data[pnn_var], bins=bins, weights=bkg_data['weight_nominal_scaled'])
                sumw2, _ = np.histogram(bkg_data[pnn_var], bins=bins, weights=bkg_data['weight_nominal_scaled']**2)

                # Add to the dictionary
                histograms[f"{bkg_proc};{process}"] = (list(hist), list(sumw2))


        # Just save this as a json file for now
        save_loc = f"{output_direc}/combine/mH{mH}_mA{mA}"
        os.makedirs(save_loc, exist_ok=True)

        with open(f"{save_loc}/backgrounds.json", "w") as f:
            json.dump(histograms, f, indent=4)