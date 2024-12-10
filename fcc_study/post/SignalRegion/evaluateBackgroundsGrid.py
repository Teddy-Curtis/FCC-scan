import numpy as np
import awkward as ak 
import pandas as pd 
import time
import sys
import argparse
import json
import glob
from tqdm import tqdm
import boost_histogram as bh
import uproot, pickle
import os, copy
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
        required=False,
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
# mH = parser.mH


# Load in the mass scaler
mass_scaler = pickle.load(open(f"{train_direc}/mass_scaler.pkl", "rb"))

# Load in the mass scan
mass_pairs = np.loadtxt(f"{output_direc}/mass_scan.txt")
mass_pairs = ak.values_astype(mass_pairs, "float32")

# mass_pairs = mass_pairs[mass_pairs[:, 0] == 72.5]# [:2] #! Change!
# mass_pairs = mass_pairs[mass_pairs[:,1] - mass_pairs[:,0] > 25] #! Change!
# mass_pairs = mass_pairs[:5]
print(f"Running over mass pairs: {mass_pairs}")

# I need to convert the mass pairs
mass_pairs_scaled  = mass_scaler.transform(mass_pairs)
mass_pairs_scaled = ak.values_astype(mass_pairs_scaled, "float32")

for m, ms in zip(mass_pairs, mass_pairs_scaled):
    print(f"{m} -> {ms}")


bins = np.linspace(0.9, 1, 16)

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

fields = ak.fields(bkg)
pnn_fields = [f for f in fields if 'pnn_output' in f]
for field in pnn_fields:
    mH = field.split("mH")[1].split("_")[0]
    mA = field.split("mA")[1]

    new_field = f"pnn_output_mH{mH}.0_mA{mA}.0"

    bkg[new_field] = bkg[field]

    # Now remove the old field
    del bkg[field]

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


bkg_proc_groups = {
    "ZZ" : ['p8_ee_ZZ'],
    "tautau" : ['wzp6_ee_tautau'],
    "WW" : ['p8_ee_WW'],
    "ee" : ['wzp6_ee_ee_Mee_30_150'],
    "mumu" : ['wzp6_ee_mumu'],
    "Higgs_X" : ['wzp6_ee_eeH', 'wzp6_ee_mumuH', 
                    'wzp6_ee_nunuH', 'wzp6_ee_tautauH']
}

def getGroup(proc):
    for group, procs in bkg_proc_groups.items():
        for p in procs:
            if proc.startswith(p):
                return group
    raise ValueError(f"Process {proc} not found in any group")


def makeEmptyHistogramDict(bins):
    # This just makes a histogram with empty bins for each process
    hist = np.zeros(len(bins)-1)

    return {proc : [copy.deepcopy(hist), copy.deepcopy(hist)] for proc in bkg_proc_groups.keys()}

def convertToRoot(hists_dict, save_loc, bins):
    # I want to make a dictionary of hists, with the correct samples combined
    # and with the correct names 
    root_hist_dict = {
        "Electron" : makeEmptyHistogramDict(bins),
        "Muon" : makeEmptyHistogramDict(bins)
    }

    for proc, hists in hists_dict.items():
        proc_name, process = proc.split(";")
        hist, sumw2 = hists

        group = getGroup(proc_name)

        root_hist_dict[process][group][0] += np.array(hist)
        root_hist_dict[process][group][1] += np.array(sumw2)

    # Also add all of the histograms together, to make data_obs
    for process in ['Electron', 'Muon']:
        root_hist_dict[process]['data_obs'] = [np.zeros(len(hist)), np.zeros(len(hist))]
        for group, hists in root_hist_dict[process].items():
            if group == "data_obs":
                continue
            root_hist_dict[process]['data_obs'][0] += hists[0]
            root_hist_dict[process]['data_obs'][1] += hists[1]

    # Now that I have the histograms, I want to save them to a root file
    with uproot.recreate(f"{save_loc}/backgrounds.root") as f:
        for process, hist_dict in root_hist_dict.items():
            for group, hists in hist_dict.items():
                hist, sumw2 = hists

                # If hist is all zeros, then just add a tiny value 
                if np.all(hist == 0):
                    hist += 1e-7

                root_hist = bh.Histogram(bh.axis.Variable(bins), 
                                        storage=bh.storage.Weight())
                root_hist[...] = np.stack([hist, sumw2], axis=-1)

                f[f"{group}_{process}"] = root_hist

# Now for each background process, and for each mass pair, I need to make the 
# histograms
# Now for each mass pair, I need to make the histograms
histograms = {}
bkg_procs = np.unique(bkg.process)
for channel in ['Electron', 'Muon']:
    channel_data = bkg[bkg[f'n_{channel.lower()}s'] > 1]

    for bkg_proc in bkg_procs:
        print(f"Process: {bkg_proc}")
        bkg_data = channel_data[channel_data['process'] == bkg_proc]

        for mH, mA in mass_pairs:
            print(f"mH: {mH}, mA: {mA}")
            pnn_var = f"pnn_output_mH{mH}_mA{mA}"

            try:
                pnn_field_data = bkg_data[pnn_var]
            except:
                print(f"WARNING: {pnn_var} not found in data!!! Skipping")
                continue

            hist, bins = np.histogram(bkg_data[pnn_var], bins=bins, weights=bkg_data['weight_nominal_scaled'])
            sumw2, _ = np.histogram(bkg_data[pnn_var], bins=bins, weights=bkg_data['weight_nominal_scaled']**2)

            # Add to the dictionary
            if f"mH{mH}_mA{mA}" in histograms:
                histograms[f"mH{mH}_mA{mA}"][f"{bkg_proc};{channel}"] = (list(hist), list(sumw2))
            else:
                histograms[f"mH{mH}_mA{mA}"] = {f"{bkg_proc};{channel}" : (list(hist), list(sumw2))}


for mass_pair, hists in histograms.items():
    save_loc = f"{output_direc}/combine/{mass_pair}"
    os.makedirs(save_loc, exist_ok=True)

    with open(f"{save_loc}/backgrounds.json", "w") as f:
        json.dump(hists, f, indent=4)

    convertToRoot(hists, save_loc, bins)

# for mH, mA in mass_pairs:
#     print(f"mH: {mH}, mA: {mA}")
#     pnn_var = f"pnn_output_mH{mH}_mA{mA}"

#     histograms = {}
#     bkg_procs = np.unique(bkg.process)
#     for bkg_proc in bkg_procs:
#         bkg_proc_data = bkg[bkg['process'] == bkg_proc]
#         print(f"Process: {bkg_proc}")
#         for process in ['Electron', 'Muon']:
#             print(f"Channel: {process}")
#             bkg_data = bkg_proc_data[bkg_proc_data[f'n_{process.lower()}s'] > 1 ]
            
#             hist, bins = np.histogram(bkg_data[pnn_var], bins=bins, weights=bkg_data['weight_nominal_scaled'])
#             sumw2, _ = np.histogram(bkg_data[pnn_var], bins=bins, weights=bkg_data['weight_nominal_scaled']**2)

#             # Add to the dictionary
#             histograms[f"{bkg_proc};{process}"] = (list(hist), list(sumw2))


#     # Just save this as a json file for now
#     save_loc = f"{output_direc}/combine/mH{mH}_mA{mA}"
#     os.makedirs(save_loc, exist_ok=True)

#     with open(f"{save_loc}/backgrounds.json", "w") as f:
#         json.dump(histograms, f, indent=4)

#     convertToRoot(histograms, save_loc, bins)