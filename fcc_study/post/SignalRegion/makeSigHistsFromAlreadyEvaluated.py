import numpy as np
import awkward as ak 
import argparse
import json
import glob
from tqdm import tqdm
import os, copy, uproot
import boost_histogram as bh

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

# Load in the mass scan
mass_pairs = np.loadtxt(f"{output_direc}/mass_scan.txt")

bins = np.linspace(0.9, 1, 10)

branches = ['n_muons', 'n_electrons', 'pnn_output_*', 'weight_nominal_scaled', 'mH', 'mA', 'Zcand_m']


def removeLowMassEE(signal):
    sig_elec = signal[signal.n_electrons == 2]
    sig_mu = signal[signal.n_muons == 2]
    sig_elec_Mll_cut = sig_elec.Zcand_m > 30
    eff = np.sum(sig_elec_Mll_cut) / len(sig_elec_Mll_cut)
    print(f"Efficiency of Mll cut: {eff}")
    sig_elec = sig_elec[sig_elec_Mll_cut]

    signal = ak.concatenate([sig_elec, sig_mu], axis=0)

    return signal

def convertToBoostHistogram(hist, sumw2, bins):
    root_hist = bh.Histogram(bh.axis.Variable(bins), 
                            storage=bh.storage.Weight())

    root_hist[...] = np.stack([hist, sumw2], axis=-1)

    return root_hist



histogram_dict = {
    "Electron": {},
    "Muon": {}
}

sig_files = glob.glob(f"{train_direc}/data/test/awkward/mH*.parquet")

for file in tqdm(sig_files):
    file_name = file.split("/")[-1]
    mH = int(file_name.split("mH")[1].split("_")[0])
    mA = int(file_name.split("mA")[1].split("_")[0])
    signal = ak.from_parquet(file, columns=branches)
    signal = removeLowMassEE(signal)

    for process in ['Electron', 'Muon']:

        events_proc = signal[signal[f'n_{process.lower()}s'] == 2]

        weights_array = events_proc.weight_nominal_scaled

        hist = np.histogram(events_proc[f"pnn_output_mH{int(mH)}_mA{int(mA)}"], bins=bins, weights=weights_array)[0]

        if f"mH{mH}_mA{mA}" in histogram_dict[process]:
            histogram_dict[process][f"mH{mH}.0_mA{mA}.0"][f"{process};idm;mH{mH}.0_mA{mA}.0"] += hist
        else:
            histogram_dict[process][f"mH{mH}.0_mA{mA}.0"] = {f"{process};idm;mH{mH}.0_mA{mA}.0" : hist}


for channel, h_dict in histogram_dict.items():
    for mass_point, h_d in h_dict.items():
        print(f"Saving for {mass_point}")
        root_histograms = {}
        for key, histogram in h_d.items():
            PROCESS, proc_name, mass = key.split(";")
            root_histogram = convertToBoostHistogram(histogram, np.zeros_like(histogram), bins)
            root_histograms[f"{proc_name}"] = root_histogram

        # Now save all of the histograms
        # make the directory
        save_direc = f"{output_direc}/combine/{mass_point}"
        os.makedirs(save_direc, exist_ok=True)

        save_loc = f"{save_direc}/{mass_point}_signal_{channel}_hists.root"

        with uproot.recreate(save_loc) as f:
            for key, hist in root_histograms.items():
                f[key] = hist