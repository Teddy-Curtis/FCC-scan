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
bins = np.linspace(0.9, 1, 16)


bkg = []
files = glob.glob(f"{train_direc}/data/test/awkward/*.parquet")
files = [f for f in files if "mH" not in f]
for file in tqdm(files): 
    bkg.append(ak.from_parquet(file))

bkg = combineInChunks(bkg)

#! Remove ee events with Mll < 30
bkg_elec = bkg[bkg.n_electrons == 2]
bkg_mu = bkg[bkg.n_muons == 2]
bkg_elec_Mll_cut = bkg_elec.Zcand_m > 30
eff = np.sum(bkg_elec_Mll_cut) / len(bkg_elec_Mll_cut)
print(f"Efficiency of Mll cut: {eff}")
bkg_elec = bkg_elec[bkg_elec_Mll_cut]

bkg = ak.concatenate([bkg_elec, bkg_mu], axis=0)


bkg_proc_groups = {
    "ZZ" : ['p8_ee_ZZ'],
    "tautau" : ['wzp6_ee_tautau'],
    "WW" : ['p8_ee_WW'],
    "ee" : ['wzp6_ee_ee_Mee_30_150'],
    "mumu" : ['wzp6_ee_mumu'],
    "Higgs_X" : ['wzp6_ee_eeH', 'wzp6_ee_mumuH', 
                    'wzp6_ee_nunuH', 'wzp6_ee_tautauH', 'wzp6_ee_qqH']
}

def getGroup(proc):
    for group, procs in bkg_proc_groups.items():
        for p in procs:
            if proc.startswith(p):
                return group
    raise ValueError(f"Process {proc} not found in any group")


def makeEmptyHistogramDict():
    # This just makes a histogram with empty bins for each process
    bins = np.linspace(0.9, 1, 16)
    hist = np.zeros(len(bins)-1)

    return {proc : [copy.deepcopy(hist), copy.deepcopy(hist)] for proc in bkg_proc_groups.keys()}

# Now for each background process, and for each mass pair, I need to make the 
# histograms
# Now for each mass pair, I need to make the histograms
for mH, mA in mass_pairs:
    print(mH, mA)
    mH, mA = int(mH), int(mA)
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


    root_hist_dict = {
        "Electron" : makeEmptyHistogramDict(),
        "Muon" : makeEmptyHistogramDict()
    }

    for proc, hists in histograms.items():
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


    # Just save this as a json file for now
    save_loc = f"{output_direc}/combine/mH{mH}.0_mA{mA}.0"
    os.makedirs(save_loc, exist_ok=True)

    # with open(f"{save_loc}/backgrounds.json", "w") as f:
    #     json.dump(histograms, f, indent=4)


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
