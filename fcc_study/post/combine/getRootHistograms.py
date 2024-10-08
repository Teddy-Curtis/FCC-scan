# Script to get the signal region histograms
import argparse
import awkward as ak 
import glob
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import mplhep as hep
import uproot, os
import boost_histogram as bh
import numpy as np
import copy



def parse_arguments():

    parser = argparse.ArgumentParser(description="Evaluate model on a dataset.")

    parser.add_argument(
        "--run_loc",
        required=True,
        default=None,
        type=str,
        help="Training directory where the model is.",
    )
    
    parser.add_argument(
        "--ecom",
        required=True,
        type=int,
        help="Center of mass energy.",
    )

    return parser.parse_args()

parser = parse_arguments()
run_loc = parser.run_loc
ecom = parser.ecom

def rescaleWeightsToCorrectLumi(events, ecom):
    if ecom == 240:
        scale = 10_800_000 / 5_000_000
    elif ecom == 365:
        scale = 3_000_000 / 5_000_000
    else:
        raise ValueError("Invalid ecom")

    events['weight_nominal'] = events['weight_nominal'] * scale
    events['weight_nominal_scaled'] = events['weight_nominal_scaled'] * scale

    return events


def getWeight(mH, mA, process, Ecom, lumi):
    print(f"Finding weight for {mH}, {mA}, {process}, {Ecom}, {lumi}")
    with open("/vols/cms/emc21/FCC/FCC-Study/Data/stage2/signalInfo.txt", "r") as f:
        lines = f.readlines()
        lines = [l.strip() for l in lines]

    for ecom, mh, ma, proc, num_evs, xs in [l.split(",") for l in lines]:
        try:
            ecom = int(ecom)
            mh = int(mh)
            ma = int(ma)
            xs = float(xs)
        except:
            continue
        if ecom == Ecom and mh == mH and ma == mA and proc == process:
            weight = xs * lumi / 500_000
            return weight
        
    print("You done fucked up")

def getSignalWeights(sig, Ecom, lumi):
    proc = sig.process[0]
    mH = int(proc.split("mH")[1].split("_")[0])
    mA = int(proc.split("mA")[1])

    #specific_procs = np.unique(list(sig.specific_proc))
    specific_procs = ['h2h2ll', 'h2h2llvv']
    scale = copy.deepcopy(sig['weight_nominal_scaled'] / sig['weight_nominal'])
    sig['weight_nominal'] = np.ones_like(sig['weight_nominal'])
    for proc in specific_procs:
        # Get the weight for that proc

        weight = getWeight(mH, mA, proc, Ecom, lumi)

        sig['weight_nominal'] = np.where(
            sig.specific_proc == proc, 
            sig['weight_nominal'] * weight, 
            sig['weight_nominal']
        )

    sig['weight_nominal_scaled'] = sig['weight_nominal'] * scale

    return sig
    



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



branches = ['n_muons', 'n_electrons', 'pnn_output_mH*', 'process', 'specific_proc', 'class', 'weight_nominal*']

test = []
#test_files = glob.glob(f"data/test/awkward/*.parquet")
sig_files = glob.glob(f"{run_loc}/data/test/awkward/mH*.parquet")
bkg_files = glob.glob(f"{run_loc}/data/test/awkward/*.parquet")
bkg_files = [f for f in bkg_files if "mH" not in f]
test_files = sig_files + bkg_files
for file in tqdm(test_files):
    test.append(ak.from_parquet(file, columns=branches))

test = combineInChunks(test)

print(f"Mean weight before lumi rescaling = {np.mean(test['weight_nominal_scaled'])}")
test = rescaleWeightsToCorrectLumi(test, ecom)
print(f"Mean weight after lumi rescaling = {np.mean(test['weight_nominal_scaled'])}")


bkg = test[test['class'] == 0]

sig_procs = sorted(np.unique(test[test['class'] == 1].process))
bkg_procs = np.unique(test[test['class'] == 0].process)

def getSlimBkgProcName(proc):
    new_proc_name = proc.replace('wzp6_', '').replace("_ecm240", "").replace("p8_", "")[3:]
    return new_proc_name

def getSigAndBkg(data, sig_proc):
    sig = data[data['process'] == sig_proc]
    bkg = data[data['class'] == 0]
    return sig, bkg

def getSig(data, sig_proc):
    sig = data[data['process'] == sig_proc]
    return sig



bins = np.linspace(0.9, 1, 26)

datacard_files = ['/vols/cms/emc21/FCC/FCC-Study/fcc_study/post/MuMu_datacard.txt',
                  '/vols/cms/emc21/FCC/FCC-Study/fcc_study/post/EE_datacard.txt',
                  '/vols/cms/emc21/FCC/FCC-Study/fcc_study/post/combined_datacard.txt']


def checkIfZeroHist(hist):
    if np.sum(hist) == 0:
        return np.ones_like(hist) * 1e-9
    return hist


for sig_proc in tqdm(sig_procs):
    #print(f"Processing signal process: {sig_proc}")
    sig = getSig(test, sig_proc)

    # Get the new sig weights

    if ecom == 240:
        print(f"Mean weight before getSignalWeights = {np.mean(sig['weight_nominal_scaled'])}")
        sig = getSignalWeights(sig, 240, 10_800_000) #! Will need to change the Ecom!
        print(f"Mean weight after getSignalWeights = {np.mean(sig['weight_nominal_scaled'])}")

    # Now loop over electrons and muons
    for PROCESS in ['electrons', 'muons']:
        sig_PROC = sig[sig[f'n_{PROCESS}'] == 2]
        bkg_PROC = bkg[bkg[f'n_{PROCESS}'] == 2]

        # Now get the signal histogram
        sig_hist = np.histogram(sig_PROC[f'pnn_output_{sig_proc}'], bins=bins, weights=sig_PROC['weight_nominal_scaled'])[0]
        sig_err = np.sqrt(np.histogram(sig_PROC[f'pnn_output_{sig_proc}'], bins=bins, weights=sig_PROC['weight_nominal_scaled'])[0])

        sig_hist = checkIfZeroHist(sig_hist)

        # Now just append to the dictionary
        signal_root_hist = bh.Histogram(bh.axis.Variable(bins), 
                                        storage=bh.storage.Weight())
        signal_root_hist[...] = np.stack([sig_hist, sig_err], axis=-1)

        # Now get the background histograms
        data_obs = np.array(np.zeros_like(sig_hist))
        bkg_hists = {}
        for bkg_proc in bkg_procs:
            bkg_specific_proc = bkg_PROC[bkg_PROC['process'] == bkg_proc]
            bkg_hist = np.histogram(bkg_specific_proc[f'pnn_output_{sig_proc}'], bins=bins, weights=bkg_specific_proc['weight_nominal_scaled'])[0]
            bkg_err = np.sqrt(np.histogram(bkg_specific_proc[f'pnn_output_{sig_proc}'], bins=bins, weights=bkg_specific_proc['weight_nominal_scaled'])[0])

            bkg_hist = np.array(bkg_hist)
            bkg_err = np.array(bkg_err)

            bkg_hist = checkIfZeroHist(bkg_hist)

            bkg_hists[bkg_proc] = bh.Histogram(bh.axis.Variable(bins),
                                                storage=bh.storage.Weight())
            bkg_hists[bkg_proc][...] = np.stack([bkg_hist, bkg_err], axis=-1)

            # Add to data_obs
            data_obs += bkg_hist

        # Now get data_obs boost histogram
        data_obs_hist = bh.Histogram(bh.axis.Variable(bins),
                                    storage=bh.storage.Weight())
        data_obs_hist[...] = np.stack([data_obs, np.sqrt(data_obs)], axis=-1)


        # Now save the histograms to a root file
        os.makedirs(f"{run_loc}/combine_old_done_again_correctLumi/{sig_proc}", exist_ok=True)
        with uproot.recreate(f"{run_loc}/combine_old_done_again_correctLumi/{sig_proc}/{sig_proc}_{PROCESS}.root") as f:
            f[f'idm'] = signal_root_hist
            f['data_obs'] = data_obs_hist
            for bkg_proc, bkg_hist in bkg_hists.items():
                f[f'{getSlimBkgProcName(bkg_proc)}'] = bkg_hist

        
        # Copy all the datacards to the right place
        for datacard in datacard_files:
            os.system(f"cp {datacard} {run_loc}/combine_old_done_again_correctLumi/{sig_proc}/.")
