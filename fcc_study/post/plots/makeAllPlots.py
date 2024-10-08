import awkward as ak 
import glob
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import mplhep as hep
import argparse, os 

def parse_arguments():

    parser = argparse.ArgumentParser(
        description="Which process, year and mass point to make the plots for.")
    
    parser.add_argument(
        "--train_direc",
        required=False,
        default=None,
        type=str,
        help="Directory that contains the histograms for combine.")
    
    parser.add_argument(
        "--ecom",
        required=True,
        default=None,
        type=int,
        help="Center of mass energy.")


    parser = parser.parse_args()
    
    parser_kwargs = parser._get_kwargs()
    for arg, val in parser_kwargs:
        print(f"{arg} : {val}")

    return parser





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


def rescaleWeightsToCorrectLumi(events, ecom):
    if ecom == 240:
        scale = 10800000 / 5000000
    elif ecom == 365:
        scale = 3000000 / 5000000
    else:
        raise ValueError("Invalid ecom")

    events['weight_nominal'] = events['weight_nominal'] * scale
    events['weight_nominal_scaled'] = events['weight_nominal_scaled'] * scale

    return events

def getSigAndBkg(data, sig_proc):
    sig = data[data['process'] == sig_proc]
    bkg = data[data['class'] == 0]
    return sig, bkg

def getSig(data, sig_proc):
    sig = data[data['process'] == sig_proc]
    return sig


def plot(sig_proc, bkg_procs, bins, save_name):
    mH = sig_proc.split("_")[0].split("mH")[1]
    mA = sig_proc.split("mA")[1]
    sig = getSig(test, sig_proc)
    bkg = test[test['class'] == 0]

    if mA - mH < 30:
        latex_proc = r"$\mu^-\mu^+$"
        sig = sig[sig['n_muons'] > 0]
        bkg = bkg[bkg['n_muons'] > 0]
    else:
        latex_proc = r"$e^-e^+/\mu^-\mu^+$"


    sig_hist = np.histogram(sig[f'pnn_output_{sig_proc}'], bins=bins, weights=sig['weight_nominal_scaled'])[0]

    bkg_hists = []
    bkg_proc_labels = []
    for proc in bkg_procs:
        # if proc in ['wzp6_ee_tautauH_ecm240', 'wzp6_ee_mumuH_ecm240', 'wzp6_ee_eeH_ecm240', 'nunuH']:
        #     continue
        mask = bkg['process'] == proc
        bkg_hist = np.histogram(bkg[mask][f'pnn_output_{sig_proc}'], bins=bins, weights=bkg[mask]['weight_nominal_scaled'])[0]
        bkg_hists.append(bkg_hist)
        new_proc = proc.replace('wzp6_', '').replace("_ecm240", "").replace("p8_", "").replace("_ecm365", "")[3:]
        if "Mee_30_150" in new_proc:
            new_proc = new_proc.replace("_Mee_30_150", "")
        bkg_proc_labels.append(new_proc)


    # Now I want to reorder these in terms of number of events
    bkg_hists = np.array(bkg_hists)
    bkg_proc_labels = np.array(bkg_proc_labels)
    bkg_order = np.argsort(np.sum(bkg_hists, axis=1))# [::-1]
    bkg_hists = bkg_hists[bkg_order]
    bkg_proc_labels = bkg_proc_labels[bkg_order]

    # combine the first 4 backgrounds into a single Other background
    other_bkg = np.sum(bkg_hists[:4], axis=0)
    bkg_hists = np.concatenate([[other_bkg], bkg_hists[4:]])
    bkg_proc_labels = np.concatenate([['Higgs + X'], bkg_proc_labels[4:]])


    for h, name in zip(bkg_hists, bkg_proc_labels):
        print(name, np.sum(h))


    # Now plot

    fig, ax = plt.subplots(figsize=(12, 8.5))
    plt.style.use(hep.style.CMS)
    #hep.cms.label(data=False, loc=0, lumi = r"10.8 $ab^-1$")
    _ = hep.histplot(bkg_hists, bins=bins, label=bkg_proc_labels, stack=True, histtype='fill', ax=ax, alpha=0.7)
    _ = hep.histplot(sig_hist, bins=bins, label=f'$M_H={mH}, M_A={mA}$ GeV', color='black', ax=ax, linestyle='--')
    plt.legend(ncol=2, loc='upper center', fontsize=16)
    ax.set_ylim(0.1, 1e6)
    ax.set_yscale('log')
    ax.set_xlabel('PNN Output')
    ax.set_ylabel('Events')
    ax.set_xlim(0.9, 1)
    #ax.set_title(f'{sig_proc} PNN Output')
    plt.text(0.01, 1.013, latex_proc, fontsize="18",
                transform=ax.transAxes)
    plt.text(0.51, 1.013, r"FCC-ee,  $\sqrt{s}$=240 GeV,  Lumi=10.8 $ab^{-1}$", fontsize="18",
                transform=ax.transAxes)
    plt.savefig(save_name, bbox_inches='tight')

parser = parse_arguments()
train_direc = parser.train_direc
ecom = parser.ecom

branches = ['n_muons', 'n_electrons', 'pnn_output_*', 'process', 'specific_proc', 'class', 'weight_nominal*', 'MET_pt', 'Zcand_m']

test = []
#test_files = glob.glob(f"data/test/awkward/*.parquet")
sig_files = glob.glob(f"{train_direc}/data/test/awkward/*.parquet")
bkg_files = glob.glob(f"{train_direc}/data/test/awkward/*.parquet")
bkg_files = [f for f in bkg_files if "mH" not in f]
test_files = sig_files + bkg_files
for file in tqdm(test_files):
    test.append(ak.from_parquet(file, columns=branches))

test = combineInChunks(test)




test = rescaleWeightsToCorrectLumi(test, ecom)

bkg_procs = np.unique(test[test['class'] == 0].process)

sig_procs = sorted(np.unique(test[test['class'] == 1].process))

save_direc = f"{train_direc}/plots/pnn_output"
os.makedirs(save_direc, exist_ok=True)
for sig_proc in tqdm(sig_procs):

    # First plot the whole pNN output
    bins = np.linspace(0, 1, 100)
    save_name = f'{save_direc}/pnn_output_{sig_proc}.png'
    plot(sig_proc, bkg_procs, bins, save_name=f'pnn_output_{sig_proc}.png')

    # Now for just the SR region
    bins = np.linspace(0.9, 1, 25)
    save_name = f'{save_direc}/pnn_output_{sig_proc}_SR.png'
    plot(sig_proc, bkg_procs, bins, save_name=save_name)
