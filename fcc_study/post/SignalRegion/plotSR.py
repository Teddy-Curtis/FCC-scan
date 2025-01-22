import awkward as ak 
import numpy as np 
import matplotlib.pyplot as plt
import glob
import mplhep as hep
import argparse, os, sys

def parse_arguments():

    parser = argparse.ArgumentParser(
        description="Masses to evaulate and interpolate pNN at.")
    
    parser.add_argument(
        "--train_direc",
        required=True,
        default=None,
        type=str,
        help="Directory that contains the limit file.")
    
    parser.add_argument(
        "--lumi",
        required=True,
        default=None,
        type=float,
        help="Luminosity to scale the limits by.")
    
    parser.add_argument(
        "--ecom",
        required=True,
        default=None,
        type=float,
        help="Center of mass energy.")

    parser.add_argument(
        "--mHs",
        required=True,
        default=None,
        type=lambda s: [int(item) for item in s.split(',')],
        help="mHs to plot.")

    parser.add_argument(
        "--mAs",
        required=True,
        default=None,
        type=lambda s: [int(item) for item in s.split(',')],
        help="mHs to plot.")

    parser.add_argument(
        "--binsLin",
        required=True,
        default=None,
        type=lambda s: [item for item in s.split(',')],
        help="binsLin to plot.")
    
    parser.add_argument(
        "--save_name",
        required=False,
        default=None,
        type=str,
        help="Extra save name.")


    return parser.parse_args()


input_command_line = " ".join(sys.argv)

parser = parse_arguments()
train_direc = parser.train_direc
lumi = parser.lumi
ecom = int(parser.ecom)
save_name = parser.save_name
mHs = parser.mHs
mAs = parser.mAs
binsLin = parser.binsLin
bins = np.linspace(float(binsLin[0]), float(binsLin[1]), int(binsLin[2]))

# Which mass points do I want to plot?
bkg_files = glob.glob(f"{train_direc}/data/*/awkward/*.parquet")
bkg_files = [f for f in bkg_files if "mH" not in f]

sig_files = [glob.glob(f"{train_direc}/data/*/awkward/*mH{mH}*mA{mA}*.parquet") for mH, mA in zip(mHs, mAs)]
sig_files = [itm for sublist in sig_files for itm in sublist]

branches = ['weight_nominal', 'process', 'specific_proc', 'n_muons']
for mH in mHs:
    branches.append(f"pnn_output_mH{mH}*")
print("Loading backgrounds:")
bkg = ak.concatenate([ak.from_parquet(f, columns=branches) for f in bkg_files])
print("Loading signals:")
signals = ak.concatenate([ak.from_parquet(f, columns=branches) for f in sig_files])

if ecom == 365:
    print("NOTE: I am scaling the lumi from 3ab to 2.7ab!!!")
    signals['weight_nominal'] = signals['weight_nominal'] * 2.7/3.0
    bkg['weight_nominal'] = bkg['weight_nominal'] * 2.7/3.0


for field in ak.fields(signals):
    print(f"Field: {field}")

def getSignal(sig, mH, mA):
    sig_cut = ak.str.starts_with(sig.process, f"mH{mH}_mA{mA}")

    return sig[sig_cut]


bkg_procs = np.unique(bkg.specific_proc)
bkg_groups = {
    "Higgs + X" : ['wzp6_ee_tautauH', 'wzp6_ee_nunuH', 'wzp6_ee_mumuH', 'wzp6_ee_eeH'],
    r"$\mu \mu$" : ['wzp6_ee_mumu'],
    "ZZ" : ['p8_ee_ZZ'],
    r"$ee$" : ['wzp6_ee_ee_Mee_30_150'],
    r"$\tau \tau$" : ['wzp6_ee_tautau'],
    "WW" : ['p8_ee_WW']
}
def getGroupInfo(events, groups):
    # Want to make new column of the group, rather than just the process
    events['group'] = [""]*len(events)

    for group in groups:
        group_mask = events['n_muons'] < -999 # All False
        for process in groups[group]:
            group_mask = group_mask | ak.str.starts_with(events.process, process)

        events['group'] = ak.where(
            group_mask,
            [group]*len(events),
            events['group']
        )

    return events

bkg = getGroupInfo(bkg, bkg_groups)


PROC_COLOURS = {
    "Higgs + X" : "#bd1f01",
    "WW" : "#92dadd",
    r"$ee$" : "#b9ac70",
    r"$\mu \mu$" : "#3f90da",
    "ZZ" : "#ffa90e",
    r"$\tau \tau$" : "#a96b59"
}
def getGroupColour(group):
    for p, col in PROC_COLOURS.items():
        if group.startswith(p):
            return col
    
    raise ValueError(f"Group {group} not found in PROC_COLOURS")


def getHistograms(signals, bkg, mH, mA, bins):

    sig = getSignal(signals, mH, mA)
    
    if mA - mH <= 30:
        bkg_to_plot = bkg[bkg.n_muons > 1]
        sig_to_plot = sig[sig.n_muons > 1]
    else:
        bkg_to_plot = bkg
        sig_to_plot = sig


    bkg_hists = []
    bkg_errs = []
    bkg_label = []
    bkg_colours = []
    for group in bkg_groups:
        mask = bkg_to_plot.group == group

        weights = bkg_to_plot.weight_nominal[mask]

        h = np.histogram(bkg_to_plot[mask][f'pnn_output_mH{mH}_mA{mA}'], bins = bins, weights = weights)[0]
        err = np.histogram(bkg_to_plot[mask][f'pnn_output_mH{mH}_mA{mA}'], bins = bins, weights = weights**2)[0]

        # if np.sum(h) < 10:
        #     print(f"Skipping {group} as not enough events") 
        #     continue

        bkg_hists.append(h)
        bkg_errs.append(err)
        bkg_label.append(group)
        bkg_colours.append(getGroupColour(group))


    bkg_hists = np.array(bkg_hists)
    bkg_errs = np.array(bkg_errs)
    tot_bkg = np.sum(bkg_hists, axis=0)
    tot_bkg_err = np.sqrt(np.sum(bkg_errs, axis=0))

    # I want to reorder in terms of number of events in background process
    new_idxs = np.argsort(np.sum(bkg_hists, axis=1))

    print(f"Colours before re-ordering:")
    for proc, col in zip(bkg_label, bkg_colours):
        print(f"{proc}: {col}")

    bkg_hists = bkg_hists[new_idxs]
    bkg_errs = bkg_errs[new_idxs]
    bkg_label = np.array(bkg_label)[new_idxs]
    bkg_colours = np.array(bkg_colours)[new_idxs]


    print(f"Colours after re-ordering:")
    for proc, col in zip(bkg_label, bkg_colours):
        print(f"{proc}: {col}")


    # Now get the signal hist
    sig_h, _ = np.histogram(sig_to_plot[f'pnn_output_mH{mH}_mA{mA}'], bins = bins, weights = sig_to_plot.weight_nominal)

    return sig_h, bkg_hists, bkg_errs, bkg_label, bkg_colours, tot_bkg, tot_bkg_err


def plot(mH, mA, sig_h, bkg_hists, bkg_errs, bkg_label, bkg_colours, tot_bkg, tot_bkg_err, bins, ecom, lumi, direc, save_name):
    # Now plot 
    plt.style.use(hep.style.CMS)
    fig, ax = plt.subplots(figsize=(13,11))
    plt.style.use(hep.style.CMS)

    # Get total background and normalise myself
    hep.histplot(bkg_hists, bins=bins, ax=ax, stack=True, label=bkg_label, 
                    alpha=1, histtype='fill', color=bkg_colours, edgecolor = 'black', linewidth=1.2)



    # Now plot the background error

    ax.fill_between(bins, np.append(tot_bkg + tot_bkg_err, 0), np.append(tot_bkg-tot_bkg_err, 0), alpha=1,
                    step='post', hatch="\\\\\\\\", facecolor='none', label='Stat. Uncert.', edgecolor="black", linewidth=0.0)


    hep.histplot(sig_h, bins=bins, ax=ax, label=f'IDM $M_H={mH}$ GeV,\n        $M_A={mA}$ GeV', 
                    histtype="step", color='black', linestyle='--', linewidth=2)


    eq1 = ("IDM:\n"
            r"$M_{H^\pm} = M_A$" + "\n"
            r"$\lambda_{345} = 1e\text{-}6$")
    # put a light box around the text
    plt.text(0.1, 0.85, eq1, fontsize="27",
                transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.5))



    ax.legend(fontsize="27", ncol=2, loc='upper right')
    ax.set_yscale('log')
    ax.set_ylabel("Events", fontsize="36")
    ax.set_xlabel("PNN Output", fontsize="36")

    ymin, ymax = ax.get_ylim()
    print(f"Current y limits: {ymin}, {ymax}")
    ax.set_ylim(1, ymax * 10)
    ax.set_xlim(np.min(bins), np.max(bins))

    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)

    # plt.tight_layout()
    if mA - mH <= 30:
        process_latex = r"$\mu^-\mu^+$"
    else:
        process_latex = r"$e^-e^+/\mu^-\mu^+$"

    plt.text(0, 1.013, process_latex,
                transform=ax.transAxes, fontsize='33')


    plt.text(1, 1.013, r"FCC-ee,  $\sqrt{s}$" + f"={ecom} GeV,  " + f"Lumi={lumi}" + "$ab^{-1}$", fontsize="33",
                transform=ax.transAxes, horizontalalignment='right')
    save_loc = f"{direc}/plots/{save_name}/mH{mH}_mA{mA}_{save_name}.pdf"
    os.makedirs(os.path.dirname(save_loc), exist_ok=True)
    plt.savefig(save_loc, bbox_inches='tight')

# Save the command line argument to the same location {direc}/plots/{save_name}/command_line.txt
command_save_loc = f"{train_direc}/plots/{save_name}/command_line.txt"
os.makedirs(os.path.dirname(command_save_loc), exist_ok=True)
with open(command_save_loc, 'w') as f:
    f.write(input_command_line)

for mH, mA in zip(mHs, mAs):
    sig_h, bkg_hists, bkg_errs, bkg_label, bkg_colours, tot_bkg, tot_bkg_err = getHistograms(signals, bkg, mH, mA, bins)
    plot(mH, mA, sig_h, bkg_hists, bkg_errs, bkg_label, bkg_colours, tot_bkg, tot_bkg_err, bins, ecom, lumi, train_direc, save_name)