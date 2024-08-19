import glob
import awkward as ak
import numpy as np
import json
from fcc_study.pNN.training.train import getRunLoc
from fcc_study.pNN.training.preprocessing_datasetClasses import getData, consistentTrainTestSplit
from fcc_study.pNN.training.preprocessing_datasetClasses import normaliseWeights, scaleFeatures, CustomDataset
from fcc_study.pNN.training.train import trainNN
import copy
import matplotlib.pyplot as plt
import mplhep as hep
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

######################## Define Hyperparams and Model #########################
base_run_dir = "runs/fcc_scan"
run_loc = getRunLoc(base_run_dir)

samples = {
    "backgrounds" : {
    "p8_ee_ZZ_ecm240": {
        "files" : ['/vols/cms/emc21/fccStudy/data/p8_ee_ZZ_ecm240.root'], 
        "xs": 1.35899
    },
    "wzp6_ee_eeH_ecm240": {
        "files" : ["/vols/cms/emc21/fccStudy/data/wzp6_ee_eeH_ecm240.root"],
        "xs": 0.0071611
    },
    "wzp6_ee_mumuH_ecm240": {
        "files" : ["/vols/cms/emc21/fccStudy/data/wzp6_ee_mumuH_ecm240.root"],
        "xs": 0.0067643
    },
    "wzp6_ee_nunuH_ecm240": {
        "files" : ["/vols/cms/emc21/fccStudy/data/wzp6_ee_nunuH_ecm240.root"],
        "xs": 0.046191
    },
    "wzp6_ee_tautauH_ecm240": {
        "files" : ["/vols/cms/emc21/fccStudy/data/wzp6_ee_tautauH_ecm240.root"],
        "xs": 0.0067518
    },
    "wzp6_ee_qqH_ecm240": {
        "files" : ["/vols/cms/emc21/fccStudy/data/wzp6_ee_qqH_ecm240.root"],
        "xs": 0.13635
    },
    "wzp6_ee_ee_Mee_30_150_ecm240": {
        "files" : ["/vols/cms/emc21/fccStudy/data/wzp6_ee_ee_Mee_30_150_ecm240.root"],
        "xs": 8.305
    },
    "wzp6_ee_mumu_ecm240": {
        "files" : ["/vols/cms/emc21/fccStudy/data/wzp6_ee_mumu_ecm240.root"],
        "xs": 5.288
    },
    "wzp6_ee_tautau_ecm240": {
        "files" : ["/vols/cms/emc21/fccStudy/data/wzp6_ee_tautau_ecm240.root"],
        "xs": 4.668
    }
    },
    "signal" : {
        "BP1" : {
            "files" : ["/vols/cms/emc21/fccStudy/data/e240_bp1_h2h2ll.root", "/vols/cms/emc21/fccStudy/data/e240_bp1_h2h2llvv.root"],
            "masses" : [80, 150],
            "xs": 0.0069
        },
        "BP2" : {
            "files" : ["/vols/cms/emc21/fccStudy/data/e240_bp2_h2h2ll.root", "/vols/cms/emc21/fccStudy/data/e240_bp2_h2h2llvv.root"],
            "masses" : [80, 160],
            "xs": 0.005895
        },
    },
    "Luminosity" : 500,
    "test_size" : 0.25 # e.g. 0.2 means 20% of data used for test set
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
 'n_photons',
 'n_muons',
 'n_electrons']

# Save the branches used for the run
with open(f"{run_loc}/branches.json", "w") as f: 
    json.dump(branches, f, indent=4)


params = {
        'hyperparams' : { 
            'epochs' : 2, 
            'early_stop' : 40,
            'batch_size': 500,
            'optimizer' : 'Adam', 
            'optimizer_params' : {
                'lr': 0.00001
            },
            'criterion' : 'WeightedBCEWithLogitsLoss',
            'criterion_params' : {
            },
            'scheduler' : 'ReduceLROnPlateau', 
            'scheduler_params' : {
                "patience" : 10, 
                'factor' : 0.1,
                'verbose' : True
            },
            "scheduler_requires_loss" : True
        },
        'model' : 'MLPRelu',
        'model_params' : {
            'input_features' : len(branches) + 2,
            'fc_params' : [(0.0, 300), (0.0, 300), (0.0, 300)],
            'output_classes' : 1,
            'num_masses' : 2,
        }
    }

print("All parameters: \n", json.dumps(params, indent=4))
# Save all the parameters to a json file
with open(f"{run_loc}/params.json", "w") as f: 
    json.dump(params, f, indent=4)


######################### Preprocessing #################################

def applyCuts(evs):
    mask = (
        (np.abs(evs.Zcand_pz) < 70)
        & (evs.jet1_e < 0)
        & (evs.n_photons == 0)
        & (evs.MET_pt > 5)
        & (evs.lep1_pt < 80)
        & (evs.lep2_pt < 60)
        & (evs.Zcand_povere > 0.1)
    )
    mask = ak.flatten(mask)
    sum_before = ak.sum(evs.weight)
    sum_after = ak.sum(evs[mask].weight)
    print(
        f"Sum before: {sum_before}, Sum after: {sum_after}, Fraction: {sum_after/sum_before}"
    )
    return evs[mask]


data = getData(samples, applyCuts)


# Now split into train and test
train_data, test_data = consistentTrainTestSplit(data, 
                                                 test_size=samples['test_size'])


# Now normalise weights so that the signal points sum to the same weight
# e.g. each BP sum to 1, then reweight backgrounds so that 
# sum(backgrounds) = sum(signal)
train_data = normaliseWeights(train_data)
test_data = normaliseWeights(test_data)

# Now scale the features so that intput features have mean 0 and std 1
# and that the masses are scaled to be between 0 and 1
train_data, test_data, feat_scaler, mass_scaler = scaleFeatures(train_data, 
                                                                test_data, 
                                                                branches,
                                                                run_loc)


# # Now put these both into helper classes for training 
train_dataset = CustomDataset(train_data, branches, feat_scaler, mass_scaler)
test_dataset = CustomDataset(test_data, branches, feat_scaler, mass_scaler)
train_dataset.shuffleMasses()
test_dataset.shuffleMasses()


######################### Training #################################

trainer = trainNN(params, branches, run_loc)


trainer.trainModel(train_dataset, test_dataset)


train_data = trainer.getProbsForEachMass(train_dataset, samples)
test_data = trainer.getProbsForEachMass(test_dataset, samples)

# Now save these
ak.to_parquet(copy.deepcopy(train_data), f"{run_loc}/train_data.parquet")
ak.to_parquet(copy.deepcopy(test_data), f"{run_loc}/test_data.parquet")

print("Done!")

# Now do some extra plotting
# Get list of all the signal names
sig_procs = np.unique(list(test_data[test_data['class'] == 1].process))
# Get list of all the background names
bkg_procs = np.unique(list(test_data[test_data['class'] == 0].process))

# Define the bins for the histogram
bins = np.linspace(0, 1, 50)
# Loop over all the signal processes and plot signal versus background
for sig_proc in sig_procs:
    plt.close()
    
    bkg_hists = []
    bkg_weights = []
    for bkg_proc in bkg_procs:
        # Get each background process
        bkg = test_data[test_data['process'] == bkg_proc]
        # Get the histogram for it
        bkg_hist = np.histogram(ak.flatten(bkg[f'pnn_output_bp{sig_proc}']), bins=bins, weights = bkg['weight'])[0]
        # append to background list
        bkg_hists.append(bkg_hist)
    
    # Plot all backgrounds, stacked on top of each other
    _ = hep.histplot(bkg_hists, bins=bins, histtype='fill', label=bkg_procs, stack=True)

    # Now get the signal and plot that on top
    signal = test_data[test_data['process'] == sig_proc]
    _ = plt.hist(signal[f'pnn_output_bp{sig_proc}'], bins=bins, histtype='step', 
                 label=sig_proc, weights = signal['weight'], color='black',
                 linestyle='--')

    plt.xlabel('PNN output')
    plt.ylabel('Events')
    plt.title(f'PNN output for {sig_proc} vs background')

    plt.legend()
    plt.yscale('log')
    plt.savefig(f'pnn_output_bp{sig_proc}.png')
    plt.show()