import glob
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import Dataset
import pickle
from tqdm import tqdm
import uproot
import awkward as ak
import json
from fcc_study.pNN.utils import convertToNumpy
import copy


def flattenFields(evs):
    for field in ak.fields(evs):
        if "var" in str(ak.type(evs[field])):
            evs[field] = ak.flatten(evs[field])
    return evs


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


def getData(samples, cuts=None):

    events = []
    # First load in the signal samples
    for sig_point, sig_dict in samples['signal'].items():
        print(f"Loading signal point: {sig_point}")
        files = sig_dict["file"]
        files = [f"{samples['directory']}/{f}" for f in files]

        for file in tqdm(files):
            print(f"Loading file: {file.split('/')[-1]}")
            with uproot.open(file, num_workers=16) as f:
                tree = f["events"]
                branches_to_load = list(tree.keys())
                branches_to_load.remove("n_seljets")
                evs = tree.arrays(branches_to_load, library="ak")

                weight = sig_dict["weight"]
                evs["weight"] = ak.ones_like(evs.Zcand_m) * weight

                if cuts is not None:
                    evs = cuts(evs)

                # I want to add some stuff here as well to make the data more useful
                class_num = ak.ones_like(evs.Zcand_m)
                # Get the BP number
                id_num = int(sig_point.split("BP")[1])
                evs["id_num"] = ak.ones_like(evs.Zcand_m) * id_num
                # Also get the masses
                mH = sig_dict["masses"][0]
                mA = sig_dict["masses"][1]
                evs["mH"] = ak.ones_like(evs.Zcand_m) * mH
                evs["mA"] = ak.ones_like(evs.Zcand_m) * mA

                process = sig_point
                specific_proc = file.split("_")[-1].split(".root")[0]

                evs["process"] = [process] * len(evs.Zcand_m)
                evs["specific_proc"] = [specific_proc] * len(evs.Zcand_m)
                evs["class"] = class_num

                # flatten all fields
                evs = flattenFields(evs)
                # Convert all to float32
                evs = ak.values_astype(evs, "float32")

                events.append(evs)

    # Now load in the background samples
    for proc, proc_dict in samples['background'].items():
        print(f"Loading background process: {proc}")
        files = proc_dict["file"]
        files = [f"{samples['directory']}/{f}" for f in files]

        for file in tqdm(files):
            print(f"Loading file: {file.split('/')[-1]}")
            with uproot.open(file, num_workers=16) as f:
                tree = f["events"]
                branches_to_load = list(tree.keys())
                branches_to_load.remove("n_seljets")
                evs = tree.arrays(branches_to_load, library="ak")

                weight = proc_dict["weight"]
                evs["weight"] = ak.ones_like(evs.Zcand_m) * weight

                if cuts is not None:
                    evs = cuts(evs)

                # I want to add some stuff here as well to make the data more useful
                class_num = ak.zeros_like(evs.Zcand_m)

                evs["mH"] = [-1] * len(evs)
                evs["mA"] = [-1] * len(evs)
                evs["id_num"] = [-1] * len(evs)

                evs["process"] = [proc] * len(evs.Zcand_m)
                evs["specific_proc"] = [proc] * len(evs.Zcand_m)
                evs["class"] = class_num

                # flatten all fields
                evs = flattenFields(evs)
                # Convert all to float32
                evs = ak.values_astype(evs, "float32")

                events.append(evs)
    
    events = ak.concatenate(events)
    return events


def consistentTrainTestSplit(
    events, test_size=0.2, random_state=42, stratify_var="process"
):

    print("Doing train test split")
    unique_strat_vars = list(np.unique(events[stratify_var]))
    print(unique_strat_vars)

    # Split using indexes as it is much faster
    indexes = np.arange(len(events))

    train_data_idxs, test_data_idxs = train_test_split(
        indexes,
        test_size=test_size,
        random_state=random_state,
        stratify=events[stratify_var],
    )

    train_data = events[train_data_idxs]
    test_data = events[test_data_idxs]

    train_data["weight_scaled"] = train_data["weight"] / (1 - test_size)
    test_data["weight_scaled"] = test_data["weight"] / test_size

    return train_data, test_data


def normaliseWeightsEqualProc(events):
    print(f"Normalising weights:")
    sig_dat = events[events["class"] == 1]
    signal_ids = np.unique(sig_dat["id_num"])

    # Want sum of weights of each signal to equal 1, then I will reweight
    # both such that the average weight = 0.001
    for id in signal_ids:

        proc = sig_dat[sig_dat["id_num"] == id]
        sum_weight = np.sum(proc["weight"])

        events["weight"] = ak.where(
            events["id_num"] == id,
            events["weight"] / sum_weight,
            events["weight"],
        )

    # now reweight so that the average weight of signal = 0.001

    mean_sig = np.mean(events[events["class"] == 1]["weight"])
    ratio = 0.001 / mean_sig
    events["weight"] = events["weight"] * ratio

    # Now do the same for the background
    # First get the sum of the signal
    sum_sig = np.sum(events[events["class"] == 1]["weight"])
    # Now find bkgrnds
    bkg_dat = events[events["class"] == 0]
    bkg_procs = np.unique(bkg_dat["process"])
    num_bkg_groups = len(bkg_procs)
    # This is the weight of each individual background process
    indi_group_sum = sum_sig / num_bkg_groups

    for proc in bkg_procs:

        proc_dat = bkg_dat[bkg_dat["process"] == proc]
        sum_bkg = np.sum(proc_dat["weight"])

        events["weight"] = ak.where(
            events["process"] == proc,
            events["weight"] * (indi_group_sum / sum_bkg),
            events["weight"],
        )

    sig = events[events["class"] == 1]
    bkg = events[events["class"] == 0]
    print(
        f"Sum sig = {np.floor(np.sum(sig['weight']))}, sum bkg = {np.floor(np.sum(bkg['weight']))}"
    )

    return events



def normaliseWeightsEqualClass(events):
    print(f"Normalising weights:")
    sig_dat = events[events["class"] == 1]
    signal_ids = np.unique(sig_dat["id_num"])

    # Want sum of weights of each signal to equal 1, then I will reweight
    # both such that the average weight = 0.001
    for id in signal_ids:

        proc = sig_dat[sig_dat["id_num"] == id]
        sum_weight = np.sum(proc["weight"])

        events["weight"] = ak.where(
            events["id_num"] == id,
            events["weight"] / sum_weight,
            events["weight"]
        )

    # now reweight so that the average weight of signal = 0.001

    mean_sig = np.mean(events[events["class"] == 1]["weight"])
    ratio = 0.001 / mean_sig
    events["weight"] = events["weight"] * ratio

    # Now do the same for the background
    # First get the sum of the signal
    sum_sig = np.sum(events[events["class"] == 1]["weight"])
    # reweight the backgrounds so sum(bkg) = sum(sig)
    sum_bkg = np.sum(events[events["class"] == 0]["weight"])

    events["weight"] = ak.where(
        events["class"] == 0,
        events["weight"] * (sum_sig / sum_bkg),
        events["weight"]
    )


    sig = events[events["class"] == 1]
    bkg = events[events["class"] == 0]
    print(
        f"Sum sig = {np.floor(np.sum(sig['weight']))}, sum bkg = {np.floor(np.sum(bkg['weight']))}"
    )

    return events


def normaliseWeights(events):
    # Normalise so the signal process sum to the same weight
    # Start by making all signal samples to sum to 1
    sig_dat = events[events["class"] == 1]
    signal_procs = np.unique(sig_dat["process"])

    # Want sum of weights of each signal to equal 1, then I will reweight
    # both such that the average weight = 0.001
    for proc in signal_procs:
        print(proc)
        proc_data = events[events["process"] == proc]
        sum_weight = np.sum(proc_data["weight"])

        events["weight"] = ak.where(
            events["process"] == proc,
            events["weight"] / sum_weight,
            events["weight"],
        )

    # now reweight so that the average weight of signal = 0.001
    mean_sig = np.mean(events[events["class"] == 1]["weight"])
    ratio = 0.001 / mean_sig
    events["weight"] = events["weight"] * ratio

    # Now get the sum of signal and reweight background to that
    sum_sig = np.sum(events[events["class"] == 1]["weight"])
    sum_bkg = np.sum(events[events["class"] == 0]["weight"])
    events["weight"] = ak.where(
        events["class"] == 0,
        events["weight"] * (sum_sig / sum_bkg),
        events["weight"],
    )

    # Now check that the sum of signal and background are the same
    sig = events[events["class"] == 1]
    bkg = events[events["class"] == 0]
    sum_sig = np.floor(np.sum(sig['weight']))
    sum_bkg = np.floor(np.sum(bkg['weight']))
    print(f"Sum sig = {sum_sig}, sum bkg = {sum_bkg}")

    if abs(sum_sig - sum_bkg) > 2:
        print("Sum sig does not equal sum background!")

    return events


def getGroup(process_array, group):
    # Want to get the mask of all the processes that are in the same group
    mask = np.zeros(len(process_array), dtype=bool)
    if isinstance(group, list):
        for p in group:
            mask = mask | (process_array == p)
    else:
        mask = (process_array == group)

    return mask


def normaliseWeightsEqualProcGroup(events, bkg_groups):
    print(f"Normalising weights:")

    sig_dat = events[events["class"] == 1]
    signal_procs = np.unique(sig_dat["process"])

    # Want sum of weights of each signal to equal 1, then I will reweight
    # both such that the average weight = 0.001
    for proc in signal_procs:
        print(proc)
        proc_data = events[events["process"] == proc]
        sum_weight = np.sum(proc_data["weight"])

        events["weight"] = ak.where(
            events["process"] == proc,
            events["weight"] / sum_weight,
            events["weight"],
        )

    # now reweight so that the average weight of signal = 0.001
    mean_sig = np.mean(events[events["class"] == 1]["weight"])
    ratio = 0.001 / mean_sig
    events["weight"] = events["weight"] * ratio

    # Now do the same for the background
    # First get the sum of the signal
    sum_sig = np.sum(events[events["class"] == 1]["weight"])
    
    num_bkg_groups = len(bkg_groups)
    group_weight = sum_sig / num_bkg_groups
    # Now find bkgrnds
    # This is to check that all the processes are in the bkg_groups
    mask = np.zeros(len(events), dtype=bool)
    for group in bkg_groups:
        print(group)
        group_mask = getGroup(events["process"], group)
        mask = mask | group_mask 
        group_data = events[group_mask]
        sum_group_weights = np.sum(group_data["weight"])
        events["weight"] = ak.where(
            group_mask,
            events["weight"] * (group_weight / sum_group_weights),
            events["weight"]
        )
    
    if not ak.all(mask | (events["class"] == 1)):
        print(f"Some processes weren't normalised! This means they were not in bkg_groups")
    
    sig = events[events["class"] == 1]
    bkg = events[events["class"] == 0]
    sum_sig = np.floor(np.sum(sig['weight']))
    sum_bkg = np.floor(np.sum(bkg['weight']))
    print(
        f"Sum sig = {sum_sig}, sum bkg = {sum_bkg}"
    )
    if abs(sum_sig - sum_bkg) > 2:
        print("Sum sig does not equal sum background!")


    return events


def applyScaler(evs, scaler, branches):
    numpy_data = convertToNumpy(evs, branches)
    trans_numpy_data = scaler.transform(numpy_data)
    for i, br in enumerate(branches):
        evs[br] = trans_numpy_data[:, i]

    return evs


def scaleFeatures(train_data, test_data, branches, run_loc="."):
    print(f"Scaling features")
    scaler = StandardScaler()
    numpy_data = convertToNumpy(train_data, branches)
    scaler.fit(numpy_data)
    del numpy_data

    train_data = applyScaler(train_data, scaler, branches)
    test_data = applyScaler(test_data, scaler, branches)

    # Also scale the masses using the minmax scaler
    # First select just the signal
    sig_data = train_data[train_data["class"] == 1]
    mass_branches = ["massSum", "massDiff"]
    mass_scaler = MinMaxScaler()
    numpy_data = convertToNumpy(sig_data, mass_branches)
    mass_scaler.fit(numpy_data)

    del numpy_data

    train_data = applyScaler(train_data, mass_scaler, mass_branches)
    test_data = applyScaler(test_data, mass_scaler, mass_branches)

    # Now save them both
    pickle.dump(scaler, open(f"{run_loc}/scaler.pkl", "wb"))
    pickle.dump(mass_scaler, open(f"{run_loc}/mass_scaler.pkl", "wb"))

    return train_data, test_data, scaler, mass_scaler


class CustomDataset(Dataset):
    def __init__(
        self,
        data,
        train_branches,
        feat_scaler,
        mass_scaler,
        mass_branches=["massSum", "massDiff"],
    ):
        print("Initialising CustomDataset")
        self.train_branches = train_branches
        self.data = data

        # Get the features and masses
        self.features = convertToNumpy(data, train_branches)
        self.masses = convertToNumpy(data, mass_branches)
        labs = convertToNumpy(data, ["class"])
        self.labels = np.reshape(labs, (len(labs), 1))
        weights = convertToNumpy(data, ["weight"])
        self.weight = np.reshape(weights, (len(weights), 1))

        if not "weight_scaled" in ak.fields(data):
            data["weight_scaled"] = data["weight"]

        weights_scaled = convertToNumpy(data, ["weight_scaled"])
        self.weight_scaled = np.reshape(weights_scaled, (len(weights_scaled), 1))

        # Now find the unique masses
        # Pick only the signal masses, not the default ones for the backgrounds
        sig_masses = self.masses[labs == 1]
        self.unique_masses = np.unique(sig_masses, axis=0)

        # Also save the scalers so that we can convert between the real data
        # and the converted data
        self.feat_scaler = feat_scaler
        self.mass_scaler = mass_scaler

    def shuffleMasses(self):
        # Want to find unique groupings of the mH,mA,mHch
        # Pick masses randomly from unique_masses
        choices = np.random.choice(
            np.arange(len(self.unique_masses)), len(self.labels)
        )

        shuffled_masses = self.unique_masses[choices]

        # Now mask so that the original signal masses are not changed
        sig_mask = self.labels == 1
        new_masses = np.where(sig_mask, self.masses, shuffled_masses)

        self.masses = new_masses

    def setAllMasses(self, masses_to_set):
        # Set all backgrounds to the same mass
        # e.g. masses_to_set = [80, 100, 160]
        # Expand out to same size of data array
        specific_masses = np.array([masses_to_set] * len(self.labels))

        # Now only apply the background samples
        sig_mask = self.labels == 1
        new_masses = np.where(sig_mask, self.masses, specific_masses)

        self.masses = new_masses

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.features[idx]
        y = self.labels[idx]
        mass = self.masses[idx]
        w = self.weight[idx]
        wc = self.weight_scaled[idx]
        # self.features = convertToNumpy(self.data[idx], self.train_branches)
        # self.masses = convertToNumpy(self.data[idx], self.mass_branches)
        # self.labels = convertToNumpy(self.data[idx], ['class'])
        # self.weight = convertToNumpy(self.data[idx], ['weight'])

        return x, y, mass, w, wc

    def toOriginalArray(self):
        # Want to change back the self.data features and masses
        features_nump = convertToNumpy(self.data, self.train_branches)
        features_nump = self.feat_scaler.inverse_transform(features_nump)

        mass_nump = convertToNumpy(self.data, self.mass_branches)
        mass_nump = self.mass_scaler.inverse_transform(mass_nump)

        for i, field in enumerate(self.train_branches):
            self.data[field] = features_nump[:, i]

        for i, field in enumerate(self.mass_branches):
            self.data[field] = mass_nump[:, i]

        return self.data
