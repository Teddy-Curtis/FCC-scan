import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import os
import pandas as pd
import copy
from sklearn.model_selection import StratifiedKFold
import glob
from collections.abc import MutableMapping
import importlib
import matplotlib.pyplot as plt
import math
import json
import awkward as ak 

from PIL import Image as pil
from pkg_resources import parse_version

if parse_version(pil.__version__) >= parse_version("10.0.0"):
    pil.ANTIALIAS = pil.LANCZOS


def getRunLoc(directory, prefix=False):
    # Get a directory for the run: There might have been previous
    # runs in the same directory, if so we don't want to overwite
    # -> automatically +1 to end of the run name
    direcs = glob.glob(f"{directory}/*/")
    runs = sorted(
        [int(direc.split("/")[-2].split("run")[1]) for direc in direcs]
    )
    if len(runs) == 0:  # if no runs yet
        run_name = f"run1"
    else:  # if there are runs, add 1 to end for new run name
        n = runs[-1] + 1
        run_name = f"run{n}"

    if prefix:
        run_name = f"{prefix}_{run_name}"

    run_loc = f"{directory}/{run_name}"
    print(f"======= Run directory: =======")
    print(run_loc)
    print(f"==============================")
    # Make the run_loc
    os.makedirs(run_loc, exist_ok=True)

    return run_loc


def flatten_dict(
    d: MutableMapping, parent_key: str = "", sep: str = "."
) -> MutableMapping:
    """
    Function that flattens dict e.g.:
    my_dict = {
            'a' : '12',
            'b' : {
                'one' : 0.5,
                'two' : 20,
            }}
    goes to:
    my_dict = {'a': '12', 'b.one': 0.5, 'b.two': 20}
    Done recursively.
    """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


class WeightedBCEWithLogitsLoss(torch.nn.Module):
    def __init__(self, **kwargs):
        super(WeightedBCEWithLogitsLoss, self).__init__()
        self.base_loss_fn = torch.nn.BCEWithLogitsLoss(
            reduction="none", **kwargs
        )

    def forward(self, inputs, targets, weights):
        # Check that weights is shape (len(weights), 1)
        # otherwise tensor multiplication is wrong!
        assert weights.size() == (
            weights.size()[0],
            1,
        ), "Shape of weights must be : (len(weights), 1)"

        per_sample_loss = self.base_loss_fn(inputs, targets)
        reweighted_loss = weights * per_sample_loss
        loss = reweighted_loss.mean()

        return loss


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float("inf")

    def is_early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


class MyWriter:
    def __init__(self, save_loc):
        self.save_loc = save_loc
        self.all_stats = {}

    def addModelGraph(self, model, data, device):
        # Get samples to pass in
        x, y, masses, w, wc = data[:2]
        x = torch.Tensor(x).to(device)
        masses = torch.Tensor(masses).to(device)

    def writeStats(self, train_stats, test_stats, lr, epoch):
        # Write all the stats to the writer
        # Note stats given as a dict -> loop over all keys
        stat_names = list(train_stats.keys())
        for stat in stat_names:

            # Also append to the stats dict (or add if it doesn't exist)
            if not stat in self.all_stats.keys():
                self.all_stats[stat] = {
                    "train": [train_stats[stat]],
                    "test": [test_stats[stat]],
                }
            else:
                self.all_stats[stat]["train"].append(train_stats[stat])
                self.all_stats[stat]["test"].append(test_stats[stat])

        if not "lr" in self.all_stats.keys():
            self.all_stats['lr'] = [lr]
        else:
            self.all_stats['lr'].append(lr)
        
        # And save the dict stats as a json to the same file
        with open(f"{self.save_loc}/stats.json", "w") as f:
            json.dump(self.all_stats, f, indent=4)

    def writeHyperparams(self, hparams, train_stats, test_stats):
        # flattemn hyperparam dict in case it contains sub dicts (e.g. scheduler
        # settings)
        hparams = flatten_dict(hparams)
        # Save the hparams
        metric_dict = {}
        # Add all the best metrics
        for stat in list(train_stats.keys()):
            metric_dict[f"best_{stat}_train"] = train_stats[stat]
            metric_dict[f"best_{stat}_test"] = test_stats[stat]

    def addWeightHistograms(self, model, epoch):
        # Get histogram of the weights for ONLY the linear layers
        # -> Might be able to include other layers later
        # num_layers = 0
        # for layer_number in range(len(model.fc_process)):
        #     # Get layer
        #     layer = model.fc_process[layer_number]

        #     if isinstance(layer, torch.nn.Linear):
        #         self.getLinearLayerHist(layer, epoch, layer_number)
        #         num_layers += 1

        #     # Also might just be a sequential
        #     elif isinstance(layer, torch.nn.Sequential):
        #         for l in layer:
        #             if isinstance(l, torch.nn.Linear):
        #                 self.getLinearLayerHist(l, epoch, layer_number)
        #                 num_layers += 1

        # # Now do for final output layer
        # layer = model.output_mlp_linear
        # self.getLinearLayerHist(layer, epoch, num_layers)
        for name, layer in model.named_modules():
            # print(i)
            # print(name)
            # print(type(layer))
            # if isinstance(layer, torch.nn.Linear):
            #     flattened_weights = layer.weight.flatten()
            #     print(len(flattened_weights))
            if hasattr(layer, "weight"):
                # layer_num = name.split(".")[1].split(".")[0]
                # print(f"layer_num = {layer_num}")
                flattened_weights = layer.weight.flatten()

            if hasattr(layer, "bias"):
                flattened_weights = layer.bias.flatten()

    def getLinearLayerHist(self, layer, epoch, layer_number):
        flattened_weights = layer.weight.flatten()
        tag = f"layer_weights_{layer_number}"
        bias = layer.bias
        tag = f"layer_bias_{layer_number}"

    def plotOutputDistribution(
        self,
        train_prob,
        train_label,
        train_weights,
        test_prob,
        test_label,
        test_weights,
        epoch,
    ):

        test_sig = test_prob[test_label == 1]
        test_bkg = test_prob[test_label == 0]

        fig = plt.figure()
        bins = np.linspace(0, 1, 100)
        _ = plt.hist(
            test_sig,
            bins=bins,
            label="Test Signal",
            density=True,
            weights=test_weights[test_label == 1],
            histtype="step",
        )
        _ = plt.hist(
            test_bkg,
            bins=bins,
            label="Test Background",
            density=True,
            weights=test_weights[test_label == 0],
            histtype="step",
        )

        train_sig = train_prob[train_label == 1]
        train_bkg = train_prob[train_label == 0]
        _ = plt.hist(
            train_sig,
            bins=bins,
            label="Train Signal",
            density=True,
            linestyle="--",
            weights=train_weights[train_label == 1],
            histtype="step",
            alpha=0.6,
        )
        _ = plt.hist(
            train_bkg,
            bins=bins,
            label="Train Background",
            density=True,
            linestyle="--",
            weights=train_weights[train_label == 0],
            histtype="step",
            alpha=0.6,
        )

        plt.legend()
        plt.yscale("log")
        plt.ylim((0.001, 100))
        plt.title(f"Signal vs Background Distributions")

    def closeWriter(self):
        pass 


def weights_init(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.kaiming_normal_(
            m.weight, mode="fan_out", nonlinearity="relu"
        )
        torch.nn.init.zeros_(m.bias)


# Class that trains the NNs
class trainNN:
    def __init__(self, params, branches, run_loc="runs"):

        self.hparams = params["hyperparams"]
        self.branches = branches
        self.run_loc = run_loc

        self.batch_size = self.hparams["batch_size"]
        self.epochs = self.hparams["epochs"]

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        print(f"Device = {self.device}")

        ########################### model #############################
        # open module model.py that has all the models in
        model_module = importlib.import_module("fcc_study.pNN.training.model")
        # Initiate specific model
        self.model = getattr(
            model_module,
            params["model"],
        )(params["model_params"])

        print(self.model)
        print(
            f"Number of parameters in model = {sum([param.nelement() for param in self.model.parameters()])}"
        )

        # initialise weights
        self.model.apply(weights_init)

        # Send model to device
        self.model.to(self.device)

        ########################### optimizer #############################
        # Select optimizer from torch optim module
        torch_optim_module = importlib.import_module("torch.optim")
        self.optimizer = getattr(
            torch_optim_module,
            self.hparams["optimizer"],
        )(self.model.parameters(), **self.hparams["optimizer_params"])

        ########################### criterion ###############################
        # Select criterion from torch nn module
        torch_nn_module = importlib.import_module("torch.nn")
        if self.hparams["criterion"] == "WeightedBCEWithLogitsLoss":
            # Not in the torch.nn module
            self.criterion = WeightedBCEWithLogitsLoss(
                **self.hparams["criterion_params"]
            )
        else:
            self.criterion = getattr(
                torch_nn_module,
                self.hparams["criterion"],
            )(**self.hparams["criterion_params"])
            # self.criterion = WeightedBCEWithLogitsLoss()

        ########################### early scheduler ##########################
        # Select scheduler from torch optim lr_scheduler module (if scheduler
        # specified)
        if "scheduler" in self.hparams.keys():
            print("Scheduler found")
            torch_lr_schedule_module = importlib.import_module(
                "torch.optim.lr_scheduler"
            )
            self.scheduler = getattr(
                torch_lr_schedule_module,
                self.hparams["scheduler"],
            )(self.optimizer, **self.hparams["scheduler_params"])

            self.scheduler_requires_loss = self.hparams[
                "scheduler_requires_loss"
            ]

        ########################### early stopper #############################
        if "early_stop" in self.hparams.keys():
            print("Early stopper found")
            self.early_stop = EarlyStopper(patience=self.hparams["early_stop"])

        ########################### Additional #############################
        # Create instance of writer class to tensorboard
        self.writer = MyWriter(self.run_loc)
        # make model weight histograms before training
        self.writer.addWeightHistograms(self.model, 0)

    def trainModel(self, train_data, test_data):
        print("Beginning training")

        test_loss_min = float("inf")  # Keep track of min of test_loss
        test_stats_best = {}
        train_stats_best = {}

        # Save model graph to tensorboard
        self.writer.addModelGraph(self.model, train_data, self.device)

        # Loop over and train the model
        for epoch in range(1, self.epochs + 1):
            
            # Shuffle background masses every epoch
            train_data.shuffleMasses()
            test_data.shuffleMasses()

            # Train model
            self.train(train_data)

            # Now test model
            train_stats, train_prob, train_label, train_weights = self.test(
                train_data
            )
            test_stats, test_prob, test_label, test_weights = self.test(
                test_data
            )

            # Check if there's a scheduler
            if hasattr(self, "scheduler"):
                print("Doing step for scheduler")
                if self.scheduler_requires_loss:
                    self.scheduler.step(test_stats["loss"])
                else:
                    self.scheduler.step()

            # Check if need to stop early
            test_loss = test_stats["loss"]
            # Check that there is an early stopper
            if hasattr(self, "early_stop"):
                if self.early_stop.is_early_stop(test_loss):
                    print(f"Stopping early, at epoch {epoch}")
                    break

            # Save model with the lowest loss
            if test_loss < test_loss_min:
                torch.save(self.model.state_dict(), f"{self.run_loc}/model.pt")
                # reset lowest loss
                test_loss_min = test_loss
                train_stats_best = train_stats
                test_stats_best = test_stats

            # make output distribution plot
            self.writer.plotOutputDistribution(
                train_prob,
                train_label,
                train_weights,
                test_prob,
                test_label,
                test_weights,
                epoch,
            )

            # Get the current lr, and save all the stats
            lr = self.optimizer.param_groups[0]['lr']
            print(f"Learning rate end of epoch = {lr}")
            # Print stats out
            self.printStats(train_stats, test_stats, epoch)
            # Save stats to writer
            self.writer.writeStats(train_stats, test_stats, lr, epoch)
            # Save model weights to writer
            self.writer.addWeightHistograms(self.model, epoch)

        print("Finished training")
        print("Now deleting last model and loading best model into GPU memory")
        self.model.load_state_dict(torch.load(f"{self.run_loc}/model.pt"))
        print("Done.")

        # Write final score and  hparams
        self.writer.writeHyperparams(
            self.hparams, train_stats_best, test_stats_best
        )
        # Close the writer
        self.writer.closeWriter()

    def train(self, train_data):
        self.model.train()
        print("Training:")
        loader = DataLoader(
            train_data,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=8,
        )
        # loader = self.batchDataStratified(train_data)

        # Iterate in batches over the training dataset.
        for x, y, masses, w, wc in tqdm(loader):
            self.optimizer.zero_grad()  # Clear gradients.
            self.model.zero_grad()
            # Transform the data
            x = x.to(self.device)
            y = y.to(self.device)
            masses = masses.to(self.device)
            w = w.to(self.device)

            out = self.model(x, masses)  # Perform a single forward pass.

            loss = self.criterion(out, y, w)  # Compute the loss.
            loss.backward()  # Derive gradients.
            self.optimizer.step()  # Update parameters based on gradients.

    def test(self, data):
        print("Testing:")
        self.model.eval()
        correct = 0
        running_loss = 0.0
        loader = DataLoader(
            data,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=8,
        )
        probability, labels, weights = [], [], []

        for x, y, masses, w, wc in tqdm(
            loader
        ):  # Iterate in batches over the training/test dataset.
            labels.append(y)
            weights.append(wc)
            # Transform the data
            x = x.to(self.device)
            y = y.to(self.device)
            masses = masses.to(self.device)
            w = w.to(self.device)
            # print(f"weights = {w}")
            # print(f"Number of samples = {len(w)}")
            # print(f"Sum of weights = {torch.sum(w)}")

            out = self.model(x, masses)
            prob = torch.sigmoid(out)
            probability.append(prob.cpu().detach().numpy())

            pred = torch.round(prob)
            correct += int(
                (pred == y).sum()
            )  # Check against ground-truth labels.

            loss = self.criterion(out, y, w)  # Compute the loss.
            num_samps_in_batch = int(len(y) + 1)
            # print(f"loss.item() = {loss.item()}")
            running_loss += loss.item() * num_samps_in_batch
            # print(f"Running_loss = {running_loss}")
            # break

        probability = np.concatenate(probability)
        labels = np.concatenate(labels)
        weights = np.concatenate(weights)

        acc = correct / len(data)  # Derive ratio of correct predictions.
        loss = running_loss / len(data)
        stats = {"acc": acc, "loss": loss}
        return stats, probability, labels, weights

    # def getAUC(self, data):
    #     fpr_train, tpr_train, thresholds = roc_curve(data['class'],
    #                                                  data[f"prob_{mass}"],
    #                                                  sample_weight=data['weight_central'])
    #     auc = integrate.trapezoid(y=tpr_train, x=fpr_train)

    def printStats(self, train_stats, test_stats, epoch):
        for stat in list(train_stats.keys()):
            print(
                f"Epoch: {epoch:03d}, Train {stat}: {train_stats[stat]:.7f}, Test {stat}: {test_stats[stat]:.7f}"
            )

    # def batchDataStratified(self, data):
    #     # Find number of splits associated roughly with batch_size

    #     skf = StratifiedKFold(n_splits=2)

    #     loader = DataLoader(data, batch_size=self.batch_size,
    #                       shuffle=True,pin_memory=True, num_workers = 8)

    #     for x, y, masses, w, wc in loader:
    #         yield x, y, masses, w, wc

    def batchDataStratified(self, data):
        # Every batch has same percentage of each process as the whole dataset
        num_samps = len(data)
        n_splits = math.ceil(num_samps / self.batch_size)
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True)

        for _, batch_idxs in skf.split(
            data.features, data.default_data.specific_proc
        ):
            x, y, mass, w, wc = data[batch_idxs]
            x = torch.tensor(x)
            y = torch.tensor(y)
            mass = torch.tensor(mass)
            w = torch.tensor(w)
            wc = torch.tensor(wc)
            yield x, y, mass, w, wc

    def getProbs(self, data):
        # Input the data, and get the probs out
        print("Finding NN output probabilities")
        probabilities = []
        loader = DataLoader(
            data,
            batch_size=10000,
            shuffle=False,
            pin_memory=True,
            num_workers=8,
        )
        self.model.eval()
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
            dataset.setAllMasses(mass)
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

            masses_strings.append(f"pnn_output_mH{ms}_mA{md}")

        print(f"Masses {masses} converted to {masses_strings}")
        return masses_strings

    def getPRcurvesForEachMass(self, train_data, test_data):
        # Every 5th epoch, find the PR curve for each mass and add to the
        # writer
        train_all_probs = self.getProbsForEachMass(train_data)
        test_all_probs = self.getProbsForEachMass(test_data)
        # Now I want to
        return False
