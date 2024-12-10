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
import boost_histogram as bh
import os

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



########## Loop over the grid and evaulate ###############
#! Link for splines: https://github.com/nucleosynthesis/EFT-Fitter/blob/addgradients/tools/rbf_spline.py 
import sys
import numpy as np
import pandas as pd
import numpy.typing as npt

# -----------------
# Basis functions
# -----------------
class radialGauss():
    def __init__(self) -> None:
        return
    def evaluate(self, 
                 input: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        return np.exp(-input)
    def getDeltaPhi(self, 
                    input: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        return -self.evaluate(input)

class radialMultiQuad():
    def __init__(self) -> None:
        return
    def evaluate(self, 
                 input: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        return np.sqrt(1+input)
    def getDeltaPhi(self, 
                    input: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        return 1/(2*self.evaluate(input))
    
class radialInverseMultiQuad():
    def __init__(self) -> None:
        return
    def evaluate(self, 
                 input: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        return np.divide(1, np.sqrt(1+input))
    def getDeltaPhi(self, 
                    input: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        return -1/(2*np.power(1+input, 3/2))

class radialLinear():
    def __init__(self) -> None:
        return
    def evaluate(self, 
                 input: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        return np.sqrt(input)
    def getDeltaPhi(self, 
                    input: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        return 1/(2*self.evaluate(input))

class radialCubic():
    def __init__(self) -> None:
        return
    def evaluate(self, 
                 input: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        return np.power(input, 3/2)
    def getDeltaPhi(self, 
                    input: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        return 3*np.sqrt(input)/2
    
class radialQuintic():
    def __init__(self) -> None:
        return
    def evaluate(self, 
                 input: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        return np.power(input, 5/2)
    def getDeltaPhi(self, 
                    input: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        return 5*np.power(input, 3/2)/2

class radialThinPlate():
    def __init__(self) -> None:
        return
    def evaluate(self, 
                 input: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        return np.multiply(input, np.log(np.sqrt(input)))
    def getDeltaPhi(self, 
                    input: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        return (np.log(input)+1)/2
# -----------------

class rbf_spline:
    def __init__(self, ndim=1) -> None:
        self._ndim = ndim
        self._initialised = False
        self._radialFuncs = dict(
            [("gaussian", radialGauss),
             ("multiquadric", radialMultiQuad),
             ("inversemultiquadric", radialInverseMultiQuad),
             ("linear", radialLinear),
             ("cubic", radialCubic),
             ("quintic", radialQuintic),
             ("thinplate", radialThinPlate)
            ])

    def _initialise(self, input_data: pd.DataFrame, target: str, 
                    eps: float, rescaleAxis: bool) -> None:
        # Parse args
        self._input_data = input_data
        self._target_col = target
        self._input_pts = input_data.drop(target, axis="columns").to_numpy()
        self._eps = eps  
        self._rescaleAxis = rescaleAxis
        self._parameter_keys = list(input_data.columns)
        self._parameter_keys.remove(target)

        # Check number of basis points
        self._M = len(input_data)
        if self._M < 1 : 
            sys.exit("Error - At least one basis point is required")
        
        # Check dimensions
        if self._ndim!=len(self._parameter_keys): 
            sys.exit(f"Error - initialise given points with more dimensions " +
                     f"({len(self._parameter_keys)}) than ndim ({self._ndim})")

        # Get scalings by axis (column)
        self._axis_pts = np.power(self._M, 1./self._ndim)
        if self._rescaleAxis:
            self._scale = np.divide(self._axis_pts, 
                                    (np.max(self._input_pts, axis=0) -
                                     np.min(self._input_pts, axis=0)))
        else:
            self._scale = 1

        self.calculateWeights()

    def initialise(self, input_data: pd.DataFrame, target_col: str, 
                   radial_func: str="gaussian", eps: float=10.,
                   rescaleAxis: bool=True) -> None:
        # Get basis function and initialise
        try:
            self.radialFunc = self._radialFuncs[radial_func]()
        except KeyError:
            sys.exit(f"Error - function '{radial_func}' not in " +
                     f"'{list(self._radialFuncs.keys())}'")
        self._initialise(input_data, target_col, eps, rescaleAxis)
        
    def initialise_text(self, input_file: str, target_col, 
                        radial_func: str="gaussian", eps: float=10.,
                        rescaleAxis: bool=True) -> None:
        df = pd.read_csv(input_file, index_col=False, delimiter=' ')
        self.initialise(df,target_col,radial_func,eps,rescaleAxis)
        
    def diff(self, points1: npt.NDArray[np.float32],
             points2: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        # Get diff between two sets of points, pairwise
        v = np.multiply(self._scale, (points1[:, np.newaxis, :] - 
                                      points2[np.newaxis, :, :]))
        return v    
    
    def diff2(self, points1: npt.NDArray[np.float32], 
              points2: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        # Get squared diff between two sets of points, pairwise
        return np.power(self.diff(points1, points2), 2)
    
    def getDistFromSquare(self, point: npt.NDArray[np.float32]):
        # Get distance between a point and the basis points, per axis
        return self.diff2(point, self._input_pts)
    
    def getRadialArg(self, 
                     d2: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        # Get arg to pass to basis functions
        return np.divide(d2, self._eps*self._eps)

    def grad_r2(self, point) -> npt.NDArray[np.float32]:
        # Calculates grad(|r|^2)
        return (2*self.diff(point, self._input_pts)*self._scale/(self._eps*self._eps))

    def evaluate(self, point: pd.DataFrame) -> npt.NDArray[np.float32]:
        # Check input is okay (can be turned off for perfomance)
        if not self._initialised:
            print("Error - must first initialise spline with set of points " + 
                  "before calling evaluate()") 
            return np.array(np.nan)
        if not set(point.keys()) == set(self._parameter_keys): 
            print(f"Error - {point.keys()} must match {self._parameter_keys}")
            return np.array(np.nan)
        
        # Evaluate spline at point
        point_arr = point.to_numpy()
        radial_arg = self.getRadialArg(np.sum(self.getDistFromSquare(point_arr), axis=-1))
        vals = self.radialFunc.evaluate(radial_arg).flatten()
        
        # Get val and grads
        weighted_vals = self._weights * vals
        ret_val = np.sum(weighted_vals)
        
        return ret_val.astype(float)
    
    def evaluate_grad(self, point: pd.DataFrame) -> npt.NDArray[np.float32]:
        # Check input is okay (can be turned off for perfomance)
        if not self._initialised:
            print("Error - must first initialise spline with set of points " + 
                  "before calling evaluate()") 
            return np.array(np.nan)
        if not set(point.keys()) == set(self._parameter_keys): 
            print(f"Error - {point.keys()} must match {self._parameter_keys}")
            return np.array(np.nan)

        # Evaluate spline at point
        point_arr = point.to_numpy()
        radial_arg = self.getRadialArg(self.getDistFromSquare(point_arr))
        delta_phi = self.radialFunc.getDeltaPhi(radial_arg)
        grad_phi = np.linalg.norm(delta_phi, axis=-1)
        grads = self.grad_r2(point_arr) * grad_phi.reshape(1, self._M, 1) * np.sign(delta_phi)
        
        # Get val and grads
        weighted_grads = np.multiply(self._weights.reshape(1, self._M, 1), grads)
        ret_grad = np.sum(weighted_grads, axis=1)
        
        return ret_grad.astype(float)
        
    def calculateWeights(self) -> None: 
        # Solve interpolation matrix equation for weights
        inp = self._input_pts
        B = self._input_data[self._target_col].to_numpy()
        d2 = np.sum(self.diff2(inp, inp), axis=2)
        A = self.radialFunc.evaluate(self.getRadialArg(d2)) 
        np.fill_diagonal(A, 1)
    
        self._interp_mat = A
        self._inv_interp_mat = np.linalg.inv(A)
        self._weights = np.dot(self._inv_interp_mat, B)
        self._initialised = True
        
    def calculateLOOCV(self) -> float:
        # Get leave-one-out cross-validation error, implementing
        # https://doi.org/10.1023/A:1018975909870]
        if not self._initialised:
            print("Error - must first initialise spline with set of points " + 
                  "before calling evaluate()") 
            return np.nan
        
        cost_vec = self._weights / self._inv_interp_mat.diagonal()       
        return np.linalg.norm(cost_vec)



def findBestEpsilon(df, target, radial_func, eps_range, rescaleAxis):
    best_eps = 0
    best_loocv = np.inf
    spline = rbf_spline(3)
    print(f"Finding the best epsilon for the spline")
    for eps in eps_range:
        spline.initialise(df, target, radial_func=radial_func, eps=eps, rescaleAxis=rescaleAxis)
        loocv = spline.calculateLOOCV()
        if loocv < best_loocv:
            best_loocv = loocv
            best_eps = eps
        print(f"LOOCV: {loocv} for epsilon: {eps}")
    print(f"Best epsilon: {best_eps}")
    return best_eps

def init_splines(signal, weights, bins, epsilon=None, spline_type="cubic", fine_search=False, channel="Electron"):
    print("Initiating Splines")
    masses = np.unique(ak.to_numpy(signal[['mH', 'mA']])).view('<f4').reshape(-1, 2)
    mHs, mAs = masses[:,0], masses[:,1]

    print(f"Unique mHs = {np.unique(mHs)}")
    print(f"Unique mAs = {np.unique(mAs)}")

    print(f"Type = {type(mHs[0])}")

    mH_spline_input = []
    mA_spline_input = []
    pNNbins = []
    pNNoutputs = []

    for mH, mA in zip(mHs, mAs):
        if (channel == "Electron") and (mA - mH <= 30):
            continue

        # get signal for that mass 
        cut = (signal['mH'] == mH) & (signal['mA'] == mA)
        signal_mass = signal[cut]

        mass = f"mH{int(mH)}_mA{int(mA)}"
        pNNout, b = np.histogram(signal_mass[f'pnn_output_{mass}'],
                                bins=bins,
                                weights=weights[cut])

        if mass == "mH80_mA150":
            print(f"pNNout")
            for i, val in enumerate(pNNout):
                print(f"Bin {i: <3}: {val}")

            # sys.exit(0)
        
        for i, (output, bin) in enumerate(zip(pNNout, bins[:-1])): # only need start edge of bin
            pNNbins.append(i)
            pNNoutputs.append(output)
            mH_spline_input.append(int(mH))
            mA_spline_input.append(int(mA - mH))

    data = {"mH" : mH_spline_input, "mA" : mA_spline_input, "pNNbin" : pNNbins,
            "pNN" : pNNoutputs}
    df = pd.DataFrame(data=data)

    if epsilon is not None:
        spline = rbf_spline(3)
        spline.initialise(df,'pNN', radial_func=spline_type, eps=epsilon, rescaleAxis=True)
        return spline, epsilon

    # Get the best value for eps
    eps_range = np.arange(0.01, 0.15, 0.01)


    best_eps = findBestEpsilon(df, 'pNN', spline_type, eps_range, rescaleAxis=True)

    # if fine_search:
    #     print(f"Doing a finer search for the best epislon")
    #     eps_range_fine = np.arange(best_eps - 0.01, best_eps + 0.1 + 0.01, 0.01)
    #     best_eps = findBestEpsilon(df, 'pNN', spline_type, eps_range_fine, rescaleAxis=True)

    spline = rbf_spline(3)
    spline.initialise(df,'pNN', radial_func=spline_type, eps=best_eps, rescaleAxis=True)

    return spline, best_eps


def interpolateSignal(spline, mH, mA, bins):
    interp_points = [spline.evaluate(pd.DataFrame(data={"mH" : [mH], "mA" : [mA], "pNNbin" : [bin]})) for bin in range(len(bins[:-1]))]

    # Make sure none are negative
    # if min(interp_points) < 0:
    #     print(f"WARNING: for mH{mH}_mA{mA}: Min histogram value < 0, hist = {interp_points}")
    interp_points = [max(x, 0) for x in interp_points]
    return interp_points


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




def convertToBoostHistogram(hist, sumw2, bins):
    root_hist = bh.Histogram(bh.axis.Variable(bins), 
                            storage=bh.storage.Weight())

    root_hist[...] = np.stack([hist, sumw2], axis=-1)

    return root_hist


def doInterpolation(train_direc, output_direc):

    # Load in the mass scaler
    mass_scaler = pickle.load(open(f"{train_direc}/mass_scaler.pkl", "rb"))

    # Load in the mass scan
    mass_pairs = np.loadtxt(f"{output_direc}/mass_scan.txt")

    bins = np.linspace(0.9, 1, 16)

    # Load the signal and weights
    branches = ['n_muons', 'n_electrons', 'pnn_output_*', 'weight_nominal_scaled', 'mH', 'mA', 'Zcand_m']

    signal = []
    #test_files = glob.glob(f"data/test/awkward/*.parquet")
    sig_files = glob.glob(f"{train_direc}/data/test/awkward/mH*.parquet")
    for file in tqdm(sig_files):
        file_name = file.split("/")[-1]
        mH = int(file_name.split("mH")[1].split("_")[0])
        mA = int(file_name.split("mA")[1].split("_")[0])
        signal.append(ak.from_parquet(file, columns=branches))

    signal = combineInChunks(signal)

    #! Remove ee events with Mll < 30
    sig_elec = signal[signal.n_electrons == 2]
    sig_mu = signal[signal.n_muons == 2]
    sig_elec_Mll_cut = sig_elec.Zcand_m > 30
    eff = np.sum(sig_elec_Mll_cut) / len(sig_elec_Mll_cut)
    print(f"Efficiency of Mll cut: {eff}")
    sig_elec = sig_elec[sig_elec_Mll_cut]

    signal = ak.concatenate([sig_elec, sig_mu], axis=0)

    # Change the mass to the unscaled mass
    masses = ak.to_numpy(signal[['mH', 'mA']]).view('<f4').reshape(-1, 2)
    unscaled_masses = mass_scaler.inverse_transform(masses)
    signal['mH'] = np.round(unscaled_masses[:, 0])
    signal['mA'] = np.round(unscaled_masses[:, 1])


    for process in ['Electron', 'Muon']:

        histogram_dict = {}

        events_proc = signal[signal[f'n_{process.lower()}s'] == 2]

        weights_array = events_proc.weight_nominal_scaled
        # initiate the splines
        spline, best_eps = init_splines(events_proc, weights = weights_array, bins = bins, fine_search = True, channel = process)

        
        # Now save straight to root files
        #with uproot.recreate(f"{output_direc}/combine/{year}/OF_histograms/{mass}_OF_CR_{year}_hists.root") as f:

        for mH, mA in tqdm(mass_pairs):
            if (process == "Electron") and (mA - mH <= 34.99):
                continue
            interp_signal = interpolateSignal(spline, mH, mA - mH, bins = bins)

            if (mH == 80) & (mA == 150):
                print(f"interp_signal")
                for i, val in enumerate(interp_signal):
                    print(f"Bin {i: <3}: {val}")

            # Now add to histogram dict
            if f"mH{mH}_mA{mA}" in histogram_dict:
                histogram_dict[f"mH{mH}_mA{mA}"][f"{process};idm;mH{mH}_mA{mA}"] = interp_signal
            else:
                histogram_dict[f"mH{mH}_mA{mA}"] = {f"{process};idm;mH{mH}_mA{mA}" : interp_signal}


        # Now that I have the histogram dict, I can save these directly to signal root files
        # Convert all the histograms to boost histograms

        for mass_point, histogram_dict in histogram_dict.items():
            print(f"Saving for {mass_point}")
            root_histograms = {}
            for key, histogram in histogram_dict.items():
                PROCESS, proc_name, mass = key.split(";")
                root_histogram = convertToBoostHistogram(histogram, np.zeros_like(histogram), bins)
                root_histograms[f"{proc_name}"] = root_histogram

            # Now save all of the histograms
            # make the directory
            save_direc = f"{output_direc}/combine/{mass_point}"
            os.makedirs(save_direc, exist_ok=True)

            save_loc = f"{save_direc}/{mass_point}_signal_{process}_hists.root"

            with uproot.recreate(save_loc) as f:
                for key, hist in root_histograms.items():
                    f[key] = hist


if __name__ == "__main__":
    parser = parse_arguments()
    train_direc = parser.train_direc
    output_direc = parser.output_direc
    doInterpolation(train_direc, output_direc)