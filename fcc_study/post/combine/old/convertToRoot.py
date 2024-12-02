""" 
Script to recursively save a dictionary as a root file. This will also maintain 
the tree structure of the root file.
For example: 
{
    "BP1" : {
        "signal" : (hist, err, bins), 
        "bkg"    : (hist, err, bins)
    }
}

will be saved as 
BP1/
    -> /signal
    -> /bkg
"""

import uproot
import boost_histogram as bh
import numpy as np

def recursivelySave(obj, f, default_bins=None, path=""):
    if isinstance(obj, dict):
        # obj is a dictionary therefore we need to go deeper into the dictionary
        for key, value in obj.items():
            recursivelySave(value, f, default_bins, f"{path}/{key}")
    else:
        # obj isn't a dict, therefore it must be a tuple with the histogram
        # information 
        if len(obj) == 3:
            (hist, err, bins) = obj
        elif len(obj) == 2:
            (hist, err) = obj
            bins = default_bins

        root_hist = bh.Histogram(bh.axis.Variable(bins), 
                                storage=bh.storage.Weight())
        root_hist[...] = np.stack([hist, err], axis=-1)

        f[path] = root_hist

def saveHistogramDictToRoot(histograms, filename, default_bins=None):
    # Want to open the file here so that we aren't opening and closing the file
    # loads of times
    with uproot.recreate(filename) as f:
        # Now recursively save the histograms
        recursivelySave(histograms, f, default_bins)