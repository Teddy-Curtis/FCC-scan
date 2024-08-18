import awkward as ak
import numpy as np


def convertToNumpy(events: ak.Array, branches: list) -> np.array:
    # if only 1 branch then don't do the view stuff below, as that returns
    # garbage
    if len(branches) == 1:
        return ak.to_numpy(events[branches[0]])

    numpy_data = (
        ak.to_numpy(events[branches]).view("<f4").reshape(-1, len(branches))
    )
    return numpy_data
