import uproot
import json
import argparse
import glob

def getLimitFromFile(file_path):
    limits_dict = {}
    with uproot.open(file_path) as f:
        print(f.keys())
        print(f['limit'].keys())
        print(f['limit']['limit'].array())
        limits = f['limit']['limit'].array()
        print(f['limit']['quantileExpected'].array())
        for quan, lim in zip(f['limit']['quantileExpected'].array(), limits):
            print(f"Quantile: {quan}, Limit: {lim}")
            limits_dict[round(float(quan), 2)] = float(lim)

    # Now return the dictionary
    return limits_dict



def parse_arguments():

    parser = argparse.ArgumentParser(description="Evaluate model on a dataset.")

    parser.add_argument(
        "--combine_direc",
        required=True,
        default=None,
        type=str,
        help="Training directory where the model is.",
    )

    parser.add_argument(
        "--extra_name",
        required=True,
        default=None,
        type=str,
        help="Name of the final limits file.",
    )

    return parser.parse_args()


parser = parse_arguments()
combine_direc = parser.combine_direc
extra_name = parser.extra_name

asymp_file = glob.glob(f"{combine_direc}/*{extra_name}*AsymptoticLimits*.root")[0]

limits = getLimitFromFile(asymp_file)

# Now save as a json
with open(f"{combine_direc}/limits_{extra_name}.json", "w") as f:
    json.dump(limits, f, indent=4)
