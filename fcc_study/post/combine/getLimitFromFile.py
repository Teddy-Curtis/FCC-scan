import uproot
import json
import argparse
import glob

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

    parser.add_argument('--object_name',
                        choices=['limits', 'significance'],
                        help='What object do you want to get from the file?')

    return parser.parse_args()

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

def getSignificanceFromFile(file_path):
    with uproot.open(file_path) as f:
        signif = f['limit']['limit'].array()[0]

        return signif




parser = parse_arguments()
combine_direc = parser.combine_direc
extra_name = parser.extra_name
object_name = parser.object_name

save_dict = {}

if object_name == 'limits':
    asymp_file = glob.glob(f"{combine_direc}/*{extra_name}*AsymptoticLimits*.root")[0]
    limits = getLimitFromFile(asymp_file)
    save_dict.update(limits)

elif object_name == 'significance':
    asymp_file = glob.glob(f"{combine_direc}/*{extra_name}*Significance*.root")[0]
    signif = getSignificanceFromFile(asymp_file)
    save_dict['significance'] = signif




# Now save as a json
with open(f"{combine_direc}/{object_name}_{extra_name}.json", "w") as f:
    json.dump(save_dict, f, indent=4)
