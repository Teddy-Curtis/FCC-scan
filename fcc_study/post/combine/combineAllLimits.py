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


    return parser.parse_args()

parser = parse_arguments()
combine_direc = parser.combine_direc

# Get all the limit files
limit_files = glob.glob(f"{combine_direc}/combine/*/limits*.json")


all_limits = {}
for file in limit_files:
    with open(file, "r") as f:
        limits = json.load(f)

    mass_point = file.split("/")[-2]
    print(mass_point)

    channel_name = file.split("/")[-1].split("limits_")[1].split(".json")[0]

    print(channel_name)

    if mass_point in all_limits:
        all_limits[mass_point][channel_name] = limits
    else:
        all_limits[mass_point] = {channel_name: limits}


print(all_limits)

# Sort the dictionary before saving
all_limits = dict(sorted(all_limits.items()))

# Now save as a json
with open(f"{combine_direc}/all_limits.json", "w") as f:
    json.dump(all_limits, f, indent=4)


# Now get all of the significances
# Get all the limit files
signif_files = glob.glob(f"{combine_direc}/combine/*/signif*.json")


all_signifs = {}
for file in signif_files:
    with open(file, "r") as f:
        signif = json.load(f)

    mass_point = file.split("/")[-2]
    print(mass_point)

    channel_name = file.split("/")[-1].split("significance_")[1].split(".json")[0]

    print(channel_name)

    if mass_point in all_signifs:
        all_signifs[mass_point][channel_name] = signif['significance']
    else:
        all_signifs[mass_point] = {channel_name: signif['significance']}


print(all_signifs)

# Sort the dictionary before saving
all_signifs = dict(sorted(all_signifs.items()))

# Now save as a json
with open(f"{combine_direc}/all_signifs.json", "w") as f:
    json.dump(all_signifs, f, indent=4)