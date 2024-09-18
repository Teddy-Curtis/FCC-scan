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
limit_files = glob.glob(f"{combine_direc}/*/limits*.json")


all_limits = {}
for file in limit_files:
    with open(file, "r") as f:
        limits = json.load(f)

    mass_point = file.split("/")[-2]
    print(mass_point)

    limit_name = file.split("/")[-1].split("limits_")[1].split(".json")[0]

    print(limit_name)

    if mass_point in all_limits:
        all_limits[mass_point][limit_name] = limits
    else:
        all_limits[mass_point] = {limit_name: limits}


print(all_limits)

# Sort the dictionary before saving
all_limits = dict(sorted(all_limits.items()))

# Now save as a json
with open(f"{combine_direc}/all_limits.json", "w") as f:
    json.dump(all_limits, f, indent=4)