import numpy as np
import matplotlib.pyplot as plt
import json, argparse
import mplhep as hep


def parse_arguments():

    parser = argparse.ArgumentParser(
        description="Masses to evaulate and interpolate pNN at.")
    
    parser.add_argument(
        "--combine_direc",
        required=True,
        default=None,
        type=str,
        help="Directory that contains the limit file.")
    
    parser.add_argument(
        "--lumi",
        required=True,
        default=None,
        type=float,
        help="Luminosity to scale the limits by.")
    
    parser.add_argument(
        "--ecom",
        required=True,
        default=None,
        type=float,
        help="Center of mass energy.")

    parser.add_argument(
        "--save_name",
        required=False,
        default=None,
        type=str,
        help="Extra save name.")

    parser.add_argument(
        "--skip_excluded",
        required=False,
        default=False,
        action="store_true",
        help="Whether to plot the excluded regions on top of the plots.")


    return parser.parse_args()



def fillMissingLimits(limits):
    for val in ['0.03', '0.16', '0.5', '0.84', '0.97']:
        if val not in limits:
            limits[val] = 0
    
    return limits



parser = parse_arguments()
combine_direc = parser.combine_direc
lumi = parser.lumi
ecom = parser.ecom
skip_excluded = parser.skip_excluded
save_name = parser.save_name


with open(f"{combine_direc}/all_signifs.json", "r") as f:
    all_signifs = json.load(f)



# make the grid
all_ms = np.loadtxt(f"{combine_direc}/mass_scan.txt")

mHs = np.unique(all_ms[:, 0])
# For a single mH, get the values of mA - mH
diff_scan = all_ms[all_ms[:, 0] == mHs[0], 1] - mHs[0]

# Now need to make a 2d grid to store the limits
grid = np.zeros((len(diff_scan), len(mHs)))


plot_grid = np.ones((len(diff_scan)+1, len(mHs))) * -1
plot_grid_for_contour = np.ones((len(diff_scan)+1, len(mHs))) * -1      

for mass_point, signif_dict in all_signifs.items():
    mH = float(mass_point.split("_")[0].split("mH")[1])
    mA = float(mass_point.split("mA")[1])


    #! To change, this is just for Mu-Mu case
    # print(mass_point)
    # try:
    #     signif = signif_dict["combined"]
    # except:
    #     signif = signif_dict["MuMu"]

    signif = signif_dict["MuMu"]

    deltaAH = int(mA - mH)

    row_idx = len(diff_scan) - np.where(diff_scan == mA - mH)[0][0]
    col_idx = np.where(mHs == mH)[0][0]

    #print(mH, mA, deltaAH)  

    #print(row_idx, col_idx)

    plot_grid[row_idx, col_idx] = signif

    plot_grid_for_contour[row_idx, col_idx] = signif

    
dx = (mHs[1]-mHs[0])/2.
dy = (diff_scan[1]-diff_scan[0])/2.
extent = [mHs[0]-dx, mHs[-1]+dx, diff_scan[0]-dy, diff_scan[-1]+dy]

plt.style.use(hep.style.CMS)
plt.rcParams['text.usetex'] = True

fig, ax = plt.subplots(figsize=(15,12))

cmap = plt.cm.viridis
cmap.set_bad(color='white')
masked_array = np.ma.masked_where(np.ones_like(plot_grid) * -1 == -1, plot_grid)
im = ax.imshow(masked_array, cmap = cmap, aspect='auto', extent=extent)

max_mH = (ecom - np.min(diff_scan)) / 2
line = plt.plot([np.min(mHs), max_mH], [ecom - 2 * np.min(mHs), np.min(diff_scan)], color='black', linestyle='--', label = f'$M_H$ + $M_A$ = {ecom} GeV')

# plt.xlim(min(mHs), max(mHs))
# plt.ylim(min(diff_scan), max(diff_scan))

plt.xlim(60, max_mH)
plt.ylim(np.min(diff_scan), ecom - 2 * np.min(mHs))


#plt.contour(plot_grid, np.array([1]) , colors=['yellow'], linewidths=[5])

legend_elements = []
legend_names = []

con = plt.contour(plot_grid_for_contour, np.array([5]) , colors=['black'], linewidths=[2], extent=extent, origin='upper')
handles_con, labels = con.legend_elements()

con_filled = plt.contourf(plot_grid_for_contour, np.array([5, np.inf]), colors=['white', 'white'],
                   hatches = ['///', '//////'], levels=[5, np.inf], alpha=0.5, extent=extent, origin='upper')
handles_con_filled, labels_filled = con_filled.legend_elements()

legend_elements += handles_con + handles_con_filled + line
legend_names += ["5$\sigma$ Discovery", "Discoverable", '$M_H$ + $M_A$ = $\sqrt{s}$']


if not skip_excluded:
    x = np.arange(50, 71, 1)
    y1 = (-5 / 4) * x + 117.5
    y2 = 150 * np.ones_like(x)
    excl_dm = plt.fill_between(x, y1, y2, color='blue', alpha=0.2, label = 'Excluded by DM Observations')
    #handles_excl_dm, labels_excl_dm = excl_dm.legend_elements()

    x2 = np.arange(50, 91, 1)
    y3 = (-5 / 4) * x2 + 117.5
    y4 = 10 * np.ones_like(x2)
    excl_LEP = plt.fill_between(x2, y3, y4, color='green', alpha=0.2, label = 'Excluded by LEP')

    legend_elements += [excl_dm, excl_LEP]
    legend_names += ["Relic Density", "LEP SUSY Recast"]


ax.legend(legend_elements,
          legend_names, 
          loc='upper right', ncol=2)


plt.xlabel("$M_H$ (GeV)")
plt.ylabel(r"$\Delta(M_A,M_H) = M_A - M_H$ (GeV)")   


plt.text(0.55, 1.013, r"FCC-ee,  $\sqrt{s}$" + f"={ecom} GeV,  " + f"Lumi={lumi}" + "$ab^{-1}$", fontsize="21",
             transform=ax.transAxes)

# Plot diagonal line for mA + mH = 240
#plt.plot([25, 0], [0, 25], color='black', linestyle='--')


eq1 = (r"\begin{eqnarray*}"
        r"\textit{Significance} = \Biggl\{"
        r"  \begin{array}{l}"
        r" e\textit{-}e/\mu\textit{-}\mu \quad \textit{if}\ \quad \Delta(M_A,M_H) \geq 30 GeV\\"
        r"  \mu\textit{-}\mu \quad \quad \; \; \, \textit{if}\ \quad \Delta(M_A,M_H) < 30 GeV"
        r"\end{array}"
       r"\end{eqnarray*}")
plt.text(0.405, 0.79, eq1, fontsize="21",
             transform=ax.transAxes)
# plt.savefig("limit.pdf", bbox_inches='tight')
#plt.title("Expected Limit, r")

if save_name is not None:
    name = save_name
else:
    name = "significance"

plt.savefig(f"{combine_direc}/{name}.pdf", bbox_inches='tight')
plt.savefig(f"{combine_direc}/{name}.png", bbox_inches='tight')
#plt.title("Expected Limit, r")


