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
        "--combine_direc_others",
        required=True,
        default=None,
        type=lambda s: [str(item) for item in s.split(',')],
        help="Directory that contains the limit file.")
    
    parser.add_argument(
        "--lumi",
        required=True,
        default=None,
        type=float,
        help="Luminosity to scale the limits by.")

    parser.add_argument(
        "--lumi_others",
        required=True,
        default=None,
        type=lambda s: [float(item) for item in s.split(',')],
        help="Luminosity to scale the limits by.")

    parser.add_argument(
        "--colour_others",
        required=True,
        default=None,
        type=lambda s: [str(item) for item in s.split(',')],
        help="Colours.")
    
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
        "--skipSigmaBands",
        required=False,
        default=False,
        action="store_true",
        help="Whether to skip the sigma bands.")
    

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


def getPlotGrid(mHs, diff_scan, all_limits):
    plot_grid = np.ones((len(diff_scan)+1, len(mHs))) * -1
    plot_grid_for_contour = np.ones((len(diff_scan)+1, len(mHs))) * 10      
    plot_grid_for_up_contour = np.ones((len(diff_scan)+1, len(mHs))) * 10 
    plot_grid_for_down_contour = np.ones((len(diff_scan)+1, len(mHs))) * 10 

    for mass_point, limit_dict in all_limits.items():
        mH = float(mass_point.split("_")[0].split("mH")[1])
        mA = float(mass_point.split("mA")[1])

        if mA - mH <= 30:
            # if small mass splitting then use the muon limits
            limits = limit_dict["MuMu"]
        else:

            try:
                limits = limit_dict['combined']
            except:
                limits = limit_dict["MuMu"]


        
        # limits = f["limit"]["limit"].array()
        limits = fillMissingLimits(limits)

        # #! To change, this is just for Mu-Mu case
        # limits = limit_dict["MuMu"]

        try:
            limit = limits['0.5']
        except:
            print(f"Couldn't find 0.5 limit for {mass_point}, skipping for now")
            limits = {
                0.5 : 0, 0.84 : 0, 0.16 : 0
            }
            # continue

        if '0.84' not in limits:
            limits['0.84'] = 0


        deltaAH = int(mA - mH)

        row_idx = len(diff_scan) - np.where(diff_scan == mA - mH)[0][0]
        col_idx = np.where(mHs == mH)[0][0]

        #print(mH, mA, deltaAH)  

        #print(row_idx, col_idx)

        plot_grid[row_idx, col_idx] = limit

        plot_grid_for_contour[row_idx, col_idx] = limit

        plot_grid_for_up_contour[row_idx, col_idx] = limits['0.84']
        plot_grid_for_down_contour[row_idx, col_idx] = limits['0.16']

    return plot_grid, plot_grid_for_contour, plot_grid_for_up_contour, plot_grid_for_down_contour
    
parser = parse_arguments()
combine_direc = parser.combine_direc
combine_direc_others = parser.combine_direc_others
lumi = parser.lumi
lumi_others = parser.lumi_others
colour_others = parser.colour_others
ecom = parser.ecom
save_name = parser.save_name
skipSigmaBands = parser.skipSigmaBands
skip_excluded = parser.skip_excluded


with open(f"{combine_direc}/all_limits.json", "r") as f:
    all_limits = json.load(f)
# make the grid
all_ms = np.loadtxt(f"{combine_direc}/mass_scan.txt")
mHs = np.unique(all_ms[:, 0])
# For a single mH, get the values of mA - mH
diff_scan = all_ms[all_ms[:, 0] == mHs[0], 1] - mHs[0]
plot_grid, plot_grid_for_contour, plot_grid_for_up_contour, plot_grid_for_down_contour = getPlotGrid(mHs, diff_scan, all_limits)
    

other_plot_grids = {}
other_plot_grids_for_contour = {}
for combine_direc_other, lumi_other in zip(combine_direc_others, lumi_others):
    with open(f"{combine_direc_other}/all_limits.json", "r") as f:
        all_limits_other = json.load(f)
    # make the grid
    all_ms = np.loadtxt(f"{combine_direc_other}/mass_scan.txt")
    mHs = np.unique(all_ms[:, 0])
    # For a single mH, get the values of mA - mH
    diff_scan = all_ms[all_ms[:, 0] == mHs[0], 1] - mHs[0]
    plot_grid_other, plot_grid_for_contour_other, plot_grid_for_up_contour_other, plot_grid_for_down_contour_other = getPlotGrid(mHs, diff_scan, all_limits_other)
    other_plot_grids[lumi_other] = plot_grid_other
    other_plot_grids_for_contour[lumi_other] = plot_grid_for_contour_other


print(plot_grid_for_contour)
print(other_plot_grids_for_contour)

dx = (mHs[1]-mHs[0])/2.
dy = (diff_scan[1]-diff_scan[0])/2.
extent = [mHs[0]-dx, mHs[-1]+dx, diff_scan[0]-dy, diff_scan[-1]+dy]

import mplhep as hep

plt.style.use(hep.style.CMS)
plt.rcParams['text.usetex'] = True

fig, ax = plt.subplots(figsize=(15,12))

cmap = plt.cm.viridis
cmap.set_bad(color='white')
masked_array = np.ma.masked_where(np.ones_like(plot_grid) * -1 == -1, plot_grid)
im = ax.imshow(masked_array, cmap = cmap, aspect='auto', extent=extent)

max_mH = (ecom - np.min(diff_scan)) / 2
print(f"max_mH = {max_mH}")
line = plt.plot([np.min(mHs), max_mH], [ecom - 2 * np.min(mHs), np.min(diff_scan)], color='black', linestyle='--', label = f'$M_H$ + $M_A$ = {ecom} GeV')

plt.xlim(60, max_mH)
plt.ylim(np.min(diff_scan), ecom - 2 * np.min(mHs))



con = plt.contour(plot_grid_for_contour, np.array([1]) , colors=['black'], linewidths=[2], extent=extent, origin='upper')
handles_con, labels = con.legend_elements()

con_filled = plt.contourf(plot_grid_for_contour, np.array([0, 1]), colors=['white', 'white'],
                hatches = ['///', '//////'], levels=[0, 1], alpha=0.5, extent=extent, origin='upper')
handles_con_filled, labels_filled = con_filled.legend_elements()

legend_elements = []
legend_names = []

if not skipSigmaBands:
    con_up = plt.contour(plot_grid_for_up_contour, np.array([1]) , colors=['red'], 
                        linewidths=[2], linestyles=['--'], extent=extent, origin='upper')
    handles_con_up, labels_con_up = con_up.legend_elements()

    con_down = plt.contour(plot_grid_for_down_contour, np.array([1]) , colors=['red'], 
                        linewidths=[2], linestyles=['--'], extent=extent, origin='upper')
    handles_con_down, labels_con_down = con_down.legend_elements()

    legend_elements += handles_con + handles_con_up + handles_con_filled  + line
    legend_names += ["Expected 95% CL", "$\pm 1 \sigma$", "Excluded", '$M_H$ + $M_A$ = $\sqrt{s}$']


else:
    legend_elements + handles_con + handles_con_filled  + line
    legend_names += ["Expected 95% CL", "$\pm 1 \sigma$", "Excluded", '$M_H$ + $M_A$ = $\sqrt{s}$']
    

# Now plot the others
handles_others = []
labels_others = []
for i, (lumi_other, plot_grid_other) in enumerate(other_plot_grids_for_contour.items()):
    color = colour_others[i]
    con_other = plt.contour(plot_grid_other, np.array([1]) , linewidths=[2], extent=extent, origin='upper',colors=[color])
    handles_con_other, labels_other = con_other.legend_elements()
    handles_others += handles_con_other
    labels_others += [f'Lumi={lumi_other}' + r'$ab^{-1}$']

legend_elements += handles_others
legend_names += labels_others

if not skip_excluded:
    x = np.arange(50, 71, 1)
    y1 = (-5 / 4) * x + 117.5
    y2 = 400 * np.ones_like(x)
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

eq1 = (r"\begin{eqnarray*}"
        r"\textit{limit} = \Biggl\{"
        r"  \begin{array}{l}"
        r" e\textit{-}e/\mu\textit{-}\mu \quad \textit{if}\ \quad \Delta(M_A,M_H) \geq 30 GeV\\"
        r"  \mu\textit{-}\mu \quad \quad \; \; \, \textit{if}\ \quad \Delta(M_A,M_H) < 30 GeV"
        r"\end{array}"
       r"\end{eqnarray*}")

plt.text(0.445, 0.77, eq1, fontsize="21",
             transform=ax.transAxes)

if save_name is not None:
    name = save_name
else:
    name = "limit"

plt.savefig(f"{combine_direc}/{name}.pdf", bbox_inches='tight')
plt.savefig(f"{combine_direc}/{name}.png", bbox_inches='tight')
#plt.title("Expected Limit, r")


