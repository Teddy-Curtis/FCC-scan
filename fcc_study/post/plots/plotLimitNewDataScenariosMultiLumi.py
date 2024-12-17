import numpy as np
import matplotlib.pyplot as plt
import json, argparse
import mplhep as hep
from scipy.interpolate import griddata



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
        "--combine_direc_others",
        required=True,
        default=None,
        type=lambda s: [str(item) for item in s.split(',')],
        help="Directory that contains the limit file.")

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



parser = parse_arguments()
combine_direc = parser.combine_direc
lumi = parser.lumi
ecom = parser.ecom
save_name = parser.save_name
skipSigmaBands = parser.skipSigmaBands
skip_excluded = parser.skip_excluded


def getGrid(all_limits, all_ms, lim_val, method = 'cubic'):
    # Sort them in the correct order
    ind = np.lexsort((all_ms[:,1],all_ms[:,0]))    
    all_ms = all_ms[ind]
    
    limits = []
    mHs = []
    diffs = []
    for mH, mA in all_ms:
        # if mA - mH <= 2:
        #     continue
        mHs.append(mH)
        diffs.append(mA - mH)

        if mA - mH <= 30:
            try: 
                lim = all_limits[f"mH{mH}_mA{mA}"]['MuMu'][lim_val]
            except:
                lim = 2
        else:
            try:
                lim = all_limits[f"mH{mH}_mA{mA}"]['combined'][lim_val]
            except:
                try:
                    lim = all_limits[f"mH{mH}_mA{mA}"]['MuMu'][lim_val]
                except:
                    lim = 2

        limits.append(lim)

    # Now get the grid
    limits = np.array(limits)
    mHs = np.array(mHs)
    diffs = np.array(diffs)

    # Now combine these so they are shape (n, 2)
    masses = np.vstack((mHs, diffs)).T
    limits[limits > 1.5] = 1.5

    grid_x, grid_y = np.meshgrid(np.arange(np.min(mHs), np.max(mHs) + 6, 1),
                                np.arange(0, np.max(diffs) + 1, 1), indexing='ij')



    grid = griddata(masses, limits, (grid_x, grid_y), method=method)

    # Make nan be 2
    grid[grid > 1.5] = 1.5
    grid[np.isnan(grid)] = 1.5

    return grid.T, grid_x.T, grid_y.T, mHs, diffs



with open(f"{combine_direc}/all_limits.json", "r") as f:
    all_limits = json.load(f)


# make the grid
all_ms = np.loadtxt(f"{combine_direc}/mass_scan.txt")
ind = np.lexsort((all_ms[:,1],all_ms[:,0]))    
all_ms = all_ms[ind]

plot_grid, grid_x, grid_y, mHs, diffs = getGrid(all_limits, all_ms, '0.5')

extent = (np.min(mHs)-1, np.max(mHs) + 5, 0, np.max(diffs))

print(f"extent = {extent}")


# Plot
plt.style.use(hep.style.CMS)
plt.rcParams['text.usetex'] = True

fig, ax = plt.subplots(figsize=(15,12))

cmap = plt.cm.viridis
cmap.set_bad(color='white')
masked_array = np.ma.masked_where(np.ones_like(plot_grid) * -1 == -1, plot_grid)
im = ax.imshow(masked_array, cmap = cmap, aspect='auto', extent=extent, origin='lower')


legend_elements = []
legend_names = []

if ecom == 240:
    line = ax.plot([70, 120], [100, 0], color='black', linestyle='--', label = f'$M_H$ + $M_A$ = {ecom} GeV')
else:
    line = ax.plot([70, 365/2], [225, 0], color='black', linestyle='--', label = f'$M_H$ + $M_A$ = {ecom} GeV')


con_filled = plt.contourf(plot_grid, np.array([0, 1]), colors=['white', 'white'],
                hatches = ['///', '//////'], levels=[-10, 1], alpha=0.5, extent=extent, origin='lower')
handles_con_filled, labels_filled = con_filled.legend_elements()


con = plt.contour(plot_grid, np.array([1]) , colors=['black'], linewidths=[2], extent=extent, origin='lower')
handles_con, labels = con.legend_elements()
legend_names += [f"95\% CL, {lumi}" + r"$fb^{-1}$"]
legend_elements += handles_con
###########################################################################
######################### Plot the other scenarios#########################
###########################################################################
###########################################################################
######################### Plot the other scenarios#########################
###########################################################################
combine_direc_others = parser.combine_direc_others
lumi_others = parser.lumi_others
colour_others = parser.colour_others
# Now for the second combine_direc
for combine_direc_other, lumi_other, colour_other in zip(combine_direc_others, lumi_others, colour_others):
    with open(f"{combine_direc_other}/all_limits.json", "r") as f:
        all_limits_other = json.load(f)
    # make the grid
    plot_grid_otherLumi = getGrid(all_limits_other, all_ms, '0.5')[0]

    con_scen2 = ax.contour(plot_grid_otherLumi, np.array([1]) , colors=['red'], linewidths=[2],
                        extent=extent, 
                        origin='lower', linestyles='dashed')
    handles_conscen2, labels_conscen2 = con_scen2.legend_elements()

    legend_elements += handles_conscen2
    legend_names += [f"95\% CL, {lumi_other}" + r"$fb^{-1}$"]


legend_elements += handles_con_filled  + line
legend_names += ["Excluded", '$M_H$ + $M_A$ = $\sqrt{s}$']



if not skip_excluded:
    x2 = np.arange(50, 95, 1)
    y3 = (-5 / 4) * x2 + 117.5
    y4 = 0 * np.ones_like(x2)
    excl_LEP = plt.fill_between(x2, y3, y4, color='green', alpha=0.2, label = 'Excluded by LEP')

    legend_elements += [excl_LEP]
    legend_names += ["LEP SUSY Recast"]



plt.xlim(np.min(grid_x), np.max(grid_x))
plt.ylim(np.min(grid_y), np.max(grid_y))

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



eq1 = ("IDM:\n"
        r"$M_{H^\pm} = M_A$" + "\n"
        r"$\lambda_{345} = 1e\textit{-}6$")

plt.text(0.8, 0.6, eq1, fontsize="21",
             transform=ax.transAxes)


if save_name is not None:
    name = save_name
else:
    name = "limit_scenarios_multiLumi"

plt.savefig(f"{combine_direc}/{name}.pdf", bbox_inches='tight')
plt.savefig(f"{combine_direc}/{name}.png", bbox_inches='tight')
#plt.title("Expected Limit, r")


