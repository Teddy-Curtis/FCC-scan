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

line = ax.plot([70, 120], [100, 0], color='black', linestyle='--', label = f'$M_H$ + $M_A$ = {ecom} GeV')


con_filled = plt.contourf(plot_grid, np.array([0, 1]), colors=['white', 'white'],
                hatches = ['///', '//////'], levels=[-10, 1], alpha=0.5, extent=extent, origin='lower')
handles_con_filled, labels_filled = con_filled.legend_elements()


con = plt.contour(plot_grid, np.array([1]) , colors=['black'], linewidths=[2], extent=extent, origin='lower')
handles_con, labels = con.legend_elements()


if not skipSigmaBands:
    plot_grid_for_up_contour, _, _, _, _ = getGrid(all_limits, all_ms, '0.84')
    con_up = plt.contour(plot_grid_for_up_contour, np.array([1]) , colors=['red'], 
                        linewidths=[2], linestyles=['--'], extent=extent, origin='lower')
    handles_con_up, labels_con_up = con_up.legend_elements()

    plot_grid_for_down_contour, _, _, _, _ = getGrid(all_limits, all_ms, '0.16')
    con_down = plt.contour(plot_grid_for_down_contour, np.array([1]) , colors=['red'], 
                        linewidths=[2], linestyles=['--'], extent=extent, origin='lower')
    handles_con_down, labels_con_down = con_down.legend_elements()

    legend_elements += handles_con + handles_con_up + handles_con_filled  + line
    legend_names += ["Expected 95% CL", "$\pm 1 \sigma$", "Excluded", '$M_H$ + $M_A$ = $\sqrt{s}$']


else:
    legend_elements + handles_con + handles_con_filled  + line
    legend_names += ["Expected 95% CL", "$\pm 1 \sigma$", "Excluded", '$M_H$ + $M_A$ = $\sqrt{s}$']
    


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



eq1 = ("Scenario 1:\n"
        r"$M_{H^\pm} = M_A$" + "\n"
        r"$\lambda_{345} = 1e\textit{-}6$")

plt.text(0.8, 0.6, eq1, fontsize="21",
             transform=ax.transAxes)

if save_name is not None:
    name = save_name
else:
    name = "limit"

plt.savefig(f"{combine_direc}/{name}.pdf", bbox_inches='tight')
plt.savefig(f"{combine_direc}/{name}.png", bbox_inches='tight')
#plt.title("Expected Limit, r")


