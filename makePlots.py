import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
from sklearn.metrics import roc_curve, roc_auc_score
from fcc_study.pNN.post.plot import getSignal, getBackground, getSigMixedBackground
from fcc_study.pNN.post.plot import plotSignificance, plotSigVsBackground, plotTrainVsTest, plotROC
import os

base_path = "/vols/cms/emc21/fccStudy/runs/fcc_scan/run14"
base_plot_path = f"{base_path}/plots"
os.makedirs(f"{base_path}/plots", exist_ok=True)

test_data = ak.from_parquet(f"{base_path}/test_data.parquet", columns=['id_num', 'process', 
                                                            "specific_proc", 'class', 
                                                            "bdt_output_*", "pnn_output_*",
                                                            "massSum", "massDiff", "weight*"])
train_data = ak.from_parquet(f"{base_path}/train_data.parquet", columns=['id_num', 'process', 
                                                            "specific_proc", 'class', 
                                                            "bdt_output_*", "pnn_output_*",
                                                            "massSum", "massDiff", "weight*"])


# Now do the plots
BPs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 18, 19, 20]
#BPs = [10]
for bp in BPs:
    print(bp)

    test = getSigMixedBackground(test_data, bp)
    train = getSigMixedBackground(train_data, bp)

    os.makedirs(f"{base_plot_path}/significance", exist_ok=True)

    save_loc = f"{base_plot_path}/significance/pnn_test_bp{bp}.pdf"
    plotSignificance(test, f"pnn_output_bp{bp}", [-0.05, 1.05], save_loc, title_prefix = f"pNN, BP{bp}, Test")
    save_loc = f"{base_plot_path}/significance/pnn_train_bp{bp}.pdf"
    plotSignificance(train, f"pnn_output_bp{bp}", [-0.05, 1.05], save_loc, title_prefix = f"pNN, BP{bp}, Train")

    save_loc = f"{base_plot_path}/significance/bdt_test_bp{bp}.pdf"
    plotSignificance(test, f"bdt_output_bp{bp}", [-1.05, 1.05], save_loc, title_prefix = f"BDT, BP{bp}, Test")
    save_loc = f"{base_plot_path}/significance/bdt_train_bp{bp}.pdf"
    plotSignificance(train, f"bdt_output_bp{bp}", [-1.05, 1.05], save_loc, title_prefix = f"BDT, BP{bp}, Train")

    plt.close()


    os.makedirs(f"{base_plot_path}/sigVsBackground", exist_ok=True)
    save_loc = f"{base_plot_path}/sigVsBackground/pnn_test_bp{bp}.pdf"
    plotSigVsBackground(test_data, bp, f"pnn_output_bp{bp}", np.linspace(0, 1, 100), save_loc, "pNN: Test")
    save_loc = f"{base_plot_path}/sigVsBackground/pnn_train_bp{bp}.pdf"
    plotSigVsBackground(train_data, bp, f"pnn_output_bp{bp}", np.linspace(0, 1, 100), save_loc, "pNN: Train")

    save_loc = f"{base_plot_path}/sigVsBackground/bdt_test_bp{bp}.pdf"
    plotSigVsBackground(test_data, bp, f"bdt_output_bp{bp}", np.linspace(-1, 1, 100), save_loc, "BDT: Test")
    save_loc = f"{base_plot_path}/sigVsBackground/bdt_train_bp{bp}.pdf"
    plotSigVsBackground(train_data, bp, f"bdt_output_bp{bp}", np.linspace(-1, 1, 100), save_loc, "BDT: Train")

    plt.close()


    os.makedirs(f"{base_plot_path}/trainVsTest", exist_ok=True)
    save_loc = f"{base_plot_path}/trainVsTest/pnn_bp{bp}.pdf"
    plotTrainVsTest(train_data, test_data, bp, f"pnn_output_bp{bp}", np.linspace(0, 1, 100), "pNN", save_loc)
    save_loc = f"{base_plot_path}/trainVsTest/bdt_bp{bp}.pdf"
    plotTrainVsTest(train_data, test_data, bp, f"bdt_output_bp{bp}", np.linspace(-1, 1, 100), "BDT", save_loc)

    os.makedirs(f"{base_plot_path}/roc", exist_ok=True)
    save_loc = f"{base_plot_path}/roc/pnn_bp{bp}.pdf"
    plotROC(train_data, test_data, bp, f"pnn_output_bp{bp}", save_loc, title_prefix = "pNN")
    save_loc = f"{base_plot_path}/roc/bdt_bp{bp}.pdf"
    plotROC(train_data, test_data, bp, f"bdt_output_bp{bp}", save_loc, title_prefix = "BDT")

    plt.close()
