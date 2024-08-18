import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
from sklearn.metrics import roc_curve, roc_auc_score
import mplhep as hep


def getSignal(data, bp_id):
    sig = data[data['id_num'] == bp_id]
    return sig

def getBackground(data):
    return data[data['id_num'] <= 0]

def getSigMixedBackground(test_data, bp):
    return test_data[(test_data['id_num'] == bp) | (test_data['id_num'] <= 0)]


BPs = [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 18, 19, 20]

bkg_procs_list = ["tautauH", "eeH", "mumuH", "nunuH", "mumu", "ZZ", "ee", "tautau", "WW"]


def getSignificance(data, discrim_var):
    fpr, tpr, thresholds = roc_curve(data['class'], data[discrim_var], sample_weight=data['weight_scaled'])

    n_sig = np.sum(data[data['class'] == 1]['weight_scaled'])
    n_bkg = np.sum(data[data['class'] == 0]['weight_scaled'])

    S = n_sig*tpr
    B = n_bkg*fpr
    metric = 2 * (np.sqrt(S+B) - np.sqrt(B))

    optimal_cut = thresholds[np.argmax(metric)]
    print(f"Base significance, with no cut at all = {metric[-1]}")
    print(f"Best significance = {metric[np.argmax(metric)]}")

    return metric, thresholds



def plotSignificance(data, discrim_var, xlims, save_loc, title_prefix):
    print("Plotting significance")
    metric, thresholds = getSignificance(data, discrim_var)

    optimal_cut = thresholds[np.argmax(metric)]
    print(f"First significance = {metric[0]}")

    plt.clf()
    plt.figure(figsize=(12, 10))
    hep.style.use("CMS")
    plt.plot(thresholds, metric)
    xlabel = discrim_var.split("_bp")[0]
    plt.xlabel(xlabel)
    plt.ylabel('$2 (\\sqrt{S+B} - \\sqrt{B})$')
    plt.xlim(xlims[0], xlims[1])

    plt.title(f"{title_prefix}: Max sig = {metric[np.argmax(metric)]:.2f}")
    print(f"Optimal cut at {optimal_cut}")
    plt.axvline(optimal_cut, color='black', linestyle='--')
    plt.tight_layout()
    plt.show()
    plt.savefig(save_loc)


def plotSigVsBackground(data, bp, discrim_var, bins, save_loc, title_prefix, ylims=[1, 1e7]):
    bkg_procs_list = ["tautauH", "eeH", "mumuH", "nunuH", "mumu", "ZZ", "ee", "tautau", "WW"]

    sig = getSignal(data, bp)
    bkg = getBackground(data)

    sig_var = sig[discrim_var]
    bkg_var = bkg[discrim_var]

    try:
        sig_var = ak.flatten(sig_var)
        bkg_var = ak.flatten(bkg_var)
    except:
        pass


    bkg_hists = []
    bkg_procs = []
    for proc in bkg_procs_list:
        proc_cut = bkg['process'] == proc
        
        hist, _ = np.histogram(bkg_var[proc_cut], bins=bins, weights = bkg[proc_cut].weight_scaled)
        bkg_hists.append(hist)
        bkg_procs.append(proc)

    
    plt.clf()
    plt.figure(figsize=(12, 10))
    hep.style.use("CMS")
    _ = plt.hist(sig_var, bins=bins, histtype='step', label=f'BP{bp}', weights = sig.weight_scaled, linewidth=2, color='black')
    _ = hep.histplot(bkg_hists, bins, label=bkg_procs, histtype='fill', stack=True, alpha=0.5)


    plt.legend(loc="upper center", ncol=2)
    plt.ylim(ylims[0], ylims[1])
    plt.yscale('log')
    plt.title(f"{title_prefix}: BP{bp}")
    plt.xlabel(discrim_var.split("_bp")[0])
    plt.ylabel("Counts")
    plt.show()
    plt.savefig(save_loc)



def plotTrainVsTest(train, test, bp, discrim_var, bins, title_prefix, save_loc, ylims=[1, 1e7]):
    sig_train = getSignal(train, bp)
    bkg_train = getBackground(train)
    sig_test = getSignal(test, bp)
    bkg_test = getBackground(test)

    sig_train_var = sig_train[discrim_var]
    bkg_train_var = bkg_train[discrim_var]
    sig_test_var = sig_test[discrim_var]
    bkg_test_var = bkg_test[discrim_var]

    try:
        sig_train_var = ak.flatten(sig_train_var)
        bkg_train_var = ak.flatten(bkg_train_var)
        sig_test_var = ak.flatten(sig_test_var)
        bkg_test_var = ak.flatten(bkg_test_var)
    except:
        pass

    plt.clf()
    plt.figure(figsize=(12, 10))
    hep.style.use("CMS")
    _ = plt.hist(sig_train_var, bins=bins, histtype='step', label=f'Train BP{bp}', weights = sig_train.weight_scaled, linestyle='--')
    _ = plt.hist(sig_test_var, bins=bins, histtype='step', label=f'Test BP{bp}', weights = sig_test.weight_scaled)
    _ = plt.hist(bkg_train_var, bins=bins, histtype='step', label=f'Train Background', weights = bkg_train.weight_scaled, linestyle='--')
    _ = plt.hist(bkg_test_var, bins=bins, histtype='step', label=f'Test Background', weights = bkg_test.weight_scaled)
    

    plt.legend(loc="upper center", ncol=2)
    plt.ylim(ylims[0], ylims[1])
    plt.yscale('log')
    plt.title(f"{title_prefix}: BP{bp}")
    plt.xlabel(discrim_var.split("_bp")[0])
    plt.ylabel("Counts")
    plt.show()
    plt.savefig(save_loc)


def plotROC(train_data, test_data, bp, discrim_var, save_loc, title_prefix):
    test = getSigMixedBackground(test_data, bp)
    train = getSigMixedBackground(train_data, bp)

    train_discrim = train[discrim_var]
    test_discrim = test[discrim_var]
    try: 
        train_discrim = ak.flatten(train_discrim)
        test_discrim = ak.flatten(test_discrim)
    except:
        pass


    fpr_train, tpr_train, _ = roc_curve(train['class'], train_discrim,
                                         sample_weight=train['weight_scaled'])
    fpr_test, tpr_test, _ = roc_curve(test['class'], test_discrim,
                                         sample_weight=test['weight_scaled'])
    
    train_auc = roc_auc_score(train['class'], train_discrim,
                                         sample_weight=train['weight_scaled'])
    test_auc = roc_auc_score(test['class'], test_discrim,
                                         sample_weight=test['weight_scaled'])

    plt.clf()
    plt.figure(figsize=(12, 10))
    hep.style.use("CMS")
    plt.plot([0, 1], [0, 1], color='grey', linestyle='--', label='Randomly guess')
    plt.plot(fpr_train, tpr_train, label=f'Train; AUC = {train_auc:.3f}')
    plt.plot(fpr_test, tpr_test, label=f'Test; AUC = {test_auc:.3f}')
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.title(f"{title_prefix}: BP{bp}")
    # We can make the plot look nicer by forcing the grid to be square
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig(save_loc)