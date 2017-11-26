import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from scipy import interp

def plot_ROC_curve(y_true,y_score, filename, title, all_class=False):
    fpr = dict()
    tpr = dict()
    plt.figure()
    
    mean_tpr = 0.0
    mean_fpr = np.linspace(0,1,100)
    for i in xrange(10):
#        fpr[i], tpr[i], _ = metrics.roc_curve(y_true, y_score[:,i], pos_label = i)
        fpr[i], tpr[i], _ = roc_curve_params(y_true, y_score[:,i], pos_label = i)
        mean_tpr += interp(mean_fpr,fpr[i],tpr[i])
        mean_tpr[0] = 0
        if all_class:
            plt.plot(fpr[i],tpr[i], label = 'ROC: class {}'.format(i))
    
    roc_auc = metrics.auc(mean_fpr,mean_tpr)    
    plt.plot([0,1],[0,1],'--', label = "Random guess")
    mean_tpr /= 10
    mean_tpr[-1] = 1
    if all_class:
        plt.plot(mean_fpr,mean_tpr, 'k--', label = 'Mean - ROC for all classes (AUC: {0:.2f})'.format(roc_auc))
    else:
        plt.plot(mean_fpr,mean_tpr, label = 'Mean - ROC for all classes (AUC: {0:.2f})'.format(roc_auc))
    plt.legend(loc="lower right", fontsize="5")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title, fontweight = 'bold')
    plt.savefig(filename, dpi = 600)
    
    
def roc_curve_params(y_true, y_score, pos_label):
    # make y_true a boolean vector
    y_true = (y_true == pos_label)

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]
    weight = 1.

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    tps = np.cumsum(y_true * weight, dtype=np.float64)[threshold_idxs]
    fps = 1 + threshold_idxs - tps
    
    fpr = fps / fps[-1]
    tpr = tps / tps[-1]
    
    return fpr, tpr, y_score[threshold_idxs]
    
    