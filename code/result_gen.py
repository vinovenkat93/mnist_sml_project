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
        fpr[i], tpr[i], _ = metrics.roc_curve(y_true, y_score[:,i], pos_label = i)
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
    