import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from scipy import interp
import itertools

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
    
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    tps = np.cumsum(y_true, dtype=np.float64)[threshold_idxs]
    fps = 1 + threshold_idxs - tps
    
    fpr = fps / fps[-1]
    tpr = tps / tps[-1]
    
    return fpr, tpr, y_score[threshold_idxs]
    
def confusion_matrix_plot(conf_mat, classes, filename):
    conf_mat = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]
    
    plt.figure()
    plt.imshow(conf_mat, interpolation='nearest', cmap=plt.cm.Reds)
    plt.title('Confusion matrix', fontweight='bold')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    fmt = '.2f'
    thresh = conf_mat.max() / 2.
    for i, j in itertools.product(range(conf_mat.shape[0]), range(conf_mat.shape[1])):
        plt.text(j, i, format(conf_mat[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if conf_mat[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(filename, dpi=600)
#    plt.show()

def plot_accuracy(accuracy_file_name, filename, plot_no):
    accuracy_values = np.loadtxt(accuracy_file_name , delimiter=',')
    '''
    accuracy_values_2 = np.loadtxt(accuracy_file_name_1 , delimiter=',')
    accuracy_values = np.zeros((accuracy_values_1.shape[0] + accuracy_values_2.shape[0], accuracy_values_1.shape[1]))
    accuracy_values[0,:] = accuracy_values_1[0,:]
    accuracy_values[1:4,:] = accuracy_values_2
    accuracy_values[4:] = accuracy_values_1[1:]
    '''
    
    if (plot_no == 1):
        n = accuracy_values.shape[1]
        alpha = np.logspace(-4,3,8)
        
        plt.figure()
        plt.errorbar(alpha,accuracy_values.mean(axis=1), yerr=accuracy_values.std(axis=1)/np.sqrt(n),
                     label='Accuracy')
        plt.grid(linestyle='--')
        plt.xscale('log')
        plt.gca().set(xlabel=r'$\alpha$ (regularization)', ylabel='Mean Accuracy (10-fold CV)')
        plt.title(r'$\alpha$ (regularization) optimization using the 10-fold CV') 
        plt.savefig(filename, dpi = 600)
        plt.show()
    
    elif (plot_no == 2):  
        n = accuracy_values.shape[1]
        act_fcns = ['logistic','tanh', 'relu']
        x = np.arange(accuracy_values.shape[0])
        
        fig, ax = plt.subplots(1,1)
        ax.set_xticks(np.arange(0,3,1))
        ax.set_xticklabels(act_fcns, rotation=20)
        ax.errorbar(x, accuracy_values.mean(axis=1), yerr=accuracy_values.std(axis=1)/np.sqrt(n),
                     label='Accuracy')
        ax.grid(linestyle='--')
        ax.set_title('Choosing activation function using the 10-fold CV')
        ax.set_ylabel('Mean accuracy') 
        plt.savefig(filename, dpi = 600)
        plt.show()
    
    elif (plot_no == 3):
        n = accuracy_values.shape[1]
        learn_rate = np.arange(0.0001, 0.0021, 0.0001)
        
        plt.figure()
        plt.errorbar(learn_rate,accuracy_values.mean(axis=1), yerr=accuracy_values.std(axis=1)/np.sqrt(n),
                     label='Accuracy')
        plt.grid(linestyle='--')
        plt.gca().set(xlabel=r'Learning rate ($\eta$)', ylabel='Mean Accuracy (10-fold CV)')
        plt.title(r'Learning rate ($\eta$) optimization using the 10-fold CV') 
        plt.savefig(filename, dpi = 600)
        plt.show()
        
    elif(plot_no == 4):      
        n = accuracy_values.shape[1]
        mean_accuracy = accuracy_values.mean(axis=1)
        yerr_std = accuracy_values.std(axis=1)/np.sqrt(n)
        C = np.logspace(-2,2,num=5)
        C = np.append(C,np.array([2,5,20,40,50,60,70,80,90]))
        
        mean_accuracy_ord = np.array([])
        mean_accuracy_ord = np.concatenate((mean_accuracy[:3],mean_accuracy[5:7],mean_accuracy[7:]))
        mean_accuracy_ord = np.insert(mean_accuracy_ord, 5,mean_accuracy[3])
        mean_accuracy_ord = np.insert(mean_accuracy_ord, 13,mean_accuracy[4])
        var_accuracy_ord = np.array([])
        var_accuracy_ord = np.concatenate((yerr_std[:3],yerr_std[5:7],yerr_std[7:]))
        var_accuracy_ord = np.insert(var_accuracy_ord, 5,yerr_std[3])
        var_accuracy_ord = np.insert(var_accuracy_ord, 13,yerr_std[4])
        C_ord = np.array([])
        C_ord = np.concatenate((C[:3],C[5:7],C[7:]))
        C_ord = np.insert(C_ord, 5,C[3])
        C_ord = np.insert(C_ord, 13,C[4])
        plt.figure()
        plt.grid(linestyle='--')
        plt.errorbar(C_ord,mean_accuracy_ord, yerr=var_accuracy_ord,
                     label='Accuracy')
        plt.plot(C_ord,mean_accuracy_ord)
        plt.gca().set(xlabel='C (regularization)', ylabel='Mean Accuracy (10-fold CV)')
        plt.title('C (regularization) optimization using the 10-fold CV')    
        plt.savefig(filename, dpi = 600)
        plt.show()
    
def main():

    '''    
    Neural Net Plots
    '''
    # Parameter tuning
    csv_filename_1 = "../experiments/expNN/neural_net_accuracy_vs_alpha.csv"
    filename_1 = '..\Results_Plots\Neural_net_alpha_mean_acc_error_bar.png'
    
    csv_filename_2 = "../experiments/expNN/neural_net_accuracy_vs_activation_fcn.csv"
    filename_2 = '..\Results_Plots\Neural_net_act_fcn_mean_acc_error_bar.png'
    
    csv_filename_3 = "../experiments/expNN/neural_net_accuracy_vs_learn_rate.csv"
    filename_3 = '..\Results_Plots\Neural_net_learn_rate_mean_acc_error_bar.png'

    plot_accuracy(csv_filename_1, filename_1, 1)
    plot_accuracy(csv_filename_2, filename_2, 2)
    plot_accuracy(csv_filename_3, filename_3, 3)
    
    '''    
    SVM Plots
    '''
    csv_filename_4 = "../experiments/expSVM/svm_accuracy_vs_C.csv"
    filename_4 = '..\Results_Plots\SVM_regularization_C_mean_acc_error_bar.png'

    plot_accuracy(csv_filename_4, filename_4, 4)

    
if __name__ == "__main__":
    main()