def plot_ROC_curve(y_true,y_score):    
    mean_tpr = 0.0
    mean_fpr = np.linspace(0,1,100)
    for i in xrange(10):
        fpr[i], tpr[i] = roc_curve_params(y_true, y_score, pos_label = i)
        mean_tpr += interp(mean_fpr,fpr[i],tpr[i])
        mean_tpr[0] = 0
 
    mean_tpr /= 10
    mean_tpr[-1] = 1