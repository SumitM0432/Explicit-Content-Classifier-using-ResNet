from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.figsize'] = [8, 6]

def metric_scores(y_true, y_pred):

    cm = metrics.confusion_matrix(y_true, y_pred)

    ax = sns.heatmap(cm, annot = True, cmap = 'YlGnBu', fmt='.2f')
    ax.set(title = "Confusion Matrix", xlabel = 'Predicted Labels', ylabel = 'True Labels')

    cls_report = metrics.classification_report(y_true, y_pred)
    
    print ("")
    print (f"Accuracy : {metrics.accuracy_score(y_true, y_pred)*100 : .3f} %") 
    print ("")
    print ("Classification Report : ")
    print (cls_report)

    plt.show()