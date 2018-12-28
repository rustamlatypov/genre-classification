import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.covariance import EllipticEnvelope
from numpy import linalg as LA
from statsmodels import robust
from sklearn.metrics import roc_auc_score
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import itertools
from sklearn import preprocessing


def matrix(filename, include_header):
    if include_header == 0:
        df = pd.read_csv(filename, header=None)
    else:
        df = pd.read_csv(filename)
    data = df.values
    m = np.array(data)
    return m


def vector(filename, include_header):
    if include_header == 0:
        df = pd.read_csv(filename, header=None)
    else:
        df = pd.read_csv(filename)
    data = df.values
    label = np.array(data.reshape(data.shape[0]))
    return label

def plotting(data, clusters):
    # this function will later on be used for plotting the clusters and centroids. But now we use it to just make a scatter plot of the data
    # Input: the data as an array, cluster means (centroids), cluster assignemnts in {0,1,...,k-1}
    # Output: a scatter plot of the data in the clusters with cluster means
    plt.style.use('seaborn-notebook')

    cmpd = ['orangered', 'dodgerblue', 'springgreen']
    alp = 0.5  # data alpha
    dt_sz = 20  # data point size
    plt.scatter(data[:, 0], data[:, 1], c=[cmpd[i] for i in clusters], s=dt_sz, alpha=alp)


# input: training data, contamination factor
# output: prediction vector (0 = outlier, 1 = inlier)
def outlier(TRAIN, contam):
    
    for i in range(TRAIN.shape[1]):
        v = TRAIN[:,i]
        v_hat = (v - np.median(v))
        TRAIN[:,i] = v_hat
        
    # model creation
    clf = EllipticEnvelope(support_fraction=1., contamination=contam, assume_centered=True)
    clf.fit(TRAIN)
    C = clf.correct_covariance(TRAIN)
    pred = clf.predict(TRAIN)

    # eigen decomposition
    E, U = LA.eig(C)
    P = U[0:2, :]
    X_hat = np.dot(TRAIN, np.transpose(P))

    # plotting
    pred += 1
    for i in range(pred.shape[0]):
        pred[i] = pred[i] // 2
    plotting(X_hat, pred)

    return pred


def plot_confusion_matrix(y_true, y_pred, doplot, normalize=False, classes=(1,2,3,4,5,6,7,8,9,10)):

    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    fmt = '.2f' if normalize else 'd'
    cmap = plt.cm.Blues
    plt.figure(figsize=(15,7.5))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title('Confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    if doplot == 1: plt.show()


def to_percent(y, position):
    # Ignore the passed in position. This has the effect of scaling the default
    # tick locations.
    s = str(round(100 * y,3))

    # The percent symbol needs escaping in latex
    if matplotlib.rcParams['text.usetex'] is True:
        return s + r'$\%$'
    else:
        return s + '%'
    

def plot_gsearch(clf, xlim, ylim, title, scoring):
    # Plots gridSearchCVs (clf is fitted gridsearch model) cross validation results

    results = clf.cv_results_

    plt.figure(figsize=(8,5))
    plt.title(title)
    plt.xlabel("C")
    plt.ylabel("Prediction rate")

    # Manually alter limits
    ax = plt.gca()
    x0 = xlim[0]
    x1 = xlim[1]
    y0 = ylim[0]
    y1 = ylim[1]
    ax.set_xlim(x0,x1)
    ax.set_ylim(y0,y1)

    # Get the regular numpy array from the MaskedArray
    # Choose 'param_[altered parameter]' accordingly
    X_axis = np.array(results['param_C'].data, dtype=float)
           
    for scorer, color in zip(sorted(scoring), ['g', 'k']):
        for sample, style in (('train', '--'), ('test', '-')):
            sample_score_mean = results['mean_%s_%s' % (sample, scorer)]
            sample_score_std = results['std_%s_%s' % (sample, scorer)]
            ax.fill_between(X_axis, sample_score_mean - sample_score_std,
                            sample_score_mean + sample_score_std,
                            alpha=0.1 if sample == 'test' else 0, color=color)
            ax.plot(X_axis, sample_score_mean, style, color=color,
                    alpha=1 if sample == 'test' else 0.7,
                    label="%s (%s)" % (scorer, sample))

        best_index = np.nonzero(results['rank_test_%s' % scorer] == 1)[0][0]
        best_score = results['mean_test_%s' % scorer][best_index]

        # Plot a dotted vertical line at the best score for that scorer marked by x
        ax.plot([X_axis[best_index], ] * 2, [0, best_score],
                linestyle='-.', color=color, marker='x', markeredgewidth=3, ms=8)

        # Annotate the best score for that scorer
        ax.annotate("%0.2f" % best_score,
                    (X_axis[best_index], best_score + 0.005))

    plt.legend(loc="best")
    plt.grid(False)
    plt.show()
   