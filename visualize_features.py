import numpy as np
import os
import json
import sys
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import manifold
from time import time
import pdb
from matplotlib import offsetbox

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

class_list = ['golf','spiking','tennis','swing','walk_dog','jumping','riding','diving','juggle','shooting','biking']
# class_list = ['brush_hair','cartwheel','catch','chew','clap','climb','test']

def plot_embedding(X,y,cutoff, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    sc = ax.scatter(X[:, 0], X[:, 1], c=y,cmap='tab10', alpha=0.75)
    for i in range(0,cutoff,25):
            plt.text(X[i, 0], X[i, 1], str(class_list[y[i]]),color=sc.to_rgba(y[i]),fontdict={'weight': 'bold', 'size': 10})
    for i in range(cutoff,X.shape[0],10):
            plt.text(X[i, 0], X[i, 1], str(class_list[y[i]]),color='red',fontdict={'weight': 'bold', 'size': 9})
    if title is not None:
        plt.title(title)

    plt.title(title)
    plt.show()

def stack_feature(f_dir,label_list):
    with open(f_dir) as json_file:
        data = json.load(json_file)
    print('total {} video features'.format(len(data)))
    list_label = pd.read_csv(label_list, sep=" ", header=None, dtype=str)
    get_label = {}
    for idx, row in list_label.iterrows():
        get_label[os.path.basename(row[0])]=row[1]
    X = []
    y = []
    for result in data:
        vid_name = str(result['video_name'])
        X.append(result['feature'])
        label = get_label[vid_name]
        # label = list_label[1][list_label[0].str.split('/')[1]==(vid_name)].item()
        y.append(class_list.index(label))
    X = np.vstack(X)
    return X, y

def sklearn_predict(clf, X_train, y_train, X_test, y_test):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc_NN = np.sum(np.equal(y_pred, y_test)) * 1.0 / len(y_pred)
    return acc_NN


def main(arg):
    f_train = arg[1]
    f_test = arg[2]
    label_list_train = arg[3]
    label_list_test = arg[4]
    print("loading feature {}".format(str(f_train)))
    X_train, y_train = stack_feature(f_train,label_list_train)
    X_test, y_test = stack_feature(f_test,label_list_test)

    # components = 100
    # pca = PCA(n_components=components).fit(X_train)
    # X_train = pca.transform(X_train)
    # X_test = pca.transform(X_test)
    # print("PCA explained ratio is {}".format(np.sum(pca.explained_variance_ratio_[:components])))

    neigh = KNeighborsClassifier(n_neighbors=5)
    acc = sklearn_predict(neigh, X_train, y_train, X_test, y_test)
    print("accuracy of nearest neighbour(N=5) classifiers is {}".format(acc))

    svc = SVC(kernel='rbf')
    acc = sklearn_predict(svc, X_train, y_train, X_test, y_test)
    print("accuracy of SVC classifiers with rbf kernel is {}".format(acc))

    clf = RandomForestClassifier()
    acc = sklearn_predict(clf, X_train, y_train, X_test, y_test)
    print("accuracy of random forest classifiers is {}".format(acc))

    # clfboost = GradientBoostingClassifier()
    # acc = sklearn_predict(clfboost, X_train, y_train, X_test, y_test)
    # print("accuracy of gradient boost classifiers is {}".format(acc))

    X = np.vstack([X_train,X_test])
    y = y_train+y_test
    print('original feature of shape {}'.format(str(X.shape)))
    print("Computing t-SNE embedding")
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    starts = time()
    X_tsne = tsne.fit_transform(X)
    plot_embedding(X_tsne,y,X_train.shape[0],"tse 粗embedding of training features")
    exc_time = time() - starts
    print("--- %s seconds ---" % exc_time)





if __name__ == "__main__":
    main(sys.argv)