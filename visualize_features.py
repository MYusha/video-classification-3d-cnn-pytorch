import numpy as np
import os
import json
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import manifold
from time import time
import pdb

class_list = ['brush_hair','cartwheel','catch','chew','clap','climb','test']

def plot_embedding(X,y,title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(X[:, 0], X[:, 1], c=y, alpha=0.75)
    plt.title(title)
    plt.show()

def main(f_dir, label_list):
    with open(f_dir) as json_file:
        data = json.load(json_file)
    print('total {} video features'.format(len(data)))
    list_label = pd.read_csv(label_list, sep=" ", header=None)
    get_label = {}
    for idx, row in list_label.iterrows():
        get_label[os.path.basename(row[0])]=row[1]
    X = []
    y = []
    # pdb.set_trace()
    for result in data:
        vid_name = result['video_name']
        X.append(result['feature'])
        label = get_label[vid_name]
        # label = list_label[1][list_label[0].str.split('/')[1]==(vid_name)].item()
        y.append(class_list.index(label))
    X = np.vstack(X)
    print('original feature of shape {}'.format(str(X.shape)))
    print("Computing t-SNE embedding")
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    starts = time()
    X_tsne = tsne.fit_transform(X)
    plot_embedding(X_tsne,y,"tsne embedding of training features")
    exc_time = time() - starts
    print("--- %s seconds ---" % exc_time)





if __name__ == "__main__":
    f_dir = sys.argv[1]
    label_list = sys.argv[2]
    main(f_dir,label_list)