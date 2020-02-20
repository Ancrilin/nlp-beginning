import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors
from TextClassifier.TextCNN import Config
import torch.nn.functional as F
import logging
import testini
import argparse
import os
import json
import nltk
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import itertools
import pandas as pd
import logging
from TextClassifier.TextESIM import Config, Model
from torch.utils.data.dataloader import DataLoader
from utils.dataset import MyDataset
from utils.util import data_processor, get_time_dif, get_pre_embedding
from run import test


def plot_confusion_matrix(y_true, y_pred, classes, cmap=plt.cm.Blues):
    """Plot a confusion matrix using ground truth and predictions."""
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    #  Figure
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    fig.colorbar(cax)

    # Axis
    plt.title("Confusion matrix")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    ax.set_xticklabels([''] + classes)
    ax.set_yticklabels([''] + classes)
    ax.xaxis.set_label_position('bottom')
    ax.xaxis.tick_bottom()

    # Values
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, f"{cm[i, j]:d} ({cm_norm[i, j]*100:.1f}%)",
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    # Display
    plt.show()

if __name__ == '__main__':
    x1 = torch.tensor([1, 2, 3, 4, 2])
    mask = (torch.tensor([2]) != x1).data.numpy()
    print(mask)
    sim = [0.1, 0.5, 0.6, 0.3, 0.8]
    sim = np.array(sim)
    threshold = 0.5
    mask *= (sim >= threshold)
    print(mask)
    print(sim < threshold)
    print(1 - mask)
    result = torch.from_numpy(
        (sim < threshold) + (1 - mask).astype(float)).float()
    print(result)
