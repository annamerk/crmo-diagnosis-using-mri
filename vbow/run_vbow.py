"""Generate all vbow models, datasets, and clusters"""
import cv2
import numpy as np
import pandas as pd
import os
import glob
import re
import pickle
from matplotlib import pyplot as plt
import sklearn
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn import datasets, neighbors, linear_model, model_selection, svm
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, KFold, learning_curve, LeavePOut, GridSearchCV, StratifiedKFold

# import private libraries with a dumb hack
import sys
sys.path.append("..")
from src import generate_features, utils, vision

def run():
    train_bp, train_ap, train_labels = generate_features.train_paths_and_labels()
    # test_bp, test_ap, test_labels = generate_features.train_paths_and_labels()
    bp, ap, labels = generate_features.augment(train_bp, train_ap, train_labels)
    for model in ['nb', 'svm']:
        for multi_class in [True, False]:
            if not multi_class:
                train_labels = [l if l == 'I' else 'SR' for l in train_labels]
            for feature_type in ['surf', 'sift', 'orb']:
                for num_clusters in [50, 100, 200, 500]:
                    features = vision.get_all_features(bp, ap, feature_type=feature_type)
                    vision.vbow_kmeans(
                        features,
                        num_clusters,
                        bp,
                        ap,
                        labels,
                        model=model,
                        feature_type=feature_type,
                        cv_method='kfold',
                        write_kmeans=True,
                        multi_class=multi_class,
                        save_img=True)


if __name__=='__main__':
    run()
