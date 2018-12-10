"""Generate all vbow models, datasets, and clusters"""
from collections import namedtuple
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
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, KFold, learning_curve, LeavePOut, GridSearchCV, StratifiedKFold

# import private libraries with a dumb hack
import sys
sys.path.append("..")
from src import generate_features, utils, vision

Model = namedtuple('Model', ['model', 'multi_class', 'feature_type', 'num_clusters'])

def run():
    train_bp, train_ap, train_labels = generate_features.train_paths_and_labels()
    test_bp, test_ap, test_labels = generate_features.test_paths_and_labels()
    bp, ap, labels = generate_features.augment(train_bp, train_ap, train_labels)
    best_f1_scores = {'multi_class': 0, 'binary': 0}
    best_models = {'multi_class': None, 'binary': None}
    for model in ['svm', 'nb']:
        for multi_class in [True, False]:
            # if not multi_class:
            #     train_labels = [l if l == 'I' else 'SR' for l in train_labels]
            for feature_type in ['sift', 'orb']:
                for num_clusters in [50, 100, 200, 500]:
                    features = vision.get_all_features(bp, ap, feature_type=feature_type)
                    estimator, score = vision.vbow_kmeans(
                        features,
                        num_clusters,
                        bp + test_bp,
                        ap + test_ap,
                        labels + test_labels,
                        model=model,
                        feature_type=feature_type,
                        cv_method='kfold',
                        write_kmeans=True,
                        multi_class=multi_class,
                        save_img=True)
                    if multi_class and score > best_f1_scores['multi_class']:
                        best_models['multi_class'] = Model(model, multi_class, feature_type, num_clusters)
                        best_f1_scores['multi_class'] = score
                        print("F1 SCORE FOR MULTI_CLASS: %s" % score)
                        print(best_models['multi_class'])
                    elif not multi_class and score > best_f1_scores['binary']:
                        best_models['binary'] = Model(model, multi_class, feature_type, num_clusters)
                        best_f1_scores['binary'] = score
                        print("F1 SCORE FOR binary: %s" % score)
                        print(best_models['binary'])
                print ("=========BEST MODELS AND F1 SCORES SO FAR=======")
                print(best_models)
                print(best_f1_scores)
    print(best_models)
    print(best_f1_scores)


def run_high_performers():
    train_bp, train_ap, train_labels = generate_features.train_paths_and_labels()
    test_bp, test_ap, test_labels = generate_features.test_paths_and_labels()
    bp, ap, labels = generate_features.augment(train_bp, train_ap, train_labels)
    high_performers = [
        # Model(model='nb', multi_class=True, feature_type='orb', num_clusters=200),
        # Model(model='nb', multi_class=True, feature_type='orb', num_clusters=500),
        # Model(model='nb', multi_class=True, feature_type='sift', num_clusters=100),
        # Model(model='nb', multi_class=True, feature_type='sift', num_clusters=100),
        # Model(model='nb', multi_class=True, feature_type='sift', num_clusters=200),
        Model(model='nb', multi_class=True, feature_type='sift', num_clusters=200),
        Model(model='nb', multi_class=False, feature_type='sift', num_clusters=100),
        # Model(model='nb', multi_class=True, feature_type='surf', num_clusters=100),
        # Model(model='svm', multi_class=False, feature_type='orb', num_clusters=50),

    ]
    new_high_performers = [
        Model(model='nb', multi_class=False, feature_type='sift', num_clusters=100),
    ]
    for model in high_performers:
        features = vision.get_all_features(bp, ap, feature_type=model.feature_type)
        best_estimator = vision.vbow_kmeans(
            features,
            model.num_clusters,
            bp + test_bp,
            ap + test_ap,
            labels + test_labels,
            model=model.model,
            feature_type=model.feature_type,
            cv_method='kfold',
            write_kmeans=True,
            multi_class=model.multi_class,
            save_img=True)

if __name__=='__main__':
    run_high_performers()
    # run()
