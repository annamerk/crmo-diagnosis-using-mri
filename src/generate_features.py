import cv2
import numpy as np
import pandas as pd
import os
import glob
import re
import pickle
import sys
from matplotlib import pyplot as plt
from sklearn import datasets, neighbors, linear_model, model_selection, svm
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split,KFold,learning_curve, LeavePOut

FILE_EXTENSION = "tif"
DIR_NAME = "legs_folder/"
TEST_PATIENT_IDS = ['2','32','24','24b','6','7', '41']

def binary_threshold(img):
    ret, threshold = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
    return threshold

def hist_features(img_path, img_post=None, show_image=False):
    img = cv2.imread(img_path, 0) # 0 means grayscale
    if img_post is not None:
        img = img_post(img)
        if show_image:
            plt.imshow(img)
            plt.show()
    # Based on my research 256 is the value to use for full range
    hist = cv2.calcHist([img],[0],None,[256],[0,256])
    return hist.flatten()

def baseline_hist_diff(img_before, img_after, img_post=None):
    return hist_features(img_before, img_post=img_post) - hist_features(img_after, img_post=img_post)

def augment(before_paths, after_paths, labels):
    """flip pairs labeled I or R and change label."""
    new_before_paths, new_after_paths, new_labels = [], [], []
    for bp, ap, l in zip(before_paths, after_paths, labels):
        new_before_paths.append(bp)
        new_after_paths.append(ap)
        new_labels.append(l)
        if l == 'I' or l == 'R':
            flipped_label = 'R' if l == 'I' else 'I'
            new_before_paths.append(ap)
            new_after_paths.append(bp)
            new_labels.append(flipped_label)
    return (new_before_paths, new_after_paths, new_labels)

def test_paths_and_labels():
    test_bp, test_ap, test_labels, patient_ids = [], [], [], []
    for index, row in pd.read_csv('../data.csv').iterrows():
        patient_id = str(row["patient_id"])
        if patient_id not in TEST_PATIENT_IDS:
            continue

        file_1 = os.path.join('..', DIR_NAME, patient_id, '.'.join((row["scan_1"], FILE_EXTENSION)))
        file_2 = os.path.join('..', DIR_NAME, patient_id, '.'.join((row["scan_2"], FILE_EXTENSION)))

        if not (os.path.isfile(file_1) and os.path.isfile(file_2)):
            continue
        test_labels.append(row["y"])
        test_bp.append(file_1[3:])
        test_ap.append(file_2[3:])
    return (test_bp, test_ap, test_labels)

if __name__ == '__main__':
    label_type = 'binary'
    input_csv_fname = 'data_binary.csv'
    if len(sys.argv) > 1 and sys.argv[1] == 'multiclass':
        label_type = 'multiclass'
        input_csv_fname = 'data.csv'
    if len(sys.argv) > 2 and sys.argv[2] == 'include_paths':
        include_paths = True
    else:
        include_paths = False

    # whether to augment data points (the training set only)
    augment = len(sys.argv) > 3 and sys.argv[3] == 'augment'

    input_csv = pd.read_csv(input_csv_fname)
    img_hist = []
    patient_ids = []
    Y = []
    before_paths, after_paths = [], []

    for index, row in input_csv.iterrows():
        patient_id = str(row["patient_id"])
        file_1 = os.path.join(DIR_NAME, patient_id, '.'.join((row["scan_1"], FILE_EXTENSION)))
        file_2 = os.path.join(DIR_NAME, patient_id, '.'.join((row["scan_2"], FILE_EXTENSION)))

        if not (os.path.isfile(file_1) and os.path.isfile(file_2)):
            continue
        Y.append(row["y"])

        # Skip if the file does not exist (due to poor quality)

        patient_ids.append(patient_id)
        diff = baseline_hist_diff(file_1, file_2, img_post=binary_threshold)
        img_hist.append(diff)
        before_paths.append(file_1)
        after_paths.append(file_2)

    column_names = ["hist" + str(i) for i in range(256)]
    df = pd.DataFrame(img_hist, columns=column_names, index=patient_ids)
    df["y"] = Y

    if include_paths:
        df["before_path"] = before_paths
        df["after_path"] = after_paths

    data = df.loc[:, (df != 0).any(axis=0)]

    # Split the data into training set and test set.
    # DONT look at what is in the test set
    test_data = data.loc[TEST_PATIENT_IDS]
    train_data = data.loc[data.index.difference(TEST_PATIENT_IDS)]

    pickle.dump(train_data, open("train_data_%s_threshold.pkl" % label_type, "wb") )
    pickle.dump(test_data, open("test_data_%s_threshold.pkl" % label_type, "wb") )

    y = train_data["y"]
    X = train_data.drop('y', axis=1)
