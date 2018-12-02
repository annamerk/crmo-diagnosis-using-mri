import cv2
import numpy as np

import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import seaborn as sn
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.svm import SVC

def show_incorrect_images(model, x_test, y_test):
    """imshow before and after images for images where model predicted
    incorrectly. x_test should have before_path and after_path columns which
    contain directory paths to images"""
    before_path = x_test['before_path']
    after_path = x_test['after_path']
    x_test = x_test.drop('after_path', axis=1).drop('before_path', axis=1)
    incorrect_indices = np.logical_not(model.predict(x_test) == y_test)
    incorrect_before = before_path[incorrect_indices]
    incorrect_after = after_path[incorrect_indices]
    incorrect_true_labels = y_test[incorrect_indices]
    for before_img, after_img, true_label in zip(incorrect_before, incorrect_after, incorrect_true_labels):
        print("Model predicted incorrectly. True label is %s" % true_label)
        print("Before: %s" % before_img)
        plt.imshow(cv2.imread(before_img, 0))
        plt.show()
        print("After: %s" % after_img)
        plt.imshow(cv2.imread(after_img, 0))
        plt.show()

def generate_validation_curve(estimator, X, y, param_name, param_range, cv,
    scoring, n_jobs, title, xlabel):

    train_scores, test_scores = model_selection.validation_curve(
        estimator, X, y, param_name=param_name, param_range=param_range,
        cv=cv, scoring=scoring, n_jobs=n_jobs)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Accuracy")
    plt.ylim(0.0, 1.1)
    lw = 2
    plt.semilogx(param_range, train_scores_mean, label="Training accuracy",
               color="darkorange", lw=lw)
    plt.fill_between(param_range, train_scores_mean - train_scores_std,
                   train_scores_mean + train_scores_std, alpha=0.2,
                   color="darkorange", lw=lw)
    plt.semilogx(param_range, test_scores_mean, label="Cross-validation accuracy",
               color="navy", lw=lw)
    plt.fill_between(param_range, test_scores_mean - test_scores_std,
                   test_scores_mean + test_scores_std, alpha=0.2,
                   color="navy", lw=lw)
    plt.legend(loc="best")
    plt.show()

def plot_confusion_matrix(y_test,y_pred):
    df_cm = confusion_matrix(y_test,y_pred)#,labels=["I", "S", "R"])
    sn.set(font_scale=1.4)#for label size
    sn.heatmap(df_cm, annot=True,annot_kws={"size": 16})

#def plot_roc_multiclass():

#def plot_roc_binary():

def do_CV(X,y, model, multi_class=True):
    # Change to 2-class
    if not multi_class:
        y = y.replace('S', 'R')
    # Split the dataset in two equal parts
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0)

    print("# Tuning hyper-parameter")
    print()

    model.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(model.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = model.cv_results_['mean_test_score']
    stds = model.cv_results_['std_test_score']
    for mean, std, p in zip(means, stds, model.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, p))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    plot_confusion_matrix(y_test, y_pred)
    if multi_class == False:
        my_dict = {'I':1, 'R':-1}
        print("ROC AUC score")
        print(roc_auc_score(np.vectorize(my_dict.get)(y_test), np.vectorize(my_dict.get)(y_pred)))
    print()

    print("This is the classification report for the training set:")
    y_train_pred = model.predict(X_train)
    print(classification_report(y_train, y_train_pred))
