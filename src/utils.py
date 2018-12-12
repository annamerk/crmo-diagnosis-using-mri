import cv2
import numpy as np
import operator

from matplotlib import pyplot as plt
import seaborn as sn
import pandas as pd
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, auc, classification_report, confusion_matrix
from itertools import cycle
from sklearn.preprocessing import label_binarize
from scipy import interp
from sklearn.svm import SVC

TEST_KEYS = ['6/10-23-14/11-16-17','7/4-10-12/5-28-15','2/12-11-12/12-17-13',
                     '24/3-11-14/8-14-14','24b/3-11-14/8-14-14','32/12-17-15/1-5-17',
                     '41/1-25-11/8-7-12']
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
    scoring, n_jobs, title, xlabel, save_img=False, img_name=None):

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
    if save_img:
        plt.savefig('./cv_images/' + img_name + '_validation_curve.png', dpi=200)
    else:
        plt.show()

def plot_confusion_matrix(y_test,y_pred, save_img=False, img_name=None):
    class_names = np.unique(y_test)
    df_cm = pd.DataFrame(
        confusion_matrix(y_test,y_pred), index=class_names, columns=class_names, 
    )
    #sn.set(font_scale=1.4)#for label size
    plt.clf()
    sn.heatmap(df_cm, annot=True,annot_kws={"size": 16})
    if save_img:
        plt.savefig('./cv_images/' + img_name + '_confusion_matrix.png', dpi=200)
    else:
        plt.show()

def do_CV(X,y, model, multi_class=True, test_size=0.3, show_incorrect=False,
          save_img=False, img_name=None):
    # Change to 2-class
    if not multi_class:
        y = y.replace('S', 'SR')
        y = y.replace('R', 'SR')
    # Split the dataset in two equal parts
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=0, stratify=y)

    print("# Tuning hyper-parameter")
    print()

    model.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    if hasattr(model, "best_params_"):
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
    plot_confusion_matrix(y_test, y_pred, save_img=save_img, img_name=img_name)
    if multi_class == False:
        my_dict = {'I':1, 'SR':-1}
        print("ROC AUC score")
        vectorized = np.vectorize(my_dict.get)(y_test)
        # sort classes by label value so they match up with vectorized
        classes = sorted(my_dict.items(), key=operator.itemgetter(0))
        print(roc_auc_score(vectorized, model.predict_proba(X_test)[:, 0]))
        plot_roc_binary(vectorized, model.predict_proba(X_test), classes,
                        save_img=save_img, img_name=img_name)
    else:
        classes = sorted(np.unique(y_test))
        plot_roc_multi(y_test, model.predict_proba(X_test),
                       classes, save_img=save_img,
                       img_name=img_name)
    print()
    print("This is the classification report for the eval set:")
    print(classification_report(y_test, y_pred))

    print("This is the classification report for the training set:")
    y_train_pred = model.predict(X_train)
    print(classification_report(y_train, y_train_pred))

    # Not using actual function here since this is being run on a google compute w.o. the images
    if show_incorrect:
        incorrect_indices = np.logical_not(model.predict(X_test) == y_test)
        print("Misclassified labels")
        print(X_test.index[incorrect_indices])
        print("Predicted class")
        print(model.predict(X_test)[incorrect_indices])
        print("Actual class")
        print(y_test[incorrect_indices])
    return model.best_estimator_

def plot_roc_binary(y, y_score,classes, save_img=False, img_name=None):
    fpr, tpr, _ = roc_curve(y, y_score[:,0])

    roc_auc = auc(fpr, tpr)
    
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve class {0} (area = {1:0.2f})' ''.format(classes[0], roc_auc))
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    if save_img:
        plt.savefig('./cv_images/' + img_name + '_roc_binary.png', dpi=200)
    else:
        plt.show()

def plot_roc_multi(y_true, y_score, classes, save_img=False, img_name=None):
    y = label_binarize(y_true, classes=classes)
    n_classes = len(classes)

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    lw = 2
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(classes[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], '', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    if save_img:
        plt.savefig('./cv_images/' + img_name + '_roc_multi.png', dpi=200)
    else:
        plt.show()
