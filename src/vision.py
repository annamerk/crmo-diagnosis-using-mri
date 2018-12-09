import cv2
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import pickle
import random
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV, LeavePOut, KFold, StratifiedKFold, train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC

from .generate_features import TEST_KEYS, train_paths_and_labels
from .utils import do_CV, _do_CV


def detect_and_draw_orb(img, draw_img=True):
    orb = cv2.ORB_create()
    # Initiate ORB detector
    # find the keypoints with ORB
    kp = orb.detect(img,None)
    # compute the descriptors with ORB
    kp, des = orb.compute(img, kp)
    # draw only keypoints location,not size and orientation
    kps_img = cv2.drawKeypoints(img, kp, None, color=(0,255,0), flags=0)
    if draw_img:
        plt.imshow(kps_img), plt.show()
    return kp, des, kps_img

def detect_and_draw_sift(img, draw_img=False):
    sift = cv2.xfeatures2d.SIFT_create()
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (kp, des) = sift.detectAndCompute(img, None)
    kps_img = cv2.drawKeypoints(img, kp, None, color=(0,255,0), flags= cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    if draw_img:
        plt.imshow(img), plt.show()
    return kp, des, kps_img

def detect_and_draw_surf(img, draw_img=False):
    surf = cv2.xfeatures2d.SURF_create(400)
    kp, des = surf.detectAndCompute(img,None)
    kps_img = cv2.drawKeypoints(img, kp, None, color=(0,255,0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    if draw_img:
        plt.imshow(img), plt.show()
    return kp, des, kps_img


def detect_and_draw(path=None, feature_type='orb', draw_img=True, path_prefix='..', img_=None):
    if img_ is None:
        path = '/'.join((path_prefix, path))
        img = cv2.imread(path, 0)
    else:
        img = img_
    if feature_type == 'orb':
        return detect_and_draw_orb(img, draw_img)
    elif feature_type == 'sift':
        return detect_and_draw_sift(img, draw_img)
    elif feature_type == 'surf':
        return detect_and_draw_surf(img, draw_img)

def extract_vbow_dataset(before_paths, after_paths, labels, kmeans_clusters, num_clusters=50,
                         feature_type='orb', path_prefix='..'):
    """Use this to generate the 'one-hot' style feature vector once you already have the kmeans clusters."""
    data = []
    indices = []
    for before_path, after_path, label in zip(before_paths, after_paths, labels):
        feature_vec = np.zeros(2 * num_clusters)
        before_kp, before_des, before_img = detect_and_draw(before_path, feature_type=feature_type, draw_img=False, path_prefix=path_prefix)
        after_kp, after_des, after_img = detect_and_draw(after_path, feature_type=feature_type, draw_img=False, path_prefix=path_prefix)
        before_clusters = kmeans_clusters.predict(before_des)
        after_clusters = kmeans_clusters.predict(after_des)
        for c in before_clusters:
            feature_vec[c] += 1
        for c in after_clusters:
            feature_vec[num_clusters + c] += 1
        data.append(feature_vec)
        _, patient_id, before_fname = before_path.split('/')
        _, _, after_fname = after_path.split('/')
        indices.append('/'.join((patient_id, before_fname, after_fname)))
    data = np.array(data)
    X = pd.DataFrame(data, index=indices)
    y = pd.DataFrame(labels, index=indices)
    return X, y

def vbow_kmeans_with_noise(before_paths, after_paths, labels, path_prefix='..', augment_factor=0,
        feature_type='orb', num_clusters=50, model='svm', multi_class=False):
    """This does the entire vbow pipeline with noise added."""
    # split out the cross validation set first so that it doesn't have augmented images
    bp, ap, y = train_paths_and_labels()
    df = pd.DataFrame({'bp': bp, 'ap': ap, 'y': y})
    df = df.reset_index()
    X_train, X_test, y_train, y_test = train_test_split(
        df, y,
        test_size=0.3,
        random_state=0,
        stratify=y)

    # read the images one at a time
    all_images = []
    for i, row in X_train.iterrows(): # p, ap, yi in zip(before_paths, after_paths, labels):
        bp, ap = row['bp'], row['ap']
        yi = y[i]
        before_img = cv2.imread('/'.join((path_prefix, bp)))
        after_img = cv2.imread('/'.join((path_prefix, ap)))
        all_images.append((before_img, after_img, yi))
        for i in range(augment_factor):
            before_img_noisy = add_noise(before_img)
            after_img_noisy = add_noise(after_img)
            all_images.append((before_img_noisy, after_img_noisy, yi))

    # ok, now all_images contains a ton of images. now we build the feature frame
    all_features = pd.DataFrame()
    for before_img, after_img, yi in all_images:
        kp, des, _ = detect_and_draw(
            feature_type=feature_type,
            draw_img=False,
            img_=before_img)
        all_features = all_features.append(pd.DataFrame(des))
        kp, des, _ = detect_and_draw(
            feature_type=feature_type,
            draw_img=False,
            img_=after_img)
        all_features = all_features.append(pd.DataFrame(des))

    # ok now all_features contains all the features
    # so now we build up our feature vectors.
    kmeans_clusters = KMeans(n_clusters=num_clusters).fit(all_features)
    data = []
    labels = []
    for before_img, after_img, yi in all_images:
        feature_vec = np.zeros(2 * num_clusters)
        kp, before_des, _ = detect_and_draw(
            feature_type=feature_type,
            draw_img=False,
            img_=before_img)
        kp, after_des, _ = detect_and_draw(
            feature_type=feature_type,
            draw_img=False,
            img_=after_img)
        all_features.append(pd.DataFrame(des))
        before_clusters = kmeans_clusters.predict(before_des)
        after_clusters = kmeans_clusters.predict(after_des)
        for c in before_clusters:
            feature_vec[c] += 1
        for c in after_clusters:
            feature_vec[num_clusters + c] += 1
        data.append(feature_vec)
        labels.append(yi)

    X_train_bow = pd.DataFrame(data)
    y_train_bow = pd.DataFrame(labels)

    # now let's get the bag-of-words vectors for the train set
    data = []
    labels = []
    for i, row in X_test.iterrows(): # p, ap, yi in zip(before_paths, after_paths, labels):
        bp, ap = row['bp'], row['ap']
        yi = y[i]
        before_img = cv2.imread('/'.join((path_prefix, bp)))
        after_img = cv2.imread('/'.join((path_prefix, ap)))
        feature_vec = np.zeros(2 * num_clusters)
        kp, before_des, _ = detect_and_draw(
            feature_type=feature_type,
            draw_img=False,
            img_=before_img)
        kp, after_des, _ = detect_and_draw(
            feature_type=feature_type,
            draw_img=False,
            img_=after_img)
        before_clusters = kmeans_clusters.predict(before_des)
        after_clusters = kmeans_clusters.predict(after_des)
        for c in before_clusters:
            feature_vec[c] += 1
        for c in after_clusters:
            feature_vec[num_clusters + c] += 1
        data.append(feature_vec)
        labels.append(yi)
    X_test_bow = pd.DataFrame(data)
    y_test_bow = pd.DataFrame(labels)

    # since the dataset is larger, use 10-fold
    augmented = 'augmented_noisy'
    cv = StratifiedKFold(n_splits=10)

    if model == 'svm':
        params = {
            'C': [.001, 0.1, 1],
            'gamma': [.0001, 0.001],
        }
        classifier = GridSearchCV(SVC(probability=True), cv=cv, param_grid=params)
    elif model == 'nb':
        params = {
            'fit_prior': [True, False],
        }
        # I think this makes all vectors binary
        classifier = GridSearchCV(BernoulliNB(binarize=0.01), cv=cv, param_grid=params)

    model_grid_search_string = model + 'GridSearch'
    class_str = 'multiclass' if multi_class else 'binary'
    model_name = "{}-{}-{}-{}-{}".format(model_grid_search_string, feature_type, str(num_clusters), class_str, augmented)
    best_params = _do_CV(
        X_train_bow,
        X_test_bow,
        y_train_bow,
        y_test_bow,
        classifier,
        multi_class=multi_class, save_img=True, img_name=model_name)
    # train model on best params
    if True: # write_kmeans:
        # write the kmeans cluster centroids
        pickle.dump(kmeans_clusters, open("./clusters/noisy-{}-{}.pkl".format(feature_type, str(num_clusters)), "wb"))
        pickle.dump(X_train.append(X_test), open("./bow_datasets/noisy-{}-{}".format(feature_type, str(num_clusters)), "wb"))
        pickle.dump(classifier, open("./models/noisy-%s.pkl" % model_name, "wb"))


def vbow_kmeans(orb_features, num_clusters, before_paths, after_paths, labels, multi_class=False, write_kmeans=False,
                feature_type='sift', path_prefix='..', cv_method='lpo', model='svm',
                save_img=False):
    # get the kmeans centroids

    kmeans_clusters = KMeans(n_clusters=num_clusters).fit(orb_features)

    # using the centroids, extract the "bag-of-words" type dataset
    X_full, y_full = extract_vbow_dataset(before_paths, after_paths, labels, kmeans_clusters, num_clusters, feature_type=feature_type)

    # drop test set data from X and y
    X = X_full.drop(index=TEST_KEYS)
    y = y_full.drop(index=TEST_KEYS)

    # pick our cross validation method and train our model
    if cv_method == 'lpo':
        augmented = 'not_augmented'
        cv = LeavePOut(3)
    else:
        augmented = 'augmented'
        cv = StratifiedKFold(n_splits=5)

    if model == 'svm':
        params = {
            'C': [1, 10],
            'gamma': [.0001, 0.001, 0.01],
        }
        classifier = GridSearchCV(SVC(probability=True), cv=cv, param_grid=params)
    elif model == 'nb':
        params = {
            'fit_prior': [True, False],
        }
        # I think this makes all vectors binary
        classifier = GridSearchCV(BernoulliNB(binarize=0.01), cv=cv, param_grid=params)

    # HACK: I <3 dynamic typing
    if type(y) != pd.DataFrame:
        y = pd.DataFrame(np.ravel(y))

    model_grid_search_string = model + 'GridSearch'
    class_str = 'multiclass' if multi_class else 'binary'
    model_name = "{}-{}-{}-{}-{}".format(model_grid_search_string, feature_type, str(num_clusters), class_str, augmented)
    best_params = do_CV(X, y, classifier, multi_class=multi_class, save_img=save_img, img_name=model_name)
    # train model on best params
    if write_kmeans:
        # write the kmeans cluster centroids
        pickle.dump(kmeans_clusters, open("./clusters/{}-{}.pkl".format(feature_type, str(num_clusters)), "wb"))
        pickle.dump(X_full, open("./bow_datasets/{}-{}".format(feature_type, str(num_clusters)), "wb"))
        pickle.dump(classifier, open("./models/%s.pkl" % model_name, "wb"))
    
def get_all_features(before_paths, after_paths, feature_type='orb', path_prefix='..', augment_with_noise_factor=0):
    """This is a generic version of get_all_orb_features"""
    all_features = pd.DataFrame()
    for before_path, after_path in zip(before_paths, after_paths):
        before_kp, before_des, before_img = detect_and_draw(before_path, feature_type=feature_type, draw_img=False)
        after_kp, after_des, after_img = detect_and_draw(after_path, feature_type=feature_type, draw_img=False)
        all_features = all_features.append(pd.DataFrame(before_des))
        all_features = all_features.append(pd.DataFrame(after_des))
    return all_features

def get_features_clusters_and_vbow_data(before_paths, after_paths, labels, num_clusters=50, feature_type='orb',augment=False):
    raw_image_features = get_all_features(before_paths, after_paths, feature_type)
    kmeans_clusters = KMeans(n_clusters=num_clusters).fit(raw_image_features)
    X, y = extract_vbow_dataset(before_paths, after_paths, labels, kmeans_clusters, num_clusters, feature_type=feature_type)
    return (raw_image_features, kmeans_clusters, X, y)

def add_noise(img):
    noise = np.zeros_like(img)
    # stretch
    x = img.shape[0]
    y = img.shape[1]
    x = round(x * random.uniform(0.5, 1.5))
    y = round(y * random.uniform(0.5, 1.5))
    # add gaussian "salt n pepper" noise
    cv2.randn(noise, 0,  np.identity(1) * 20)
    return cv2.resize(img + noise, (x, y))


if __name__ == '__main__':

    pass
