import cv2
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import pickle
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV, LeavePOut, KFold
from sklearn.svm import SVC

from .utils import do_CV


def detect_and_draw_orb(path, draw_img=True):
    orb = cv2.ORB_create()
    img = cv2.imread(path,0)
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

def detect_and_draw_sift(path, draw_img=False):
    sift = cv2.xfeatures2d.SIFT_create()
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.imread(path, 0)
    (kp, des) = sift.detectAndCompute(img, None)
    kps_img = cv2.drawKeypoints(img, kp, None, color=(0,255,0), flags= cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    if draw_img:
        plt.imshow(img), plt.show()
    return kp, des, kps_img

def detect_and_draw_surf(path, draw_img=False):
    surf = cv2.xfeatures2d.SURF_create(400)
    img = cv2.imread(path, 0)
    kp, des = surf.detectAndCompute(img,None)
    kps_img = cv2.drawKeypoints(img, kp, None, color=(0,255,0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    if draw_img:
        plt.imshow(img), plt.show()
    return kp, des, kps_img


def detect_and_draw(path, feature_type='orb', draw_img=True, path_prefix='..'):
    path = '/'.join((path_prefix, path))
    if feature_type == 'orb':
        return detect_and_draw_orb(path, draw_img)
    elif feature_type == 'sift':
        return detect_and_draw_sift(path, draw_img)
    elif feature_type == 'surf':
        return detect_and_draw_surf(path, draw_img)

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
    y = labels
    return X, y

def vbow_kmeans(orb_features, num_clusters, before_paths, after_paths, labels, multi_class=False, write_kmeans=False,
                feature_type='sift', path_prefix='..', cv_method='lpo', model='svm'):
    # get the kmeans centroids

    kmeans_clusters = KMeans(n_clusters=num_clusters).fit(orb_features)

    # using the centroids, extract the "bag-of-words" type dataset
    X, y = extract_vbow_dataset(before_paths, after_paths, labels, kmeans_clusters, num_clusters, feature_type=feature_type)

    # pick our cross validation method and train our model
    if cv_method == 'lpo':
        augmented = 'not_augmented'
        cv = LeavePOut(3)
    else:
        augmented = 'augmented'
        cv = KFold(n_splits=5)
    params = {
        'C': [0.1, 1, 10, 100],
        'gamma': [.0001, 0.001, 0.01, 0.1, 1],
    }
    svm_clf = GridSearchCV(SVC(probability=True), cv=cv, param_grid=params)
    best_params = do_CV(X, y, svm_clf, multi_class=multi_class)
    # train model on best params
    if write_kmeans:
        class_str = 'multiclass' if multi_class else 'binary'
        # write the kmeans cluster centroids
        pickle.dump(kmeans_clusters, open("./clusters/{}-{}.pkl".format(feature_type, str(num_clusters)), "wb"))
        pickle.dump(X, open("./bow_datasets/{}-{}".format(feature_type, str(num_clusters)), "wb"))
        model_grid_search_string = model + 'GridSearch'
        pickle.dump(svm_clf, open("./models/{}-{}-{}-{}-{}.pkl".format(model_grid_search_string, feature_type, str(num_clusters), class_str, augmented), "wb"))
    
def get_all_orb_features(before_paths, after_paths):
    all_orb_features = pd.DataFrame()
    for before_path, after_path in zip(before_paths, after_paths):
        before_kp, before_des, before_img = detect_and_draw_orb(before_path)
        after_kp, after_des, after_img = detect_and_draw_orb(after_path)
        all_orb_features = all_orb_features.append(pd.DataFrame(before_des))
        all_orb_features = all_orb_features.append(pd.DataFrame(after_des))
    return all_orb_features

def get_all_features(before_paths, after_paths, feature_type='orb', path_prefix='..'):
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

if __name__ == '__main__':

    pass
