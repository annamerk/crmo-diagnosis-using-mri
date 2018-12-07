This directory contains three sub-directories:

* clusters
* bow_datasets
* models

`clusters` contains the kmeans cluster centroids obtained by running kmeans on
a dataset of features generated from images. So it depends on initial feature
sets and number of clusters and has num_clusters centroids.

`bow_datasets` contains the dataset obtained by assigning the features of the
images to each cluster.

`models` contains the serialized model which comes from fitting some model on
one of the datasets from `bow_datasets`.
