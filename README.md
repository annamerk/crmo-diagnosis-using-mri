# crmo-diagnosis-using-mri

## Setup

1. Install [Miniconda](https://conda.io/docs/user-guide/install/index.html#regular-installation)
  - Conda is a package manager that sandboxes your project’s dependencies in a virtual environment
  - Miniconda contains Conda and its dependencies with no extra packages by default (as opposed to Anaconda, which installs some extra packages)
2. cd into src, run `conda env create -f environment.yml`
  - This creates a Conda environment called `cs229-project`
3. Run `source activate cs229-project`
  - This activates the `cs229-project` environment
  - Do this each time you want to write/test your code

## Data
- *data.csv*: list of image pairings and class curated from radiologist information
- *legs_folder*: excluded from github for privacy, but expected to contain MRIs.
## Baseline
Initial experimentation with histograms is contained in Baseline.ipynb
## Inception (folder /inception)
Jupyter notebooks as well as pickled output data and generated models are found in the inception folder.
Key files:
- *Tensorflow Inception v3 extraction.ipynb*: contains pre-processing code for input images, and feature generation through inception model.
- *inception_cnn_features2.pkl*: selected set of features for images generated from tensorflow (dissimilarity of feature vectors, augmented, normalized and feature reduced)
- *Inception CNN retraining.ipynb*: various models trained on data generated with tensorflow.
## Ensemble
- *Ensemble Voter.ipynb*: contains various models produced by voting ensembling inception and vbow models.
