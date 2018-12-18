# aweml
This is a new classifier based on taking a weighted average of different combinations of features

# Introduction

This is a new classifier based on taking a weighted average of different combinations of features as described in: https://arxiv.org/abs/1710.10301

It is designed for categorical data. Continuous valued data can be handled using a built-in binning function. It can handle missing features. Missing values should be encoded as np.nan when using the supplied binning functions.

If the binning function is not used to preprocess data, categorical features should be encoded from 0 to N-1 for each feature and missing features should be encoded as -1.

# Installation

Installation requires Python 3.

A binary distribution can be created for Red Hat 6 if requested.

Source installation requires a C++11 compatible C compiler (The compiler should be configured in the path in Linux as needed so that pip can find the compiler). Numpy should be installed prior to installation.

The classifier can be installed with pip:
```python
pip3 install awe_ml-0.3.3.tar.gz
```

Prerequisites: Python 3, numpy>1.10, scikit-learn, pandas>0.21.0, C++11 for source installation

# Usage

The classifier is fully compatible with sckit-learn.

## Instantiation

It can be instantiated as follows:
```python
from awe_ml import AWE_ML, bin_data
clf = AWE_ML(max_fully_connected_depth=2, max_depth = 4, features_per_node=5, n_max_classify=10, n_noise=0.5, usefulness_model="simple", n_min_to_add_leaf=2, node_split_model="gini_random", node_split_fraction=0.4, probability_scaling_method= "imbalanced_reciprocal", noise_exponent=0.5, nbins=5, max_bins=20, categorical_feature_inds=None, feature_names_txt=None, categorical_bin_list=None, numeric_bin_list=None)
```

All the settings above are meta parameters for the classifier. Most can be left at the default values. They are described below.

### Meta-Parameters

The meta-parameters are described below. The can be grouped as follows:

* Three meta-parameters describing the construction of the classifier tree need to be set correctly **(max_fully_connected_depth, max_depth, features_per_node)**
* Two meta-parameters describing the noise in the data need to be correctly set **(n_max_classify, n_noise)** 
* The **usefulness_model** (“simple”, “scaled, or “none”) can also have an impact on accuracy.
* Three meta-parameters control how the data should be binned, if at all: **nbins**, **max_bins**, and **categorical_feature_inds**.
* The classifier also optionally takes three values that specify labels for the explainability results: **feature_names_txt, categorical_bin_list, numeric_bin_list**.
* The rest of the parameters can be left at default values **(node_split_model, node_split_fraction, probability_scaling_method, and noise_exponent)**.

For a given dataset, the meta-parameters will ideally be set by using a cross-validation search (i.e. GridSearchCV in scikit-learn)

#### Tree Construction Meta Parameters

The classifier based on building a tree of probabilities that are combined into a weighted average as illustrated in Fig 1.



### Data Binning

### Explainability Labels

## Training

## Prediction

## Explaining a Prediction (Explicating)
