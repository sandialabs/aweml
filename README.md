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

![](https://raw.githubusercontent.com/sandialabs/aweml/master/tree1.png)
*Fig 1: The tree probabilities are illustrated (weights are left out of the equations for clarity)*

The construction of the tree is specified as follows:

* **max_depth=4**: max depth of the tree
* **max_fully_connected_depth=2**: The top layers of the tree can be fully connected and use all possible combinations of features
* **features_per_node=5**: For levels that are not fully connected an additional features_per_node children will be created for each node in the level above
* **node_split_model="gini_random"**: This specifies how the features at each node will be chosen. **"gini"** will use the top **features_per_node** features that most reduce the gini coefficient. **“gini_random”** will first randomly select **node_split_fraction** of the possible features and then choose the top features by gini coefficient.
* **node_split_fraction=0.4** What fraction of features to select from when using "gini_random"

In general, larger trees will give more accurate results, but will slow the classifier down. Typically, **max_fully_connected_depth** will be set to 1 or 2 depending on the dataset.

If **max_fully_connected_depth=1**, a deeper tree with a larger **max_depth** (~ 6-8) and fewer **features_per_node** (~1-4) is often optimal. This gives a deeper, but sparser tree.

If **max_fully_connected_depth=2**, a shallower but denser tree is often optimal with **max_depth** ~ 4-6 and **features_per_node** ~ 4-10.

The **node_split_model** can be left as **"gini_random"** as it mostly gives identical accuracies to **"gini"**. For deep sparse trees **gini_random** can help improve the accuracy. If **node_split_fraction** is not too small (i.e.>0.2), the classifier is not very sensitive to the exact value and so the default value can be used.

#### Classification Meta Parameters

When computing the weights used in classification there are several important meta parameters:

* **n_max_classify=10**: Each combination of features is weighted to account for the amount of data backing up a given probability estimate. That weight is given by: min( n^**noise_exponent**, **n_max_classify**^**noise_exponent**) where n is the number of training examples with a given combination of features. If n > **n_max_classify**, then it’s assumed that there is sufficient data supporting the combination of features, otherwise it’s weight is reduced.
* **noise_exponent=0.5**: This is the noise exponent used in the weight above. 0.5 seems to generally give the best accuracy, but systematic tests are needed to confirm
* **usefulness_model="simple"**: Each combination of features can optionally be weighted by a usefulness weight based on how different the weight is from its parent. By default, the model is turned on (“simple”). It can be turned off by setting **usefulness_model="none"**. It seems to help for some datasets but not all. Another model, **“scaled”** is also helpful for some datasets (It’s possible that a better usefulness model can be created)
* **probability_scaling_method="imbalanced_reciprocal"**: As a probability (p) approaches zero or one its weight should be increased. The weight can increase following 1/p for the **"reciprocal"** option or it can increase slower following logistic function for the **“logit”** option. Class imbalance can also be accounted for in setting limits on the weight with the option **“imbalanced_reciprocal”**. In general, **"imbalanced_reciprocal"** performs the best.
* **n_noise=0.5**: n_noise roughly indicates the average number of training examples that can go wrong due to noise. When calculating a weight based on how close a probability is near 0 or 1, we need to avoid giving too much weight to P=1 or P=0 based on a single training example. Consequently, the probability used to compute a weight is limited to the range [n_noise/n, 1- n_noise/n] where n is the number of training examples with a given combination of features.

### Data Binning

As the classifier is designed for categorical data, continuous valued data needs to be binned. This can be done in one of two ways. The classifier can internally bin the data, or the user can externally bin the data. While it is easier to let awe-ml internally bin the data, it is more efficient to externally bin the data when the data is reused (for instance in a grid search)

#### Automatic Data Binning

The easiest option to is automatically bin the data. In order to bin the data three metaparameters specifying how to bin the data need to be set:

* **nbins=5**: nbins specifies the number of bins to create when binning continuous data. Setting **nbins=0** disables automatic binning
* **categorical_feature_inds=None**: In order to determine which features are already categorical and which need to be binned, two settings are used. **categorical_feature_inds** takes a list of column indices that explicitly specify which features are categorical and should not be binned.
* **max_bins=20**: Additionally, any feature that has unique values <= max_bins is assumed to be categorical and is not binned.

When using automatic binning, categorical data can have text based values and continuous data should not be normalized. The classifier will use the raw feature data to extract labels for each bin to be used with the explainability models. Missing data should be encoded as NaN: np.nan

#### Manual Data Binning

If re-using the same data multiple times, it is more efficient to manually bin the data once at the beginning. The user may also have better custom binning methods.

First automatic binning should be disabled by setting **nbins=0**. Next the data should be binned and preprocessed so that all features are encoded from 0 to N-1 for each category and missing features are encoded as -1. Two functions are provided to do this:

```python
from awe_ml import bin_data, bin_data_given_bins
data = bin_data (data, nbins = 4, max_bins =20, categorical_feature_inds = None, binning_method = "qcut", retbins=False)
```

The function bin_data will take the data and bin it, returning a numpy array with the binned data. **nbins**, **max_bins**, and **categorical_feature_inds** are described above. The data can be a numpy array or pandas dataframe.

**binning_method** should be left as default/ignored as there is currently only one binning method implemented, “qcut”. Data is divided into equal sized bins. If most frequent value is the first/last data value and has more counts than n_entries / nbins a single bin is created for the most frequent value the other values are split equally. This tries handle cases where most of the data might be zero and only a few values are nonzero.

Setting **retbins=True** returns two variables that contain information on how to encode and bin the data:

```python
data, categorical_bin_list, numeric_bin_list = bin_data (data, nbins = 4, max_bins =20, categorical_feature_inds = None, binning_method = "qcut", retbins=False)
```

The additional return variables are:
* **categorical_bin_list**: A list of tuples for the categorical features. Each tuple contains: (feature column, dictionary mapping category number to category text)
* **numeric_bin_list**: A list of tuples for the numeric features. Each tuple contains: (feature column, numpy array of bin edges). Each bin includes the right edge, but not the left edge (the first bin includes both edges)

These two lists can be passed to the classifier to label the explainability results or to bin new data as follows:

```python
data = bin_data_given_bins(data, categorical_bin_list, numeric_bin_list):
```

This returns a numpy array of binned data based on **categorical_bin_list** and **numeric_bin_list** as defined above. If a new feature value or out of range feature value is included it is encoded as a missing value, -1.

### Explainability Labels

In order to label the features and the individual feature values up to three variables need to be set. These can be set during the classifier instantiation or during the fit method.

* **feature_names_txt=None**: This specifies a text label for each feature and so be passed in as a list, i.e. [“feature1”, “feature2”,…] If a value is not passed in, the feature number will be used by default

The next two values, **categorical_bin_list**, **numeric_bin_list** only need to be set if manual data binning is used. They can be directly generated from the bin_data function. Failing to set them will result in category numbers being used.

* **categorical_bin_list=None**: This specifies the labels of each category for each categorical feature. It is a list of tuples for the categorical features. Each tuple contains: (feature column, dictionary mapping category number to category text)
* **numeric_bin_list=None**: This specifies the labels of each bin for each continuous valued feature. It is a list of tuples for the numeric features. Each tuple contains: (feature column, numpy array of bin edges). Each bin includes the right edge, but not the left edge. The first bin includes both edges.

## Training

The classifier can be fit as follows:

```python
clf.fit(train_data,train_targets, feature_names_txt=None, categorical_bin_list=None, numeric_bin_list=None)
```

**train_targets** can contain numeric or text data. As described above, if automatic binning is used (nbins>0), train_data can contain both text and numeric data.

If automatic binning is not used, features should be encoded from 0 to N-1 for each feature category and missing features should be encoded as -1.

**feature_names_txt**, **categorical_bin_list** and **numeric_bin_list** can be optionally set here (or during initialization) to provide labels for the explainability model. If automatic binning is used **categorical_bin_list** and **numeric_bin_list** are ignored.

## Prediction

The classifier can predict probabilities as follows:

```python
probabilities = clf.predict_proba(test_data)
```

The predicted probabilities can be used to predict class labels:

```python
predictions = clf.classify(probabilities)
```

The classes can be directly predicted (internally **predict_proba** and **classify** are called)

```python
predictions = clf.predict(test_data)
```

## Explaining a Prediction (Explicating)

Explainability data can be generated for a single input (**single_example**) as follows:

```python
combos_df, features_df, top_of_tree_df = clf.analyze_single_probability(single_example,filename_prefix=None)
```

This will return three pandas dataframes: **combos_df**, **features_df**, **top_of_tree_df**.

If a string is specified for filename_prefix, the data will also be saved to three files:

* **combos_df , [filename_prefix]**_combos.csv
  * This lists out every combination of features that contributes to the result, the probability measured from the training data given that combination of features and the weight assigned to the feature combination. The results are sorted by weight. Looking at the top few feature combos will indicate what were the most important. The columns are specified as follows:
    * level: What level of the tree is this combination of features on
    * F0, F1, … : The particular feature in the combination of features being analyzed
    * #0, #1, … : The count of training examples with the specified features in class 0, 1,…
    * P0, P1, … : The probability of each class in training data (derived from the counts)
    * W(C0), W(C1),.. : The weight assigned to this combination of features for each class. When there are more than two classes, the classifier can assign different weights depending on the class being predicted.
    *Note: The following two outputs are still in development*
    * W(C0_F0), … : The fraction of weight W(C0) that comes from feature F0
    * W(C0_FX), …: The fraction of weight W(C0) that comes from child feature combinations that have an identical probability as W(C0).
* **features_df, [filename_prefix]**_by_feature_shared.csv
  * This takes each feature combo and splits its weight evenly across the contributing features. The features are then ranked by weight to determine the most important features. The columns are specified as follows:
    * feature: The feature being analyzed
    * weighted_probability: The weighted probability using only feature combinations that contain the given feature for the predicted class
    * original_probability: The probability measured from the training data of the predicted class using only data that has the given feature. This can be very different than the weighted probability as the weighted_probability takes into account the particular combinations of features present in this example.
    * weight: The total weight assigned to feature combinations with this feature
* **top_of_tree_df , [filename_prefix]**_top_of_tree.csv (for debugging only)
  * This is the top level of the classification tree and is the classifier’s estimate of the probability given a particular feature. This is useful for digging into how exactly a weight was computed and can be used with W(C0 F0), W(C0 F1),… from results_combos.csv. W(C0 F0), W(C0 F1),… brakes each weight into its contributions to the weights specified in this file. The columns are:
    * feature: The feature being analyzed
    * weighted_probability: The estimated probability given the selected feature in the top of the classification tree
    * original_probability: The probability measured from the training data of the predicted class using only data that has the given feature. This can be very different than the Probability in Tree as the weighted_probability takes into account the particular combinations of features present in this example.
    * weight: The total weight to this feature in the tree
