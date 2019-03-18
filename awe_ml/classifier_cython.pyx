#distutils: language = c++
#cython: profile=False,wraparound=False, cdivision=True, boundscheck=False, initializedcheck=False, language_level=3


#NOPE #cython: c_string_type=unicode, c_string_encoding=utf8


#NO distutils: include_dirs=/usr/local/lib/python3.6/site-packages/numpy/core/include

#######NOTATION /  Variable name convention
# feature_pair are a C++ pair of integers specifying feature number and category number: pair[np_long,np_long] feature_pair
# feature_num is the number of a feature only, no category
# category_num is the category number for a particular feature
# comb_feat_ind is the combined feature index representing a feature_num and category_num as a single number
# label represents a list of comb_feat_ind labeling a node

#  _ctr is used for small integers that count from zero
# array means numpy or memview array 2d array
#full_tree is the data from training, i.e. all possible features are included


import numpy as np
cimport numpy as np
cimport cython

from warnings import warn

from sklearn.utils.multiclass import unique_labels

from libc.math cimport log, lround, sqrt
from libcpp.vector cimport vector
from libcpp.unordered_map cimport unordered_map
from libcpp.map cimport map
from libcpp.pair cimport pair
from libcpp.unordered_set cimport unordered_set
from libc.stdlib cimport rand, RAND_MAX
from libcpp.set cimport  set
from cython.operator cimport dereference as deref, preincrement as inc
from libcpp cimport bool
from libcpp.string cimport string
from libc.math cimport isnan
from libc.stdlib cimport malloc, free
from cython cimport view
from sklearn.utils.validation import check_X_y, check_array # check_is_fitted

import pandas as pd
from io import BytesIO


#  all integers are 32 bit numpy ints and all floats are 64 bit numpy floats
NP_LONG = np.int32
NP_FLOAT= np.float64


ctypedef np.int32_t np_long
ctypedef np.float64_t np_float


cdef extern from "<string>" namespace "std" nogil:  #cython bug, import sort directly
    string to_string (int);
    string to_string (double);

cdef extern from "<algorithm>" namespace "std" nogil:  #cython bug, import sort directly
    void sort[Iter](Iter first, Iter last)
    OutputIterator set_intersection[InputIterator1, InputIterator2,OutputIterator] (InputIterator1 first1, InputIterator1 last1,InputIterator2 first2, InputIterator2 last2,OutputIterator result);
    OutputIter copy[InputIter,OutputIter](InputIter,InputIter,OutputIter)
    Iter unique[Iter](Iter first, Iter last)
    bool includes[InIterator1, InIterator2] (InIterator1 f1, InIterator1 l1, InIterator2 f2, InIterator2 l2)
    FwdIter max_element[FwdIter](FwdIter first, FwdIter last)


# load custom C++ class classification values
cdef extern from "classification_values.h" nogil:

    cdef cppclass AccessData[T]:
            T & operator[](long) except +
            T * ptr(long) except +
            T * ptr() except +

    cdef cppclass ClassificationValues[T]:
        ClassificationValues() except +
        ClassificationValues(long, long) except +
        void initialize(long n_nodes, long n_classes)
        void set_node(long local_tree_index)



        AccessData[T] estimated_p
        AccessData[T] p_sum
        AccessData[T] weight_sum
        AccessData[T] max_child_noise
        void setstate(pair[vector[long],vector[T]] & state_data)
        pair[vector[long],vector[T]] getstate()
        vector[T] data


cdef extern from "full_tree_data.h" nogil:

    cdef cppclass FullTreeData[Tfloat, Tint]:
        FullTreeData() except +
        FullTreeData(long, long, long) except +
        void initialize(long n_nodes, long n_classes, long level)
        void set_node(long full_tree_index)
        long n_nodes

        AccessData[Tint] label
        AccessData[Tint] counts
        AccessData[Tint] parent_indicies
        AccessData[Tfloat] p0
        AccessData[Tfloat] noise_weight

        void setstate(pair[vector[long],pair[vector[Tfloat],vector[Tint]] ] & state_data)
        pair[vector[long],pair[vector[Tfloat],vector[Tint]]] getstate()

# from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
# from libcpp.algorithm cimport copy
# from libcpp.iterator cimport iterator, inserter
# from libcpp.algorithm cimport sort
# from libcpp cimport bool



########################################################################

cdef void pair_sort (pair[np_long,np_long] & pair_to_sort):
    """
    Sort a pair of long in place
    :param pair_to_sort: 
    :return: 
    """
    cdef np_long temp
    if pair_to_sort.second<pair_to_sort.first: #sort the label pair
        temp = pair_to_sort.first
        pair_to_sort.first = pair_to_sort.second
        pair_to_sort.second = temp

########### Define helper functions used to calculate intermediate values (usefulnes, scaling weight)
include "helperfunctions.pxi"

########## include data structures for trees
include "data_structures.pxi"

########################################################################
## define cython class to accelerate awe_ml

@cython.auto_pickle(False)
cdef class AWE_ML_Cython:

    cdef np_long n_features
    cdef np_float ave_corr
    cdef list feature_names_txt
    cdef public object classes_  #cython error checking does not want this set before fit, but cython needs it declared.  modifed cython error checking to avoid error
    cdef readonly np_long n_classes,
    cdef string independence_model_c

    cdef public np_long max_fully_connected_depth, n_max_classify, max_depth, n_min_to_add_leaf, features_per_node #n_jobs


    cdef public str node_split_model, probability_scaling_method, usefulness_model, independence_model, _estimator_type

    cdef public np_float node_split_fraction, noise_exponent, n_noise, noise_scale#, noise_exponent2
    cdef public object noise_exponent2
    cdef np_long[:,:] X
    cdef np_long[:] y


    cdef vector [np_long]  n_feature_categories_
    cdef vector[ClassificationValues[np_float]] classification_values_list
    cdef vector[FullTreeData[np_float,np_long]] full_tree_vector

    cdef vector[vector[np_long]] full_tree_indicies_list, full_tree_to_local_tree_lookup_list
    cdef vector[vector[vector[vector[np_float]]]] local_weight_tree
        #level->node-> parent->class->weight
    cdef vector [node_info_t] node_info


    cdef vector[vector[vector[np_long]]] children_list #1st level: tree level 2nd level: node 3rd level: children in node
    cdef map [pair[np_long,np_long],np_long] feature_pair_to_comb_feat_ind
    cdef map [np_long,pair[np_long,np_long]] comb_feat_ind_to_feature_pair
    cdef map[pair[np_long,np_long],np_float] correlation_map



    cdef vector[map[vector[np_long],np_long]] node_label_to_index_dict_list
    cdef vector [map[vector[np_long], pair[vector[np_long],vector[np_long] ] ] ] nodes
    #1st level: tree level 2nd level: map of node label to pair (node info, children feature num)


    cdef np_long nbins, max_bins
    cdef object categorical_feature_inds
    cdef str binning_method
    cdef list categorical_bin_list,numeric_bin_list

    #hack to store init values as objects for scikit-learn checks
    cdef object noise_exponent_obj, node_split_fraction_obj, n_noise_obj


    def __init__(self, np_long max_fully_connected_depth=2, np_long max_depth = 4, np_long features_per_node=5, np_long n_max_classify=10,
                 object n_noise=0.5, str usefulness_model = "simple",  #simple
                 np_long n_min_to_add_leaf=2, str node_split_model = "gini_random", object node_split_fraction=0.4,
                 str probability_scaling_method = "imbalanced_reciprocal", object noise_exponent=0.5, object noise_exponent2=None,  #imbalanced_reciprocal
                 str independence_model = "none",
                 np_long nbins = 5, np_long max_bins =20, object categorical_feature_inds = None, str binning_method = "qcut",
                 list feature_names_txt=None, list categorical_bin_list=None, list numeric_bin_list=None): #n_jobs=1, numerical_features = None
        """
        Assumes all categorical features are enumerated from 0 to N-1.  Will fail if they are not correctly enumerated
        Missing values are encoded as -1


        :param max_fully_connected_depth: the max depth of fully connected layers (2 means all pairs of features)
        :param n_max_classify:
        :param n_jobs: number of parallel job to run
        :param max_depth: Max depth of the tree including entropy and other feature based nodes, 0 means unlimited
        :param n_min_to_add_leaf: minimum number of data points to add a leaf node.
        :param numerical_features:  which features are continuous. By default, it's assumed all features are categorical.
        :param node_split_model: What method to use when choosing features at each node.  Options are:
            gini:  gini coeficient
            gini_random:  randomly choose node_split_fraction of features and then take top features by gini
        :param node_split_fraction: what fraction of features to use when doing gini random
        :param probability_scaling_method: method for scaling the raw probabilites.  options are "logit" and "reciprocal"
        :param noise_exponent:  exponent used when deciding how much to weight child vs parent node when child does not have enough data
        :param usefulness_model: what model to use to account for the usefulness of a feature.  options are, "none", "simple", and "KL"
        :param n_noise:  The number of examples that might be wrong due to noise.  Used to limit the probability scaling
        :param features_per_node: The number of features to add levels that are not fully connected
        :param independence_model:  What model to use for feature independence, options are "none" and "standard"

        :param nbins: how many bins to divide data into, 0 means do not bin data
        :param max_bins:  if num bins in a column <= max bins do not bin data, if max_bins==0, assume all data is categorical
        :param categorical_feature_inds: what features are categorical for binning
        :param binning_method:  method used to bin data, only option currently is "qcut": Data is divided into equal sized bins.
                            If most frequent value is first/last data value and has more counts than n_entries / nbins create
                            single bin for most frequent value and split other values equally

        :param feature_names_txt: list of strings of labels for each feature for explainability output
        :param categorical_bin_list: list of tuples (col, dictionary mapping category number to category)
        :param numeric_bin_list: list of tuples (col, numpy array of bins) Each bin includes rightmost edge, not left (leftmost edge is also included)
        :return:
        """

    # def __init__(self, long max_fully_connected_depth, long max_depth, long features_per_node, long n_max_classify, double n_noise,
    #              str usefulness_model, long n_min_to_add_leaf, str node_split_model, double node_split_fraction,
    #              str probability_scaling_method, double noise_exponent, object noise_exponent2, str independence_model, list feature_names_txt,
    #              np_long nbins, np_long max_bins, object categorical_feature_inds, str binning_method): #long n_jobs,

        #store parameters to self #TODO: convert python string options to C variables
        self.max_fully_connected_depth = max_fully_connected_depth
        self.n_max_classify = n_max_classify
        self.max_depth  = max_depth
        self.n_min_to_add_leaf = n_min_to_add_leaf
        self.features_per_node =features_per_node
        self.node_split_model = node_split_model
        self.node_split_fraction = node_split_fraction
        self.probability_scaling_method = probability_scaling_method
        self.noise_exponent=noise_exponent
        self.noise_exponent2=noise_exponent2
        self.usefulness_model = usefulness_model
        self.n_noise= n_noise
        self.feature_names_txt=feature_names_txt
        self.independence_model = independence_model

        self.nbins = nbins
        self.max_bins= max_bins
        self.categorical_feature_inds=categorical_feature_inds

        self.binning_method=binning_method
        self.categorical_bin_list=categorical_bin_list
        self.numeric_bin_list=numeric_bin_list

        # self.n_jobs =n_jobs
        self._estimator_type = "classifier"

        ### initialize memory view objects
        self.X= np.zeros((0,0), dtype=NP_LONG)
        self.y = np.zeros((0), dtype=NP_LONG)

        ####hack for sci-kitlearn checks
        self.noise_exponent_obj  =noise_exponent
        self.node_split_fraction_obj = node_split_fraction
        self.n_noise_obj = n_noise


        ####  TEST new class
        # cdef ClassificationValues classification_obj, classification_obj1, classification_obj2
        # classification_obj.initialize(10,3)
        # classification_obj.set_node(2)
        # classification_obj.weight_sum[1]=1.2
        # classification_obj.weight_sum[0]=2.45
        # classification_obj.estimated_p[1]=11.45
        # classification_obj.set_node(1)
        # classification_obj.max_child_noise[0]=432432.23432
        # classification_obj.set_node(0)
        # classification_obj.estimated_p[0]=89
        # print("object0=\n\t",classification_obj.data)
        # print("object1=\n\t",classification_obj1.data)
        # classification_obj1=classification_obj
        # print("object1 after assignment=\n\t",classification_obj1.data)
        # classification_obj1.set_node(1)
        # classification_obj1.max_child_noise[0]=999
        #
        # print("object0 after write=\n\t",classification_obj.data)
        # print("object1 after write=\n\t",classification_obj1.data)
        # cdef tuple state
        # state = classification_obj1.getstate()
        # print("object2 before set state=\n\t",classification_obj2.data)
        # classification_obj2.setstate(state)
        # print("object2 after set state=\n\t",classification_obj2.data)

        # cdef vector[ClassificationValues] test_obj
        # test_obj.resize(4)
        # test_obj[1].initialize(100,3)

    def __getstate__(self):
        state_dict = dict()
        state_dict["n_features"]=self.n_features
        state_dict["ave_corr"]=self.ave_corr

        # state_dict["parents_list"]=self.parents_list
        # state_dict["p0_list"]=self.p0_list
        # state_dict["noise_weight_list"]=self.noise_weight_list
        # state_dict["counts_list"]=self.counts_list
        # state_dict["node_label_list"]=self.node_label_list

        state_dict["feature_names_txt"]=self.feature_names_txt


        state_dict["classes_"]=self.classes_

        state_dict["n_classes"]=self.n_classes
        state_dict["independence_model_c"]=self.independence_model_c


        state_dict["max_fully_connected_depth"]=self.max_fully_connected_depth
        state_dict["n_max_classify"]=self.n_max_classify
        state_dict["max_depth"]=self.max_depth
        state_dict["n_min_to_add_leaf"]=self.n_min_to_add_leaf
        state_dict["features_per_node"]=self.features_per_node

        state_dict["node_split_model"]=self.node_split_model
        state_dict["probability_scaling_method"]=self.probability_scaling_method
        state_dict["usefulness_model"]=self.usefulness_model
        state_dict["independence_model"]=self.independence_model

        state_dict["node_split_fraction"]=self.node_split_fraction
        state_dict["node_split_fraction_obj"]=self.node_split_fraction_obj
        state_dict["noise_exponent"]=self.noise_exponent
        state_dict["noise_exponent_obj"]=self.noise_exponent_obj
        state_dict["n_noise"]=self.n_noise
        state_dict["n_noise_obj"]=self.n_noise_obj
        state_dict["noise_scale"]=self.noise_scale
        state_dict["noise_exponent2"]=self.noise_exponent2



        state_dict["X"]=np.asarray(self.X)
        state_dict["y"]=np.asarray(self.y)


        state_dict["n_feature_categories_"]=self.n_feature_categories_

        ####Save custom data structures
        ###########
        cdef list classification_values_list_python=[]
        for ind in range(self.classification_values_list.size()):
            classification_values_list_python.append(self.classification_values_list[ind].getstate())
        state_dict["classification_values_list"]=classification_values_list_python
        ###########
        cdef list full_tree_list_python=[]
        for ind in range(self.full_tree_vector.size()):
            full_tree_list_python.append(self.full_tree_vector[ind].getstate())
        state_dict["full_tree_vector"]=full_tree_list_python
        ###########

        state_dict["full_tree_indicies_list"]=self.full_tree_indicies_list
        state_dict["full_tree_to_local_tree_lookup_list"]=self.full_tree_to_local_tree_lookup_list
        state_dict["local_weight_tree"]=self.local_weight_tree
        state_dict["node_info"]=self.node_info  #TODO: Verify this sets correctly


        state_dict["children_list"]=self.children_list
        state_dict["feature_pair_to_comb_feat_ind"]=self.feature_pair_to_comb_feat_ind
        state_dict["comb_feat_ind_to_feature_pair"]=self.comb_feat_ind_to_feature_pair
        state_dict["correlation_map"]=self.correlation_map

        state_dict["nbins"]=self.nbins
        state_dict["max_bins"]=self.max_bins
        state_dict["categorical_feature_inds"]=self.categorical_feature_inds
        state_dict["binning_method"]=self.binning_method

        state_dict["categorical_bin_list"]=self.categorical_bin_list
        state_dict["numeric_bin_list"]=self.numeric_bin_list
        # state_dict[""]=self.

        #########
        cdef map[vector[np_long],np_long] node_label_to_index_map
        cdef pair[vector[np_long],np_long] label_index_pair
        node_label_to_index_list_by_level  =[]

        for node_label_to_index_map in self.node_label_to_index_dict_list: # iterate over levels
            node_label_to_index_list = [] #reset list that holds map
            for label_index_pair in node_label_to_index_map: # iterate over each entry in map
                node_label_to_index_list.append(label_index_pair) # add key, value pair to list
            node_label_to_index_list_by_level.append(node_label_to_index_list) #add list for each level


        state_dict["node_label_to_index_dict_list"]=node_label_to_index_list_by_level
        #########

        nodes_list_by_level  =[]
        cdef map[vector[np_long], pair[vector[np_long],vector[np_long] ] ] nodes_map
        cdef pair[vector[np_long], pair[vector[np_long],vector[np_long] ]] node_pair


        for nodes_map in self.nodes: # iterate over levels
            nodes_list= [] #reset list that holds map
            for node_pair in nodes_map: # iterate over each entry in map
                nodes_list.append(node_pair) # add key, value pair to list
            nodes_list_by_level.append(nodes_list) #add list for each level

        state_dict["nodes"]=nodes_list_by_level

        return state_dict

    def __setstate__(self, state_dict):

        self._estimator_type = "classifier" #fixed value

        self.n_features=state_dict["n_features"]
        self.ave_corr=state_dict["ave_corr"]

        # self.parents_list=state_dict["parents_list"]


        self.feature_names_txt=state_dict["feature_names_txt"]


        self.classes_=state_dict["classes_"]

        self.n_classes=state_dict["n_classes"]
        self.independence_model_c=state_dict["independence_model_c"]


        self.max_fully_connected_depth=state_dict["max_fully_connected_depth"]
        self.n_max_classify=state_dict["n_max_classify"]
        self.max_depth=state_dict["max_depth"]
        self.n_min_to_add_leaf=state_dict["n_min_to_add_leaf"]
        self.features_per_node=state_dict["features_per_node"]

        self.node_split_model=state_dict["node_split_model"]
        self.probability_scaling_method=state_dict["probability_scaling_method"]
        self.usefulness_model=state_dict["usefulness_model"]
        self.independence_model=state_dict["independence_model"]

        self.node_split_fraction=state_dict["node_split_fraction"]
        self.node_split_fraction_obj = state_dict['node_split_fraction_obj']
        self.noise_exponent=state_dict["noise_exponent"]
        self.noise_exponent_obj =state_dict["noise_exponent_obj"]
        self.n_noise=state_dict["n_noise"]
        self.n_noise_obj = state_dict["n_noise_obj"]
        self.noise_scale=state_dict["noise_scale"]
        self.noise_exponent2=state_dict["noise_exponent2"]


        self.X=state_dict["X"]
        self.y=state_dict["y"]

        self.n_feature_categories_=state_dict["n_feature_categories_"]

        ##### load custom data structures
        ###########################
        cdef list classification_values_list_python #single_class_val
        cdef np_long ind
        classification_values_list_python = state_dict["classification_values_list"]
        self.classification_values_list.clear()

        self.classification_values_list.resize(len(classification_values_list_python))

        for ind in range( len(classification_values_list_python) ):
            self.classification_values_list[ind].setstate(classification_values_list_python[ind])
        ###########################
        cdef list full_tree_list_python #single_class_val
        full_tree_list_python = state_dict["full_tree_vector"]
        self.full_tree_vector.clear()

        self.full_tree_vector.resize(len(full_tree_list_python))

        for ind in range( len(full_tree_list_python) ):
            self.full_tree_vector[ind].setstate(full_tree_list_python[ind])
        #########################



        self.full_tree_indicies_list=state_dict["full_tree_indicies_list"]
        self.full_tree_to_local_tree_lookup_list=state_dict["full_tree_to_local_tree_lookup_list"]
        self.local_weight_tree=state_dict["local_weight_tree"]
        # self.parents_list=state_dict["parents_list"]
        self.node_info=state_dict["node_info"]



        self.children_list=state_dict["children_list"]
        self.feature_pair_to_comb_feat_ind=state_dict["feature_pair_to_comb_feat_ind"]
        self.comb_feat_ind_to_feature_pair=state_dict["comb_feat_ind_to_feature_pair"]
        self.correlation_map=state_dict["correlation_map"]

        self.nbins=state_dict["nbins"]
        self.max_bins=state_dict["max_bins"]
        self.categorical_feature_inds=state_dict["categorical_feature_inds"]
        self.binning_method=state_dict["binning_method"]

        self.categorical_bin_list=state_dict["categorical_bin_list"]
        self.numeric_bin_list=state_dict["numeric_bin_list"]

        #########
        node_label_to_index_list_by_level= state_dict["node_label_to_index_dict_list"]
        cdef list node_label_to_index_list
        cdef pair[vector[np_long],np_long] label_index_pair
        cdef map[vector[np_long],np_long] node_label_to_index_map
        cdef vector[map[vector[np_long],np_long]] node_label_to_index_dict_vector

        for node_label_to_index_list in node_label_to_index_list_by_level:
            # convert list back into map
            for label_index_pair in node_label_to_index_list: #iterate over key, value pairs in list and convert to mappable key, value pairs
                node_label_to_index_map.insert(label_index_pair)
            #add map to vector
            node_label_to_index_dict_vector.push_back(node_label_to_index_map)
        self.node_label_to_index_dict_list = node_label_to_index_dict_vector


        #########
        nodes_list_by_level = state_dict["nodes"]
        cdef list nodes_list
        cdef pair[vector[np_long], pair[vector[np_long],vector[np_long] ]] node_pair
        cdef map[vector[np_long], pair[vector[np_long],vector[np_long] ] ] nodes_map
        cdef vector [map[vector[np_long], pair[vector[np_long],vector[np_long] ] ] ] nodes #1st level: tree level 2nd level: map of node label to pair (node info, children feature num)

        for nodes_list in nodes_list_by_level:
            #convert list back into map
            for node_pair in nodes_list:
                nodes_map.insert(node_pair)
            # add map to vector
            nodes.push_back(nodes_map)
        self.nodes=nodes





    def get_params(self, deep=True):

        return {"max_fully_connected_depth": self.max_fully_connected_depth, "n_max_classify": self.n_max_classify,
                "max_depth":self.max_depth, "n_min_to_add_leaf":self.n_min_to_add_leaf,"features_per_node":self.features_per_node,
                "node_split_model":self.node_split_model,"node_split_fraction":self.node_split_fraction_obj,"probability_scaling_method":self.probability_scaling_method,
                "noise_exponent":self.noise_exponent_obj,"noise_exponent2":self.noise_exponent2,"usefulness_model":self.usefulness_model,
                "n_noise":self.n_noise_obj,"independence_model":self.independence_model, "feature_names_txt":self.feature_names_txt,
                "nbins":self.nbins,"max_bins":self.max_bins,"categorical_feature_inds":self.categorical_feature_inds,"binning_method":self.binning_method}


    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            # setattr(self, parameter, value)

            if hasattr(self,parameter):
                setattr(self, parameter, value)
                #hack to store float values as objects for skikit-learn checks
                if parameter=="noise_exponent":
                    self.noise_exponent_obj=value
                elif parameter=="node_split_fraction":
                    self.node_split_fraction_obj=value
                elif parameter=="n_noise":
                    self.n_noise_obj = value

            else:
                if parameter!="noise_exponent2": ## allow for none value
                    raise AttributeError("Cannot set parameter {}, no such parameter exists".format(parameter))
        return self


    cdef string get_name(self, np_long comb_feat_ind):
        """
        returns a C++ string with the text name for feature ind
        
        :param comb_feat_ind: 
        :return: 
        """
        cdef str name_txt, feature_txt, cat_txt
        cdef np_long feature_num
        cdef dict bin_dict
        cdef np_float[:] bin_list
        cdef pair[np_long,np_long] feature_pair

        feature_pair  =  self.comb_feat_ind_to_feature_pair[comb_feat_ind]

        # get name of feature
        if self.feature_names_txt is not None:
            feature_txt  = str(self.feature_names_txt[feature_pair.first])
        else:
            feature_txt  = "x"+str(feature_pair.first)

        #get name of category
        cat_txt = str(feature_pair.second)
        if self.categorical_bin_list is not None: #search categorical variables
            for feature_num, bin_dict in self.categorical_bin_list:
                if feature_num==feature_pair.first:
                    feature_bin = bin_dict.get(feature_pair.second,None)
                    if feature_bin is not None:
                        cat_txt=str(feature_bin)
                    else:
                        cat_txt="bin #{0}".format(feature_pair.first)
                    break
            else: #search numeric features if categorical variable not found
                if self.numeric_bin_list is not None:
                    for feature_num, bin_list in self.numeric_bin_list:
                        if feature_num==feature_pair.first:
                            if feature_pair.second==0:
                                cat_txt="[{0:.4g} : {1:.4g}]".format(bin_list[0],bin_list[1])
                            else:
                                cat_txt="({0:.4g} : {1:.4g}]".format(bin_list[feature_pair.second],bin_list[feature_pair.second+1])
                            break


        name_txt=feature_txt+"="+cat_txt

        cdef string name_txt_cpp
        name_txt_cpp=name_txt.encode('UTF-8')

        return name_txt_cpp



    cpdef fit(self, X_in, y_in, list feature_names_txt=None, list categorical_bin_list=None, list numeric_bin_list=None):
        """
        Train the classifier on data X with targets y
        Assume X is integer array with features numbered 0 to N, -1 is missing feature
        assume classes (y) are sequentially ordered from  0 to N

        :param X_in:
        :type X_in: np.ndarray
        :param y:
        :param categorical_bin_list: list of categorical bin tuples, if passed overrides bins derived from bin_data function
        :param numeric_bin_list: list of numeric bin tuples, if passed overrides bins derived from bin_data function
        :return: the classifier object itself i.e. "self"
        """
        # print('X= ',X_in,'\ny= ',y_in)

        cdef np_long[:,:] X
        cdef np_long[:] y

        #Check that X and y have correct shape
        # X_in, y_in = check_X_y(X_in, y_in, y_numeric=False, force_all_finite=True) #checks needed to pass scikit-learn validation



        ############# bin input data if needed.
        if self.nbins>0:
            X_in, y_in = check_X_y(X_in, y_in, y_numeric=False, force_all_finite=False, dtype=None) # disable checks to allow for missing values as np.nan
            X,categorical_bin_list,numeric_bin_list = bin_data(X_in,nbins=self.nbins,max_bins=self.max_bins,categorical_feature_inds=self.categorical_feature_inds,
                           binning_method=self.binning_method, retbins=True)
            # X = df.values.astype(np.int32)

            # # save bins values
            # self.numeric_bin_list=numeric_bin_list2
            # self.categorical_bin_list=categorical_bin_list2

        else:
            X, y_in = check_X_y(X_in, y_in, dtype=np.int32, y_numeric=False) # use all checks if data is already binned
            # X = X_in.astype(np.int32)


        ## Save  feature names needed for explainability
        if feature_names_txt is not None:
            self.feature_names_txt=feature_names_txt
        if (self.feature_names_txt is not None) and len(self.feature_names_txt) != X.shape[1]:
                warn("feature feature name dimension, {}, does not agree with array shape {}".format(len(self.feature_names_txt),X.shape[1]))
                self.feature_names_txt=None
        if categorical_bin_list is not None:
            self.categorical_bin_list=categorical_bin_list
        if numeric_bin_list is not None:
            self.numeric_bin_list=numeric_bin_list


        ##############convert classification categories to integers

        #encode y into an integer array from 0 to N:
        self.classes_ = unique_labels(y_in)

        # create dict to look up class number based on class text
        classes_dict = {self.classes_[ind]:ind for ind in range(len(self.classes_))}
        # using dict encode y into integers
        y = np.array( [classes_dict[val] for val in y_in], dtype=np.int32) #store numpy array to memview

        # print(y)
        self.X = X
        self.y = y

        ###############
        self.independence_model_c = self.independence_model.encode('UTF-8')


        if self.independence_model_c !=b"none" and self.max_fully_connected_depth<2:
            raise ValueError("Can only use independence models when max_fully_connected_depth>=2")

        self.noise_scale = self.n_max_classify**self.noise_exponent

        cdef np_long feature_num
        cdef np_long ind

        # TODO: check for -1 labels for unlabeled data (useful once independence measure implemented)

        # Store the classes seen during fit
        self.n_classes = np.amax(y)+1

        # TODO: add checks that parameter are valid

        self.n_features = X.shape[1]

        # create a list of integers to count the number of features per category
        self.n_feature_categories_.resize(self.n_features,0)

        # count all categorical features:
        cdef np_long[:] feature_col
        for feature_num in range(self.n_features):
            feature_col = X[:,feature_num] # should be a view into X
            self.n_feature_categories_[feature_num]=np.max(feature_col)+1

        ######### encode feature number + value into integers and create lookup dicts:
        cdef np_long category_num, count
        cdef pair[np_long,np_long] feature_pair

        count= 0
        #initialize lookup dicts
        self.feature_pair_to_comb_feat_ind.clear() # feature_pair = tuple of (feature num, category num)
        self.comb_feat_ind_to_feature_pair.clear() # for debugging

        #iterate over features
        cdef np_long min_cat, n_categories
        #include category =-1 when calculating independence to represent all categories
        if self.independence_model_c==b"none":
            min_cat=0
        else:
            min_cat=-1

        for feature_num in range(self.n_features):
            n_categories=self.n_feature_categories_[feature_num]
            for category_num in range(min_cat,n_categories):
                count+=1
                feature_pair.first = feature_num
                feature_pair.second = category_num
                self.comb_feat_ind_to_feature_pair[count]= feature_pair
                self.feature_pair_to_comb_feat_ind[feature_pair] = count


        ###############
        #create a empty list of dicts to store all the nodes by level (stores Node objects)
        self.nodes.resize(self.max_depth+1)
            #1st level: tree level 2nd level: map of node label to pair (node info, children feature num)

        #create indicies for accessing node data
        self.node_info.resize(self.max_depth+1)
        for ind in range(self.max_depth+1):
            initialize_node(self.node_info[ind],self.n_classes,ind)


        ############################### create the classifier tree
        self.create_classifier_tree()
        # fills in self.nodes object with nodes at each level of the tree

        # #########  DEBUG
        # ## print classifier tree
        # for level2 in range(self.nodes.size()):
        #     for key_pair in self.nodes[level2]:
        #         label_str = ""
        #         for comb_feat_ind in key_pair.first:
        #             label_str+=self.get_name(comb_feat_ind)+", "
        #         print("label = ",label_str, " node info= ",key_pair.second.first, " children feature num",key_pair.second.second)


        #TODO: Optional delete X and y from self (X & y stored to self for easy access by recursive functions)

        #########################################
        ###### preprocesss tree for classification


        cdef FullTreeData[np_float,np_long ] * full_tree_data


        cdef map[vector[np_long],pair[vector[np_long],vector[np_long]]] node_map
        # map of node label to pair (node info, children feature num)
        cdef np_long node_num=-1
        cdef np_long class_ind, label_size, feature, feature_ctr
        cdef np_long level, n_features_combos, n_levels, parent_num, parent_feature_ctr, node_index

        cdef pair[vector[np_long],pair[vector[np_long],vector[np_long]]] node_row
        cdef vector[np_long] parent_vec
        cdef vector[np_long] total_count


        #### iterate through each level of tree and store values
        n_levels = self.nodes.size()


        #intialize vector objects
        self.node_label_to_index_dict_list.resize(n_levels)
        self.children_list.resize(n_levels) #intialize children vec size
        self.full_tree_vector.resize(n_levels)

        for level in range(n_levels):
            node_map = self.nodes[level]
            n_features_combos = node_map.size()

            ### intitialize single level arrays that will be stored
            full_tree_data = & self.full_tree_vector[level]
            full_tree_data[0].initialize(n_features_combos, self.n_classes, level)
            # print("level= ",level," n_features_combos= ",n_features_combos)

            self.node_label_to_index_dict_list[level].clear()
            self.children_list[level].resize(n_features_combos)

            ### iterate over each node in the list
            node_num=-1
            # print("fit, level= ",level," node_map size= ",node_map.size())
            for node_row in node_map:
                # print("node row= ",node_row)
                node_num+=1
                #set node in full tree
                full_tree_data[0].set_node(node_num)

                #store children
                self.children_list[level][node_num]=node_row.second.second
                sort(self.children_list[level][node_num].begin(),self.children_list[level][node_num].end()) #sort children

                #store counts
                for class_ind in range(self.n_classes):
                    full_tree_data[0].counts[class_ind]=node_row.second.first[self.node_info[level].counts_index+class_ind] # store counts

                ### iterate over each feature in node and store feature label
                label_size = node_row.first.size()
                for feature_ctr in range( label_size ):
                    feature = node_row.first[feature_ctr]
                    full_tree_data[0].label[feature_ctr]=feature #store feature label (sorted by feature when nodes were created)

                ### convert all features in node into a hashable set and store to dict
                self.node_label_to_index_dict_list[level][node_row.first]=node_num

                ### store parents
                if level==1: # need to handle level 1 separately due to having unlabeled parent
                    full_tree_data[0].parent_indicies[0] = 0
                elif level>1:
                    for parent_num in range(0,level):  # will skip for level=0 due to having no parents (range(0) = none)
                        if node_row.second.first[self.node_info[level].parents_index_begin +self.node_info[level].parents_index_step*parent_num]==-1:
                            break # no more parents
                        parent_vec.clear()
                        for parent_feature_ctr in range(self.node_info[level].parents_index_step):
                            parent_vec.push_back(node_row.second.first[self.node_info[level].parents_index_begin +self.node_info[level].parents_index_step*parent_num + parent_feature_ctr])

                        full_tree_data[0].parent_indicies[parent_num] = self.node_label_to_index_dict_list[level-1][parent_vec]

        #     print("level= ",level," child list= ", self.children_list[level])
        #
        # for level in range(n_levels):
        #     print("fit3: level= ",level," child list= ", self.children_list[level])
        #
        #
        # for level in range(n_levels):
        #     full_tree_data = & self.full_tree_vector[level]
        #     node_map = self.nodes[level]
        #     n_features_combos = node_map.size()

            #### compute probability and noise weight for all nodes

            total_count.assign(n_features_combos,0)
            for node_index in range (n_features_combos):
                full_tree_data[0].set_node(node_index) #set stored node index
                # print("node_index= ", node_index, " full tree data ",full_tree_data[0].getstate())


                for class_ind in range(self.n_classes):
                    total_count[node_index]+=full_tree_data[0].counts[class_ind]

                for class_ind in range(self.n_classes):
                    full_tree_data[0].p0[class_ind]=full_tree_data[0].counts[class_ind]/<np_float>total_count[node_index]

                # print("noise weight ",full_tree_data[0].noise_weight[0])
                # print("new  weight  ",self.compute_noise_value(total_count[node_index]))
                full_tree_data[0].noise_weight[0]=self.compute_noise_value(total_count[node_index])
                # print("saved weight ",full_tree_data[0].noise_weight[0])
                #
                # print("full tree data post write      ",full_tree_data[0].getstate())


        # for level in range(n_levels):
        #     print("level",level)
        #
        # for level in range(n_levels):
        #     print("fit: level= ",level," child list= ", self.children_list[level])
        #     print("fit: full tree vector row ",level,": ", self.full_tree_vector[level].getstate())


            # print (full_tree_data[0].getstate())
        #######debug!!
        # print("train: full tree vector row 1 ",self.full_tree_vector[1].getstate())
        # print("train: full tree vector row 2 ",self.full_tree_vector[2].getstate())

        # raise ValueError
        ###################
        ################### compute counts needed for finding independence

        cdef map [pair[np_long,np_long],vector[np_long]] independence_count_map

        cdef vector[np_long] counts, label_vec, summed_counts
        cdef pair[np_long,np_long] comb_feat_pair, feature_pair1, feature_pair2
        cdef map[vector[np_long],np_long] * node_label_to_index_dict_ptr
        cdef np_long comb_feat_ind1, comb_feat_ind2, feature_num1, feature_num2

        # cdef vector[np_long] *
        cdef np_long * n_xx_vec_ptr
        cdef np_long * n_1x_vec_ptr
        cdef np_long * n_x1_vec_ptr
        cdef np_long * n_11_vec_ptr
        cdef vector[np_long] n_11_vec

        cdef np_float  n_xx, n_1x, n_x1, n_11
        cdef np_long comb_feat_ind1all,comb_feat_ind2all

        cdef np_float a, b, corr
        cdef np_float corr_sum, corr_count

        if self.independence_model_c!=b"none":
            counts.assign(self.n_classes,0)

            node_label_to_index_dict_ptr = & self.node_label_to_index_dict_list[2]

            full_tree_data = & self.full_tree_vector[2] # use second level of tree

            #create map to store counts per class for computing independence
            independence_count_map.clear()

            #iterate through all features
            for feature_num1 in range(self.n_features):
                n_categories1=self.n_feature_categories_[feature_num1]

                # iterate through second set of features
                for feature_num2 in range(self.n_features):
                    if feature_num1==feature_num2: #skip to next feature if f1=f2
                        continue
                    n_categories2=self.n_feature_categories_[feature_num2]
                    #initialize vector to store counts of feature1, all cat1, feature2, all cat2
                    summed_counts.assign(self.n_classes,0)

                    #iterate through categories for each feature
                    for category_num1 in range (n_categories1):
                        #find single # label for feature, cat combo
                        feature_pair.first = feature_num1
                        feature_pair.second = category_num1
                        comb_feat_ind1 = self.feature_pair_to_comb_feat_ind.at(feature_pair)
                        #initialize vector to store counts of feature1, cat1, feature2, all cat2
                        counts.assign(self.n_classes,0)

                        for category_num2 in range (n_categories2):
                            #find single # label for feature, cat combo
                            feature_pair.first = feature_num2
                            feature_pair.second = category_num2
                            comb_feat_ind2 = self.feature_pair_to_comb_feat_ind.at(feature_pair)

                            #find the full tree index by creating a vector with the label
                            label_vec.clear()
                            if comb_feat_ind1<comb_feat_ind2:
                                label_vec.push_back(comb_feat_ind1)
                                label_vec.push_back(comb_feat_ind2)
                            else:
                                label_vec.push_back(comb_feat_ind2)
                                label_vec.push_back(comb_feat_ind1)

                            # print("feature {0}, cat {1} and feature {2} cat {3}".format(feature_num1,category_num1,feature_num2,category_num2))
                            # print("f1= {0}, f2= {1}".format(comb_feat_ind1,comb_feat_ind2))

                            if node_label_to_index_dict_ptr[0].count(label_vec)==1: # only add counts if node exists / counts nonzero
                                full_tree_index = node_label_to_index_dict_ptr[0].at(label_vec)
                                full_tree_data[0].set_node(full_tree_index)
                                for class_ind in range(self.n_classes):
                                    counts[class_ind]+=full_tree_data[0].counts[class_ind]

                                # print("full_tree_index=",full_tree_index)
                                # print("counts=",self.counts_list[2][full_tree_index,:])

                        #store counts to the map
                        comb_feat_pair.first = comb_feat_ind1
                        feature_pair.first = feature_num2
                        feature_pair.second = -1
                        comb_feat_pair.second = self.feature_pair_to_comb_feat_ind[ feature_pair]
                        if comb_feat_pair.second<comb_feat_pair.first: #sort the label pair
                            temp = comb_feat_pair.first
                            comb_feat_pair.first = comb_feat_pair.second
                            comb_feat_pair.second = temp

                        # print("f1={}, f2={}, counts={}\n".format(comb_feat_pair.first, comb_feat_pair.second,counts))
                        independence_count_map[comb_feat_pair]=counts

                        for class_ind in range(self.n_classes):
                            summed_counts[class_ind]+=counts.at(class_ind)
                    #store summed counts to the map
                    comb_feat_pair.first = comb_feat_ind1

                    feature_pair.first = feature_num1
                    feature_pair.second = -1
                    comb_feat_pair.first  = self.feature_pair_to_comb_feat_ind[ feature_pair]
                    # print("label",label)
                    feature_pair.first = feature_num2
                    # print("label",label)
                    comb_feat_pair.second = self.feature_pair_to_comb_feat_ind[ feature_pair]
                    pair_sort(comb_feat_pair)

                    # ##DEBUG:
                    # if independence_count_map.count(comb_feat_pair)==1:
                    #     print("old sum =",independence_count_map.at(comb_feat_pair))
                    # print("f1={}, f2={}, summed counts={}\n\n\n".format(comb_feat_pair.first, comb_feat_pair.second,summed_counts))

                    independence_count_map[comb_feat_pair]=summed_counts

            ################### compute correlations needed for finding independence

            # cdef vector[np_long] * n_xx_vec_ptr, n_1x_vec_ptr, n_x1_vec_ptr, n_11_vec_ptr
            # cdef np_long  n, n_1x, n_x1, n_11


            #iterate over all combinations of features
            for full_tree_index in range(full_tree_data[0].n_nodes):
                full_tree_data[0].set_node(full_tree_index) #set node being used in tree
                #find labels / feature numbers needed to load counts
                comb_feat_ind1= full_tree_data[0].label[0]
                comb_feat_ind2= full_tree_data[0].label[1]



                # feature_pair1= self.comb_feat_ind_to_feature_pair.at(comb_feat_ind1)
                # assign each element in pair to avoid some weird windows compiling error
                feature_pair1.first = self.comb_feat_ind_to_feature_pair.at(comb_feat_ind1).first
                feature_pair1.second = self.comb_feat_ind_to_feature_pair.at(comb_feat_ind1).second



                feature_num1 = feature_pair1.first
                feature_pair1.second = -1
                comb_feat_ind1all = self.feature_pair_to_comb_feat_ind.at(feature_pair1)

                # assign each element in pair to avoid some weird windows compiling error
                # feature_pair2= self.comb_feat_ind_to_feature_pair.at(comb_feat_ind2)
                feature_pair2.first= self.comb_feat_ind_to_feature_pair.at(comb_feat_ind2).first
                feature_pair2.second= self.comb_feat_ind_to_feature_pair.at(comb_feat_ind2).second

                feature_num2 = feature_pair2.first
                feature_pair2.second = -1
                comb_feat_ind2all = self.feature_pair_to_comb_feat_ind.at(feature_pair2)

                #find counts needed to compute correlations

                comb_feat_pair.first = comb_feat_ind1
                comb_feat_pair.second = comb_feat_ind2all
                pair_sort(comb_feat_pair)
                n_1x_vec_ptr = & independence_count_map.at(comb_feat_pair).at(0)

                comb_feat_pair.first = comb_feat_ind1all
                comb_feat_pair.second = comb_feat_ind2
                pair_sort(comb_feat_pair)
                n_x1_vec_ptr = & independence_count_map.at(comb_feat_pair).at(0)


                comb_feat_pair.first = comb_feat_ind1all
                comb_feat_pair.second = comb_feat_ind2all
                pair_sort(comb_feat_pair)
                n_xx_vec_ptr = & independence_count_map.at(comb_feat_pair).at(0)


                n_11_vec_ptr = & full_tree_data[0].counts[0]  # is pointer to an integer /c-array

                ### sum counts over classes
                n_xx=0
                n_1x=0
                n_x1=0
                n_11=0
                for class_ind in range(self.n_classes):
                    n_xx+=n_xx_vec_ptr[class_ind]
                    n_1x+=n_1x_vec_ptr[class_ind]
                    n_x1+=n_x1_vec_ptr[class_ind]
                    n_11+=n_11_vec_ptr[class_ind]


                comb_feat_pair.first = comb_feat_ind1
                comb_feat_pair.second = comb_feat_ind2
                pair_sort(comb_feat_pair)

                if n_xx!=n_1x and n_xx!=n_x1:
                    a = n_1x*n_x1/n_xx
                    b = n_x1
                    # corr = abs((n_xx*n_11-n_1x*n_x1)/sqrt(n_1x*n_x1*(n_xx-n_1x)*(n_xx-n_x1) ))
                    # corr = abs(1/(a**2*b-a*b**2)*( (b-2*a)*n_11**2+ (2*a**2-b**2)*n_11 )-1)
                    corr = min(abs((n_11 -a )/(b-a)),1)
                    # corr =1
                    self.correlation_map[comb_feat_pair]=corr
                # else:
                #     self.correlation_map[comb_feat_pair]=0

                    if self.correlation_map.at(comb_feat_pair)>=0 and self.correlation_map.at(comb_feat_pair)<=1:
                        pass
                    else:
                        print((n_xx*n_11-n_1x*n_x1))
                        print(n_1x*n_x1*(n_xx-n_1x)*(n_xx-n_x1))
                        print(sqrt(n_1x*n_x1*(n_xx-n_1x)*(n_xx-n_x1)))
                        print("\nf1=",feature_num1, " f2=",feature_num2, "f1_comb=",comb_feat_ind1, " f2_comb=",comb_feat_ind2, )
                        print("n_xx=",n_xx," n_1x=",n_1x," n_x1=",n_x1," n_11=",n_11)
                        print("correlation = ",self.correlation_map.at(comb_feat_pair))
                        print("a= ",a,"b= ",b)
                        print("This should be -1:",1/(a**2*b-a*b**2)*( (b-2*a)*0**2+ (2*a**2-b**2)*0 )-1 )
                        print("This should be 0:",1/(a**2*b-a*b**2)*( (b-2*a)*a**2+ (2*a**2-b**2)*a )-1 )
                        print("This should be 1:",1/(a**2*b-a*b**2)*( (b-2*a)*b**2+ (2*a**2-b**2)*b )-1 )

                        # raise ValueError ("GAAHHHHHHHHHH")

            #### find average correlation
            corr_sum=0
            corr_count=0
            for correlation_val in self.correlation_map:
                corr_count+=1
                corr_sum+=correlation_val.second

            self.ave_corr = corr_sum / corr_count
            print("average correlation = ", self.ave_corr)#, "indep model= ", self.independence_model_c)


        # print("child list size = ",self.children_list.size())
        # for level in range  (n_levels):
        #     print("fit: level= ",level," child list= ", self.children_list[level])

        return self



    cdef inline np_float compute_noise_value(self,const np_long & total_count):
        return min((total_count**self.noise_exponent) / self.noise_scale , 1 )


    #void
    cdef create_and_process_node(self, np_long & child_depth, vector[np_long] & child_label, vector[vector[np_long] ] & data_by_class, vector[np_long] & parent_label, np_long & child_comb_feat_ind):#, np_long & parent_depth, vector[np_long] & parent_label):
        """
        child is the node to be created
        parent is the parent of the node to be created
        :param child_depth: depth of child
        :param child_label: node label  C++ vector of features (each label is encoded as a comb_feat_ind)
        :param data_by_class: data sorted by class
        :param parent_depth: depth of parent
        :param parent_label:node label (tuple) of parent
        :param child_comb_feat_ind: the unique new feature added the child that is not in parent
        """


        #### count data
        cdef np_long class_ind, class_counts, total_data_count = 0, num_nonzero_classes =0
        for class_ind in range(self.n_classes):
            class_counts = data_by_class[class_ind].size()
            total_data_count+=class_counts #count total data
            if class_counts>0:
                num_nonzero_classes+=1  # count number of nonzero classes

        #### do not create node if there is not enough data
        if total_data_count<self.n_min_to_add_leaf:
            return

        #### create new node
        cdef pair[vector[np_long],vector[np_long]] new_node
        cdef vector[np_long] * new_node_data_ptr

        self.nodes[child_depth][child_label] = new_node #perform a copy of new node to create new node
        new_node_data_ptr = & self.nodes[child_depth][child_label].first
        new_node_data_ptr[0].resize(self.node_info[child_depth].size,-1)  #intialize vector to the correct size

        # update counts of node
        cdef np_long class_num
        for class_num in range(self.n_classes):
            new_node_data_ptr[0][self.node_info[child_depth].counts_index+class_num]=data_by_class[class_num].size()

        # add parent label to child
        self.add_parent_label(child_label, child_depth, parent_label, child_comb_feat_ind)


        # only create children if there is data in more than one class
        if num_nonzero_classes>1:
            self.create_children(child_depth,child_label, data_by_class)


    #void
    cdef create_children(self,np_long & parent_depth, vector[np_long] & parent_label, vector[vector[np_long] ] & parent_data_by_class):
        """
        create a function that can be called recursively to fill in nodes

        :param parent_depth: 
        :param parent_label: 
        :param parent_data_by_class: 
        :return: 
        """

        cdef np_long child_depth
        child_depth=parent_depth+1

        #check if the children are too deep
        if self.max_depth>0 and child_depth>self.max_depth:
            return

        # check that parent has not been updated (it must exist)
        cdef np_long parent_updated_index
        parent_updated_index = self.node_info[parent_depth].updated_index

        if self.nodes[parent_depth][parent_label].first[parent_updated_index] ==1:
            return
        else:
            self.nodes[parent_depth][parent_label].first[parent_updated_index]=1

        #################################################
        ####### Create a list of all possible child labels
        #TODO: Potentially more efficient to create list once and then exclude features based on parents

        cdef vector[np_long] childrens_comb_feat_inds             # create a list of all child labels to add
        cdef vector[np_long] comb_features_to_add
        cdef vector[np_long] potential_comb_feat_inds   # list of all category coded features that can be added to the children
        cdef vector[np_long] potential_features # list of all possible features (no category nums)
        cdef pair[np_long,np_long] feature_pair

        cdef np_long category_num, comb_feat_ind, n_categories

        # for each feature, iterate over all possible categories
        cdef np_long feature_num
        for feature_num in range(self.n_features):
            n_categories=self.n_feature_categories_[feature_num]
            comb_features_to_add.resize(0)


            #determine if a given feature is in the label (by checking each category of that feature) and if not,
                # add all indices for the given feature to potential_comb_feat_inds
            for category_num in range (n_categories):
                #convert pair (feature_num, category_num) into a single number index and add to list comb_features_to_add
                feature_pair.first = feature_num
                feature_pair.second = category_num
                comb_feat_ind = self.feature_pair_to_comb_feat_ind[ feature_pair]
                comb_features_to_add.push_back(comb_feat_ind)

                #if comb_feat_ind in parent features skip this feature
                for parent_num in range(parent_label.size() ):
                    if comb_feat_ind== parent_label[parent_num]:
                        break #if parent found break twice
                else: # no parent found so go to next iteration
                    continue
                break # parent found so break again(broke out of previous loop thus reaching here)
            else: # no parent found, add features to potential_comb_feat_inds
                potential_comb_feat_inds.insert(potential_comb_feat_inds.end(), comb_features_to_add.begin(), comb_features_to_add.end())
                potential_features.push_back(feature_num)


        ##############################################
        ########  determine what children to create using list of all possible children:

        cdef np_long datum_index, num_random, n_unshuffled, begin, rand_num, temp, feature_ind, n_potential_features
        cdef vector[np_float] gini_by_features
        cdef np_long datum_category, total_count, feature_ctr
        cdef vector[vector[np_long]] data_counts
        cdef vector[np_long] data_counts_vec, count_per_cat, selected_features
        cdef np_float weighted_gini, gini_per_cat, p_i
        cdef np_long[:] gini_inds

        n_potential_features = potential_features.size()

        if (child_depth<=self.max_fully_connected_depth) or (n_potential_features<self.features_per_node): # create all fully connected children
            childrens_comb_feat_inds = potential_comb_feat_inds
        else:
            if self.node_split_model =="gini" or self.node_split_model =="gini_random":
                if self.node_split_model == "gini_random":
                    ### randomly choose  subset of features, use  Fisher-Yates shuffle:

                    #compute how many features to use
                    num_random = lround(n_potential_features*self.node_split_fraction)

                    n_unshuffled = n_potential_features #number of unshuffled features
                    for begin in range(num_random):
                        # find unbiased random number in range [0,n_unshuffled)
                        while True:
                            rand_num = rand()
                            if rand_num< (RAND_MAX - RAND_MAX % n_unshuffled):
                                break
                        rand_num = rand_num%n_unshuffled

                        #swap value at rand_num with value at begin
                        temp = potential_features[begin]
                        potential_features[begin] = potential_features[rand_num+begin]
                        potential_features[rand_num+begin] = temp

                        #decrement the number of unshuffled values and increment begin
                        n_unshuffled-=1

                    #limit potential_features to only the randomly selected values
                    potential_features.resize(num_random)
                    #TODO:  DEBUG / TEST THIS CODE / USE BETTER RANDOM NUMBERS

                # find gini coeficient (or entropy) of every potential feature:
                gini_by_features.clear()
                gini_by_features.resize(n_potential_features,1)
                for feature_ind in range(n_potential_features):   #iterate over each possible feature and analyze categories in the feature
                    feature = potential_features[feature_ind]

                    n_categories= self.n_feature_categories_[feature]

                    # count data by classification category in a given feature
                    # data_counts.clear()
                    data_counts.assign(n_categories,vector[np_long](self.n_classes, 0 ) )

                    # data_counts = np.zeros( (self.n_classes,n_categories),dtype=int )
                    for class_num in range(self.n_classes):
                        for datum_index in parent_data_by_class[class_num]:
                            datum_category = self.X[datum_index,feature]# find the category of feature "feature"
                            # if the category is not missing:
                            if datum_category>=0:
                                data_counts[datum_category][class_num]+=1

                    ###### gini coeficient
                    # count_per_cat = np.sum(data_counts,axis=0)
                    count_per_cat.assign(n_categories,0)
                    total_count = 0
                    for category_num in range(n_categories):
                        for class_num in range(self.n_classes):
                            count_per_cat[category_num] += data_counts[category_num][class_num]
                        total_count+=count_per_cat[category_num]

                    if total_count==0:
                        gini_by_features[feature_ind] = 10000  #if there are no features (i.e. all are missing, set gini to >1 so that this feature is not selected)
                                                             #max gini is 1
                    else:
                        weighted_gini = 0
                        for category_num in range(n_categories):
                            gini_per_cat=1
                            if count_per_cat[category_num]>0:
                                for class_num in range(self.n_classes):
                                    p_i = <np_float> data_counts[category_num][class_num]/count_per_cat[category_num]
                                    gini_per_cat -= p_i**2
                            ### could alternatively calculate entropy here #TODO add option for entropy

                            weighted_gini+=  <np_float>count_per_cat[category_num] /total_count*gini_per_cat

                        gini_by_features[feature_ind]=weighted_gini

                # find the features that minimize the gini coeficcient
                gini_inds = np.argsort(gini_by_features).astype(NP_LONG)

                for feature_ctr in range(self.features_per_node):
                    feature_num = gini_inds[feature_ctr]
                    selected_features.push_back(potential_features[feature_num])

            else:
                raise ValueError("Unkown node split model "+str(self.node_split_model))

            # ########   DEBUG
            # print("potential features = ",potential_features)
            # print("selected_features = ",selected_features)

            # add all possible children for selected features
            for feature_num in selected_features: #iterate over features
                n_categories = self.n_feature_categories_[feature_num]
                comb_features_to_add.clear()

                for category_num in range(n_categories):
                    feature_pair.first = feature_num
                    feature_pair.second = category_num
                    comb_features_to_add.push_back(self.feature_pair_to_comb_feat_ind[feature_pair])
                # childrens_comb_feat_inds.extend(comb_features_to_add)
                childrens_comb_feat_inds.insert(childrens_comb_feat_inds.end(), comb_features_to_add.begin(), comb_features_to_add.end())

        ###############################################
        ################  Given list of children to add, actually add children

        cdef vector[np_long] child_label, data_list
        cdef vector[vector[np_long] ] data_by_class

        for comb_feat_ind in childrens_comb_feat_inds:
            child_label = parent_label # copy parent label

            child_label.push_back(comb_feat_ind) # add child to parent label
            sort(child_label.begin(),child_label.end()) # sort label

            # check if child node exists and if not, create child node by calling create_and_process_node.
            if self.nodes[child_depth].find(child_label)==self.nodes[child_depth].end():

                # iterate through data and find all data that has the child feature in it
                feature_pair = self.comb_feat_ind_to_feature_pair[comb_feat_ind]
                child_feature = feature_pair.first
                child_category = feature_pair.second
                #### select data to pass to child

                #reset data_by_class to zero
                data_by_class.clear()
                data_by_class.resize(self.n_classes)

                for class_num in range(self.n_classes):
                    for datum_index in parent_data_by_class[class_num]: # iterate through each data row in each class

                            if self.X[datum_index,child_feature] ==child_category:  #if the category of label[0] in datum is the same as the category, label[1], add data
                                                            # automatically excludes NaNs
                                data_by_class[class_num].push_back(datum_index)

                self.create_and_process_node(child_depth, child_label, data_by_class, parent_label, comb_feat_ind)#, parent_depth=parent_depth, parent_label=parent_label)

            else: # child node already exists, add it to node lists
                self.add_parent_label(child_label,child_depth,parent_label, comb_feat_ind)



    cdef void add_parent_label(self, vector[np_long] & child_label, np_long & child_depth, vector[np_long] & parent_label, np_long & child_comb_feat_ind):
        """
        add parent to child by iterating through possible parents
        add child to parent
        
        Assumes parent /child pair has not been added to labels
        
        :param child_label: 
        :param child_depth: 
        :param parent_label: 
        :return: 
        """
        cdef np_long parent_num, local_parent_ind, parent_feature_ctr
        cdef vector[np_long] * child_data

        if child_depth<=0:
            return # no parents for top level node

        #add child to parent list
        self.nodes[child_depth-1][parent_label].second.push_back(child_comb_feat_ind)

        #add parent to child by iterating through possible parents and adding if not present
        child_data = & self.nodes[child_depth][child_label].first
        for parent_num in range(child_depth):
            #find index of start of parent label
            local_parent_ind = self.node_info[child_depth].parents_index_begin+parent_num*self.node_info[child_depth].parents_index_step
            if child_data[0][local_parent_ind]==-1:
                # no parents left, match not found, save parent
                for parent_feature_ctr in range(self.node_info[child_depth].parents_index_step):
                    child_data[0][local_parent_ind+parent_feature_ctr]=parent_label[parent_feature_ctr]
                break # we're done

            ### following code not needed as we're assuming parent has not been added yet
            # else:
            #     for parent_feature_ctr in range(self.node_info[child_depth].parents_index_step):
            #         if child_data[0][local_parent_ind]!=parent_label[parent_feature_ctr]:
            #             break #if feature does not match, break and search next parent
            #     else:
            #         print("Parent found")
            #         break
            #         #return # if it fully matches break out of for loop, no need to add parent

    #void
    cdef  create_classifier_tree(self):
        """
        # intially create a simple tree of all nodes up to max depth, eventually use more sophisticated feature selection
        features are labeled by tuples (feature #, category #)

        Also updates counts during tree creation as data needs to be split during creation

        :param data_list:
        :return:
        """

        cdef vector[np_long]  node_label, parent_label #default_node,
        cdef np_long current_depth, sample_ind, class_num, n_data, child_comb_feat_ind=0

        # create top level node
        current_depth = 0
        node_label.resize(0)
        parent_label = node_label

        # sort data into lists by class

        cdef vector[vector[np_long] ] data_by_class
        data_by_class.resize(self.n_classes)

        n_data = self.X.shape[0]
        for sample_ind in range(n_data):
            class_num = self.y[sample_ind]
            data_by_class[class_num].push_back(sample_ind)

        self.create_and_process_node(current_depth,node_label,data_by_class,parent_label,child_comb_feat_ind)#, parent_depth, default_node)




    cdef  find_used_feature_combos (self, np_long[:] example, vector[vector[np_long]] & full_tree_indicies_list,
                                   vector[vector[np_long]] & full_tree_to_local_tree_lookup_list,
                                    vector[vector[vector[np_long]]] & compressed_comb_feat_ind_child_tree, vector[vector[vector[np_long]]] & compressed_full_tree_ind_child_tree):
        """
        find combinations of features used in the particular example by following children in tree, also store children

        :param example:
        :param full_tree_indicies_list: return variable for used indicies in full tree

        :return:
        """
        # print("starting example",np.asarray(example),'\n' )
        #############################################
        ### find the nodes that contribute to the example
        # Convert example into a list feature indicies
        cdef np_long feature_num, comb_feat_ind
        cdef pair[np_long,np_long] feature_pair
        cdef map [pair[np_long,np_long],np_long].iterator label_itr
        cdef vector[np_long] comb_feats_vec

        for feature_num in range(self.n_features): # iterate through all features in the example
            # exclude missing valued features:
            if example[feature_num]>=0:
                #find comb_feat_ind or return -1 if it doesn't exist
                feature_pair.first = feature_num
                feature_pair.second = example[feature_num]
                label_itr = self.feature_pair_to_comb_feat_ind.find(feature_pair)
                if label_itr==self.feature_pair_to_comb_feat_ind.end():
                    comb_feat_ind = -1
                    # warn("Feature Num "+str(feature_num)+" with value "+str(example[feature_num])+" has not been seen before, it is being ignored")
                else:
                    comb_feat_ind = deref(label_itr).second
                    comb_feats_vec.push_back(comb_feat_ind)

        # sort feature list for fast set_intersection later
        sort(comb_feats_vec.begin(),comb_feats_vec.end())


        #########################################################
        # find combinations of feature indicies used in the particular example
        cdef np_long level
        # cdef np_long[:,:]  node_labels, parent_node_labels
        cdef np_long row,col

        cdef vector[np_long].iterator end_it, used_children_it
        cdef vector[np_long] used_children, child_label, parent_label, parent_indicies, child_full_tree_indicies
        cdef vector[np_long] * child_ptr
        cdef np_long * child_index
        cdef np_long ctr, index, parent_index_full_tree, label_ind, tree_depth, local_index, local_tree_size
        cdef vector[np_long] * full_tree_indicies_ptr
        cdef vector[np_long] * full_tree_to_local_tree_indicies_lookup_ptr
        cdef FullTreeData[np_float,np_long ] * parent_full_tree_data


        tree_depth = self.full_tree_vector.size()


        # create lists for values needed for classification
        full_tree_to_local_tree_lookup_list.resize(tree_depth) # create a vector of vectors that go from mask indicies to indicies in classification_values
        full_tree_indicies_list.resize(tree_depth)
        compressed_comb_feat_ind_child_tree.clear()
        compressed_comb_feat_ind_child_tree.resize(tree_depth)
        compressed_full_tree_ind_child_tree.clear()
        compressed_full_tree_ind_child_tree.resize(tree_depth)


        #store the full tree indicies for top level node:
        full_tree_indicies_list[0].push_back(0)

        # for level in range  (tree_depth):
            # print("find combos: level= ",level," child list= ", self.children_list[level])

        # determine which feature combos are in the particular example
        for level in range(tree_depth):
            # node_labels = self.node_label_list[level]
            # print("find_combos1")

            full_tree_indicies_ptr = & full_tree_indicies_list[level]
            if level !=0:# if not on top level, use indicies from previous level to search for children
                parent_full_tree_data = & self.full_tree_vector[level-1]

                # parent_node_labels = self.node_label_list[level-1]
                parent_indicies = full_tree_indicies_list[level-1]
                full_tree_indicies_ptr[0].clear()
                # print("find_combos2")

                #iterate through parents
                for parent_index_full_tree in parent_indicies:
                    parent_full_tree_data[0].set_node(parent_index_full_tree) #TODO: make it so that set_node is optional, data can be accessed without it
                    child_ptr = & self.children_list[level-1][parent_index_full_tree]
                    # print("find_combos2a",parent_index_full_tree,"max size",self.children_list[level-1].size())
                    # print("child_size = ",child_ptr[0].size()," level= ",level)
                    # print(self.children_list[level-1])

                    #find children of each parent and find intersection with features uses
                    used_children.resize( min(child_ptr[0].size(),comb_feats_vec.size()))
                    used_children_it = used_children.begin()
                    # print(comb_feats_vec)
                    # print(used_children)
                    # print(child_ptr[0])
                    end_it=set_intersection(child_ptr[0].begin(),child_ptr[0].end(),comb_feats_vec.begin(),comb_feats_vec.end(),used_children_it )
                    used_children.resize(end_it - used_children_it )
                    ###### iterate over used children, save label and add to full_tree_indicies vector
                    child_full_tree_indicies.clear()
                    while used_children_it != end_it:
                        ## copy parent label to child label
                        child_label.resize(level)
                        for label_ind in range(level-1):
                            child_label[label_ind]=parent_full_tree_data[0].label[label_ind] #TODO: make it so that set_node is optional, data can be accessed without it
                        ## add child feature to label and sort
                        child_label[level-1] = deref(used_children_it)
                        sort(child_label.begin(),child_label.end()) # sort child label

                        # find index of said label
                        child_index = & self.node_label_to_index_dict_list[level][child_label]

                        # add index to full_tree_indicies_set
                        full_tree_indicies_ptr[0].push_back(child_index[0])
                        child_full_tree_indicies.push_back(child_index[0])

                        inc(used_children_it)
                    #### store children to the parents
                    compressed_comb_feat_ind_child_tree.at(level-1).push_back(used_children)
                    compressed_full_tree_ind_child_tree.at(level-1).push_back(child_full_tree_indicies)

                # print("find_combos3")

                # sort and deduplicate full_tree_indicies_ptr
                sort(full_tree_indicies_ptr[0].begin(),full_tree_indicies_ptr[0].end())
                end_it = unique(full_tree_indicies_ptr[0].begin(),full_tree_indicies_ptr[0].end())
                full_tree_indicies_ptr[0].resize(end_it - full_tree_indicies_ptr[0].begin() )

            ######### create a vector to translate from mask indicies to compressed classification values indicies
            full_tree_to_local_tree_indicies_lookup_ptr = & full_tree_to_local_tree_lookup_list[level]
            # print("level=",level)
            # print("n_nodes = ",self.full_tree_vector[level].n_nodes)
            full_tree_to_local_tree_indicies_lookup_ptr[0].assign(self.full_tree_vector[level].n_nodes,-1)

            local_tree_size = full_tree_indicies_ptr[0].size()
            for local_index in range(local_tree_size):
                full_tree_to_local_tree_indicies_lookup_ptr[0][full_tree_indicies_ptr[0][local_index]]=local_index

            # print("finished level")

    def predict(self, X):
        """

        :param X: data to predict on
        :return: predicted clasees
        """
        probabilities= self.predict_proba(X)
        return self.classify(probabilities)


    def classify(self, probabilities= None, threshold = None):
        """
        :param probabilities: probabilities on which to classify
        :param threshold:  Threshold for classifying binary data, if None, 0.5 is used
        :return:
        """
        result = []

        if self.n_classes==2:  # use computed threshold if 2 classes.  If more than 2 classes, take class with highest predicted probability
            if threshold is None:
                threshold = 0.5
            for example in probabilities:
                if example[1]>threshold:
                    result.append(self.classes_[1])
                else:
                    result.append(self.classes_[0])
        else:

            for example in probabilities:
                loc = np.argmax(example)
                result.append(self.classes_[loc])

        result = np.array(result)
        return result



    cdef np_long[:,:] validate_X(self,X):
        """
        Checks X for validity and bins input if necessary, and returns integer datatype
        :param X: 
        :return: 
        """
        # Check if fit had been called
        if self.X.size<=0:
            raise ValueError("Trying to predict without calling fit")


        cdef result
        if type(X)!=np.ndarray:
            X=np.array(X)

        if len(X.shape)==1:
            X=X[np.newaxis,:]

        # Input validation
        if X.shape[1]!=self.n_features:
            raise ValueError("Input does not have correct number of features")

        # bin input data if needed
        ############# bin input data if needed.
        if self.nbins>0:
            X = check_array(X,dtype=None,force_all_finite=False)
            X = bin_data_given_bins(X,self.categorical_bin_list,self.numeric_bin_list) #returns memview type
        else:
            X = check_array(X)
            X=X.astype(NP_LONG, copy=False)  #convert to integer type.  Binning automatically converts to integer type

        return X


    cpdef predict_proba(self, X):
        """
        returns the probability of each class
        :param X:
        :type X: np.ndarray
        :return:
        """
        cdef np_long[:,:] X_view
        cdef np_long[:] example

        ## needed to pass scikit-learn validation checks
        # X = check_array(X,force_all_finite=True)


        X_view =self.validate_X(X)


        # if self.n_jobs>1:
        #     raise NotImplementedError("multicore support with cython not yet implemented")
        #     # with Pool(processes=self.n_jobs,initializer=init_parallel_threads,initargs=(self,)) as pool:
        #     #     probability_array = pool.map(predict_single_probability,X_view)
        # else:

        probability_array = []
        cdef np.ndarray[np_float, ndim=1] single_result
        for ind, example in enumerate(X_view): # iterate through all the rows in the data
            single_result = self.find_probability(example)
            single_result = single_result/np.sum(single_result) # normalize probabilities to sum to 1
            probability_array.append(single_result)

        return np.array(probability_array)



    cpdef np.ndarray[np_float, ndim=1] find_probability(self, np_long[:] example, bint save_intermediates = False):
        """
        run classifier and find probability given a single example        

        :param example: 
        :param save_intermediates: if True save classification_values_list to self for use in explainability
        :return: np array of probabilities for each class
        """

        #############
        # set references to the appropriate scaling & usefulness functions  (make it easier for c++ compiler?)
        cdef  string usefulness_model = self.usefulness_model.encode('UTF-8')
        cdef  string probability_scaling_method = self.probability_scaling_method.encode('UTF-8')


        #################
        #### find feature combinations used in example
        cdef vector[vector[np_long]] full_tree_indicies_list, full_tree_to_local_tree_lookup_list
        cdef vector[vector[vector[np_long]]] compressed_comb_feat_ind_child_tree # comb_feat_ind of children, #level, local index, children
        cdef vector[vector[vector[np_long]]] compressed_full_tree_ind_child_tree # full tree indicies of children, #level, local index, children
                                                        # use two vectors to save computation at the expense of memory

        self.find_used_feature_combos(example, full_tree_indicies_list, full_tree_to_local_tree_lookup_list, compressed_comb_feat_ind_child_tree,compressed_full_tree_ind_child_tree)

        #####################
        cdef np_long level
        cdef np_long[:,:] parent_array, counts_array
        cdef vector[np_long] full_tree_indicies, parent_full_tree_to_local_tree_lookup


        cdef FullTreeData[np_float,np_long ] * child_full_tree_data
        cdef FullTreeData[np_float,np_long ] * parent_full_tree_data


        cdef np_long local_tree_index, full_tree_index, class_ind, child_full_tree_index, parent_ctr, local_tree_size
        cdef np_long parent_index_full_tree, parent_index_local, total_count
        cdef bint need_to_compute_noise_ratio
        cdef np_float weight, child_noise, weight2
        cdef np_long row_index, p_sum_start, weight_sum_start,max_child_noise_ind, parent_row_index, parent_p_sum_start, parent_weight_sum_start, parent_max_child_noise_ind, #estimated_p_start
        cdef ClassificationValues[np_float] * classification_values
        cdef ClassificationValues[np_float] * parent_classification_values
        cdef vector[ClassificationValues[np_float]] classification_values_list
        cdef np_long tree_depth = full_tree_indicies_list.size()
        cdef np_float noise_exponent2_validated

        if self.noise_exponent2 is None:
            noise_exponent2_validated = self.noise_exponent
        else:
            noise_exponent2_validated = self.noise_exponent2


        #######
        ####### compute independence scaling values
        cdef vector[np_long] * child_comb_feat_ind_list_ptr
        cdef vector[np_long] * child_full_tree_ind_list_ptr
        cdef np.ndarray[np_float, ndim=2] independence_array
        cdef list independence_list = []
        cdef np_float child, correlation_sum, independence
        cdef vector[np_long] child_feature_list
        cdef np_long child_ctr, child_comb_feat, n_children, feature1, feature2, parent_local_tree_index, child_local_tree_index, parent_full_tree_index, parent_index_ctr
        cdef pair [np_long,np_long] feature_num_pair
        cdef  map[pair[np_long,np_long],np_float].iterator corr_iter

        # print(self.correlation_map)

        if self.independence_model_c!=b"none":
            #intialize independence list
            for level in range(tree_depth):
                if level==0:
                    independence_array = np.full( (1,1),1, order = 'C', dtype=NP_FLOAT) # null values are assigned to -1
                else:
                    local_tree_size = full_tree_indicies_list[level].size()
                    independence_array = np.full( (local_tree_size,level),-1, order = 'C', dtype=NP_FLOAT) # null values are assigned to -1
                independence_list.append(independence_array)


            #iterate over parents and update independence_array for children
            # for level in [0]:

            for level in range(tree_depth-1):
                # print("level=",level)
                local_tree_size = full_tree_indicies_list[level].size()


                ########  iterate through each used parent node at the given level and compute independence for its children:
                for parent_local_tree_index in range(local_tree_size):
                    # print("parent_local_tree_index=",parent_local_tree_index)


                    child_comb_feat_ind_list_ptr = & compressed_comb_feat_ind_child_tree.at(level).at(parent_local_tree_index)
                    child_full_tree_ind_list_ptr = & compressed_full_tree_ind_child_tree.at(level).at(parent_local_tree_index)
                    n_children = child_comb_feat_ind_list_ptr[0].size()

                    # find feature nums for all children
                    child_feature_list.clear()

                    for child_ctr in range(n_children ): #child is a full tree index
                        child_comb_feat = child_comb_feat_ind_list_ptr[0][child_ctr]
                        child_feature_list.push_back(child_comb_feat )

                    # iterate over each child
                    # print("n children=",n_children)
                    for child_ctr in range(n_children ): #child is a full tree index
                        # print("child_ctr=",child_ctr)
                        child_full_tree_index = child_full_tree_ind_list_ptr[0][child_ctr]
                        # iterate all feature nums and sum correlation values
                        correlation_sum=0
                        for feature_ctr in range(n_children):
                            if child_ctr==feature_ctr:
                                correlation_sum+=1
                            else:
                                feature_num_pair.first=child_feature_list.at(child_ctr)
                                feature_num_pair.second=child_feature_list.at(feature_ctr)
                                pair_sort(feature_num_pair)

                                # print("feature pair= ",feature_num_pair)

                                #only add to correlation if it exists, if not, assume correlation of 0
                                corr_iter = self.correlation_map.find(feature_num_pair)
                                if corr_iter!=self.correlation_map.end():
                                    correlation_sum+=deref(corr_iter).second
                                else:
                                    correlation_sum+=self.ave_corr
                                # correlation_sum+=self.ave_corr

                                # else:
                                    # print("feature pair",feature_num_pair," not found")

                        # compute independence value for child
                        independence = 1/correlation_sum
                        # print("independence=",independence)

                        ##### store independence value in location corresponding to appropriate parent
                        #convert from parent_local_tree_index to parents full_tree_index
                        parent_full_tree_index = full_tree_indicies_list[level][parent_local_tree_index]


                        # find location of parent full_tree_index in parent array and store independence at that location
                        # use child_full_tree_index to get list of parents from parent_array
                        child_full_tree_data = & self.full_tree_vector[level+1]
                        child_full_tree_data[0].set_node(child_full_tree_index)
                        for parent_index_ctr in range(level+1):
                            if child_full_tree_data[0].parent_indicies[parent_index_ctr]==parent_full_tree_index:
                                break
                        else:
                            raise ValueError("parent not found something is broken")

                        # print("parent_index_ctr=",parent_index_ctr)
                        #find child local tree index
                        child_local_tree_index=full_tree_to_local_tree_lookup_list[level+1][child_full_tree_index]
                        independence_list[level+1][child_local_tree_index,parent_index_ctr] = independence


                        # print("made it to the end")
                        # print(self.correlation_map)

                # print(independence_list)
                # raise ValueError("Hakuna ...")
        ###################
        ##### compute probability given used features

        #### initialize values
        cdef np_float noise_ratio=0
        cdef vector[np_float] child_scaling_weight, usefulness_vec, child_scaled_p, parent_scaled_p
        child_scaling_weight.assign(self.n_classes,0)
        usefulness_vec.assign(self.n_classes,0)
        child_scaled_p.assign(self.n_classes,0)
        parent_scaled_p.assign(self.n_classes,0)

        ## create vector to store p0 and copy over from full_tree_vector
        cdef vector[np_float] top_p0
        top_p0.resize(self.n_classes)
        self.full_tree_vector[0].set_node(0)
        for class_ind in range(self.n_classes):
            top_p0[class_ind]=self.full_tree_vector[0].p0[class_ind]


        #initialize classification values:
        # cdef classification_vals_info_t child_inds, parent_inds
        classification_values_list.resize(tree_depth)
        for level in range(tree_depth): #zero defaults
            classification_values_list[level].initialize(full_tree_indicies_list[level].size(), self.n_classes)


        # initialize local_weight_tree
        if save_intermediates:
            self.local_weight_tree.resize(tree_depth)
            for level in range(tree_depth):
                local_tree_size = full_tree_indicies_list[level].size()
                self.local_weight_tree[level].resize(local_tree_size)
                for local_tree_index in range(local_tree_size):
                    self.local_weight_tree[level][local_tree_index].resize(level)
                    for parent_ctr in range(level):
                        self.local_weight_tree[level][local_tree_index][parent_ctr].assign(self.n_classes,0)



        ####comment out for compilier error checking only
        classification_values =  & classification_values_list[0]


        # iterate through each level of the tree bottom to top
        for level in range(tree_depth-1, -1,-1):

            child_full_tree_data = & self.full_tree_vector[level]

            # load full tree indicies and find parent indicies (copy into a new array)
            full_tree_indicies = full_tree_indicies_list[level]
            local_tree_size = full_tree_indicies.size()

            # # set references to arrays for computed classification values
            classification_values =  & classification_values_list[level]

            ########  iterate through each used node at the given level and compute estimated probability:
            for local_tree_index in range(local_tree_size):
                ##set the current node being used:
                classification_values[0].set_node(local_tree_index)
                # compute_classification_vals_indicies(child_inds,local_tree_index, self.n_classes)
                child_full_tree_data[0].set_node(full_tree_indicies[local_tree_index])
                need_to_compute_noise_ratio = True

                ###### iterate through each class and calculate estimated probability
                # if total child weight is zero, use raw probability, else use combination of raw and estimated based on noise_ratio
                for class_ind  in range(self.n_classes):
                    weight = classification_values[0].weight_sum[class_ind]# [child_inds.weight_sum_start+class_ind]
                    if weight==0:
                        #store probability
                        classification_values[0].estimated_p[class_ind]= child_full_tree_data[0].p0[class_ind]
                    else:
                        if need_to_compute_noise_ratio:
                            noise_ratio = classification_values[0].max_child_noise[0] / child_full_tree_data[0].noise_weight[0]
                            need_to_compute_noise_ratio = False

                        classification_values[0].estimated_p[class_ind] = noise_ratio * classification_values[0].p_sum[class_ind] / \
                            classification_values[0].weight_sum[class_ind]+(1 - noise_ratio) * child_full_tree_data[0].p0[class_ind]

            ####### process parent values  (iterate through tree by updating parents of each node -  allows for fixed memory allocations)
            if level>0:
                # set references to arrays for computed classification values for parent
                parent_classification_values =  & classification_values_list[level-1]
                parent_full_tree_data = & self.full_tree_vector[level-1]
                parent_full_tree_to_local_tree_lookup = full_tree_to_local_tree_lookup_list[level - 1]

                ### iterate through each child node and update all of it's parents
                for local_tree_index in range(local_tree_size):
                    classification_values[0].set_node(local_tree_index)
                    child_full_tree_data[0].set_node(full_tree_indicies[local_tree_index])


                    #compute parent independent values
                    child_noise = child_full_tree_data[0].noise_weight[0]

                    # compute scaling weight based on new estimated probability:
                    total_count = 0
                    for class_ind in range(self.n_classes):
                        total_count+=child_full_tree_data[0].counts[class_ind]

                    # choose probability scaling model based on settings (loop should be unswitched by compiler)
                    if probability_scaling_method ==b"reciprocal":
                        compute_scaling_weights_reciprocal(self.n_noise, classification_values[0].estimated_p.ptr(0), total_count, child_scaling_weight)
                    elif probability_scaling_method == b"imbalanced_reciprocal":
                        compute_scaling_weights_reciprocal_class_imbalance(self.n_noise, classification_values[0].estimated_p.ptr(0), total_count,top_p0, child_scaling_weight)


                    elif probability_scaling_method ==b"logit":
                        compute_scaling_weights_logit(self.n_noise, classification_values[0].estimated_p.ptr(0), total_count, child_scaling_weight)
                    else:
                        raise ValueError("Unknown Scaling Model "+str(self.probability_scaling_method))

                    #####compute child scaling probabilities
                    if usefulness_model==b"scaled":# or usefulness_model==b"scaled_per_class":
                        for class_ind in range(self.n_classes):
                            child_scaled_p[class_ind]=child_scaling_weight[class_ind]*(classification_values[0].estimated_p[class_ind]-0.5)

                    # iterate through all parents
                    for parent_ctr in range(level):
                        # get parent index and check if it exists
                        parent_index_full_tree = child_full_tree_data[0].parent_indicies[parent_ctr]
                        if parent_index_full_tree>=0:
                            parent_full_tree_data[0].set_node(parent_index_full_tree) #set node in parent data

                            # set references to arrays for computed classification values for parent
                            parent_index_local = parent_full_tree_to_local_tree_lookup[parent_index_full_tree]
                            parent_classification_values[0].set_node(parent_index_local)

                            #find max child noise
                            parent_classification_values[0].max_child_noise[0]=max(
                                parent_classification_values[0].max_child_noise[0],child_noise) # keep track of the max child noise

                            # find independence weight
                            if self.independence_model_c == b"none":
                                independence = 1
                            elif self.independence_model_c == b"standard":
                                independence = independence_list[level][local_tree_index,parent_ctr]
                            else:
                                raise ValueError("unknown independence model"+str(self.independence_model))
                            #DEBUG:
                            if independence<=0 or independence>1:
                                raise ValueError ("something is wrong with yo independence, code broken")
                            # compute usefulness  (compiler should unswitch the if statements from the loop, but it needs to be tested)
                                #choose usefulness computation model based on settings:
                                #self.usefulness_vec is a preallocated output vec
                            if usefulness_model == b"none":
                                calculate_usefulness_none(classification_values[0].estimated_p.ptr(0), parent_full_tree_data[0].p0.ptr(0), usefulness_vec)

                            elif usefulness_model ==b"simple":
                                calculate_usefulness_simple(classification_values[0].estimated_p.ptr(0), parent_full_tree_data[0].p0.ptr(0), usefulness_vec)

                            elif usefulness_model==b"scaled": #scaled_per_class
                                if probability_scaling_method ==b"reciprocal":
                                    compute_scaling_weights_reciprocal(self.n_noise, parent_full_tree_data[0].p0.ptr(0), total_count, parent_scaled_p)
                                elif probability_scaling_method == b"imbalanced_reciprocal":
                                    compute_scaling_weights_reciprocal_class_imbalance(self.n_noise, parent_full_tree_data[0].p0.ptr(0), total_count,top_p0, parent_scaled_p)
                                    # level 0 of self.full_tree_vector always has only 1 node and so we can refer to top level p0 this way

                                for class_ind in range(self.n_classes):
                                    parent_scaled_p[class_ind]*=(parent_full_tree_data[0].p0[class_ind]-0.5)
                                calculate_usefulness_scaled_per_class(&child_scaled_p[0], & parent_scaled_p[0], usefulness_vec)
                            else:
                                raise ValueError("Unknown Usefulness Model "+str(self.usefulness_model))

                            for class_ind in range(self.n_classes):

                                ###TODO:  prestore scaling weight???
                                weight2 = child_scaling_weight[class_ind] *usefulness_vec[class_ind]*\
                                          min((total_count**noise_exponent2_validated) / self.noise_scale , 1 )*independence

                                parent_classification_values[0].weight_sum[class_ind]+=weight2
                                parent_classification_values[0].p_sum[class_ind]+=weight2*\
                                                classification_values[0].estimated_p[class_ind]

                                if save_intermediates:
                                   self.local_weight_tree[level][local_tree_index][parent_ctr][class_ind]=weight2


        if save_intermediates==True: #TODO: change to avoid copies
            self.classification_values_list=classification_values_list # saves a copy



            self.full_tree_indicies_list = full_tree_indicies_list
            self.full_tree_to_local_tree_lookup_list=full_tree_to_local_tree_lookup_list


        ############# DEBUGGING
            # print("pre  data0= ",classification_values_list[0].data )
            # print("post data0= ",self.classification_values_list[0].data )
            # print("pre  data1= ",classification_values_list[1].data )
            # print("post data1= ",self.classification_values_list[1].data )
            # print("pre  data2= ",classification_values_list[2].data )
            # print("post data2= ",self.classification_values_list[2].data )
            # raise ValueError("FDASDFd")

        # if False:
        #     ###### Print full tree nodes
        #     print("Full Tree Properties")
        #     for level in range(tree_depth):
        #         print("level ",level,"# feature combos = ",self.node_label_list[level].shape[0])
        #         for full_tree_index in range(self.node_label_list[level].shape[0]):
        #             label_str = ""
        #             label=self.node_label_list[level][full_tree_index,:]
        #             for comb_feat_ind in label:
        #                 label_str+=self.get_name(comb_feat_ind)+", "
        #             print("label = ", label_str)
        #
        #
        #     ##### CHECK TREE
        #     "Local Tree Nodes"
        #
        #     cdef classification_vals_info_t class_val_inds
        #
        #     for level in range(tree_depth):
        #         print("\n\nlevel ",level,"# feature combos = ",full_tree_indicies_list[level].size())
        #         print("Internal node numbers ", full_tree_indicies_list[level])
        #         print("classification values ",classification_values[level])
        #
        #         for local_tree_index in range(<np_long> full_tree_indicies_list[level].size()):
        #             compute_classification_vals_indicies(class_val_inds,local_tree_index, self.n_classes)
        #
        #             full_tree_index = full_tree_indicies_list[level][local_tree_index]
        #
        #             ## create string with node label
        #             label_str = ""
        #             label=self.node_label_list[level][full_tree_index,:]
        #             for comb_feat_ind in label:
        #                 label_str+=self.get_name(comb_feat_ind)+", "
        #             label_str = "{:<14}".format(label_str) #make fixed width
        #
        #             ## find classification_values:
        #             p_string = ""
        #             weight_sum_str = ""
        #             p_sum_str = ""
        #             for class_ind  in range(self.n_classes):
        #                 p_string+="p{0:<2g}={1:7.4f}, ".format(class_ind, classification_values[level][class_val_inds.estimated_p_start+class_ind])
        #                 weight_sum_str+="w_sum{0:<2g}={1:8.4f}, ".format(class_ind, classification_values[level][class_val_inds.weight_sum_start+class_ind])
        #                 p_sum_str+="p_sum{0:<2g}={1:8.4f}, ".format(class_ind, classification_values[level][class_val_inds.p_sum_start+class_ind])
        #
        #
        #             child_noise_str = "child_noise ={0:7.4f}".format(classification_values[level][class_val_inds.max_child_noise_ind])
        #             self_noise_str = "self noise ={:7.4f}".format(self.noise_weight_list[level][full_tree_index])
        #             print("label = ", label_str,p_string,p_sum_str,weight_sum_str,child_noise_str,self_noise_str)
        #
        #
        #
        #     print("estimated P= ",np.array(classification_values[0][child_inds.estimated_p_start:(child_inds.estimated_p_start+self.n_classes)],dtype=NP_FLOAT))
        #     raise ValueError("We Done")

        cdef np.ndarray return_values = np.zeros(self.n_classes, dtype=NP_FLOAT)
        for class_ind in range(self.n_classes):
            return_values[class_ind]=classification_values[0].estimated_p[class_ind]
        return return_values


    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef analyze_single_probability(self,example, str filename_prefix="./results"):#None):#
        """
        Analyzes a prediction and breaks down the components that go into the prediction
        :param example:
        :return:
        """

        #### Go through a classification and store weights for each feature combination in weights_tree
        #    weights_tree = level, node, feature ctr#, class, pair (child weights, self weights) -> weights as used in the classifier
            # weights are divided equally

        cdef np_long[:,:] example_2d
        cdef np_long[:] example_view

        example_2d = self.validate_X(example) #validate X assumes multiple examples and so we need to convert back down to a single example
        example_view = example_2d[0,:]

        cdef vector[np_long] full_tree_indicies

        # cdef classification_vals_info_t child_inds, parent_inds
        cdef np_long parent_ind, p_ind, class_num
        cdef bool result
        cdef vector[np_float] predicted_probability
        cdef np_long * child_label_ptr
        cdef np_long * parent_label_ptr
        cdef np_long feature_ind, level_ind, num_features = 0




        predicted_probability=self.find_probability(example_view,save_intermediates=True)


        cdef np_long tree_depth, local_tree_size, local_tree_index,row_index, estimated_p_start, p_sum_start, \
            weight_sum_start, max_child_noise_ind, child_full_tree_index, class_ind, parent_ctr, comb_feat_ind, \
            child_feature_ctr

        cdef np_float parent_tree_weight, parent_weight_sum, child_weight, child_tree_weight
        cdef bint nonzero_self_weight


        cdef FullTreeData[np_float,np_long ] * child_full_tree_data
        cdef FullTreeData[np_float,np_long ] * parent_full_tree_data
        cdef FullTreeData[np_float,np_long ] * full_tree_data


        tree_depth = self.full_tree_vector.size()

        #store weights in a nested vector (inefficient)
        cdef vector[vector[vector[vector[ pair[np_float,np_float] ]]]] weights_tree
        #    level, node, feature ctr#, class, pair (child weights, self weights)

        weights_tree.resize(tree_depth)
        cdef np_long level, feature_ctr, parent_index_full_tree
        cdef np_float noise_ratio

        # cdef  np_float[:] noise_weight_array
        # cdef np_long[:,:] parent_array
        # cdef np_long [:,:] label_array, parent_label_array

        #iterate through each level
        for level in range(tree_depth):
            full_tree_indicies = self.full_tree_indicies_list[level]
            local_tree_size = full_tree_indicies.size()
            child_full_tree_data = & self.full_tree_vector[level]

            #
            # parent_array = self.parents_list[level]
            # noise_weight_array= self.noise_weight_list[level]
            # label_array=self.node_label_list[level]

            if level!=0: parent_full_tree_to_local_tree_lookup = self.full_tree_to_local_tree_lookup_list[level - 1]

            weights_tree[level].resize(local_tree_size)

            # iterate through each node at level
            for local_tree_index in range(local_tree_size):

                #resize weights tree to have correct number of features
                weights_tree[level][local_tree_index].resize(max(level,1))

                #resize weights tree to have correct number of classes per feature
                for feature_ctr in range(max(level,1)):
                    weights_tree[level][local_tree_index][feature_ctr].resize(self.n_classes)
                    # initialize all values:
                    for class_num in range(self.n_classes):
                        weights_tree[level][local_tree_index][feature_ctr][class_num].first =0
                        weights_tree[level][local_tree_index][feature_ctr][class_num].second =0


                ##create indicies for each value:
                self.classification_values_list[level].set_node(local_tree_index)
                child_full_tree_data[0].set_node(full_tree_indicies[local_tree_index])

                # child_full_tree_index =




                #compute noise ratio, i.e what fraction of weight is attributed to self vs children
                noise_ratio = self.classification_values_list[level].max_child_noise[0] / child_full_tree_data[0].noise_weight[0]

                if level ==0:
                    for class_num in range(self.n_classes):
                        weights_tree[level][0][0][class_num].first =noise_ratio
                        weights_tree[level][0][0][class_num].second =(1-noise_ratio)
                else:
                    child_label_ptr = child_full_tree_data[0].label.ptr() #find child label
                    parent_full_tree_data = & self.full_tree_vector[level-1]
                    # parent_label_array = self.node_label_list[level-1]

                    # iterate through all parents to get the weight from each parent that is passed down to the children
                    nonzero_self_weight = False
                    for parent_ctr in range(level):

                        # get parent index and check if it exists
                        parent_index_full_tree = child_full_tree_data[0].parent_indicies[parent_ctr]
                        parent_full_tree_data.set_node(parent_index_full_tree)

                        if parent_index_full_tree>=0:

                            parent_index_local = parent_full_tree_to_local_tree_lookup[parent_index_full_tree]
                            self.classification_values_list[level-1].set_node(parent_index_local) #set parent node
                            # compute_classification_vals_indicies(parent_inds,parent_index_local, self.n_classes)


                            #get parent label
                            parent_label_ptr = parent_full_tree_data.label.ptr() #find parent label
                            child_feature_ctr =0

                            # iterate through each feature in the parent label so that the weight from each parent can be divided to the feature / top of tree location
                            for parent_feature_ind in range(max(level-1,1)):
                                #find child_feature_ctr that indicates the location of the parent feature (for level=1, child_feature_ctr=0)
                                if level>1:
                                    comb_feat_ind=parent_label_ptr[parent_feature_ind]

                                    #find match in child label, aka feature #.  Labels are ordered and so feature ind does not need to be reset
                                    while child_label_ptr[child_feature_ctr]!=comb_feat_ind:
                                        child_feature_ctr+=1


                                #iterate through classes (calculate each class independently)
                                for class_num in range(self.n_classes):
                                    #for each parent find the parent  weight sum and the individual child weight and the noise ratio
                                    parent_weight_sum = self.classification_values_list[level-1].weight_sum[class_num]
                                    child_weight= self.local_weight_tree[level][local_tree_index][parent_ctr][class_num]


                                    parent_tree_weight = weights_tree[level-1][parent_index_local][parent_feature_ind][class_num].first

                                    # weight per class & feature num+= parent weight per (class & feature num) * child weight / parent weight sum
                                    #compute how much each parent tree weight scaled by
                                    if child_weight==0:
                                        child_tree_weight=0
                                    else:
                                        child_tree_weight = parent_tree_weight*child_weight/parent_weight_sum

                                    #store child and self weights based on noise ratio
                                    # only assign weight to child if the child weight sum for the particular class is nonzero
                                    if self.classification_values_list[level].weight_sum[class_num]!=0:
                                        weights_tree[level][local_tree_index][child_feature_ctr][class_num].first += child_tree_weight*noise_ratio
                                        weights_tree[level][local_tree_index][child_feature_ctr][class_num].second += child_tree_weight*(1-noise_ratio)
                                    else:
                                        weights_tree[level][local_tree_index][child_feature_ctr][class_num].second += child_tree_weight

                                    # pws2 = self.classification_values_list[level][child_inds.weight_sum_start+class_num]
                                    # ptw2 = weights_tree[level][local_tree_index][child_feature_ctr][class_num].first
                                    #
                                    # if (ptw2!=0) and (pws2==0):
                                    #     print("weight sum = ",pws2)
                                    #     print("weight assigned to child is ",ptw2)
                                    #     print("noise ratio1",self.classification_values_list[level][child_inds.max_child_noise_ind])
                                    #     print("noise ratio2",noise_ratio)
                                    #
                                    #     print("child_tree_weight",child_tree_weight)
                                    #     print("parent_tree_weight",parent_tree_weight)
                                    #     print("child_weight",child_weight)
                                    #     print("parent_weight_sum",parent_weight_sum)
                                    #
                                    #     for class_num2 in range(self.n_classes):
                                    #         print("weight sum(",class_num2,")= ",self.classification_values_list[level][child_inds.weight_sum_start+class_num2])
                                    #     raise ValueError ("Should be no weight going to child")



                                    if weights_tree[level][local_tree_index][child_feature_ctr][class_num].second!=0:
                                        nonzero_self_weight=True
                    if nonzero_self_weight:
                        num_features+=1

            # print(weights_tree)
        ###############################################333333
        # ### convert weights_tree to the total weight per feature combination (i.e. account for parent weights)

        cdef np_long counts_ind,weight_ind, n_features_in_level,weight_by_feature_ind
        cdef np_float weight_sum, p_sum, local_weight
        cdef np_long n_explainable_cols
        #count number of rows in numpy array /  store indicies
        level_ind =0
        feature_ind =level_ind+1
        counts_ind = feature_ind+ tree_depth-1
        p_ind = counts_ind+self.n_classes
        weight_ind = p_ind+self.n_classes
        weight_by_feature_ind = weight_ind+self.n_classes
        n_explainable_values_cols = weight_ind + self.n_classes*(tree_depth+1)


        #convert weights tree to a numpy array with relevant data
        cdef np_float [:,:] explainable_values
        explainable_values = np.zeros((num_features,n_explainable_values_cols),dtype = NP_FLOAT)
        cdef np_long parent_level, feature_end_ind
        cdef np_long * label_ptr
        cdef np_long * label_end_ptr
        cdef np_long parental_size
        cdef np_float * p0_ptr
        cdef np_float [:]  parental_label
        # cdef np_long[:,:] upper_label_array, counts_array
        # cdef np_float[:,:] upper_p0_array
        cdef np_long full_tree_index=0, label_size
        row_index=-1


        for level in range(tree_depth):
            n_features_in_level = weights_tree[level].size()
            full_tree_indicies = self.full_tree_indicies_list[level]
            # label = self.node_label_list[level][full_tree_index,:]

            full_tree_data = & self.full_tree_vector[level]
            # upper_label_array = self.node_label_list[level]
            # upper_p0_array =self.p0_list[level]
            # counts_array = self.counts_list[level]


            for local_tree_index in range(n_features_in_level ):
                #check if the node has nonzero self weights:
                nonzero_self_weight = False
                for feature_ctr in range(level):
                    for class_num in range(self.n_classes):
                        if weights_tree[level][local_tree_index][feature_ctr][class_num].second!=0:
                            nonzero_self_weight=True


                # if nonzero self weights then store node:
                if nonzero_self_weight:
                    full_tree_index = full_tree_indicies[local_tree_index]
                    full_tree_data.set_node(full_tree_index)

                    #create pointers for fast access
                    label_ptr =  full_tree_data.label.ptr()
                    label_size = level
                    label_end_ptr = full_tree_data.label.ptr(label_size)
                    p0_ptr = full_tree_data.p0.ptr()

                    #### check if label is contains any already stored nodes
                    ###TODO: use numpy functions for now, change to C++ later
                    for parent_ind in range(row_index+1):

                        parent_level= <np_long> explainable_values[parent_ind,level_ind]
                        if parent_level<level:
                            parental_label = explainable_values[parent_ind,feature_ind:(feature_ind+parent_level)]
                            parental_size = parental_label.shape[0]

                            #check if parent label is fully contained within label, i.e. is parent label a parent of label?
                            #Can pass memory view to C++ function by pointer.
                            result = includes(label_ptr,label_end_ptr, &parental_label[0], &parental_label[parental_size])

                            if result == True: #if parent_label is a parent
                                ## check if probabilities are equal for all classes
                                equal_prob=True
                                for class_num in range(self.n_classes):
                                    if explainable_values[parent_ind,p_ind+class_num]!= p0_ptr[class_num]:
                                        equal_prob=False
                                if equal_prob: # parent is the same prob as child, update parent and not child
                                    #add weights to parent, store all weight in Fx
                                    #TODO:  Check if weight should be divided elsewhere
                                    for class_num in range(self.n_classes):
                                        weight_sum =0
                                        for feature_ctr in range(level):
                                            local_weight =  weights_tree[level][local_tree_index][feature_ctr][class_num].second
                                            explainable_values[parent_ind,weight_by_feature_ind+(class_num+1)*(tree_depth)-1] += local_weight
                                            weight_sum+=local_weight
                                        explainable_values[parent_ind,weight_ind+class_num]+=weight_sum
                                    nonzero_self_weight = False
                                    break

                if nonzero_self_weight:
                    row_index+=1
                    explainable_values[row_index,level_ind]=level
                    # store features in label
                    for feature_ctr in range(tree_depth-1):
                        if feature_ctr<label_size:
                            explainable_values[row_index,feature_ind+feature_ctr]=label_ptr[feature_ctr]
                        else:
                            explainable_values[row_index,feature_ind+feature_ctr]=-1

                    # store counts and probabilites
                    for class_num in range(self.n_classes):
                        explainable_values[row_index,counts_ind+class_num]=full_tree_data.counts[class_num]
                        # use p0 for self values
                        explainable_values[row_index,p_ind+class_num]=full_tree_data.p0[class_num]

                    # store weights by feature
                    for class_num in range(self.n_classes):
                        weight_sum =0
                        for feature_ctr in range(level):
                            explainable_values[row_index,weight_by_feature_ind+class_num*(tree_depth)+feature_ctr] =  weights_tree[level][local_tree_index][feature_ctr][class_num].second
                            weight_sum+=explainable_values[row_index,weight_by_feature_ind+class_num*(tree_depth)+feature_ctr]
                        explainable_values[row_index,weight_ind+class_num]=weight_sum


        #########################

        n_unique_combos = row_index+1
        explainable_values=explainable_values[:n_unique_combos,:]



        # #####DEBUG: Check for nan values in explainable_values
        # if np.isnan(explainable_values).any():
        #     print(weights_tree)
        #     raise ValueError("Nan Values in explainable_values")
        #
        #
        # ###### sanity checks:
        # for class_num in range(self.n_classes):
        #     weight_sum=0
        #     p_sum = 0
        #     print("Class ",class_num)
        #     for row_index in range(n_unique_combos):
        #         weight_sum+=explainable_values[row_index,weight_ind+class_num]
        #         p_sum+=explainable_values[row_index,p_ind+class_num]*explainable_values[row_index,weight_ind+class_num]
        #     print("computed p= ",p_sum," weight_sum= ",weight_sum)
        #     print("predicted p = ",predicted_probability[class_num])
        # #######
        #

        #########sort by the most probable class

        #### find the class that has the highest predicted probability
        cdef np_float max_prob=0
        cdef np_long predicted_class=0
        for prob_ent in range(predicted_probability.size()):
            if max_prob < predicted_probability[prob_ent]:
                max_prob = predicted_probability[prob_ent]
                predicted_class = prob_ent

        inds = np.argsort(explainable_values[:,weight_ind+predicted_class])
        rev_inds = inds[::-1]

        ############## save all feature combos

        #write value
        cdef str header
        header = "level, "
        for feature_ctr in range(tree_depth-1):
            header+='F'+str(feature_ctr)+', '
        for class_num in range(self.n_classes):
            header+='#'+str(class_num)+', '
        for class_num in range(self.n_classes):
            header+='P'+str(class_num)+', '
        for class_num in range(self.n_classes):
            header+='W(C'+str(class_num)+'), '
        for class_num in range(self.n_classes):
            for feature_ctr in range(tree_depth-1):
                header+='W(C'+str(class_num)+'_F'+str(feature_ctr)+'), '
            header+='W(C'+str(class_num)+'_FX), '
        header+='\n'

        ## python strings reallocate every time string is appended, use c++ string to avoid this
        cdef string data_string_cpp
        #create string to store all data and then write to file:
        data_string_cpp  = header.encode('UTF-8')
        cdef np_long col_ind
        cdef np_long row
        for row in rev_inds:

            for col_ind in range(n_explainable_values_cols):
                if col_ind>=feature_ind and col_ind<counts_ind:
                    if explainable_values[row,col_ind]>=0:
                        data_string_cpp+=self.get_name(<np_long> explainable_values[row,col_ind])
                        data_string_cpp+=<char*>", "
                    else:
                        data_string_cpp+=<char*>", "
                else:
                    data_string_cpp+=to_string(explainable_values[row,col_ind])
                    data_string_cpp+=<char*>", "
            data_string_cpp+=<char*>'\n'

        if filename_prefix is not None:
            with open(filename_prefix+"_combos.csv", 'w') as f:
                f.write(data_string_cpp.decode('UTF-8'))

        ##### save to a pandas object
        combos_df = pd.read_csv(BytesIO(data_string_cpp))


        ########################### print top level features:
        cdef np_float weight, probability
        n_features_in_level = weights_tree[1].size()

        cdef np_float label0
        cdef np.ndarray[np_float, ndim=2] top_level_values = np.zeros((n_features_in_level,4),dtype = NP_FLOAT)
        #cdef np_float[:,:] top_level_values = np_top_level_values

        for local_tree_index in range(n_features_in_level):

            #find index into classification values
            self.classification_values_list[1].set_node(local_tree_index)
            # compute_classification_vals_indicies(child_inds,local_tree_index, self.n_classes)

            #load weight and probability
            weight = weights_tree[1][local_tree_index][0][predicted_class].first+weights_tree[1][local_tree_index][0][predicted_class].second
            probability = self.classification_values_list[1].estimated_p[predicted_class]
            #find label
            full_tree_index = self.full_tree_indicies_list[1][local_tree_index]
            self.full_tree_vector[1].set_node(full_tree_index)
            label0 = self.full_tree_vector[1].label[0]

            top_level_values[local_tree_index,0]=label0
            top_level_values[local_tree_index,1]=probability
            top_level_values[local_tree_index,2]= self.full_tree_vector[1].p0[predicted_class]
            top_level_values[local_tree_index,3]=weight


        # sort by weight
        inds = top_level_values[:,3].argsort()
        top_level_values = top_level_values[inds[::-1]]

        #create string to store all data and then write to file:
        #use C++ string for fast appends
        data_string_cpp  = <string> b"feature, weighted_probability, original_probability, weight\n"
        for local_tree_index in range(n_features_in_level):
            comb_feat_ind= <np_long> top_level_values[local_tree_index,0]
            data_string_cpp+=self.get_name(comb_feat_ind)
            data_string_cpp+=<char*>", "
            data_string_cpp+=to_string(top_level_values[local_tree_index,1])
            data_string_cpp+=<char*>", "
            data_string_cpp+=to_string(top_level_values[local_tree_index,2])
            data_string_cpp+=<char*>", "
            data_string_cpp+=to_string(top_level_values[local_tree_index,3])
            data_string_cpp+=<char*>"\n"

        # print (data_string)

        if filename_prefix is not None:
            with open(filename_prefix+"_top_of_tree.csv", 'w') as f:
                f.write(data_string_cpp.decode('UTF-8'))


        ### save to pandas dataframe
        top_of_tree_df = pd.read_csv(BytesIO(data_string_cpp))



        ################################# print weight summed by feature (credit split evenly over all parents):

        # find the total weight by feature
        # feature_weights = np.zeros(self.n_features)
        # weighted_probability = np.zeros(self.n_features)
        #original_probability = np.zeros(self.n_features)

        # create a map between the feature # and feature weight/ probability
        cdef map[np_long, np_float] feature_weights
        cdef map[np_long, np_float] weighted_probability

        cdef np_float node_weight, node_probability
        cdef np_long explainable_level

        for feature_ctr in range(n_unique_combos):
            explainable_level  = <np_long> explainable_values[feature_ctr,0]
            node_weight = explainable_values[feature_ctr,weight_ind+predicted_class]
            node_probability = explainable_values[feature_ctr,p_ind+predicted_class]


            #iterate over features and split weight and probability over all parents
            for parent_ctr in range(explainable_level):
                comb_feat_ind = <np_long> explainable_values[feature_ctr,feature_ind+parent_ctr]
                if feature_weights.find(comb_feat_ind) == feature_weights.end():
                    feature_weights[comb_feat_ind]=node_weight/explainable_level
                    weighted_probability[comb_feat_ind]=node_probability*node_weight/explainable_level
                else:
                    feature_weights[comb_feat_ind]+=node_weight/explainable_level
                    weighted_probability[comb_feat_ind]+=node_probability*node_weight/explainable_level


        # convert maps to strings

        n_features_in_level = weights_tree[1].size()
        cdef np.ndarray[np_float, ndim=2] explain_by_feature = np.zeros((n_features_in_level,4),dtype = NP_FLOAT)
        #cdef np_float[:,:] explain_by_feature = np_explain_by_feature

        cdef map[np_long, np_float].iterator feature_weights_it
        cdef map[np_long, np_float].iterator weighted_probability_it

        feature_weights_it = feature_weights.begin()
        weighted_probability_it = weighted_probability.begin()

        for feature_ctr in range(n_features_in_level):
            comb_feat_ind = deref(weighted_probability_it).first
            explain_by_feature[feature_ctr,0]=comb_feat_ind
            explain_by_feature[feature_ctr,1]=deref(weighted_probability_it).second/deref(feature_weights_it).second
            explain_by_feature[feature_ctr,3]=deref(feature_weights_it).second
            #go from label to global tree index to p0

            #search node_label_list to find full_tree_ind
            for full_tree_index in range(self.full_tree_vector[1].n_nodes):
                self.full_tree_vector[1].set_node(full_tree_index)
                if self.full_tree_vector[1].label[0]==comb_feat_ind:
                    break

            explain_by_feature[feature_ctr,2]= self.full_tree_vector[1].p0[predicted_class]


            inc(feature_weights_it)
            inc(weighted_probability_it)

        # print(explain_by_feature)

        # sort be weight
        inds = explain_by_feature[:,3].argsort()
        explain_by_feature = explain_by_feature[inds[::-1]]

        #create string to store all data and then write to file:
        data_string_cpp  = <string> b"feature, weighted_probability, original_probability, weight\n"
        for local_tree_index in range(n_features_in_level):
            comb_feat_ind= <np_long> explain_by_feature[local_tree_index,0]
            data_string_cpp+=self.get_name(comb_feat_ind)
            data_string_cpp+=<char*>", "
            data_string_cpp+=to_string(explain_by_feature[local_tree_index,1])
            data_string_cpp+=<char*>", "
            data_string_cpp+=to_string(explain_by_feature[local_tree_index,2])
            data_string_cpp+=<char*>", "
            data_string_cpp+=to_string(explain_by_feature[local_tree_index,3])
            data_string_cpp+=<char*>"\n"

        # print (data_string)

        if filename_prefix is not None:
            with open(filename_prefix+"_by_feature_shared.csv", 'w') as f:
                f.write(data_string_cpp.decode('UTF-8'))

        ### save to pandas dataframe
        features_df = pd.read_csv(BytesIO(data_string_cpp))

        return combos_df, features_df, top_of_tree_df
