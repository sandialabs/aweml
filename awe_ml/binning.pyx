#distutils: language = c++
#cython: profile=False,wraparound=False, cdivision=True, boundscheck=False, initializedcheck=False, language_level=3

#nope cython: c_string_type=unicode, c_string_encoding=utf8

from pandas import DataFrame
import numpy as np
cimport numpy as np
cimport cython
from cython.operator cimport dereference as deref, preincrement as inc
from libcpp.string cimport string

import pandas as pd
from warnings import warn
from libcpp.map cimport map



# ctypedef np.int32_t np_long
# ctypedef np.float64_t np_float
NP_LONG = np.int32
NP_FLOAT= np.float64


cpdef bin_data(data, np_long nbins = 4, np_long max_bins =20, object categorical_feature_inds = None, str binning_method = "qcut",bint retbins=False):
    """
    if retbins=True return two lists of of tuples (col, bin_data)  for categorical data bin_data is a dict, else a list of bin edges

    :param data: contains training data in a numpy array or pandas dataframe, np.nan represents missing data, missing values will be converted to -1
    :type data: DataFrame
    :param nbins: how many bins to divide data into, 0 means do not bin data
    :param max_bins:  if num bins in a column <= max bins do not bin data, if max_bins==0, assume all data is categorical
    :param categorical_feature_inds:  features explicitly labeled as categorical to avoid binning
    :param binning_method:  method used to bin data, only option currently is "qcut": Data is divided into equal sized bins.  
                        If most frequent value is first/last data value and has more counts than n_entries / nbins create 
                        single bin for most frequent value and split other values equally
    :param retbins: If true, return two lists of of tuples (col, bin_data)  for categorical data bin_data is a dict, else a list of bin edges 
    :return: numpy array of long with binned data
    """
    data = pd.DataFrame(data)
    if nbins<=0:
        if not retbins:
            return data
        else:
            raise ValueError("Data is not binned as nbins<=0, cannot return bins")

    if categorical_feature_inds is None:
        categorical_feature_inds = np.array([])
    else:
        categorical_feature_inds = np.array(categorical_feature_inds)

    categorical_bin_list = []
    numeric_bin_list = []

    n_entries = data.shape[0]
    if binning_method =="qcut":
        for col in data.columns:
            is_categorical=False

            col_ind = data.columns.get_loc(col)
            if col_ind in categorical_feature_inds:
                is_categorical=True
            else:
                nvals = data[col].nunique(dropna=True)
                #check if data has fewer bins than max_bins
                if nvals<=max_bins or max_bins==0:
                    is_categorical = True
                else:
                    # check if there is a single dominant category:
                    counts = data[col].value_counts()
                    most_counts = counts.iloc[0]
                    most_frequent_val = counts.index[0]
                    if most_counts > n_entries / nbins:
                        # remove most frequent entry
                        # verify most frequent value is min or max value
                        if most_frequent_val == data[col].min() or most_frequent_val == data[col].min():
                            # remove value
                            train_most_freq_locs = (data[col] == most_frequent_val)
                            data[col].replace(most_frequent_val, np.nan, inplace=True)


                            # run qcut
                            data[col], bins = pd.qcut(data[col] , nbins-1, labels=False, retbins=True, duplicates = 'drop')
                            #  bins include right edge but not left edge (except for 1st bin which includes both edges)


                            # restore removed values
                            column_values = data[col].values
                            column_values[train_most_freq_locs]=-1  #missing values are still np.nan
                            column_values+=1

                            data[col]=column_values

                            ##append most frequent val to bin list, need to adjust bin at edge to midpoint
                            if most_frequent_val<bins.min():
                                bins[0]=(bins[0]+most_frequent_val)/2
                                bins=np.insert(bins,0,most_frequent_val)
                            elif most_frequent_val>bins.max():
                                nbins2=bins.size
                                bins[nbins2-1]=(bins[nbins2-1]+most_frequent_val)/2
                                bins= np.append(bins,most_frequent_val)
                            else:
                                raise NotImplementedError("Can't insert most frequent val into middle of bin list")

                        else:
                            # warn("Can't handle most frequent category in middle of data, defaulting to standard qcut")
                            # run qcut
                            data[col], bins = pd.qcut(data[col] , nbins, labels=False, retbins=True, duplicates = 'drop')

                    else:
                        # run qcut
                        data[col], bins = pd.qcut(data[col] , nbins, labels=False, retbins=True, duplicates = 'drop')


                    data[col].fillna(-1, inplace=True)# encode missing values to -1

                    # if np.any(np.isnan(data[col].values)):
                    #     print("Has NAN\n",data[col])


            col_num= data.columns.get_loc(col)
            if is_categorical:
                # encode categorical variables into a dictionary
                data[col]=data[col].astype('category')
                # category_dict = dict( zip(data[col].cat.categories,range(data[col].cat.categories.size)) )
                category_dict = dict( enumerate(data[col].cat.categories) )
                data[col] = data[col].cat.codes  # missing values are encoded to -1, will be treated as own category unless explicitly handled by classifier
                categorical_bin_list.append( (col_num,category_dict))
            else:
                numeric_bin_list.append((col_num,bins))


    else:
        raise ValueError("Unknown binning method"+str(binning_method))
    if retbins:
        return data.values.astype(NP_LONG), categorical_bin_list,numeric_bin_list
    else:
        return data.values.astype(NP_LONG)




cpdef np_long[:,:] bin_data_given_bins(object data,list categorical_bin_list,list numeric_bin_list):
    """
    Apply bins determined from bin_data to new data.  New values are assigned -1
    :param data: contains data in a numpy array or pandas dataframe, modified in place
    :param categorical_bin_list: list of tuples (col, dictionary mapping category number to category)
    :param numeric_bin_list: list of tuples (col, ndarray of bins).  Each bin includes rightmost edge, not left 
    :return: numpy array of binned data
    """
    cdef np_long row, col, nrows
    cdef dict bin_dict

    data = pd.DataFrame(data)

    nrows = data.shape[0]
    if categorical_bin_list is None or numeric_bin_list is None:
        raise ValueError("Bins must be specified")

    # set categorical values
    for (col, bin_dict) in categorical_bin_list:
        # invert bin dict
        bin_dict = {cat: cat_num for cat_num, cat in bin_dict.items()}
        #replace data
        data.iloc[:,col].replace(bin_dict,inplace=True)

    # set continuous values
    for (col, bin_list) in numeric_bin_list:
        binned_col=pd.cut(data.iloc[:,col],bin_list,labels=False,include_lowest=True)
        binned_col.replace(np.nan,-1,inplace=True) #encode out of range values to missing, use separate variable to manage data types (ints don't have nan)
        data.iloc[:, col] =binned_col

    return data.values.astype(NP_LONG)