#distutils: language = c++
#cython: profile=False,wraparound=False, cdivision=True, boundscheck=False, initializedcheck=False, language_level=3

from libcpp.string cimport string
cimport numpy as np
ctypedef np.int32_t np_long
ctypedef np.float64_t np_float

cpdef bin_data(train_data, np_long nbins=*, np_long max_bins=*, object categorical_feature_inds=*, str binning_method=*,bint retbins=*)

cpdef np_long[:,:] bin_data_given_bins(object data,list categorical_bin_list,list numeric_bin_list)
