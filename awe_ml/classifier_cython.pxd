#cython: profile=False,wraparound=False, cdivision=True, boundscheck=False, initializedcheck=False, language_level=3

from  awe_ml.binning cimport bin_data, bin_data_given_bins