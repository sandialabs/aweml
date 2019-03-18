
__author__ = 'sagarwa'

from awe_ml.classifier_cython import AWE_ML_Cython

class AWE_ML(AWE_ML_Cython):#BaseEstimator, ClassifierMixin

    def __init__(self,*args,**kwargs):  # n_jobs=1, numerical_features = None
        super().__init__(*args,**kwargs)

    def fit(self, X, y, *args,**kwargs):

        return super().fit(X,y,*args,**kwargs)  ## repeated declaration here for scikit learn error checking


