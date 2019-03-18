from setuptools import setup, find_packages
from setuptools.extension import Extension
# from Cython.Build import cythonize
import numpy as np  #THIS MEANS NUMPY MUST BE INSTALLED FIRST (FIND A BETTER WAY)




sourcefiles1 = ['awe_ml/classifier_cython.cpp']
sourcefiles2 = ['awe_ml/binning.cpp']


extensions = [
    Extension("awe_ml.classifier_cython", sourcefiles1, include_dirs=[np.get_include()]),
    Extension("awe_ml.binning", sourcefiles2, include_dirs=[np.get_include()]),
]

setup(name='awe_ml',
      version='0.3.3',
      description='Averaged Weights for Explainable Machine Learning',
      long_description='Averaged Weights for Explainable Machine Learning, Compatible with Sci-Kit-Learn',
      author='Sandia National Laboratories',
      author_email='sagarwa@sandia.gov',
      install_requires=[
          'numpy>1.10','scikit-learn','pandas>=0.21.0'],
      python_requires='>=3',

      packages=find_packages(exclude=[]),
      include_package_data=True,
      ext_modules = extensions,
      zip_safe=False
)

