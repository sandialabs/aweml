from setuptools import setup, find_packages
from setuptools.extension import Extension
import Cython
from Cython.Build import cythonize
import numpy as np
import os

# os.environ["CC"] = "/usr/local/bin/gcc-7"
# os.environ["CXX"] = "/usr/local/bin/g++-7"


dir_path = os.path.dirname(os.path.realpath(__file__))
extensions = [
    Extension("awe_ml.binning", [os.path.join(dir_path, "awe_ml/binning.pyx")],
              include_dirs=[np.get_include(), "."]),
    Extension("awe_ml.classifier_cython",
              [os.path.join(dir_path, 'awe_ml/classifier_cython.pyx')],
              include_dirs=[np.get_include(), "."]),

]
#         language="c++",
#         include_dirs = ['/usr/local/lib/python3.6/site-packages/numpy/core/include']
#         libraries = [...],
#         library_dirs = [...]),

# if USE_CYTHON:
extensions = cythonize(extensions, compiler_directives={'profile':False,'wraparound':False,'cdivision':True,'boundscheck':False, 'initializedcheck':False,'language_level':3 })
# package_data = {'awe_ml': ['*.pxd']}

setup(name='awe_ml',
      version='0.3.3',
      description='Averaged Weights for Explainable Machine Learning',
      long_description='Averaged Weights for Explainable Machine Learning, Compatible with Sci-Kit-Learn',
      author='Sandia National Laboratories',
      author_email='sagarwa@sandia.gov',
      install_requires=[
          'numpy>1.10', 'scikit-learn','pandas>=0.21.0'],
      python_requires='>=3',


      packages=find_packages(exclude=[]),
      include_package_data=True,
      # package_data=package_data,
      # include_dirs=["."],
      ext_modules = extensions,
      zip_safe=False
)
# os.system("gcc --version")

# ext_modules = cythonize(extensions, extra_compile_args=["-Wconversion"]), gdb_debug=True),
# url = 'http://cross-sim.sandia.gov',
# license = 'BSD'