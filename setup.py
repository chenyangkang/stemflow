from setuptools import setup, find_packages
import codecs
import os

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()

VERSION = '0.0.12'
DESCRIPTION = 'A package for Adaptive Spatio-Temporal Model (AdaSTEM) in python'
LONG_DESCRIPTION = 'stemflow is a toolkit for Adaptive Spatio-Temporal Model (AdaSTEM) in python. A typical usage is daily abundance estimation using eBird citizen science data. It leverages the "adjacency" information of surrounding target values in space and time, to predict the classes/continues values of target spatial-temporal point. In the demo, we use a two-step hurdle model as "base model", with XGBoostClassifier for occurence modeling and XGBoostRegressor for abundance modeling.'

# Setting up
setup(
    name="stemflow",
    version=VERSION,
    author="Yangkang Chen",
    author_email="chenyangkang24@outlook.com",
    url='https://github.com/chenyangkang/stemflow',
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=long_description,
    packages=find_packages(),
    install_requires=['numpy>=1.24.3',
                      'matplotlib>=3.7.1',
                      'pandas>=2.0.3',
                      'geopandas>=0.11.1',
                      'tqdm>=4.65.0',
                      'h3pandas>=0.2.3',
                      'scikit-learn>=1.2.2',
                      'seaborn>=0.11.2',
                      'xgboost>=2.0.0',
                      'watermark>=2.4.3'],
    keywords=['python', 'spatial-temporal model', 'ebird', 'citizen science', 'spatial temporal exploratory model',
              'STEM','AdaSTEM','abundance','phenology'],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)