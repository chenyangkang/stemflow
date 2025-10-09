import codecs
import os

from setuptools import find_packages, setup

from stemflow.version import __version__

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()

requirements = open(os.path.join(here, "requirements.txt"), "r", encoding="utf-8").read().strip().split("\n")
requirements = [i for i in requirements if not i.startswith("#")]

DESCRIPTION = "A package for Adaptive Spatio-Temporal Exploratory Model (AdaSTEM) in python"
LONG_DESCRIPTION = '**Stemflow** is a toolkit for Adaptive Spatio-Temporal Exploratory Model (AdaSTEM [1,2]) in python. A typical usage is daily abundance estimation using eBird citizen science data. It leverages the "adjacency" information of surrounding target values in space and time to predict the classes/continuous values of target spatial-temporal points. In the demo, we use a two-step hurdle model as "base model", with XGBoostClassifier for occurrence modeling and XGBoostRegressor for abundance modeling.'

# Setting up
setup(
    name="stemflow",
    version=__version__,
    author="Yangkang Chen",
    author_email="chenyangkang24@outlook.com",
    url="https://github.com/chenyangkang/stemflow",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=long_description,
    packages=find_packages(),
    package_dir={"stemflow": "stemflow"},
    python_requires=">=3.8.0",
    install_requires=requirements,
    platforms="any",
    keywords=[
        "python",
        "spatial-temporal model",
        "ebird",
        "citizen science",
        "spatial temporal exploratory model",
        "STEM",
        "AdaSTEM",
        "abundance",
        "phenology",
    ],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ],
)
