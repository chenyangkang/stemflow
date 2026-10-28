import codecs
import os
from runpy import run_path

from setuptools import find_packages, setup

here = os.path.abspath(os.path.dirname(__file__))
__version__ = run_path(os.path.join(here, "stemflow", "version.py"))["__version__"]

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
    project_urls={
        "Documentation": "https://chenyangkang.github.io/stemflow/",
        "Source": "https://github.com/chenyangkang/stemflow",
        "Issues": "https://github.com/chenyangkang/stemflow/issues",
        "Changelog": "https://github.com/chenyangkang/stemflow/blob/main/NEWS.md",
    },
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=long_description,
    packages=find_packages(include=["stemflow", "stemflow.*"]),
    package_dir={"stemflow": "stemflow"},
    python_requires=">=3.9.0",
    install_requires=requirements,
    extras_require={"tuning": ["xgboost>=1.7.6", "optuna>=3.0"],
                    "monitoring": ["psutil>=5.9"]},
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
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.15",
        "Programming Language :: Python :: 3.14",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ],
)
