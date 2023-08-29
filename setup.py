from setuptools import setup, find_packages
import codecs
import os

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()

VERSION = '0.0.2'
DESCRIPTION = 'AdaSTEM model for daily abundance estimation using eBird citizen science data'
LONG_DESCRIPTION = 'TBD'

# Setting up
setup(
    name="BirdSTEM",
    version=VERSION,
    author="Yangkang Chen",
    author_email="chenyangkang24@outlook.com",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=long_description,
    packages=find_packages(),
    install_requires=[''],
    keywords=['python', 'ebird', 'spatial-temporal model', 'citizen science', 'spatial temporal exploratory model',
              'STEM','AdaSTEM','abundance','phenology'],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)