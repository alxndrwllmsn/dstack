from setuptools import setup, find_packages
from codecs import open
import os
import re

with open("requirements.txt", "r") as f:
    requirements = f.read().splitlines()

def get_version():
    version_file = os.path.join(os.path.abspath(os.path.dirname(__file__)),'dstack','__version__.py')

    with open(version_file, "r") as f:
        lines = f.read()
        version = re.search(r"^_*version_* = ['\"]([^'\"]*)['\"]", lines, re.M).group(1)
        f.close()

        return version

dstack_version = get_version()

setup(
    name='dstack',
    packages=find_packages(),
    version=dstack_version,
    author='Kristof Rozgonyi',
    author_email='rstofi@gmail.com',
    description='Python package to stack images and grids for the DINGO survey',
    install_requires=requirements,
    entry_points={
    'console_scripts': [
        'dstacking = dstack.applications:dstacking',
        'dparset = dstack.applications:dparset',
        'cim2fits = dstack.applications:cim2fits',
        'sdplots = dstack.applications:sdplots',
        'cimRMS = dstack.applications:cimRMS',
        'initTSF = dstack.applications:initTSF',
        'dsSoFiAtparset = dstack.applications:dsSoFiAtparset']
        }
    )
