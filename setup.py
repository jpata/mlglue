#!/usr/bin/env python

from distutils.core import setup

setup(name='mlglue',
    version='0.1',
    description='Glue between machine learning libraries',
    author='Joosep Pata',
    author_email='pata@phys.ethz.ch',
    url='http://github.com/jpata/mlglue',
    packages=['mlglue'],
    install_requires=['sklearn', 'numpy', 'xgboost'],
)

