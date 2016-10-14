# mlglue

This library contains tools to simplify exporting between different machine learning libraries.
In particular, it contains implementations for gradBDT-s, ...

### Installation

Clone this repository, install with
~~~
python setup.py install
~~~

### Quickstart

Check out `test/test_sklearn_to_tmva.py` for a quick look into how to use this code.

`test/test_xgboost_tmva.py` contains some example code on how to convert an xgboost tree into the internal structure.

## Features

### Gradient boosted decision trees

Converts sklearn trees (classification and regression) to an internal representation, which can then be exported to other formats

 * sklearn -> TMVA: binary classification, multiclass, regression via conversion to a TMVA XML. On the sklearn-side, it needs a special evaluation of the decision trees, which uses the same transformation functions as TMVA

# LICENSE

This program is licensed under the GPLv3 license, see LICENSE.md for details.

