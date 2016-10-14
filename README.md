# mlglue

This library contains tools to simplify exporting between different machine learning libraries.
In particular, it contains implementations for the following methods

### Gradient boosted decision trees

Converts sklearn trees (classification and regression) to an internal representation, which can then be exported to other formats

 * sklearn -> TMVA: binary classification, multiclass, regression via conversion to a TMVA XML. On the sklearn-side, it needs a special evaluation of the decision trees, which uses the same transformation functions as TMVA


# LICENSE

This program is licensed under the GPLv3 license, see LICENSE.md for details.

