# mlglue

This library contains tools to simplify conversion between models in different machine learning libraries.
In particular, it contains conversion code Gradient Boosted Decision Trees for `sklearn -> TMVA` and `xgboost -> TMVA`.
Binary classification, multiclass and regression trees are supported.

Exporting to TMVA XML:
~~~
╔════════════╦═════════╦═════════╗
║    type    ║ sklearn ║ xgboost ║
╠════════════╬═════════╬═════════╣
║ binary     ║ x       ║ x       ║
║ multiclass ║ x       ║         ║
║ regression ║ x       ║         ║
╚════════════╩═════════╩═════════╝
~~~

### Installation

Clone this repository, install with
~~~
pip install -r requirements.txt
pip install .
~~~

When installing in the CMSSW software environment, you may want to install with
~~~
pip install . --user
~~~
Also, you need to make sure that you are loading the patched version of xgboost using
~~~
PYTHONPATH=~/.local/lib/python2.7/site-packages/xgboost-0.6-py2.7.egg/:$PYTHONPATH python test/test_all.py
~~~

#### xgboost installation
Please note that you will need to increase the precision of the xgboost textual output in order to be able to import the trees from xgboost -> TMVA. Do this with the following patch in the xgboost directory:
~~~
diff --git a/src/tree/tree_model.cc b/src/tree/tree_model.cc
index 06fb005..03cd8cf 100644
--- a/src/tree/tree_model.cc
+++ b/src/tree/tree_model.cc
@@ -19,6 +19,7 @@ void DumpRegTree2Text(std::stringstream& fo,  // NOLINT(*)
                       const RegTree& tree,
                       const FeatureMap& fmap,
                       int nid, int depth, bool with_stats) {
+  fo.precision(std::numeric_limits<double>::max_digits10 + 2);
   for (int i = 0;  i < depth; ++i) {
     fo << '\t';
   }
~~~
and then reinstall xgboost with
~~~
make -j4
cd python-package
python setup.py install --user
~~~

### Quickstart

Here is a small example
~~~
model = xgboost.XGBClassifier(n_estimators=10)
model.fit(data_x, data_y_binary)

num_features = data_x.shape[1]
features = ["feat{0}".format(nf) for nf in range(num_features)]
target_names = ["cls0", "cls1"]

bdt = BDTxgboost(model, features, target_names)
bdt.to_tmva("test.xml")
bdt.setup_tmva("test.xml")
for irow in range(data_x.shape[0]):
    predA = bdt.eval_tmva(data_x[irow, :])
    predB = bdt.eval(data_x[irow, :])
~~~

Check out `test/test_all.py` for an overview and the testsuite. You can download the test dataset using `test/data.sh`.

# LICENSE

This program is licensed under the GPLv3 license, see LICENSE.md for details.

