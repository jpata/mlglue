import sys

import mlglue
from mlglue import sklearn_to_tmva

import sklearn
from sklearn import datasets
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

import pandas, math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

import ROOT, array
from ROOT import TMVA
from sklearn.datasets import fetch_mldata

from sklearn.datasets import load_svmlight_files

def test_bdt(data_x, data_y, kind, ntrees=10):
    """Simple test function for a scikit-learn BDT.
    Trains the BDT via scikit-learn, exports to TMVA, compares the scikit-learn and TMVA evaluation
    and calculates the total deviance (sum of suqares of differences).
    if kind is "classification" and the target array contains multiple kinds of targets,
    multiclass training is performed. If the targets are binary, then a binary classifier is trained.
    Regression can be specified as well, in which case, the 
    
    Args:
        ntrees (int): Number of trees
        data_x (numpy array): (n_samples, n_features) array of the training data
        data_y (numpy array): (n_samples, n_targets) array of the targets
        kind (string): "classification" or "regression"
    
    Returns:
        TYPE: Description
    """

    #number of samples
    N_points = data_x.shape[0]

    #generate variable names
    vars_x = ["var{0}".format(i) for i in range(data_x.shape[1])]

    #per-class weights
    weights = np.ones(N_points)

    if kind == "classification":
        classes = np.unique(data_y)
        #reweight each class to the same sum of weights
        for cl in classes:
            ncl = data_y == cl
            weights[ncl] = 1.0 / np.sum(ncl)

        cls = GradientBoostingClassifier(
            max_depth=4,
            learning_rate=0.01,
            n_estimators=ntrees,
            verbose=True,
            #min_samples_leaf=1,
            #min_samples_split=2,
            loss = "deviance"
        )
    elif kind == "regression":
        classes = [0]
        if len(data_y.shape) == 2:
            classes = range(data_y.shape[1])

        cls = GradientBoostingRegressor(
            max_depth=4,
            learning_rate=0.01,
            n_estimators=ntrees,
            verbose=True,
            #min_samples_leaf=1,
            #min_samples_split=2,
        )

    cls.fit(data_x, data_y, weights)

    #convert to TMVA
    sklearn_to_tmva.gradbdt_to_tmva(
        cls,
        "test.xml",
        feature_names = vars_x,
        target_names = ["class{0}".format(iclass) for iclass in range(len(classes))]
    )

    #Now book the TMVA MVA
    reader = TMVA.Reader("!V")
    vardict = {}

    #all variables must be float32
    for varname in vars_x:
        vardict[varname] = np.array([0], dtype=np.float32)
        reader.AddVariable(varname, vardict[varname])
    mva = reader.BookMVA("testmva", "test.xml")

    #helper function to evaluate TMVA
    def eval_tmva(xs):
        for ivar, varname in enumerate(vars_x):
            vardict[varname][0] = xs[0, ivar]
        if kind == "classification":
            if len(classes)>2:
                ret = reader.EvaluateMulticlass("testmva")
                ret = np.array([r for r in ret])
            else:
                ret = reader.EvaluateMVA("testmva")
        elif kind == "regression":
            ret = reader.EvaluateRegression("testmva")
            ret = np.array([r for r in ret])
        return ret

    #calculate the total sum of squares of differences between sklearn and TMVA
    tot_dev = 0
    for i in range(N_points):
        xs = data_x[i, :]
        #note that we use our own evaluation function on the input trees, so that
        #we could have the exact same distribution as in TMVA
        #this transformation does not change the final discrimination, but simply
        #the "shape" of the discriminant
        v1 = sklearn_to_tmva.evaluate_sklearn(cls, xs.reshape(1, -1))[0]
        v2 = eval_tmva(xs)
        tot_dev += np.sum(np.power(v1 - v2, 2))

    return tot_dev/float(N_points)



import unittest

class TestScikitLearnTMVA(unittest.TestCase):

    def setUp(self):
        # self.usps = fetch_mldata("usps")
        # self.data_x = self.usps.data 
        # self.data_y = self.usps.target 
        data= load_svmlight_files(("usps", "usps.t"))
        self.data_x, self.data_y = data[0].todense(), data[1]


    # def test_cls_to_nodetree(self):

    #     cls = GradientBoostingClassifier(
    #         max_depth=2,
    #         learning_rate=0.01,
    #         n_estimators=1,
    #         verbose=True,
    #         loss = "deviance"
    #     )
    #     cls.fit(self.data_x, self.data_y > 5)

    #     nodetree = {}
    #     sklearn_to_tmva.cls_to_nodetree(
    #         cls,
    #         nodetree,
    #         cls.estimators_[0,0],
    #         0, -1, -1
    #     )

    #do a binary classification
    def test_classify_binary(self):
        print "testing test_classify_binary"
        #convert multilabel data to binary
        data_y = (self.data_y>5).astype(np.int32)
        dev = test_bdt(self.data_x, data_y, "classification")
        self.assertTrue(dev < 0.00001)

    #multi-class classification
    def test_classify_mc(self):
        print "testing test_classify_mc"

        dev = test_bdt(self.data_x, self.data_y, "classification")
        self.assertTrue(dev < 0.00001)

    #do a regression to a single variable (scalar regression)
    def test_regression_scalar(self):
        print "testing test_regression_scalar"

        dev = test_bdt(self.data_x, self.data_y, "regression")
        self.assertTrue(dev < 0.00001)

if __name__ == "__main__":

    data = load_svmlight_files(("usps", "usps.t"))
    data_x, data_y = data[0].todense(), data[1]
    data_y = (data_y>5).astype(np.int32)
    print data_x.shape, data_y.shape
    v = test_bdt(data_x, data_y, "classification")
    print v
    #unittest.main()