from tree import Node, tree_to_tmva

import numpy as np
from sklearn.tree import _tree
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor


def cls_to_nodetree(cls, nodetree, sklearn_tree, node_id, parent_id, depth):
    """Recursively converts a sklearn GradientBoosting{Classifier,Regressor} to a generic representation
    
    Args:
        nodetree (dict id->Node): The output dictionary with the nodes
        sklearn_tree (scikit learn DecisionTree): The input decision tree
        node_id (int): the id of the root node
        parent_id (int): the id of the parent node
        depth (int): The current depth
    
    Returns:
        dict int->Node: The output node tree
    """

    #if the left (or right) child node id is -1, then this node is already a leaf node
    if sklearn_tree.children_left[node_id] == _tree.TREE_LEAF:
        n = Node(
            node_id,
            [],
            parent_id,
            depth,
            ("val", sklearn_tree.value[node_id][0,0])
        )
        nodetree[node_id] = n
        if nodetree.has_key(parent_id):
            nodetree[parent_id].children += [node_id]
    #this is not a leaf node
    else:
        n = Node(
            node_id,
            [],
            parent_id,
            depth,
            ("cut", sklearn_tree.feature[node_id], sklearn_tree.threshold[node_id])
        )
        nodetree[node_id] = n
        if nodetree.has_key(parent_id):
            nodetree[parent_id].children += [node_id]

    left_child = sklearn_tree.children_left[node_id]
    right_child = sklearn_tree.children_right[node_id]
    if left_child != _tree.TREE_LEAF:
        cls_to_nodetree(cls, nodetree, sklearn_tree, left_child, node_id, depth+1)
    if right_child != _tree.TREE_LEAF:
        cls_to_nodetree(cls, nodetree, sklearn_tree, right_child, node_id, depth+1)

    return nodetree
    
def gradbdt_to_tmva(cls, outfile_name, feature_names, target_names, mva_name="bdt"):
    """Converts a sklearn GradientBoostingClassifier (binary or multiclass) or GradientBoostingRegressor to a TMVA-compatible XML
    
    Args:
        outfile_name (string): Output XML file
        feature_names (list of strings): Description
        target_names (list of strings): Description
        mva_name (str, optional): Description
    
    Returns:
        nothing
    
    """

    #Check if we are dealing with regression or classification
    #Depending on this, we have to write a different XML file
    if isinstance(cls, GradientBoostingRegressor):
        kind = "regression"
    elif isinstance(cls, GradientBoostingClassifier):
        kind = "classification"

    #Create list of variables
    #we assume that all variables are 'simple', that is, not expressions
    varstring = ""
    for i in range(cls.n_features):
        varstring += '<Variable VarIndex="{0}" Expression="{1}" Label="{1}" Title="{1}" Unit="" Internal="{1}" Type="F" Min="{2:.64E}" Max="{3:.64E}"/>\n'.format(
            i, feature_names[i], 0, 0
        )

    if kind == "regression":
        class_string = ""
        num_classes = 1
        analysis_type = "Regression"

        #for regression, just one class
        for icls, clsname in enumerate(["Regression"]):
            class_string += '<Class Name="{0}" Index="{1}"/>\n'.format(
                clsname, icls
            )

        #as many targets as given (n>1: vector valued regression)
        target_string = ""
        num_targets = len(target_names)
        if num_targets > 1:
            raise Exception("TMVA does not support regression with vector values, need to specify a scalar target")
        for itgt, tgtname in enumerate(target_names):
            target_string += '<Target Name="{0}" TargetIndex="{1}" Expression="{0}" Label="{0}" Title="{0}" Unit="" Internal="{0}" Type="F" Min="{2:.64E}" Max="{3:.64E}"/>\n'.format(
                tgtname, itgt, 0.0, 0.0
            )
        ntrees = cls.estimators_.shape[0] * cls.estimators_.shape[1]

    elif kind == "classification":
        class_string = ""
        num_classes = len(target_names)

        #Decide between multiclass or binary
        if len(target_names) == 2:
            analysis_type = "Classification"
        elif len(target_names) > 2:
            analysis_type = "Multiclass"

        for icls, clsname in enumerate(target_names):
            class_string += '<Class Name="{0}" Index="{1}"/>\n'.format(
                clsname, icls
            )
        num_targets = 0
        target_string = ""

        #TMVA saves trees for different classes separately, like [tree1_cls1, tree1_cls2, ...]
        #so total number of trees = num_classes * num_trees_per_class
        ntrees = cls.estimators_.shape[0] * cls.estimators_.shape[1]
      
    outfile = open(outfile_name, "w")
    outfile.write(
    """
    <?xml version="1.0"?>
    <MethodSetup Method="BDT::{mva_name}">
    <GeneralInfo>
    <Info name="TMVA Release" value=""/>
    <Info name="ROOT Release" value=""/>
    <Info name="Creator" value="mlglue"/>
    <Info name="Date" value=""/>
    <Info name="Host" value=""/>
    <Info name="Dir" value=""/>
    <Info name="Training events" value="-1"/>
    <Info name="TrainingTime" value="-1"/>
    <Info name="AnalysisType" value="{analysis_type}"/>
    </GeneralInfo>
    <Options>
    <Option name="NTrees" modified="Yes">{ntrees}</Option>
    <Option name="MaxDepth" modified="Yes">{maxdepth}</Option>
    <Option name="BoostType" modified="Yes">Grad</Option>
    <Option name="Shrinkage" modified="Yes">{learnrate}</Option>
    <Option name="UseNvars" modified="Yes">{usenvars}</Option>
    </Options>

    <Variables NVar="{nvars}">
    {varstring}
    </Variables>

    <Classes NClass="{nclasses}">
    {class_string}
    </Classes>

    <Targets NTrgt="{ntargets}">
    {target_string}
    </Targets>

    <Transformations NTransformations="0"/>
    <MVAPdfs/>
    <Weights NTrees="{ntrees}" AnalysisType="1">
    """.format(**{
            "analysis_type": analysis_type,
            "mva_name": mva_name,
            "ntrees": ntrees,
            "maxdepth":cls.max_depth,
            "maxdepth":cls.max_depth,
            "usenvars":cls.max_features,
            "nvars": cls.n_features,
            "varstring": varstring,
            "learnrate": cls.learning_rate,
            
            "nclasses": num_classes,
            "class_string": class_string,

            "ntargets": num_targets,
            "target_string": target_string

            }
        )
    )

    itree = 0
    for trees in cls.estimators_:
        for class_tree in trees:
            outfile.write(
                '<BinaryTree type="DecisionTree" boostWeight="0.0" itree="{0}">\n'.format(
                    itree, cls.learning_rate
                )
            )
            nodetree = {}
            cls_to_nodetree(cls, nodetree, class_tree.tree_, 0, -1, -1)
            tree_to_tmva(outfile, nodetree, 0, 1.0 / cls.n_estimators)
            outfile.write('</BinaryTree>\n')
            itree += 1
    outfile.write("""
      </Weights>
    </MethodSetup>
    """)
    outfile.close()

def evaluate_sklearn(cls, vals):
    """A TMVA-compatible evaluation function for a scikit-learn classifier
    
    Args:
        vals (numpy array): An array (n_samples, n_features) of the input variables
    
    Returns:
        numpy array: (n_samples, n_classes) array of the output
    """
    
    #need to scale the same way as done in TMVA    
    scale = 1.0 / cls.n_estimators

    if isinstance(cls, GradientBoostingClassifier):
        #multiclass classification
        #according to TMVA::MethodBDT::GetMulticlassValues()
        if cls.n_classes_ > 2:
            ret = np.zeros((vals.shape[0], cls.n_classes_))
            for iclass in range(cls.n_classes_):
                for itree, t in enumerate(cls.estimators_[:, iclass]):
                    r = t.predict(vals)
                    ret[:, iclass] += r * scale

            norm = np.zeros(ret.shape)
            for i in range(cls.n_classes_):
                for j in range(cls.n_classes_):
                    if i != j:
                        norm[:, i] += np.exp(ret[:, j] - ret[:, i])

            ret = 1.0 / (1.0 + norm)        
            return ret
        #binary classification
        elif cls.n_classes_ == 2:
            ret = np.zeros(vals.shape[0])

            for itree, t in enumerate(cls.estimators_[:, 0]):
                r = t.predict(vals)
                ret += r * scale
            return 2.0/(1.0 + np.exp(-2.0 * ret)) - 1
    if isinstance(cls, GradientBoostingRegressor):
        ret = np.zeros((vals.shape[0], cls.n_classes_))
        for iclass in range(cls.n_classes_):
            for itree, t in enumerate(cls.estimators_[:, iclass]):
                r = t.predict(vals)
                ret[:, iclass] += r * scale
        return ret


