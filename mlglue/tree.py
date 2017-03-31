import re
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.tree import _tree

class Tree:
    """Represents a node in a decision tree, identified by a unique integer id
    
    Attributes:
        children (list of int): The id-s of the children associated with this node
        depth (int): The depth of this node in the tree
        id (int): The unique id of this node
        parent (id): The id of the parent node
        payload (tuple): Describes what this node does,
            i.e. is it a non-terminal node (cut) or a terminal (leaf) 
    """
    def __init__(self, id, children, parent, depth, payload):
        self.id = id
        self.children = children
        self.parent = parent
        self.depth = depth
        self.payload = payload

    def __repr__(self):
        return "Tree id:{id} children:{children} parent:{parent}, depth:{depth}, payload:{payload}".format(**{
            "id": self.id,
            "children": str(self.children),
            "parent": self.parent,
            "depth": self.depth,
            "payload": self.payload,
        })

    def print_out(self, node_dict):
        """Recursively prints a node and its children, given a dictionary with all the available nodes
        
        Args:
            node_dict (dict id->node): All the available nodes
        
        Returns:
            nothing
        """
        print (self.depth + 1) * "-" + str(self)
        for ch in self.children:
            node_dict[ch].print_out(node_dict)

    def to_tmva(self, nodetree, scale):
        """Writes out a TMVA-compatible XML string for a given node in the decision tree
        
        Args:
            nodetree (dict int->Tree): The dictionary of the full tree
            scale (float): A scaling coefficient for the TMVA leaves (TMVA = sklearn * scale)
        
        Returns:
            string: XML with the node
        
        """

        kind = "c"
        if self.parent != -1:
            idx = nodetree[self.parent].children.index(self.id)
            if idx == 0:
                kind = "l"
            elif idx == 1:
                kind = "r"
        
        #handle leaf (terminal) node
        if len(self.children) == 0:

            return '<Node pos="{0}" depth="{1}" NCoef="0" \
    IVar="{2}" Cut="{3:17E}" cType="1" \
    res="{4:17E}" rms="0.0e-00" \
    purity="{5:.8E}" nType="-99">'.format(
                kind,
                self.depth + 1,
                -1,
                0.0,
                self.payload[1] * scale,
                0.0
            )
        #handle non-leaf node
        else:
            return '<Node pos="{0}" depth="{1}" NCoef="0" \
    IVar="{2}" Cut="{3:17E}" cType="1" \
    res="{4:17E}" rms="0.0" \
    purity="{5:.8E}" nType="0">'.format(
            kind,
            self.depth + 1,
            self.payload[1],
            self.payload[2],
            0.0, 0.0
        )

def sklearn_to_nodetree(cls, nodetree, sklearn_tree, node_id=0, parent_id=-1, depth=-1):
    """Recursively converts a sklearn GradientBoosting{Classifier,Regressor} to a generic representation
    
    Args:
        nodetree (dict id->Node): The output dictionary with the nodes
        sklearn_tree (DecisionTreeRegressor): The input decision tree
        node_id (int): the id of the root node
        parent_id (int): the id of the parent node
        depth (int): The current depth
    
    Returns:
        dict int->Tree: The output node tree
    """

    #if the left (or right) child node id is -1, then this node is already a leaf node
    if sklearn_tree.children_left[node_id] == _tree.TREE_LEAF:
        n = Tree(
            node_id,
            [],
            parent_id,
            depth,
            ("val", sklearn_tree.value[node_id][0,0]/cls.n_estimators)
        )
        nodetree[node_id] = n
        if nodetree.has_key(parent_id):
            nodetree[parent_id].children += [node_id]
    #this is not a leaf node
    else:
        n = Tree(
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
        sklearn_to_nodetree(cls, nodetree, sklearn_tree, left_child, node_id, depth+1)
    if right_child != _tree.TREE_LEAF:
        sklearn_to_nodetree(cls, nodetree, sklearn_tree, right_child, node_id, depth+1)

    return nodetree

def xgbtree_to_nodetree(tree):
    """Converts an xgboost tree dump to an internal Tree representation
    
    Args:
        tree (string): The model dump from xgboost using model.booster().get_dump()[ntree]
    
    Returns:
        dict int->Tree: The tree structure
    """
    _NODEPAT = re.compile(r'(\d+):\[(.+)\]')
    _LEAFPAT = re.compile(r'(\d+):(leaf=.+)')

    parent_stack = []
    prev_depth = -1
    prev_index = -1
    nodes = {}

    for node in tree.split("\n"):
        node_depth = node.count("\t")

        is_node = False
        is_leaf = False

        match = _NODEPAT.match(node.strip())
        if match is not None:
            node_index = int(match.group(1))
            node_variable, threshold = match.group(2).split("<")
            node_variable = int(node_variable.replace("f", ""))
            threshold = float(threshold)
            is_node = True

        match = _LEAFPAT.match(node.strip())
        if match is not None:
            node_index = int(match.group(1))
            val = float(match.group(2).split("=")[1])
            is_leaf = True

        if not (is_node or is_leaf):
            continue

        #keep track of the parent of this node
        istack = prev_depth
        while istack < node_depth:
            parent_stack += [prev_index]
            istack += 1
        istack = node_depth
        while istack < prev_depth:
            parent_stack.pop()
            istack += 1
        my_parent = parent_stack[-1]

        #create the node
        if is_node:
            nodes[node_index] = Tree(node_index, [], my_parent, node_depth, ("cut", node_variable, threshold))
        elif is_leaf:
            nodes[node_index] = Tree(node_index, [], my_parent, node_depth, ("val", val))

        #insert node into final node dict
        if nodes.has_key(my_parent):
            nodes[my_parent].children += [node_index]

        prev_depth = node_depth
        prev_index = node_index

    #nodes[0].print_out(nodes)

    return nodes

class BDT(object):
    def __init__(self, trees, kind, feature_names, target_names, max_depth, learning_rate):
        self.trees = trees
        self.kind = kind
        self.ntrees = len(trees)

        self.feature_names = feature_names
        self.target_names = target_names

        self.max_depth = max_depth
        self.learning_rate = learning_rate



    def to_tmva(self, outfile_name, mva_name="bdt"):

        #Create list of variables
        #we assume that all variables are 'simple', that is, not expressions
        varstring = ""
        for i in range(len(self.feature_names)):
            varstring += '<Variable VarIndex="{0}" Expression="{1}" Label="{1}" Title="{1}" Unit="" Internal="{1}" Type="F" Min="{2:.64E}" Max="{3:.64E}"/>\n'.format(
                i, self.feature_names[i], 0, 0
            )

        if self.kind == "regression":
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
            num_targets = len(self.target_names)
            if num_targets > 1:
                raise Exception("TMVA does not support regression with vector values, need to specify a scalar target")
            for itgt, tgtname in enumerate(self.target_names):
                target_string += '<Target Name="{0}" TargetIndex="{1}" Expression="{0}" Label="{0}" Title="{0}" Unit="" Internal="{0}" Type="F" Min="{2:.64E}" Max="{3:.64E}"/>\n'.format(
                    tgtname, itgt, 0.0, 0.0
                )

        elif self.kind == "binary" or self.kind == "multiclass":
            class_string = ""
            num_classes = len(self.target_names)

            #Decide between multiclass or binary
            if self.kind == "binary":
                analysis_type = "Classification"
            elif self.kind == "multiclass":
                analysis_type = "Multiclass"

            for icls, clsname in enumerate(self.target_names):
                class_string += '<Class Name="{0}" Index="{1}"/>\n'.format(
                    clsname, icls
                )
            num_targets = 0
            target_string = ""

          
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
                "ntrees": self.ntrees,
                "maxdepth": self.max_depth,
                "usenvars": len(self.feature_names),
                "nvars": len(self.feature_names),
                "varstring": varstring,
                "learnrate": self.learning_rate,
                
                "nclasses": num_classes,
                "class_string": class_string,

                "ntargets": num_targets,
                "target_string": target_string

                }
            )
        )

        #Loop over decision trees, in scikit that's a 2D array (N_estimators, N_classes)
        #if binary classification, N_classes = 1
        itree = 0
        for tree in self.trees:
            outfile.write(
                '<BinaryTree type="DecisionTree" boostWeight="0.0" itree="{0}">\n'.format(
                    itree, self.learning_rate
                )
            )

            #convert internal representation to TMVA tree
            #re-weight each node by 1/N (N - num trees per class)
            tree_to_tmva(outfile, tree, 0, 1.0)

            outfile.write('</BinaryTree>\n')
            itree += 1

        #done with output
        outfile.write("""
          </Weights>
        </MethodSetup>
        """)
        outfile.close()

    def setup_tmva(self, bdtfile):
        from ROOT import TMVA
        self.reader = TMVA.Reader("!V")

        self.vardict = {}
        #all variables must be float32
        for ivar in range(0, len(self.feature_names)):
            self.vardict[ivar] = np.array([0], dtype=np.float32)
            self.reader.AddVariable("f{0}".format(ivar), self.vardict[ivar])
        self.tmva = self.reader.BookMVA("bdt", bdtfile)

    def eval_tmva(self, features):
        for ivar, varname in enumerate(self.feature_names):
            self.vardict[ivar][0] = features[0, ivar]

        if self.kind == "multiclass":
            ret = self.reader.EvaluateMulticlass("bdt")
            ret = np.array([r for r in ret])
        elif self.kind == "binary":
            ret = self.reader.EvaluateMVA("bdt")
        elif self.kind == "regression":
            ret = self.reader.EvaluateRegression("bdt")
            ret = np.array([r for r in ret])
        return ret

class BDTxgboost(BDT):
    def __init__(self, model, feature_names, target_names):
        
        self.model = model
        kind = None
        if model.objective.startswith("binary:logistic"):
            kind = "binary"
        elif model.objective.startswith("multiclass"):
            kind = "multiclass"
        else:
            kind = "regression"
        print model.objective, kind

        trees = []
        for tree_dump in model.booster().get_dump():
            tree = xgbtree_to_nodetree(tree_dump)
            trees += [tree]

        super(BDTxgboost, self).__init__(trees, kind, feature_names, target_names, model.max_depth, model.learning_rate)

    def eval(self, features):
        proba = self.model.predict_proba(features)[:, 1]

        #invert sigmoid
        proba = -np.log(1.0/proba - 1.0)

        #apply TMVA transformation
        proba = 2.0 / (1.0 + np.exp(-2.0*proba)) - 1
        
        return proba

class BDTsklearn(BDT):

    def __init__(self, model, feature_names, target_names):
        
        self.model = model

        kind = None
        if isinstance(model, GradientBoostingRegressor):
            kind = "regression"
        elif isinstance(model, GradientBoostingClassifier):
            if len(target_names) == 2:
                kind = "binary"
            else:
                kind = "multiclass"

        trees = []
        #Loop over decision trees, in scikit that's a 2D array (N_estimators, N_classes)
        for sklearn_trees in model.estimators_:
             #write trees for different classes next to each other
            for class_tree in sklearn_trees:
                nodetree = {}
                sklearn_to_nodetree(model, nodetree, class_tree.tree_, 0, -1, -1)
                trees += [nodetree]

        super(BDTsklearn, self).__init__(trees, kind, feature_names, target_names, model.max_depth, model.learning_rate)


    def eval(self, vals):
        """A TMVA-compatible evaluation function for a scikit-learn classifier
        
        Args:
            vals (numpy array): An array (n_samples, n_features) of the input variables
        
        Returns:
            numpy array: (n_samples, n_classes) array of the output
        """
        
        #need to scale the same way as done in TMVA    
        scale = 1.0 / self.model.n_estimators

        if isinstance(self.model, GradientBoostingClassifier):
            #multiclass classification
            #according to TMVA::MethodBDT::GetMulticlassValues()
            if self.model.n_classes_ > 2:
                ret = np.zeros((vals.shape[0], self.model.n_classes_))
                for iclass in range(self.model.n_classes_):
                    for itree, t in enumerate(self.model.estimators_[:, iclass]):
                        r = t.predict(vals)
                        ret[:, iclass] += r * scale

                norm = np.zeros(ret.shape)
                for i in range(self.model.n_classes_):
                    for j in range(self.model.n_classes_):
                        if i != j:
                            norm[:, i] += np.exp(ret[:, j] - ret[:, i])

                ret = 1.0 / (1.0 + norm)        
                return ret
            #binary classification
            elif self.model.n_classes_ == 2:
                ret = np.zeros(vals.shape[0])

                for itree, t in enumerate(self.model.estimators_[:, 0]):
                    r = t.predict(vals)
                    ret += r * scale
                return 2.0/(1.0 + np.exp(-2.0 * ret)) - 1
        elif isinstance(self.model, GradientBoostingRegressor):
            ret = np.zeros((vals.shape[0], self.model.n_classes_))
            for iclass in range(self.model.n_classes_):
                for itree, t in enumerate(self.model.estimators_[:, iclass]):
                    r = t.predict(vals)
                    ret[:, iclass] += r * scale
            return ret

def tree_to_tmva(outfile, nodetree, current_node, scale):
    """Recursively writes out a decision tree as an XML
    
    Args:
        outfile (TYPE): Output file, must be writeable
        nodetree (TYPE): The dictionary with the nodes
        current_node (int): current node ID
        scale (float): The scale factor for each leaf
    
    Returns:
        nothing
    """
    outfile.write((nodetree[current_node].depth + 1)*"    " + nodetree[current_node].to_tmva(nodetree, scale) + "\n")
    for child in nodetree[current_node].children:
        tree_to_tmva(outfile, nodetree, child, scale)
    outfile.write((nodetree[current_node].depth + 1)*"    " + "</Node>\n")