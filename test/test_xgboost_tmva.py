import xgboost
import numpy as np
from sklearn.datasets import fetch_mldata
import re
from mlglue.tree import Node


usps = fetch_mldata("usps")
data_x, data_y = usps.data, usps.target 
data_y = (data_y>5).astype(np.int32)

model = xgboost.XGBClassifier(n_estimators=10)
model.fit(data_x, data_y)
dump = model.booster().get_dump()

_NODEPAT = re.compile(r'(\d+):\[(.+)\]')
_LEAFPAT = re.compile(r'(\d+):(leaf=.+)')

#loop over all trees in the dump
for tree in dump:

    print "==="
    nodelist = []
    print tree

    parent_stack = []
    prev_depth = -1
    prev_index = -1

    nodes = {}

    #loop over all the nodes in the tree
    for node in tree.split("\n"):
        node_depth = node.count("\t")

        is_node = False
        is_leaf = False

        match = _NODEPAT.match(node.strip())
        if match is not None:
            node_index = int(match.group(1))
            node_variable, threshold = match.group(2).split("<")
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
            nodes[node_index] = Node(node_index, [], my_parent, node_depth, ("cut", node_variable, threshold))
        elif is_leaf:
            nodes[node_index] = Node(node_index, [], my_parent, node_depth, ("val", val))

        #insert node into final node dict
        if nodes.has_key(my_parent):
            nodes[my_parent].children += [node_index]

        prev_depth = node_depth
        prev_index = node_index

    for n in nodes.values():
        print n

