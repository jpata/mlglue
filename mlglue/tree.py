class Node:
    """Represents a node in a decision tree, identified by a unique integer id
    
    Attributes:
        children (list of int): The id-s of the children associated with this node
        depth (int): The depth of this node in the tree
        id (int): The unique id of this node
        parent (id): The id of the parent node
        payload (tuple): Describes what this node does, i.e. is it a non-terminal node (cut) or a terminal (leaf) 
    """
    def __init__(self, id, children, parent, depth, payload):
        self.id = id
        self.children = children
        self.parent = parent
        self.depth = depth
        self.payload = payload

    def __repr__(self):
        return "Node id:{id} children:{children} parent:{parent}, depth:{depth}, payload:{payload}".format(**{
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
            nodetree (dict int->Node): The dictionary of the full tree
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
    IVar="{2}" Cut="{3:.8E}" cType="1" \
    res="{4:.8E}" rms="0.0e-00" \
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
    IVar="{2}" Cut="{3:.8E}" cType="1" \
    res="{4:.8E}" rms="0.0" \
    purity="{5:.8E}" nType="0">'.format(
            kind,
            self.depth + 1,
            self.payload[1],
            self.payload[2],
            0.0, 0.0
        )

def tree_to_tmva(outfile, nodetree, node, scale):
    outfile.write((nodetree[node].depth + 1)*"    " + nodetree[node].to_tmva(nodetree, scale) + "\n")
    for child in nodetree[node].children:
        tree_to_tmva(outfile, nodetree, child, scale)
    outfile.write((nodetree[node].depth + 1)*"    " + "</Node>\n")