EDIT: Instead of daily notes, I am going to take topic-wise notes.

Started with decision trees today! I found resources on the theory of decision trees but I am having trouble with implementing the theory. So I googled some resources on "decision trees from scratch" and found a couple of notebooks (I will attach links at the bottom). Every one has implemented it on their own but after reading what I understand is:
 - I need to calculate entropy for a node, where entropy = $$-\sum_{i=1} ^ {n} (p_i * log_2p_i)$$
 - Calculate information gain for a parent node having left and right children nodes, where  
    $$IG(parent) = entropy(parent) - \frac{n_{left}}{n_{parent}} * entropy(left) - \frac{n_{right}}{n_{parent}} * entropy(right)$$
 - A node in the tree will contain a feature, left subtree, right subtree and information gain. A leaf node will contain the value.  
 - The decision tree will have a max depth and a minimum sample split as the stopping criteria.   

 So far I have implemented this. I need to complete the best sample split function and then start building the tree, fitting and prediction.

10/3/26
I finished the best split finder function. 
 - For each column (feature), I need to iterate over all the unique values. 
 - For each unique value, I will assume it to be the threshold as the parent node and check the information gain for all the values to the left (more) and right (less) of this node. If information gain is more than before, I will use that threshold as the splitting value.  

To build the tree:
 - Check if the max_depth has reached or the min_samples_split for the parent node has reached. 
     * If yes: 
        * return the leaf node with the most common label at that particular split
     * else:
        * return the splitting internal node.  

After building the tree, prediction is an easy task. Make a recursive call to a function which checks the value of the current node (starting with root). 
 - If the node contains any value (a leaf node), return the value.
 - Else:
      * If the feature value for the internal node is less than the node threshold, traverse left.
      * Else traverse right.



Links:
 - https://github.com/daradecic/BDS-articles/blob/main/013_MML_Decision_Trees.ipynb
 - https://github.com/eriklindernoren/ML-From-Scratch/blob/master/mlfromscratch/supervised_learning/decision_tree.py 