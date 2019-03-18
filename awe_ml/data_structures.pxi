

cdef struct node_info_t:
    np_long counts_index, parents_index_begin, parents_index_end, updated_index, size, depth, parents_index_step

@cython.profile(False)
cdef void initialize_node(node_info_t & node, const np_long & n_classes, const np_long & depth ):
    node.depth = depth
    node.counts_index = 0
    node.parents_index_step = depth-1  #for level 1 the parent has no label, i.e. there is no feature # for top level node
    node.parents_index_begin = n_classes
    node.parents_index_end = n_classes + depth*node.parents_index_step
    node.updated_index = node.parents_index_end
    node.size = node.updated_index+1

#### define data structure to store indicies into a classfication values list

# cdef struct classification_vals_info_t:
#     np_long row_index, estimated_p_start, p_sum_start, weight_sum_start, max_child_noise_ind
#
# @cython.profile(False)
# cdef void compute_classification_vals_indicies(classification_vals_info_t & node, const np_long & local_tree_index, const np_long & n_classes):
#     """
#
#     :param node: node where all values are stored
#     :param local_tree_index: index of where in local tree we are
#     :param n_classes: number of classes
#     :return: nothing, node is updated by reference
#     """
#
#     ##create indicies for each value:
#     node.row_index = (3*n_classes+1) * local_tree_index
#     node.estimated_p_start = node.row_index+0                             # stores the estimate probability
#     node.p_sum_start = node.row_index+n_classes                      # stores the sum of p*weight for children
#     node.weight_sum_start = node.row_index+2*n_classes               # stores the total weight of the children
#     node.max_child_noise_ind = node.row_index + 3*n_classes          # stores the max noise value of children
