

########### Define helper functions used to calculate intermediate values (usefulnes, scaling weight)
@cython.profile(False)
cdef inline void calculate_usefulness_simple(np_float * p_child, np_float * p_parent, vector[np_float] & return_vec):
    """
    Calculate simple usefulness model (all input vectors of length self.n_classes) cython calls python when retuning a memview 
        and so pass return vec as an argument
    :param p_child: vector of child probabilities
    :param p_parent: vector of parent probabilities
    :return: vector of usefulness values
    """
    cdef unsigned int class_num
    cdef np_float usefulness=0

    for class_num in range(return_vec.size()):

        if p_child[class_num] !=p_parent[class_num]:
            usefulness+= abs(p_child[class_num]-p_parent[class_num])/(p_child[class_num]+p_parent[class_num])

    # store result to pre-initialized data result memory
    for class_num in range(return_vec.size()):
        return_vec[class_num]=usefulness

@cython.profile(False)
cdef inline void calculate_usefulness_scaled_per_class(np_float * child_scaled_p, np_float * parent_scaled_p, vector[np_float] & return_vec):
    """
    Calculate usefulness model based on probability scaling independently for each class
    """
    cdef unsigned int class_num

    for class_num in range(return_vec.size()):
        return_vec[class_num]= abs(child_scaled_p[class_num]-parent_scaled_p[class_num])




@cython.profile(False)
cdef inline void calculate_usefulness_none(np_float * p_child, np_float * p_parent, vector[np_float] & return_vec):
    # store 1 to pre-initialized data result memory
    cdef unsigned int class_num
    for class_num in range(return_vec.size()):
        return_vec[class_num]=1


# @cython.profile(False)
# cdef inline void calculate_usefulness_scaled(np_float * child_scaled_p, np_float * parent_scaled_p, vector[np_float] & return_vec):
#     """
#     Calculate usefulness model based on probability scaling
#     """
#     cdef unsigned int class_num
#     cdef np_float usefulness=0
#
#     for class_num in range(return_vec.size()):
#         usefulness+= abs(child_scaled_p[class_num]-parent_scaled_p[class_num])
#
#     # store result to pre-initialized data result memory
#     for class_num in range(return_vec.size()):
#         return_vec[class_num]=usefulness


@cython.profile(False)
cdef inline void compute_scaling_weights_reciprocal(const np_float & n_noise, np_float * p, const np_float & total_count, vector[np_float] & result):
    """
    computes the weight scaling that increases the weight as the probability approaches 0 or 1

    :param p: list of probabilities by output class
    :param total_count: total number of examples supporting a given combination of features
    :type total_count: int
    :return:
    """
    # clip probability for computing scale based on the counts
    cdef np_float clipped_p, p_limit = 0.5

    if total_count>2*n_noise:
        p_limit =  n_noise/total_count

    for class_ind in range(result.size()):
        if p[class_ind]<p_limit:
            clipped_p = p_limit
        elif p[class_ind]>(1-p_limit):
            clipped_p = 1-p_limit
        else:
            clipped_p = p[class_ind]

        result[class_ind]= 1/(clipped_p*(1-clipped_p))


@cython.profile(False)
cdef inline void compute_scaling_weights_reciprocal_class_imbalance (const np_float & n_noise, np_float * p, const np_float & total_count, const vector[np_float] & class_p, vector[np_float] & result):
    """
    computes the weight scaling that increases the weight as the probability approaches 0 or 1

    :param p: list of probabilities by output class
    :param total_count: total number of examples supporting a given combination of features
    :type total_count: int
    :param class_p: pointer to vector of the overall probability for each class
    :return:
    """
    # clip probability for computing scale based on the counts
    cdef np_float clipped_p, p_limit_max, p_limit_min


    for class_ind in range(result.size()):
        p_limit_max =  max(1-class_p[class_ind]*n_noise/total_count,0.5)
        p_limit_min =  min((1-class_p[class_ind])*n_noise/total_count,0.5)


        if p[class_ind]<p_limit_min:
            clipped_p = p_limit_min
        elif p[class_ind]>p_limit_max:
            clipped_p = p_limit_max
        else:
            clipped_p = p[class_ind]

        result[class_ind]= 1/(clipped_p*(1-clipped_p))

#
# cdef inline void compute_scaled_p_for_logit_usefulness (const np_float & n_noise, np_float * p, const np_float & total_count, np_float * class_p, vector[np_float] & result):
#     """
#     computes the weight scaling that increases the weight as the probability approaches 0 or 1
#
#     :param p: list of probabilities by output class
#     :param total_count: total number of examples supporting a given combination of features
#     :type total_count: int
#     :param class_p: pointer to vector of the overall probability for each class
#     :return:
#     """
#     # clip probability for computing scale based on the counts
#     cdef np_float clipped_p, p_limit_max, p_limit_min
#
#     for class_ind in range(result.size()):
#         p_limit_max =  max(1-class_p[class_ind]*n_noise/total_count,0.5)
#         p_limit_min =  min((1-class_p[class_ind])*n_noise/total_count,0.5)
#
#
#         if p[class_ind]<p_limit_min:
#             clipped_p = p_limit_min
#         elif p[class_ind]>p_limit_max:
#             clipped_p = p_limit_max
#         else:
#             clipped_p = p[class_ind]
#
#         result[class_ind]= log(clipped_p/(1-clipped_p))





@cython.profile(False)
cdef inline void compute_scaling_weights_logit(const np_float & n_noise, np_float * p, const np_float & total_count, vector[np_float] & result):
    # clip probability for computing scale based on the counts
    cdef np_float clipped_p, p_limit = 0.5

    if total_count>2*n_noise:
        p_limit =  n_noise/total_count

    for class_ind in range(result.size()):
        if p[class_ind]<p_limit:
            clipped_p = p_limit
        elif p[class_ind]>(1-p_limit):
            clipped_p = 1-p_limit
        else:
            clipped_p = p[class_ind]

        if clipped_p==0.5:
            result[class_ind] = 4
        else:
            result[class_ind] = log(clipped_p/(1-clipped_p)) / (clipped_p-0.5)
