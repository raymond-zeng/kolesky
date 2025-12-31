from .maxheap cimport Heap
from libcpp.vector cimport vector
from scipy.spatial import KDTree
import numpy as np
cimport numpy as np

cdef inline void update_dists(
    Heap heap,
    double[:, ::1] dists,
    double[::1] dists_k,
    long[::1] js,
):
    """Update the distance table and heap."""
    cdef:
        int p, index, i, insert
        long j
        float d

    p = dists.shape[1]
    for index in range(js.shape[0]):
        j = js[index]
        d = dists_k[index]
        # insert d into dists[j], pushing out the largest value
        i = 0
        for i in range(p):
            if d <= dists[j, i]:
                break
        insert = i
        for i in range(p - 1, insert, -1):
            dists[j, i] = dists[j, i - 1]
        if insert < p:
            dists[j, insert] = d
        heap.__decrease_key(j, dists[j, p - 1])

np.import_array()
cdef inline double[::1] _distance_vector(double[:, ::1] points, double[::1] point):
   cdef:
       int n, i, j
       double dist, d
       double *start
       double *p
       np.ndarray[np.float64_t, ndim=1] dists
   n = points.shape[1]
   start = &points[0, 0]
   p = &point[0]
   dists = np.empty(points.shape[0], np.float64)
   for i in range(points.shape[0]):
       dist = 0
       for j in range(n):
           d = (start + i*n)[j] - p[j]
           dist += d*d
       dists[i] = dist
   np.sqrt(dists, out=dists)
   return dists

cpdef tuple reverse_maximin(np.ndarray[np.float64_t, ndim=2] points, double[:, ::1] initial = None):
    cdef:
        int n, i, k, index, j, start
        double lk
        long[::1] indices
        double[::1] lengths
        Heap heap
        double[::1] dists
        list js
    n = points.shape[0]
    indices = np.empty(n, dtype = np.long)
    lengths = np.empty(n, dtype = np.float64)
    if initial is None or initial.shape[0] == 0:
        k = 0
        dists = _distance_vector(points, points[0])
        indices[n - 1] = k
        lengths[n - 1] = np.inf
        start = n - 2
    else:
        initial_tree = KDTree(initial)
        dists = initial_tree.query(points)[0]
        start = n - 1
    tree = KDTree(points)
    heap = Heap(dists, np.arange(n))
    for i in range(start, -1, -1):
        lk, k = heap.pop()
        indices[i] = k
        lengths[i] = lk
        js = tree.query_ball_point(points[k], lk)
        dists = _distance_vector(points[js], points[k])
        for index in range(len(js)):
            j = js[index]
            heap.decrease_key(j, dists[index])
    return indices, lengths

cpdef tuple p_reverse_maximin(np.ndarray[np.float64_t, ndim=2] points, double[:, ::1] initial = None, int p = 1):
    cdef:
        int n, i, k
        double inf, lk
        long[::1] indices
        double[::1] lengths
        double[:, ::1] dists
        double[::1] dists_k
        Heap heap
        list js
    inf = 1e6
    n = points.shape[0]
    indices = np.empty(n, dtype = np.int64)
    lengths = np.empty(n, dtype = np.float64)
    if initial is None or initial.shape[0] == 0:
        dists = np.array([[-i + inf] * p for i in range(n)])
    else:
        initial_tree = KDTree(initial)
        dists = (initial_tree.query(points, p)[0]).reshape(n, p)
    tree = KDTree(points)
    heap = Heap(np.max(dists, axis=1), np.arange(n))
    for i in range(n - 1, -1, -1):
        lk, k = heap.pop()
        indices[i] = k
        if lk < inf - n:
            lengths[i] = lk
        else:
            lengths[i] = np.inf
        js = tree.query_ball_point(points[k], lk)
        dists_k = _distance_vector(points[js], points[k])
        update_dists(heap, dists, dists_k, np.array(js, dtype=np.int64))
    return indices, lengths

cpdef object[::1] sparsity_pattern(double[:, ::1] points, double[::1] lengths, double rho):
    cdef:
        int n, i, offset
        long j
        double length_scale
        object[::1] sparsity
    n = points.shape[0]
    tree = KDTree(points)
    offset = 0
    length_scale = lengths[0]
    sparsity = np.empty(n, dtype = object)
    for i in range(n):
        if lengths[i] > 2 * length_scale:
            tree = KDTree(points[i:])
            offset = i
            length_scale = lengths[i]
        sparsity[i] = [
            offset + j
            for j in tree.query_ball_point(points[i], rho * lengths[i])
            if offset + j >= i
        ]
    return sparsity

cpdef tuple optimized_sparsity_pattern(double[:, ::1] points, double[::1] lengths, double rho):
    cdef:
        int n = points.shape[0]
        int i, k, offset, count
        long j, real_index
        double length_scale, r
        
        # CSR Row Pointers
        long[::1] indptr = np.zeros(n + 1, dtype=np.int64)
        
        # C++ Vector for indices (Automatic dynamic array)
        vector[int] indices
        
        # Tree variables
        object tree
        list neighbors
    
    tree = KDTree(points)
    offset = 0
    length_scale = lengths[0]
    indptr[0] = 0

    for i in range(n):
        if lengths[i] > 2 * length_scale:
            tree = KDTree(points[i:])
            offset = i
            length_scale = lengths[i]
        
        neighbors = tree.query_ball_point(points[i], rho * lengths[i])
        
        count = 0
        for j in neighbors:
            real_index = offset + j
            if real_index >= i:
                indices.push_back(real_index)
                count += 1
        
        indptr[i + 1] = indptr[i] + count
    
    cdef long[:] indices_view = np.empty(indices.size(), dtype=np.int64)
    for i in range(indices.size()):
        indices_view[i] = indices[i]

    return indices_view, indptr

cpdef tuple supernodes(object sparsity, double[::1] lengths, double lamb):
    cdef int n = lengths.shape[0]
    cdef set candidates = set(range(n))
    
    # Use C++ vectors for dynamic appending
    cdef vector[int] groups_data_vec
    cdef vector[int] groups_indptr_vec
    
    # Store lengths for each node, then build indptr via cumsum
    cdef int[:] agg_lengths = np.zeros(n, dtype=np.int32)
    
    # Store (node_index, s_list) pairs to fill agg_data later
    cdef list node_slists = []
    
    cdef int i = 0
    cdef int j, k, offset
    cdef list group
    cdef list s_list
    cdef set s_set
    cdef dict positions
    
    groups_indptr_vec.push_back(0)
    
    while len(candidates) > 0:
        while i not in candidates:
            i += 1
            
        group = sorted([j for j in sparsity[i] 
                        if lengths[j] <= lamb * lengths[i] and j in candidates])
        
        # Append group members to groups_data
        for j in group:
            groups_data_vec.push_back(j)
        groups_indptr_vec.push_back(groups_data_vec.size())
        
        candidates -= set(group)
        
        # Compute aggregate sparsity
        s_set = set()
        for j in group:
            for k in sparsity[j]:
                s_set.add(k)
        s_list = sorted(list(s_set))
        
        positions = {k: idx for idx, k in enumerate(s_list)}
        
        # Store length for each group member and save data for later
        for j in group:
            offset = positions[j]
            agg_lengths[j] = len(s_list) - offset
            node_slists.append((j, s_list, offset))
    
    # Build agg_indptr from agg_lengths via cumsum
    cdef int[:] agg_indptr = np.zeros(n + 1, dtype=np.int32)
    cdef int cumsum = 0
    for i in range(n):
        agg_indptr[i] = cumsum
        cumsum += agg_lengths[i]
    agg_indptr[n] = cumsum
    
    # Now fill agg_data using the stored node_slists
    cdef int[:] agg_data = np.zeros(cumsum, dtype=np.int32)
    cdef int node_idx, slist_offset
    cdef list slist
    for item in node_slists:
        node_idx = item[0]
        slist = item[1]
        slist_offset = item[2]
        for k in range(len(slist) - slist_offset):
            agg_data[agg_indptr[node_idx] + k] = slist[slist_offset + k]
    
    # Convert vectors to memoryviews
    cdef int[:] groups_data = np.zeros(groups_data_vec.size(), dtype=np.int32)
    cdef int[:] groups_indptr = np.zeros(groups_indptr_vec.size(), dtype=np.int32)
    
    for i in range(<int>groups_data_vec.size()):
        groups_data[i] = groups_data_vec[i]
    for i in range(<int>groups_indptr_vec.size()):
        groups_indptr[i] = groups_indptr_vec[i]

    return groups_data, groups_indptr, agg_data, agg_indptr