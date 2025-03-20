from .maxheap cimport Heap
from scipy.spatial import KDTree
import numpy as np
cimport numpy as np

cpdef void update_dists(
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
cdef double[::1] _distance_vector(double[:, ::1] points, double[::1] point):
   cdef:
       int n, i, j
       double dist, d
       double *start
       double *p
       double[::1] dists
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
   dists = np.sqrt(dists)
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
    indices = np.empty(n, dtype = np.long)
    lengths = np.empty(n, dtype = np.float64)
    if initial is None or initial.shape[0] == 0:
        dists = np.array([[-i + inf] * p for i in range(n)])
    else:
        initial_tree = KDTree(initial)
        dists = initial_tree.query(points, p)[0]
        dists = dists.reshape(n, p)
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