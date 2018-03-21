import numpy as np
from sklearn.neighbors import BallTree
import queue 
## distance functions
def point2dist(point, line):
    """calculate distance from a point to a line"""
    linelen = np.linalg.norm(line[1]-line[0])
    if linelen == 0:
        return 0
    else:
        return np.linalg.norm(np.cross(point-line[0], line[1]-line[0]))/linelen

def vertical_distance(x, y):
    """vertical distance as defined in paper"""
    if np.linalg.norm(x[1]-x[0]) < np.linalg.norm(y[1]-y[0]):
        dist_end1 = point2dist(x[0], y)
        dist_end2 = point2dist(x[1], y)
    else:
        dist_end1 = point2dist(y[0], x)
        dist_end2 = point2dist(y[1], x)
    dist = dist_end1+dist_end2
    if dist == 0:
        return 0
    else:
        return (dist_end1**2 + dist_end2**2)/dist

def parallel_distance(x, y):
    l2square = np.dot(y[1]-y[0], y[1]-y[0])
    if l2square != 0:
        l1 = np.linalg.norm(np.dot(x[0]-y[0], y[1]-y[0])/l2square*(y[1]-y[0]))
        l2 = np.linalg.norm(y[1] - y[0] - np.dot(x[1]-y[0], y[1]-y[0])/l2square*(y[1]-y[0]))
        return min(l1, l2)
    else:
        return 0

def angular_distance(x, y):
    ylen = np.linalg.norm(y[1]-y[0])
    xlen = np.linalg.norm(x[1]-x[0])
    shorter_len = min(xlen, ylen)
    longer_len = max(xlen, ylen)
    dist = np.dot(x[1]-x[0], y[1]-y[0])
    if dist < 0:
        return shorter_len
    else:
        return 0 if longer_len == 0 else np.linalg.norm(np.cross(x[1]-x[0], y[1]-y[0]))/longer_len
    
def line_distance(x, y):
    return vertical_distance(x, y)+parallel_distance(x, y)+angular_distance(x, y)


# MDL cost
def mdl(data, model_indexes):
    """
    calculate mdl cost
    data the raw data
    model_indexes indexes into the raw data, data points choose by indexes are used to represent the raw data
    """
    # L(H) model cost could have multiple segments
    lh = 0
    for i in range(len(model_indexes)-1):
        dist = np.linalg.norm(data[model_indexes[i]]-data[model_indexes[i+1]])
        lh = lh + np.log2(dist) if dist!=0 else 0
    # L(D|H)
    ldh = 0
    # the index into the model_indexes, NOT into raw data
    model_index = 0
    for i in range(len(data)-1):
        if i > model_indexes[model_index+1]:
            model_index = model_index + 1

        model_i_start = model_indexes[model_index]
        model_i_end = model_indexes[model_index+1]
        if i != model_i_start or i+1 != model_i_end:
            dist = vertical_distance((data[model_i_start], data[model_i_end]),(data[i], data[i+1]))* \
                angular_distance((data[model_i_start], data[model_i_end]), (data[i], data[i+1]))
            ldh = ldh + np.log2(dist) if dist != 0 else 0
    return lh + ldh

## MDL partition algorithm
def mdl_partition(path):
    partition_set = [path[0]]
    start_index = 0
    length = 1
    while start_index + length < len(path):
        curr_index = start_index + length
        # MDL cost if curr_index is included as a characteristic point
        mdlparcost = mdl(path[start_index:curr_index+1], (0, curr_index-start_index))
        # no partition, segmented line
        mdlnoparcost = mdl(path[start_index:curr_index+1], np.arange(curr_index-start_index+1))
        if mdlparcost > mdlnoparcost:
            partition_set = partition_set + [path[curr_index-1]]
            start_index = curr_index-1
            length = 1
        else:
            length = length + 1
    partition_set = partition_set + [path[len(path)-1]]
    return partition_set

NUM_INDEX = 2
def segment_line_dist(xl, yl):
    xl = xl[:-NUM_INDEX]
    yl = yl[:-NUM_INDEX]
    # hacky, this function will be called by DistanceMetric.get_metric,
    # which will supply a random xl vector of length 10
    # we have to limit the dimension to 3D in order to do a cross product
    dimension = min(len(xl)/2, 3)
    #pdb.set_trace()
    return line_distance((xl[:dimension], xl[dimension:2*dimension]), 
            (yl[:dimension], yl[dimension:2*dimension]))


def segment_clustering(segments, eps, minlines):
    """
    cluster line segment
    segments: line segments to be clustered. last two columns are clusterid and pathid. 
    clusterid: -1 means not clusterd, -2 means noise
    eps: eps-neighbour
    minlines: threshold, mimimum number of lines
    """
    tree = BallTree(segments, leaf_size=10, metric=segment_line_dist)
    clusters = -np.ones(len(segments))
    clusterid = 0
    q = queue.Queue()
    for i in np.arange(len(segments)):
        if clusters[i]==-1:            
            segment = segments[i]
            neighbour_index = tree.query_radius(segment.reshape(1, -1), eps)[0]
            #pdb.set_trace()
            if len(neighbour_index) >= minlines:
                #pdb.set_trace()
                clusters[neighbour_index] = clusterid
                [q.put(x) for x in neighbour_index if x != i]
                while not q.empty():
                    m = segments[q.get()]
                    mneighbour_index = tree.query_radius(m.reshape(1, -1), eps)[0]
                    if len(mneighbour_index) >= minlines:
                        for idx in mneighbour_index:
                            if clusters[idx] < 0:
                                clusters[idx] = clusterid
                            if clusters[idx] == -1:
                                q.put(idx)
                clusterid = clusterid + 1
            else:
                clusters[i]=-2
                
    return clusters
