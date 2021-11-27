from re import X
from bindings import *
import random
import math
import conversions
import networkx as nx
import sklearn.neighbors
import numpy as np
import time

from rod.solvers.prm_basic import point_d_to_arr, calc_bbox
from rod.solvers.prm_knn_angle import custom_dist_with_angle
from rod.solvers.prm_smart_rotation import add_edge_if_motion_is_valid_smart_rotate
from rod.solvers.prm_dynamic_epsilon import DynamicCollisionDetector


# the radius by which the rod will be expanded
eps = [FT(0.4), FT(0.2), FT(0.1)]

# The steering const
steering_const = 3

def sample_free_point(x_range, y_range, z_range, length, cd):
    rand_x = FT(random.uniform(x_range[0], x_range[1]))
    rand_y = FT(random.uniform(y_range[0], y_range[1]))
    rand_z = FT(random.uniform(z_range[0], z_range[1]))

    # If not valid, try again
    while not cd.is_rod_position_valid(rand_x, rand_y, rand_z, length):
        rand_x = FT(random.uniform(x_range[0], x_range[1]))
        rand_y = FT(random.uniform(y_range[0], y_range[1]))
        rand_z = FT(random.uniform(z_range[0], z_range[1]))

    p = Point_d(3, [rand_x, rand_y, rand_z])
    return p

def steer(p, q, steering_const):
    _p, _q = point_d_to_arr(p), point_d_to_arr(q)
    distance = custom_dist_with_angle(_p, _q)
    if distance == 0:
        return None
    if distance <= steering_const:
        return q
    steered_vector = [FT(p_i + (q_i - p_i) * steering_const / distance) for p_i, q_i in zip(_p, _q)]
    return Point_d(3, steered_vector)

# User defined metric cannot be used with the kd_tree algorithm
nearest_neighbors = sklearn.neighbors.NearestNeighbors(n_neighbors=1, metric=custom_dist_with_angle, algorithm='auto')

def get_nearest_neighbor(points, query):
    global nearest_neighbors
    # sklearn (which we use for nearest neighbor search) works with numpy array
    # of points represented as numpy arrays
    _points = np.array([point_d_to_arr(p) for p in points])
    # Numpy stuff
    if len(points) == 1:
        _points = _points.reshape(1, -1)

    nearest_neighbors.fit(_points)
    neighbors = nearest_neighbors.kneighbors(np.array(point_d_to_arr(query)).reshape(1, -1), return_distance=False)[0]
    result = points[neighbors[0]]
    return result


def generate_path(length, obstacles, origin, destination, argument, writer, isRunning):
    t0 = time.perf_counter()
    # Parsing of arguments
    path = []
    try:
        num_iterations = int(argument)
    except Exception as e:
        print("argument is not an integer", file=writer)
        return path

    print('Initializing RRT', file=writer)

    polygons = [conversions.tuples_list_to_polygon_2(p) for p in obstacles]
    bbox = calc_bbox(polygons)
    x_range = (bbox[0].to_double(), bbox[1].to_double())
    y_range = (bbox[2].to_double(), bbox[3].to_double())
    z_range = (0, 2 * math.pi)

    begin = Point_d(3, [FT(origin[0]), FT(origin[1]), FT(origin[2])])
    end = Point_d(3, [FT(destination[0]), FT(destination[1]), FT(destination[2])])

    # Initiate the graph
    G = nx.DiGraph()
    G.add_nodes_from([begin])
    points = [begin]

    # Initiate the collision detector
    cd = DynamicCollisionDetector(eps, polygons)

    # Try to run RRT
    print('Running RRT', file=writer)
    for i in range(num_iterations):
        x_rand = sample_free_point(x_range, y_range, z_range, length, cd)
        x_near = get_nearest_neighbor(points, x_rand)
        x_new = steer(x_near, x_rand, steering_const)
        if x_new and add_edge_if_motion_is_valid_smart_rotate(G, cd, x_near, x_new, length):
            points.append(x_new)

        if (i + 1) % 100 == 0:
            print('Iterated RRT', (i+1), 'times', file=writer)
            # Try to add the final point every once in a while
            x_near = get_nearest_neighbor(points, end)
            if add_edge_if_motion_is_valid_smart_rotate(G, cd, x_near, end, length):
                points.append(end)
                print("Added final point to RRT Tree", file=writer)
                break
    
    if end not in points:
        # Final try to add the final point every
        x_near = get_nearest_neighbor(points, end)
        if add_edge_if_motion_is_valid_smart_rotate(G, cd, x_near, end, length):
            points.append(end)
            print("Added final point to RRT Tree", file=writer)    
        else:
            print(f"Failed to connect final point to tree. Final point: {end}, closest point: {x_near}", file=writer)
            return path

    # Check for path
    if nx.has_path(G, begin, end):
        shortest_path = nx.shortest_path(G, begin, end)
        print("path found", file=writer)
        print("distance:", nx.shortest_path_length(G, begin, end, weight='weight'), file=writer)

        if len(shortest_path) == 0:
            return path
        first = shortest_path[0]
        path.append((first[0], first[1], first[2], True))
        for i in range(1, len(shortest_path)):
            last = shortest_path[i-1]
            next = shortest_path[i]
            # determine correct direction
            clockwise = G.get_edge_data(last, next)["clockwise"]
            path.append((next[0], next[1], next[2], clockwise))
    else:
        print("no path was found", file=writer)
    t1 = time.perf_counter()
    print("Time taken:", t1 - t0, "seconds", file=writer)
    return path
