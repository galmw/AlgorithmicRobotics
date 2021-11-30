from re import X
from bindings import *
import random
import math
import conversions
import networkx as nx
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
    rand_x, rand_y, rand_z = None, None, None
    # Generate valid point
    while rand_x is None or not cd.is_rod_position_valid(rand_x, rand_y, rand_z, length):
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


def get_nearest_neighbor(points, _points, query):
    _query = point_d_to_arr(query)
    index = min(enumerate(_points), key=lambda _ip: custom_dist_with_angle(_ip[1], _query))[0]
    return points[index]


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
    _end = point_d_to_arr(end)
    # Initiate the graph
    G = nx.DiGraph()
    G.add_nodes_from([begin])
    points = [begin]
    _points = [point_d_to_arr(p) for p in points]

    # Initiate the collision detector
    cd = DynamicCollisionDetector(eps, polygons)

    # Try to run RRT
    print('Running RRT', file=writer)
    for i in range(num_iterations):
        x_rand = sample_free_point(x_range, y_range, z_range, length, cd)
        x_near = get_nearest_neighbor(points, _points, x_rand)
        x_new = steer(x_near, x_rand, steering_const)
        if x_new and add_edge_if_motion_is_valid_smart_rotate(G, cd, x_near, x_new, length):
            points.append(x_new)
            _x_new = point_d_to_arr(x_new)
            _points.append(_x_new)
            # Try to add the final point if it's close enough
            if custom_dist_with_angle(_x_new, _end) <= steering_const:
                if add_edge_if_motion_is_valid_smart_rotate(G, cd, x_new, end, length):
                    points.append(end)
                    break
        if (i + 1) % 100 == 0:
            print('Iterated RRT', (i+1), 'times', file=writer)

    # Check for path
    if end in points and nx.has_path(G, begin, end):
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
