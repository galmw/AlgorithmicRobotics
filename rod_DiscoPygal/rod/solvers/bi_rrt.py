from re import X
from bindings import *
import math
import conversions
import networkx as nx
import numpy as np
import time

from rod.solvers.prm_smart_rotation import add_edge_if_motion_is_valid_smart_rotate
from rod.solvers.prm_dynamic_epsilon import DynamicCollisionDetector

from rod.solvers.prm_basic import calc_bbox
from rod.solvers.rrt import sample_free_point, steer, get_nearest_neighbor

# the radius by which the rod will be expanded
eps = [FT(0.4), FT(0.2), FT(0.1)]

# The steering const
steering_const = 3


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

    # Initiate the graphs
    G1 = nx.DiGraph()
    G1.add_nodes_from([begin])
    g1_points = [begin]

    G2 = nx.DiGraph()
    G2.add_nodes_from([end])
    g2_points = [end]

    # Initiate the collision detector
    cd = DynamicCollisionDetector(eps, polygons)

    # Try to run Bi-RRT
    print('Running Bi-RRT', file=writer)
    curr_graph, curr_points, other_graph, other_points = G1, g1_points, G2, g2_points

    graphs_connection_point = None
    for i in range(num_iterations):
        x_rand = sample_free_point(x_range, y_range, z_range, length, cd)
        x_near = get_nearest_neighbor(curr_points, x_rand)
        x_new = steer(x_near, x_rand, steering_const)
        if x_new and add_edge_if_motion_is_valid_smart_rotate(curr_graph, cd, x_near, x_new, length):
            curr_points.append(x_new)
            # Try to connect the graphs
            other_x_near = get_nearest_neighbor(other_points, x_new)
            other_x_new = steer(other_x_near, x_new, steering_const)
            if other_x_new and add_edge_if_motion_is_valid_smart_rotate(other_graph, cd, other_x_near, other_x_new, length):
                other_points.append(other_x_new)
                # Check if done
                if other_x_new == x_new:
                    graphs_connection_point = x_new
                    break
        if (i + 1) % 100 == 0:
            print('Iterated Bi-RRT', (i+1), 'times', file=writer)
        # Swap graphs
        curr_graph, curr_points, other_graph, other_points = other_graph, other_points, curr_graph, curr_points

    # Check for path
    if graphs_connection_point:
        assert nx.has_path(G1, begin, graphs_connection_point) and nx.has_path(G2, end, graphs_connection_point)
        shortest_path1 = nx.shortest_path(G1, begin, graphs_connection_point)
        shortest_path2 = nx.shortest_path(G2, end, graphs_connection_point)

        print("path found", file=writer)
        print("distance:", nx.shortest_path_length(G1, begin, graphs_connection_point, weight='weight') + 
                           nx.shortest_path_length(G2, end, graphs_connection_point, weight='weight'), file=writer)

        if len(shortest_path1) == 0 and len(shortest_path2) == 0:
            return path
        first = shortest_path1[0]
        path.append((first[0], first[1], first[2], True))
        for i in range(1, len(shortest_path1)):
            last = shortest_path1[i - 1]
            next = shortest_path1[i]
            # determine correct direction
            clockwise = G1.get_edge_data(last, next)["clockwise"]
            path.append((next[0], next[1], next[2], clockwise))
        for i in range(len(shortest_path2) - 1, 0, -1):
            last = shortest_path2[i]
            next = shortest_path2[i - 1]
            # determine correct direction (flip)
            clockwise = not G2.get_edge_data(next, last)["clockwise"]
            path.append((next[0], next[1], next[2], clockwise))
        
    else:
        print("no path was found", file=writer)
    t1 = time.perf_counter()
    print("Time taken:", t1 - t0, "seconds", file=writer)
    return path
