"""
    Derived from demand_first_waypoints.py and waypoint_greedy_multipath.py.
    Follows the same procedure as them, assigning each demand the ideal waypoint (can be none),
    but allows splitting the traffic between the waypoint route and the original one,
    using the ideal split fraction in regard to the mlu at the moment.
    The ideal split fraction is computed using properties of the mlu and link utilization functions
    as shown in the thesis.
    Still uses ECMP.
    Link weights are as given or all set to 1.
"""

import time

import networkit as nk
import numpy as np

from algorithm.generic_sr import GenericSR

class IdealWaypointOptimization(GenericSR):
    BIG_M = 10 ** 9

    def __init__(self, nodes: list, links: list, demands: list, weights: dict = None, waypoints: dict = None, **kwargs):
        self.nodes = nodes
        self.links = links
        self.demands = demands
        self.weights = weights if weights else {(u, v): 1. for u, v, _ in links}
        self.__capacities = self.__extract_capacity_dict(links)
        self.__links = list(self.__capacities.keys())
        self.__n = len(nodes)
        self.__capacity_map = None
        self.__demands = demands
        self.__weights = self.weights
        self.__g = None
        self.__apsp = None

        # Add additional node
        self.__additional_node = self.__n
        self.__n += 1

        self.__init_graph()
        self.__init_capacity_map()
        

    @staticmethod
    def __extract_capacity_dict(links):
        return {(u, v): c for u, v, c in links}

    def __init_capacity_map(self):
        self.__capacity_map = np.ones((self.__n, self.__n), np.float)
        for u, v in self.__links:
            self.__capacity_map[u][v] = self.__capacities.get((u, v), 1.0)
        # Add capacity for links involving the additional node
        for u in range(self.__n - 1):
            self.__capacity_map[u][self.__additional_node] = self.__capacities.get((u, self.__additional_node), 1.0)
            self.__capacity_map[self.__additional_node][u] = self.__capacities.get((self.__additional_node, u), 1.0)

    def __init_graph(self):
        self.__g = nk.Graph(weighted=True, directed=True, n=self.__n)
        for u, v in self.__links:
            self.__g.addEdge(u, v, self.__weights.get((u, v), 1.0))
        # Add edge for the additional node
        for u in range(self.__n - 1):
            self.__g.addEdge(u, self.__additional_node, self.__weights.get((u, self.__additional_node), 1.0))
            self.__g.addEdge(self.__additional_node, u, self.__weights.get((self.__additional_node, u), 1.0))
        self.__apsp = nk.distance.APSP(self.__g)

   

    def __compute_distances(self):
        self.__apsp.run()
        return self.__apsp.getDistances()

    def __get_shortest_path_fraction_map(self, distances):
        link_fraction_map = np.zeros((self.__n, self.__n, self.__n, self.__n), np.float)
        for s in range(self.__n):
            u_map = dict(zip(range(self.__n), np.array(distances[s]).argsort()))
            for t in range(self.__n):
                if s == t:
                    continue
                node_fractions = np.zeros(self.__n, np.float)
                node_fractions[s] = 1
                for u_idx in range(self.__n - 1):
                    u = u_map[u_idx]
                    fraction = node_fractions[u]
                    if not fraction:
                        continue
                    successors = list(v for v in self.__g.iterNeighbors(u) 
                                      if (u, v) in self.__weights and self.__weights[(u, v)] == distances[u][t] - distances[v][t])
                    new_fraction = fraction / len(successors) if len(successors) != 0 else fraction
                    for v in successors:
                        link_fraction_map[s][t][u][v] = new_fraction
                        node_fractions[v] += new_fraction if v != t else 0.
        return link_fraction_map

    def __get_flow_map(self, sp_fraction_map):
        flow_map = np.zeros((self.__n, self.__n), np.float)
        for s, t, d in self.__demands:
            flow_map += sp_fraction_map[s][t] * d
        return flow_map

    def __compute_utilization(self, flow_map):
        util_map = (flow_map / self.__capacity_map)
        objective = np.max(util_map)
        return util_map, objective

    def __update_flow_map(self, sp_fraction_map, flow_map, s, t, d, waypoint, multipath_split=1.0):
        new_flow_map = flow_map - sp_fraction_map[s][t] * (d * multipath_split)
        new_flow_map += sp_fraction_map[s][waypoint] * (d * multipath_split)
        new_flow_map += sp_fraction_map[waypoint][t] * (d * multipath_split)
        return new_flow_map

    def __adaptive_traffic_splitting(self, current_best_util_map, s, t, d, waypoint, sp_fraction_map, best_flow_map):
        def binary_search_split(low, high):
            best_split = low
            best_objective = float('inf')
            while low <= high:
                mid = (low + high) / 2
                test_flow_map = self.__update_flow_map(sp_fraction_map, best_flow_map, s, t, d, waypoint, mid)
                test_util_map, test_objective = self.__compute_utilization(test_flow_map)
                if test_objective < best_objective:
                    best_split = mid
                    best_objective = test_objective
                    high = mid - 0.1
                else:
                    low = mid + 0.1
            return best_split
        return binary_search_split(0.1, 1.0)

    def __waypoint_multipath_adaptive(self):
        distances = self.__compute_distances()
        sp_fraction_map = self.__get_shortest_path_fraction_map(distances)
        best_flow_map = self.__get_flow_map(sp_fraction_map)
        best_util_map, best_objective = self.__compute_utilization(best_flow_map)
        current_best_flow_map = best_flow_map
        current_best_util_map = best_util_map
        current_best_objective = best_objective

        waypoints = dict()
        sorted_demand_idx_map = dict(zip(range(len(self.__demands)), np.array(self.__demands)[:, 2].argsort()[::-1]))
        for d_map_idx in range(len(self.__demands)):
            d_idx = sorted_demand_idx_map[d_map_idx]
            s, t, d = self.__demands[d_idx]
            current_best_waypoint = None
            current_best_split = 0

            # Check if any link exceeds MLU threshold
            if np.max(self.__compute_utilization(self.__get_flow_map(sp_fraction_map))[0]) > 0.8:
                # Route some traffic through the additional node
                current_best_split = self.__adaptive_traffic_splitting(current_best_util_map, s, t, d, self.__additional_node, sp_fraction_map, best_flow_map)
                test_flow_map = self.__update_flow_map(sp_fraction_map, best_flow_map, s, t, d, self.__additional_node, current_best_split)
            else:
                # Standard routing without additional node
                current_best_split = self.__adaptive_traffic_splitting(current_best_util_map, s, t, d, current_best_waypoint, sp_fraction_map, best_flow_map)
                test_flow_map = self.__update_flow_map(sp_fraction_map, best_flow_map, s, t, d, current_best_waypoint, current_best_split)

            test_util_map, test_objective = self.__compute_utilization(test_flow_map)

            if test_objective < current_best_objective:
                current_best_flow_map = test_flow_map
                current_best_util_map = test_util_map
                current_best_objective = test_objective
                current_best_waypoint = self.__additional_node

            if current_best_waypoint is not None:
                waypoints[d_idx] = ([(s, current_best_waypoint), (current_best_waypoint, t)], current_best_split)
            else:
                waypoints[d_idx] = ([(s, t)], 0)

            best_flow_map = current_best_flow_map
            best_util_map = current_best_util_map
            best_objective = current_best_objective

        self.__loads = {(u, v): best_util_map[u][v] for u, v, in self.__links}
        return self.__loads, waypoints, best_objective

    def solve(self):
        self.__start_time = t_start = time.time()
        pt_start = time.process_time()
        loads, waypoints, objective = self.__waypoint_multipath_adaptive()
        pt_duration = time.process_time() - pt_start
        t_duration = time.time() - t_start

        solution = {
            "objective": objective,
            "execution_time": t_duration,
            "process_time": pt_duration,
            "waypoints": waypoints,
            "weights": self.__weights,
            "loads": loads,
        }

        return solution



    def get_name(self):
        """ returns name of algorithm """
        return f"idealwaypoint_optimization"
