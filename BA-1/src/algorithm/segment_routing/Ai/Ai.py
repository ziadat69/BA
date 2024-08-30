import time
import networkit as nk
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import pickle
import os

def save_historical_data(filename, data):
    try:
        with open(filename, 'wb') as file:
            pickle.dump(data, file)
        print(f"Data successfully saved to {filename}")
    except Exception as e:
        print(f"An error occurred while saving data: {e}")

def load_historical_data(filename):
    if os.path.exists(filename):
        try:
            with open(filename, 'rb') as file:
                return pickle.load(file)
        except Exception as e:
            print(f"An error occurred while loading data: {e}")
            return [], {}, {}
    else:
        print(f"File {filename} does not exist. Returning empty data.")
        return [], {}, {}

class WaypointMultipathAi:
    BIG_M = 10 ** 9

    def __init__(self, nodes: list, links: list, demands: list, weights: dict = None, waypoints: dict = None, **kwargs):
        self.__capacities = self.__extract_capacity_dict(links)
        self.__links = list(self.__capacities.keys())
        self.__n = len(nodes)
        self.__capacity_map = None
        self.__demands = demands
        self.__weights = weights if weights else {(u, v): 1. for u, v in self.__links}
        self.__g = None
        self.__apsp = None
        self.__historical_data_file = 'historical_data.pkl'
        self.__historical_data, self.__historical_fraction_maps, self.__historical_splits = load_historical_data(self.__historical_data_file)
        self.__init_graph()
        self.__init_capacity_map()
        self.__model = self.__train_ml_model()

    @staticmethod
    def __extract_capacity_dict(links):
        return {(u, v): c for u, v, c in links}

    def __init_capacity_map(self):
        self.__capacity_map = np.ones((self.__n, self.__n), np.float)
        for u, v in self.__links:
            self.__capacity_map[u][v] = self.__capacities[u, v]

    def __init_graph(self):
        self.__g = nk.Graph(weighted=True, directed=True, n=self.__n)
        for u, v in self.__links:
            self.__g.addEdge(u, v, self.__weights[u, v])
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
                    successors = [v for v in self.__g.iterNeighbors(u) if self.__weights[(u, v)] == distances[u][t] - distances[v][t]]
                    new_fraction = fraction / len(successors) if successors else fraction
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
        util_map = np.zeros((self.__n, self.__n), np.float)
        for u, v in self.__links:
            if (u, v) in self.__capacities:
                util_map[u][v] = flow_map[u][v] / self.__capacities[u, v]
            else:
                print(f"Capacity for link ({u}, {v}) not found.")
                util_map[u][v] = 0
        objective = np.max(util_map)

        return util_map, objective

    def __update_flow_map(self, sp_fraction_map, flow_map, s, t, d, waypoint, multipath_split=1.0):
        new_flow_map = flow_map - sp_fraction_map[s][t] * (d * multipath_split)
        new_flow_map += sp_fraction_map[s][waypoint] * (d * multipath_split)
        new_flow_map += sp_fraction_map[waypoint][t] * (d * multipath_split)
        return new_flow_map

    def __train_ml_model(self):
        X = []
        y = []

        # 
        for s, t, d in self.__historical_data:
            if (s, t) in self.__historical_fraction_maps and (s, t) in self.__historical_splits:
                flow_map = self.__get_flow_map(self.__historical_fraction_maps[s, t])
                util_map, _ = self.__compute_utilization(flow_map)
                if (s, t) in self.__capacities:
                    features = [util_map[s][t], self.__capacities[(s, t)], d]
                    target = self.__historical_splits[(s, t)]
                    X.append(features)
                    y.append(target)
                
        # 
        if len(X) > 0 and len(y) > 0:
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X, y)
            return model
        else:
            return None

    def __predict_split_ratio(self, features):
        if self.__model:
            return self.__model.predict([features])[0]
        else:
            # 
            return 0

    def __adaptive_traffic_splitting(self, current_best_util_map, s, t, d, waypoint, sp_fraction_map, best_flow_map):

        if (s, t) in self.__capacities:
            features = [current_best_util_map[s][t], self.__capacities[(s, t)], d]
            best_split = self.__predict_split_ratio(features)

            return best_split
        else:
            return 0  

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
            for waypoint in range(self.__n):
                if waypoint == s or waypoint == t:
                    continue
                current_best_split = self.__adaptive_traffic_splitting(current_best_util_map, s, t, d, waypoint, sp_fraction_map, best_flow_map)
                test_flow_map = self.__update_flow_map(sp_fraction_map, best_flow_map, s, t, d, waypoint, current_best_split)
                test_util_map, test_objective = self.__compute_utilization(test_flow_map)
                print(f"Waypoint: {waypoint}, Test Objective: {test_objective}, Current Best Objective: {current_best_objective}")

                if test_objective < current_best_objective:
                    current_best_flow_map = test_flow_map
                    current_best_util_map = test_util_map
                    current_best_objective = test_objective
                    current_best_waypoint = waypoint

            if current_best_waypoint is not None:
                waypoints[d_idx] = ([(s, current_best_waypoint), (current_best_waypoint, t)], current_best_split)
            else:
                waypoints[d_idx] = ([(s, t)], 0)
            best_flow_map = current_best_flow_map
            best_util_map = current_best_util_map
            best_objective = current_best_objective

            # download the data
            self.__historical_data.append((s, t, d))
            self.__historical_fraction_maps[s, t] = sp_fraction_map
            self.__historical_splits[s, t] = current_best_split

       

        # save the data
        save_historical_data(self.__historical_data_file, (self.__historical_data, self.__historical_fraction_maps, self.__historical_splits))

        self.__loads = {(u, v): best_util_map[u][v] for u, v, in self.__links}
        return self.__loads, waypoints, best_objective

    def solve(self) -> dict:
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
        return "waypoint_multipath_Ai"
