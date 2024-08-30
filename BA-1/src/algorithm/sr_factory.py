""" Factory for segment routing algorithms"""
import random

from algorithm.generic_sr import GenericSR
from algorithm.segment_routing.demand_first_waypoints import DemandsFirstWaypoints
from algorithm.segment_routing.heur_ospf_weights import HeurOSPFWeights
from algorithm.segment_routing.inverse_capacity import InverseCapacity
from algorithm.segment_routing.segment_ilp import SegmentILP
from algorithm.segment_routing.uniform_weights import UniformWeights
from algorithm.segment_routing.binary_search_adaptive import BinarySearchAdaptiveRouting
from algorithm.segment_routing.waypoint_multipath_adaptive import  WaypointMultipathAdaptive
from algorithm.segment_routing.idealwaypoint_with_failure_handling import WaypointMultipathWithFailureHandling
from algorithm.segment_routing.idealwaypoint_optimization import IdealWaypointOptimization
from algorithm.segment_routing.waypoint_multipath_two_phase_optimization import WaypointMultipathTwoPhaseOptimization
from algorithm.segment_routing.sequential_combination import SequentialCombination

from algorithm.segment_routing.idealwaypoint_optimization2 import IdealWaypointOptimization2


def get_algorithm(algorithm_name: str, nodes: list, links: list, demands: list, weights=None, waypoints=None,
                  seed: float = 42, ilp_method: str = None, time_out: int = None, sf: int = 100) -> GenericSR:
    priorities = []
    for i, d in enumerate(demands):
        random.seed(seed + i)
        priorities.append(random.randint(0, 19) == 0)

    algorithm_name = algorithm_name.lower()
    if algorithm_name == "demand_first_waypoints":
        algorithm = DemandsFirstWaypoints(nodes, links, demands, weights, waypoints)
    elif algorithm_name == "heur_ospf_weights":
        algorithm = HeurOSPFWeights(nodes, links, demands, weights, waypoints, seed=seed, time_out=time_out)
    elif algorithm_name == "inverse_capacity":
        algorithm = InverseCapacity(nodes, links, demands, weights, waypoints, seed=seed)
    elif algorithm_name == "segment_ilp":
        algorithm = SegmentILP(nodes, links, demands, weights, waypoints, waypoint_count=1, method=ilp_method,
                               splitting_factor=sf, time_out=time_out)
    elif algorithm_name == "uniform_weights":
        algorithm = UniformWeights(nodes, links, demands, weights, waypoints, seed=seed)
        
    elif algorithm_name == "binary_search_adaptive":
        algorithm = BinarySearchAdaptiveRouting(nodes, links, demands, weights, waypoints, seed=seed)
    elif algorithm_name == "idealwaypoint_optimization":
        algorithm = IdealWaypointOptimization(nodes, links, demands, weights, waypoints, seed=seed)
    
    elif algorithm_name == "idealwaypoint_optimization2":
        algorithm = IdealWaypointOptimization2(nodes, links, demands, weights, waypoints, seed=seed)
        

    elif algorithm_name == "waypoint_multipath_adaptive":
        algorithm =  WaypointMultipathAdaptive(nodes, links, demands, weights, waypoints, seed=seed)


    elif algorithm_name == "idealwaypoint_with_failure_handling":
        algorithm = WaypointMultipathWithFailureHandling(nodes, links, demands, weights, waypoints, seed=seed)
    
    elif algorithm_name == "waypoint_multipath_two_phase_optimization":
        algorithm = WaypointMultipathTwoPhaseOptimization(nodes, links, demands, weights, waypoints, seed=seed)
        
    elif algorithm_name == "sequential_combination":
        algorithm = SequentialCombination(nodes, links, demands, weights, waypoints, seed=seed, time_out=time_out,
                                          first_algorithm="heur_ospf_weights", second_algorithm="demand_first_waypoints")
    else:
        err_msg = f"algorithm not found: {algorithm_name}"
        raise Exception(err_msg)
    return algorithm
