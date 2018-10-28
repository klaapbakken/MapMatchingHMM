from data_interface import query_ways_postgis_db
from data_interface import query_nodes_postgis_db

from data_wrangling import get_accepted_highways
from data_wrangling import create_node_dict
from data_wrangling import create_highway_dict
from data_wrangling import get_required_nodes
from data_wrangling import find_intersections
from data_wrangling import create_state_space_representations

from simulation import simulate_route
from simulation import simulate_gps_signals

from hmm import observation_emissions
from hmm import transition_probabilities
from hmm import viterbi
from hmm import backward_recursions
from hmm import forward_recursions

from visualization import plot_road_network
from visualization import plot_route

from tools import state_sequence_to_node_sequence

import random
import numpy as np

import sys

password = sys.argv[1]

ways = query_ways_postgis_db([9.738344,61.590963,9.777225,61.604356], password)

accepted_highways = get_accepted_highways(ways)

required_nodes = get_required_nodes(accepted_highways)

highway_dict = create_highway_dict(accepted_highways)

nodes = query_nodes_postgis_db(required_nodes, password)

node_dict = create_node_dict(nodes)

intersections = find_intersections(highway_dict, node_dict)

starting_highway = random.choice(list(highway_dict.keys()))
starting_node = random.choice(highway_dict[starting_highway]['data']['nd'])

simulated_route = simulate_route(highway_dict, starting_node, starting_highway, intersections, 100)
simulated_measurements = simulate_gps_signals(simulated_route, node_dict, 5, 1, [10]*len(simulated_route))

state_space = create_state_space_representations(accepted_highways, node_dict)

P = transition_probabilities(state_space, 5, 100)
l = observation_emissions(simulated_measurements, state_space, 5)

N = len(state_space)
alpha = forward_recursions(P, l, np.array([1/N]*N))

beta = backward_recursions(P, l, alpha)

estimated_states = viterbi(alpha, beta, P, l, np.array([1/N]*N))
estimated_route = state_sequence_to_node_sequence(estimated_states, state_space)


print(estimated_route, simulated_route)
plot_road_network(accepted_highways, node_dict, intersections)
plot_route(accepted_highways, node_dict, simulated_route, measurements=simulated_measurements)
plot_route(accepted_highways, node_dict, estimated_route, measurements=simulated_measurements)


