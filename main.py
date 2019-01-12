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

from visualization import plot_results

from tools import state_sequence_to_node_sequence
from tools import get_accuracy_of_estimate
from tools import edges_to_states

import random
import numpy as np

import sys

password = sys.argv[1]
P_source = sys.argv[2]

random.seed(1234)

ways = query_ways_postgis_db([10.391731,63.411357,10.432919,63.42397], password)

accepted_highways = get_accepted_highways(ways)

required_nodes = get_required_nodes(accepted_highways)

highway_dict = create_highway_dict(accepted_highways)

nodes = query_nodes_postgis_db(required_nodes, password)

node_dict = create_node_dict(nodes)

state_space = create_state_space_representations(accepted_highways, node_dict)

intersections = find_intersections(highway_dict, node_dict)

starting_highway = random.choice(list(highway_dict.keys()))
starting_node = random.choice(highway_dict[starting_highway]['data']['nd'])


if P_source == 'cache':
    #P = np.load("P.npy")
    P = np.ones((len(state_space), len(state_space)))/len(state_space)
else:
    P = transition_probabilities(state_space, 5, 100)
    np.save('P', P)

intersections = find_intersections(highway_dict, node_dict)

starting_highway = random.choice(list(highway_dict.keys()))
starting_node = random.choice(highway_dict[starting_highway]['data']['nd'])

simulated_route = simulate_route(highway_dict, starting_node, starting_highway, intersections, 50)
simulated_measurements, measurement_edges = simulate_gps_signals(simulated_route, node_dict, 10, 1/10, [5]*len(simulated_route))
measurement_states = edges_to_states(measurement_edges, state_space)

l = observation_emissions(simulated_measurements, state_space, 10)

N = len(state_space)
alpha = forward_recursions(P, l, np.array([1/N]*N))

beta = backward_recursions(P, l, alpha)

estimated_states = viterbi(alpha, beta, P, l, np.array([1/N]*N))