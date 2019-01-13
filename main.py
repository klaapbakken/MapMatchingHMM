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
from hmm import alternative_transition_probabilties
from hmm import viterbi
from hmm import backward_recursions
from hmm import forward_recursions

from visualization import plot_results

from tools import state_sequence_to_node_sequence
from tools import get_accuracy_of_estimate
from tools import edges_to_states

from naive_estimation import spatially_closest_states

import random
import numpy as np

import sys

password = sys.argv[1]
P_source = sys.argv[2]

random.seed(3265)

ways = query_ways_postgis_db([10.3930385052,63.4313222082,10.4054088532,63.4347422694], password)

accepted_highways = get_accepted_highways(ways)

required_nodes = get_required_nodes(accepted_highways)

highway_dict = create_highway_dict(accepted_highways)

nodes = query_nodes_postgis_db(required_nodes, password)

node_dict = create_node_dict(nodes)

state_space = create_state_space_representations(accepted_highways, node_dict)

print("Size of state space: {}".format(len(state_space)))

intersections = find_intersections(highway_dict, node_dict)

starting_highway = random.choice(list(highway_dict.keys()))
starting_node = random.choice(highway_dict[starting_highway]['data']['nd'])

speed = 5
frequency = 1/15
measurement_variance = 5
transition_variance = 2
maximum_distance = speed/frequency * 3


if P_source == 'cache':
    # = np.load("P.npy")
    P = np.ones((len(state_space), len(state_space)))/len(state_space)
else:
    P = alternative_transition_probabilties(state_space, speed, frequency, transition_variance, maximum_distance)
    np.save('P', P)

intersections = find_intersections(highway_dict, node_dict)

starting_highway = random.choice(list(highway_dict.keys()))
starting_node = random.choice(highway_dict[starting_highway]['data']['nd'])

simulated_route = simulate_route(highway_dict, starting_node, starting_highway, intersections, 100)
simulated_measurements, measurement_edges = simulate_gps_signals(simulated_route, node_dict, measurement_variance, frequency, [speed]*len(simulated_route))
measurement_states = edges_to_states(measurement_edges, state_space)

l = observation_emissions(simulated_measurements, state_space, measurement_variance)

N = len(state_space)
alpha = forward_recursions(P, l, np.array([1/N]*N))

beta = backward_recursions(P, l, alpha)

estimated_states = viterbi(alpha, beta, P, l, np.array([1/N]*N))

naive_estimate = spatially_closest_states(simulated_measurements, state_space)

print("Accuracy with naive method: {}".format(np.mean(measurement_states == naive_estimate)))
print("Accuracy with hidden markov model: {}".format(np.mean(estimated_states == measurement_states)))