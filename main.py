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
from hmm import transition_probabilties_by_weighting_route_length
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

print("Fetching and processing data..")

bbox = [10.366042,63.421885,10.408271,63.435746]
ways = query_ways_postgis_db(bbox, password)

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
beta = 1/100
maximum_distance = 200

print("Calculating transition probabilities..")
if P_source == 'cache':
    # = np.load("P.npy")
    P = np.ones((len(state_space), len(state_space)))/len(state_space)
else:
    P = transition_probabilties_by_weighting_route_length(state_space, beta, maximum_distance)
    np.save('P', P)

intersections = find_intersections(highway_dict, node_dict)

starting_highway = random.choice(list(highway_dict.keys()))
starting_node = random.choice(highway_dict[starting_highway]['data']['nd'])

print("Simulating route..")

simulated_route = simulate_route(highway_dict, starting_node, starting_highway, intersections, 500)
simulated_measurements, measurement_edges = simulate_gps_signals(simulated_route, node_dict, measurement_variance, frequency, [speed]*len(simulated_route))
measurement_states = edges_to_states(measurement_edges, state_space)

l = observation_emissions(simulated_measurements, state_space, 1)

print("Running Forward-backward algorithm..")
N = len(state_space)
alpha = forward_recursions(P, l, np.array([1/N]*N))
beta = backward_recursions(P, l, alpha)

print("Running Viterbi..")
estimated_states = viterbi(alpha, beta, P, l, np.array([1/N]*N))

naive_estimate = spatially_closest_states(simulated_measurements, state_space)

print("Accuracy with naive method: {}".format(np.mean(measurement_states == naive_estimate)))
print("Accuracy with hidden markov model: {}".format(np.mean(estimated_states == measurement_states)))