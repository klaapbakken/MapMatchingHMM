from data_interface import query_osm_api

from data_wrangling import get_accepted_highways
from data_wrangling import create_node_dict
from data_wrangling import create_highway_dict
from data_wrangling import get_required_nodes
from data_wrangling import find_intersections
from data_wrangling import create_state_space_representations

from simulation import simulate_route
from simulation import simulate_observations

from hmm import transition_probabilties_by_weighting_route_length
from hmm import viterbi
from hmm import backward_recursions
from hmm import forward_recursions

from hmm_extensions import emission_probabilities

from visualization import plot_results

from tools import state_sequence_to_node_sequence
from tools import get_accuracy_of_estimate
from tools import generate_base_locations

from naive_estimation import spatially_closest_states

import random
import numpy as np

import sys

print("Fetching and processing data..")

#bbox = [10.366042,63.421885,10.408271,63.435746]
bbox = [10.411165,63.415631,10.432451,63.425788]
nodes, ways = query_osm_api(bbox)

accepted_highways = get_accepted_highways(ways)

required_nodes = get_required_nodes(accepted_highways)

highway_dict = create_highway_dict(accepted_highways)

node_dict = create_node_dict(nodes)

state_space = create_state_space_representations(accepted_highways, node_dict)

print("Size of state space: {}".format(len(state_space)))

intersections = find_intersections(highway_dict, node_dict)

starting_highway = random.choice(list(highway_dict.keys()))
starting_node = random.choice(highway_dict[starting_highway]['data']['nd'])

speed_limit = 5
polling_frequency = 1/15
gps_variance = 1
transition_decay = 1/100
maximum_route_length = 200
no_of_bases = 1
base_max_range = 500
route_length = 50

print("Simulating route..")

base_locations = generate_base_locations(bbox, no_of_bases)

simulated_route = simulate_route(highway_dict, starting_node, starting_highway, intersections, route_length)
gps_measurements, signal_measurements, measurement_states = simulate_observations(simulated_route, node_dict, gps_variance, polling_frequency,\
 [speed_limit]*len(simulated_route), base_locations, np.array([base_max_range]*no_of_bases), state_space)


print("Calculating transition probabilities..")
tp = transition_probabilties_by_weighting_route_length(state_space, transition_decay, maximum_route_length)

print("Calculating emission probabilities..")
ep = emission_probabilities(gps_measurements, gps_variance, signal_measurements, base_locations, np.array([500]*no_of_bases), state_space)

print("Running Forward-backward algorithm..")

N = len(state_space)
alpha = forward_recursions(tp, ep, np.array([1/N]*N))
beta = backward_recursions(tp, ep, alpha)

print("Running Viterbi..")
estimated_states = viterbi(alpha, beta, tp, ep, np.array([1/N]*N))

naive_estimate = spatially_closest_states(gps_measurements, state_space)

print("Accuracy with naive method: {}".format(np.mean(measurement_states == naive_estimate)))
print("Accuracy with hidden markov model: {}".format(np.mean(estimated_states == measurement_states)))

from visualization import MapMatchingVisualization

viz1 = MapMatchingVisualization(accepted_highways, node_dict, state_space, (25, 15))
viz1.plot_road_network('black', 0.5, 0.4)
viz1.plot_node_sequence(simulated_route, 'blue')
viz1.plot_bases(base_locations, 'green', 10)
viz1.plot_base_range(base_locations, base_max_range)

viz2 = MapMatchingVisualization(accepted_highways, node_dict, state_space, (25, 15))
viz2.plot_road_network('black', 0.5, 0.4)
viz2.plot_state_sequence(np.array(measurement_states), 'magenta', 0.5, label=True)
viz2.plot_coordinate_array(gps_measurements, 'black', 10, signal_measurements=signal_measurements)
viz2.shrink_to_fit_state_sequence(np.array(measurement_states), 0)

viz3 = MapMatchingVisualization(accepted_highways, node_dict, state_space, (25, 15))
viz3.plot_road_network('black', 0.5, 0.4)
viz3.plot_estimation_performance(estimated_states.astype(int), np.array(measurement_states).astype(int), 5, 0.3)
viz3.shrink_to_fit_state_sequence(estimated_states.astype(int), 0)

viz4 = MapMatchingVisualization(accepted_highways, node_dict, state_space, (25, 15))
viz4.plot_road_network('black', 0.5, 0.4)
viz4.plot_estimation_performance(naive_estimate.astype(int), np.array(measurement_states).astype(int), 5, 0.3)
viz4.shrink_to_fit_state_sequence(naive_estimate.astype(int), 0)