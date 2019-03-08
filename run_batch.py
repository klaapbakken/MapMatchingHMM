from data_interface import query_ways_postgis_db
from data_interface import query_nodes_postgis_db

from data_wrangling import get_accepted_highways
from data_wrangling import create_node_dict
from data_wrangling import create_highway_dict
from data_wrangling import get_required_nodes
from data_wrangling import find_intersections
from data_wrangling import create_state_space_representations
from data_wrangling import remove_unconnected_states
from data_wrangling import remove_unconnected_highways

from simulation import simulate_route
from simulation import simulate_observations

from hmm import viterbi
from hmm import backward_recursions
from hmm import forward_recursions
from hmm import transition_probabilties_by_weighting_route_length

from hmm_extensions import emission_probabilities

from visualization import plot_results

from tools import state_sequence_to_node_sequence
from tools import get_accuracy_of_estimate
from tools import generate_base_locations

from naive_estimation import spatially_closest_states

from batch_processing import simulate_routes
from batch_processing import simulate_measurements
from batch_processing import get_estimates
from batch_processing import results_as_dataframe

from itertools import product

import random
import pickle
import sys
import os

import numpy as np
import pandas as pd

random.seed(3265)
np.random.seed(3265)

password = sys.argv[1]

print("Fetching and processing data..")

bbox = [10.3914799928,63.4271680224,10.4036679506,63.4323125008]
ways = query_ways_postgis_db(bbox, password)

accepted_highways = get_accepted_highways(ways)

required_nodes = get_required_nodes(accepted_highways)

nodes = query_nodes_postgis_db(required_nodes, password)

node_dict = create_node_dict(nodes)

untrimmed_state_space = create_state_space_representations(accepted_highways, node_dict)
state_space = remove_unconnected_states(untrimmed_state_space)

highways_in_state_space = remove_unconnected_highways(accepted_highways, state_space)

print("Size of state space: {}".format(len(state_space)))

highway_dict = create_highway_dict(highways_in_state_space)
intersections = find_intersections(highway_dict, node_dict)

n = 5
route_length = 200

routes = simulate_routes(n, highway_dict, intersections, route_length)

base_max_range = 100
base_locations = [generate_base_locations(bbox, 0), generate_base_locations(bbox, 50)]

polling_frequencies = [1/15]
missing_data = [True, False]
observation_simulation_parameters = list(product(polling_frequencies, base_locations, missing_data))

measurements = list()

gps_variance = 5
speed_limit = 8

base_locations_list = list()
for i, parameter_list in enumerate(observation_simulation_parameters):
    print("Measurement run #{}".format(i + 1))
    
    polling_frequency = parameter_list[0]
    base_locations = parameter_list[1]
    missing_data = parameter_list[2]

    gps_measurements_list, signal_measurements_list, measurement_states_list =\
        simulate_measurements(node_dict, state_space, polling_frequency, missing_data, routes, base_locations, base_max_range, gps_variance, speed_limit)


    base_locations_list.append(base_locations)
    measurements.append([gps_measurements_list, signal_measurements_list, measurement_states_list])


pickling_routes_on = open("routes.p", "wb")
pickle.dump(routes, pickling_routes_on)
pickling_measurements_on = open("measurements.p", "wb")
pickle.dump(measurements, pickling_measurements_on)
pickling_routes_on.close()
pickling_measurements_on.close()


maximum_route_length = speed_limit/min(polling_frequencies)*2
emission_variances = [1/2, 1, 2]
transition_decays = [1/100, 1/250, 1/500]
estimation_parameters = list(product(emission_variances, transition_decays))

pickling_estimation_parameters_on = open("estimation_parameters.", "wb")
pickling_simulation_parameters_on = open("simulation_parameters.p", "wb")

pickle.dump(estimation_parameters, pickling_estimation_parameters_on)
pickle.dump(observation_simulation_parameters, pickling_simulation_parameters_on)

print("Estimating states..")

estimates = list()
naive_estimates = list()

for i, measurement in enumerate(measurements):
    print("Measurement run #{}".format(i + 1))
    gps_measurements_list = measurement[0]
    signal_measurements_list = measurement[1]
    base_locations = base_locations_list[i]
    for j, parameter_list in enumerate(estimation_parameters):
        print("Estimation run #{}".format(j + 1))
        emission_variance = parameter_list[0]
        transition_decay = parameter_list[1]
        print("Estimation..")
        estimated_states_list, naive_estimates_list =\
            get_estimates(state_space, gps_measurements_list, signal_measurements_list, emission_variance,\
                transition_decay, maximum_route_length, base_locations, base_max_range)
        print("Benchmark estimation..")
        estimates.append(estimated_states_list)
        naive_estimates.append(naive_estimates_list)

pickling_estimates_on = open("estimates.p", "wb")
pickle.dump(estimates, pickling_estimates_on)
pickling_naive_estimates_on = open("naive_estimates.p", "wb")
pickle.dump(naive_estimates, pickling_naive_estimates_on)
pickling_estimates_on.close()
pickling_naive_estimates_on.close()

measurements_df = results_as_dataframe(measurements, estimates, naive_estimates, \
                                      observation_simulation_parameters, estimation_parameters, state_space)

measurements_df.to_csv(os.path.join(os.curdir, "measurements.csv"))