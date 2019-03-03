import numpy as np

import random

from simulation import simulate_route
from simulation import simulate_observations

from hmm import transition_probabilties_by_weighting_route_length

from hmm_extensions import emission_probabilities

from naive_estimation import spatially_closest_states

def simulate_routes(n, highway_dict, intersections, route_length):
    routes = list()
    for i in range(n):
        
        starting_highway = random.choice(list(highway_dict.keys()))
        starting_node = random.choice(highway_dict[starting_highway]['data']['nd'])
        routes.append(simulate_route(highway_dict, starting_node, starting_highway, intersections, route_length))

    return routes

def simulate_measurements(polling_frequency, missing_data, routes, base_locations, base_max_range, gps_variance, speed_limit):
    print(base_locations.shape)
    gps_measurements_list = list()
    signal_measurements_list = list()
    measurement_states_list = list()
    

    for i, route in enumerate(routes):
        print("Route #{}".format(i + 1))
        gps_measurements, signal_measurements, measurement_states = simulate_observations(route, node_dict, gps_variance, polling_frequency,\
             [speed_limit]*len(route), base_locations, np.array([base_max_range]*base_locations.shape[0]), state_space)
        gps_measurements_list.append(gps_measurements)
        signal_measurements_list.append(signal_measurements)

        if missing_data:
            N = gps_measurements.shape[0]
            missing_indices = np.random.choice(np.arange(N), np.floor(N/5).astype(int), replace=False)
            gps_measurements[missing_indices, :] = np.nan

        measurement_states_list.append(measurement_states)

    return gps_measurements_list, signal_measurements_list, measurement_states_list

def get_estimates(gps_measurements_list, signal_measurements_list, emission_variance, transition_decay, maximum_route_length, base_locations, base_max_range):
    estimated_states_list = list()
    naive_estimates_list = list()
    
    i = 0
    for gps_measurements, signal_measurements in zip(gps_measurements_list, signal_measurements_list):
        print("Route #{}".format(i + 1))

        print("Transition probabilities..")

        tp = transition_probabilties_by_weighting_route_length(state_space,\
                                                           transition_decay, maximum_route_length)

        print("Emission probabilities..")
        ep = emission_probabilities(gps_measurements, emission_variance, signal_measurements,\
                                base_locations, np.array([base_max_range]*base_locations.shape[0]), state_space)

        print("Viterbi..")
        pi = np.ones((len(state_space), ))/len(state_space)
        estimated_states = viterbi(tp, ep, pi)
        estimated_states_list.append(estimated_states)

        naive_estimate = spatially_closest_states(gps_measurements, state_space)
        naive_estimates_list.append(naive_estimate)

        i += 1
    
    return estimated_states_list, naive_estimates_list

def results_as_dataframe(measurements, estimates, naive_estimates,\
                         simulation_parameters, estimation_parameters):
    measurements_df = pd.DataFrame(columns=["route_id", "polling_frequency", "no_of_bases",\
                                            "missing_data", "transition_decay", "emission_variance", \
                                           "hmm_accuracy", "benchmark_accuracy"])
    row = 0
    number_of_routes = len(measurements[0][0])
    for n in range(number_of_routes):
        for i, measurement in enumerate(measurements):

            polling_frequency = simulation_parameters[i][0]
            bases = observation_simulation_parameters[i][1].shape[0]
            missing_data = observation_simulation_parameters[i][2]
            
            
            m = len(estimation_parameters)
            k = 0
            for j in range(m*i, m*(i+1)):
                hmm_acc = np.mean(estimates[j][n] == measurement[2][n])
                benchmark_acc = np.mean(naive_estimates[j][n] == measurement[2][n])
                emission_variance = estimation_parameters[k][0]
                transition_decay = estimation_parameters[k][1]
                measurements_df.loc[row] = [n + 1, polling_frequency, bases, missing_data,\
                                          transition_decay, emission_variance,
                                         hmm_acc, benchmark_acc]
                k += 1
                row += 1
                
    return measurements_df