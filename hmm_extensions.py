from hmm import closest_point

import numpy as np


from scipy.stats import beta, bernoulli

def probability_of_signal_given_state(signal_strength, closest_z, base_position, max_range):
    closest_x, closest_y = closest_z
    #Idea: Use measurement position instead of closest_point_in_state
    #Problem: Missing data will cascade through algorithm
    #Solution: Use measurement position if available. Otherwise current solution
    distance_to_state = np.linalg.norm(base_position - np.array([closest_x, closest_y]))
    if np.isnan(signal_strength):
        return 1
    elif signal_strength == 0:
        return bernoulli.pmf(0, max(0, 1 - distance_to_state/max_range))
    else:
        return bernoulli.pmf(1, max(0, 1 - distance_to_state/max_range))*beta.pdf(signal_strength, 2, 5*distance_to_state/max_range)

def probability_of_position_given_state(position, state, variance, position_measurements, position_index):
    if np.isnan(position).any():
        variance = 1e12
        position = locate_last_non_missing_position(position_index, position_measurements)
    closest_x, closest_y = closest_point(state["function"], state["domain"], position)
    return 1/(np.sqrt(2*np.pi)*variance)*np.exp(-1*np.linalg.norm([position[0] - closest_x, position[1] - closest_y])/(2*variance**2))

def locate_last_non_missing_position(position_index, position_measurements):
    for position in reversed(position_measurements[:position_index, :]):
        if not np.isnan(position).any():
            return position
    for position in position_measurements[position_index:, :]:
        if not np.isnan(position).any():
            return position
    return np.apply_along_axis(np.mean, 0, position_measurements[np.invert(np.isnan(position_measurements))].reshape(-1, 2))

def emission_probabilities(position_measurements, variance, signal_measurements, base_locations, base_ranges, state_space):
    ep = np.ones((position_measurements.shape[0], len(state_space)))
    points_closest_to_base_array = points_closest_to_bases(state_space, base_locations)
    for row, position in enumerate(position_measurements):
        for column, state in enumerate(state_space):
            for i, signal_strength in enumerate(signal_measurements[row, :]):
                ep[row, column] = ep[row, column]*probability_of_signal_given_state(signal_strength,\
                     points_closest_to_base_array[column, i], base_locations[i, :], base_ranges[i])
            ep[row, column] = ep[row, column]*probability_of_position_given_state(position,\
                 state, variance, position_measurements, row)
    return ep

def points_closest_to_bases(state_space, base_locations):
    points_closest_to_base_array = np.zeros((len(state_space), base_locations.shape[0], 2))
    for i, state in enumerate(state_space):
        for j in range(base_locations.shape[0]):
            closest_x, closest_y = closest_point(state['function'], state['domain'], base_locations[j, :])
            points_closest_to_base_array[i, j, :] = np.array([closest_x, closest_y])

    return points_closest_to_base_array
                
    #Return, for each observation (vector), find the probability of making 
    #this observation given each state (so observation x state space)
    
    #For each possible state
        #Calculate the probability that one would observe the n signal strengths and
        #the position measurement
        #In the case of missing data, use high-variance with mean measurment, or high-variance with last observation


