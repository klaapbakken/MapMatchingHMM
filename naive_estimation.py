from hmm import closest_point

from hmm_extensions import locate_last_non_missing_position

import numpy as np

def spatially_closest_states(measurement_array, state_space):
    closest_states = list()
    for row in range(measurement_array.shape[0]):
        z = measurement_array[row, :]
        if np.isnan(z).any():
            z = locate_last_non_missing_position(row, measurement_array)
        print(z)
        distance_to_states = dict()
        distance_to_closest_state = np.inf
        id_of_closest_state = None
        for state in state_space:
            closest_x, closest_y = closest_point(state['function'], state['domain'], z)
            distance = np.linalg.norm(z - np.array([closest_x, closest_y]))
            if distance < distance_to_closest_state:
                distance_to_closest_state = distance
                id_of_closest_state = state['id']
        closest_states.append(id_of_closest_state)
    return np.array(closest_states)