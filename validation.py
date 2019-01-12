import numpy as np

def estimation_accuracy(measurement_edges, estimated_states):
    N = estimated_states.shape[0]
    estimated_edges = np.array([state_space[int(state)]['edge'] for state in estimated_states]) 
    no = 0
    for i in range(N):
        no += int(np.all(np.array(measurement_edges)[i, :] == estimated_edges[i, :]))
    return no/N