import numpy as np

INF = 1e9 # Using a more standard large number instead of an integer

def generate_initial_solution_northwest(supply_vector, demand_vector):
    """Generates an initial solution using the Northwest Corner Rule."""
    supply = supply_vector.copy()
    demand = demand_vector.copy()
    num_sources, num_destinations = len(supply), len(demand)
    solution_matrix = np.zeros((num_sources, num_destinations))

    i, j = 0, 0
    while i < num_sources and j < num_destinations:
        quantity = min(supply[i], demand[j])
        solution_matrix[i, j] = quantity
        supply[i] -= quantity
        demand[j] -= quantity

        if np.isclose(supply[i], 0):
            i += 1
        else:
            j += 1
    return solution_matrix

def generate_initial_solution_minimum_cost(cost_matrix, supply_vector, demand_vector):
    """Generates an initial solution using the Minimum Cost Method."""
    costs = cost_matrix.copy()
    supply = supply_vector.copy()
    demand = demand_vector.copy()
    num_sources, num_destinations = costs.shape
    solution_matrix = np.zeros((num_sources, num_destinations))
    
    while np.any(supply > 0) and np.any(demand > 0):
        # Find the cell with the minimum cost that can still be supplied/demanded
        min_cost = np.inf
        r_min, c_min = -1, -1
        for r in range(num_sources):
            for c in range(num_destinations):
                if supply[r] > 0 and demand[c] > 0 and costs[r, c] < min_cost:
                    min_cost = costs[r, c]
                    r_min, c_min = r, c
        
        if r_min == -1: # No more valid moves
            break

        quantity = min(supply[r_min], demand[c_min])
        solution_matrix[r_min, c_min] = quantity
        supply[r_min] -= quantity
        demand[c_min] -= quantity
        
        # Mark the cell's cost as infinite to avoid re-selecting it
        costs[r_min, c_min] = INF
        
    return solution_matrix
