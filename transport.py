import numpy as np
import pandas as pd

inf = np.inf

def print_txt_table(table, filename, iteration, clean=False, print_output='n'):
    if print_output == 'y':
        mode = 'w' if clean else 'a'
        with open(filename, mode) as f:
            f.write(f"Iteration {iteration}:\n")
            np.savetxt(f, table, fmt='%5d')
            f.write("\n")

def print_xlsx_table(table, filename, iteration, clean=False, print_output='n'):
    if print_output == 'y':
        df = pd.DataFrame(table)
        with pd.ExcelWriter(filename, mode='a' if not clean else 'w', engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name=f"Iteration_{iteration}", index=False, header=False)


def verify_col(matrix, col):
    col_ = matrix[:, col]
    
    single_one = np.sum(col_ == 1) == 1
    remainder_zeros = np.sum(col_ == 0) == (col_.size - 1)
    
    return single_one and remainder_zeros

def turn_into_canonical_shape(solutions, c, A, B, print_output='n'):
    cost = 0
    iteration = 0
    clean = True
    cs_file = "Canonical_shape.txt"

    flatten_solutions = solutions.flatten()
    basic_variables_indexes = np.where(flatten_solutions > 0)[0]

    table = np.hstack((A, B.reshape(-1, 1)))
    print_txt_table(table, cs_file, iteration, clean, print_output)

    available_rows = np.arange(B.size)
    
    for i in basic_variables_indexes:
        if not verify_col(A, i):
            clean = False
            iteration += 1

            eq1_rows_indexes = np.where(A[:, i] == 1)[0]
            available_eq1_rows = np.intersect1d(eq1_rows_indexes, available_rows)

            filtered_B = B[available_eq1_rows]
            j = available_eq1_rows[np.argmin(filtered_B)]

            non_zero_indexes = np.nonzero(A[:, i])[0]
            turning_zero_indexes = non_zero_indexes.tolist() 
            turning_zero_indexes.remove(j)

            available_rows = available_rows[available_rows != j]

            for k in turning_zero_indexes:
                factor = A[k, i]
                A[k, :] -= A[j, :] * factor
                B[k] -= B[j] * factor
            
            table = np.hstack((A, B.reshape(-1, 1)))
            print_txt_table(table, cs_file, iteration, clean, print_output)

    for i in basic_variables_indexes:
        if(c[i] != 0):
            factor = c[i]
            j = np.argmax(A[:, i])
            c -= A[j, :]*factor
            cost -= B[j]*factor

    return cost

def prepare_restrictions(costs, supplies, demands):
    n_source, m_destinations = costs.shape
    c = costs.flatten()

    A = []
    B = []

    #Supply Restrictions
    for i in range(n_source):
        restrictions = [0] * (n_source * m_destinations)
        for j in range(m_destinations):
            restrictions[i * m_destinations + j] = 1
        A.append(restrictions)
        B.append(supplies[i])

    #Demand Restrictions
    for j in range(m_destinations):
        restrictions = [0] * (n_source * m_destinations)
        for i in range(n_source):
            restrictions[i * m_destinations + j] = 1
        A.append(restrictions)
        B.append(demands[j])

    A = np.array(A)
    B = np.array(B)
    return c, A, B

def generate_inicial_solution_northwest_rule(solutions, supplies, demands, print_output='n'):
    nr_file = "Northwest_rule.txt"
    
    supplies_copy = supplies.copy()
    demands_copy = demands.copy()
    iterations = 0
    clean = True  

    solutions_with_demands = np.vstack([solutions, demands_copy])
    solutions_table = np.hstack([solutions_with_demands, np.append(supplies_copy, [0]).reshape(-1, 1)])

    print_txt_table(solutions_table, nr_file, iterations, clean, print_output)
    clean = False

    for row in range(supplies.size):
        for col in range(demands.size):
            previous_solutions = solutions.copy()
            if supplies_copy[row] >= demands_copy[col]:
                solutions[row, col] = demands_copy[col]
                supplies_copy[row] -= demands_copy[col]
                demands_copy[col] = 0
            else:
                solutions[row, col] = supplies_copy[row]
                demands_copy[col] -= supplies_copy[row]
                supplies_copy[row] = 0
            if not np.array_equal(solutions, previous_solutions):
                iterations += 1
                print_txt_table(solutions, nr_file, iterations, clean, print_output)
                clean = False  
                previous_solutions = solutions.copy()

def generate_inicial_solution_minimal_cost(costs, solutions, supplies, demands, print_output='n'):
    mcct_file = "Minimal_costs_costs_table.txt"
    mcst_file = "Minimal_costs_solutions_table.txt"

    costs_copy = costs.copy()
    supplies_copy = supplies.copy()
    demands_copy = demands.copy()

    iteration_solutions = 0
    iteration_costs = 0
    clean = True  

    costs_with_demands = np.vstack([costs_copy, demands_copy])
    costs_table = np.hstack([costs_with_demands, np.append(supplies_copy, [0]).reshape(-1, 1)])

    solutions_with_demands = np.vstack([solutions, demands_copy])
    solutions_table = np.hstack([solutions_with_demands, np.append(supplies_copy, [0]).reshape(-1, 1)])

    print_txt_table(costs_table, mcct_file, iteration_costs, clean, print_output)
    print_txt_table(solutions_table, mcst_file, iteration_solutions, clean, print_output)
    clean = False
 
    while np.any(demands_copy > 0) and np.any(supplies_copy > 0):
        costs_table_copy = costs_table.copy()
        solutions_table_copy = solutions_table.copy()

        i_min = np.argmin(costs_copy)
        pos = np.unravel_index(i_min, costs_copy.shape)

        row, col = pos
        if supplies_copy[row] >= demands_copy[col]:
            solutions[row, col] = demands_copy[col]
            supplies_copy[row] -= demands_copy[col]
            demands_copy[col] = 0
        else:
            solutions[row, col] = supplies_copy[row]
            demands_copy[col] -= supplies_copy[row]
            supplies_copy[row] = 0
        costs_copy[row, col] = inf

        costs_with_demands = np.vstack([costs_copy, demands_copy])
        costs_table = np.hstack([costs_with_demands, np.append(supplies_copy, [0]).reshape(-1, 1)])

        solutions_with_demands = np.vstack([solutions, demands_copy])
        solutions_table = np.hstack([solutions_with_demands, np.append(supplies_copy, [0]).reshape(-1, 1)])
        
        if not np.array_equal(solutions_table_copy, solutions_table):
            iteration_solutions += 1
            print_txt_table(solutions_table, mcst_file, iteration_solutions, clean, print_output)

        if not np.array_equal(costs_table_copy, costs_table):
            iteration_costs += 1
            print_txt_table(costs_table, mcct_file, iteration_costs, clean, print_output)


def execute_simplex(cost, c, A, B, print_output='n'):
    iteration = 0
    simplex_file = "Simplex.xlsx"
    #fatores_file = "Fatores.txt"

    c_with_cost = np.append(c, cost)
    table = np.hstack((A, B.reshape(-1, 1)))
    table = np.vstack((c_with_cost, table))
    clean = True

    print_xlsx_table(table, simplex_file, iteration, clean, print_output)
    while np.any(c > 0):
        iteration += 1
        
        i = np.argmax(c)
        indexes_vector = np.where(A[:, i] == 1)[0]
        values_of_b = B[indexes_vector]
        min_value_index = np.argmin(values_of_b)
        j = indexes_vector[min_value_index]

        non_null_vector = np.where(A[:, i] != 0)[0]
        copy = non_null_vector.tolist() 
        copy.remove(j)

        factor = c[i]
        cost -= factor * B[j]
        c -= factor * A[j, :]

        cost -= c[i] * B[j]
        c -= c[i] * A[j, :]
        for k in copy:
            factor = A[k, i]
            A[k, :] -= A[j, :] * factor
            B[k] -= B[j] * factor

        c_expanded = np.append(c, cost)
        table = np.hstack((A, B.reshape(-1, 1)))
        table = np.vstack((c_expanded, table))
        print_xlsx_table(table, simplex_file, iteration, False, print_output)
    
    return cost

def main(costs, demands, supplies, method='northwest', print_output='n'):
    solutions = np.zeros_like(costs)

    if method == 'minimal_cost':
        generate_inicial_solution_minimal_cost(costs, solutions, supplies, demands, print_output)
    elif method == 'northwest':
        generate_inicial_solution_northwest_rule(solutions, supplies, demands, print_output)
    else:
        raise ValueError("Not recognized method. Use 'minimal_cost' ou 'northwest'.")

    c_, A, B = prepare_restrictions(costs, supplies, demands)
    c = c_ * -1

    cost = turn_into_canonical_shape(solutions, c, A, B, print_output)

    result = execute_simplex(cost, c, A, B, print_output)

    print(f"Final result: {result}")
    return result

def run():
    print("Enter the values for the costs matrix (e.g., '20 15 10; 12 8 16'):")
    costs_input = input().strip()
    costs = np.array([list(map(int, row.split())) for row in costs_input.split(';')])

    print("Enter the values for the demands (e.g., '20 40 60'):")
    demands = np.array(list(map(int, input().strip().split())))

    print("Enter the values for the supplies (e.g., '50 70'):")
    supplies = np.array(list(map(int, input().strip().split())))

    print("Do you want to print the output to files? (y/n):")
    print_output = input().strip().lower()

    print("Choose the method to generate the initial solution (minimal_cost/northwest):")
    method = input().strip().lower()

    main(costs, demands, supplies, method, print_output)

if __name__ == "__main__":
    run()
