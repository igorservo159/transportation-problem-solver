import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import simpledialog

INF = 1000000000000

# Save tables to .txt files
def save_table_to_txt(matrix, filename, iteration, overwrite=False, save_output='n'):
    if save_output == 'y':
        mode = 'w' if overwrite else 'a'
        with open(filename, mode) as file:
            file.write(f"Iteration {iteration}:\n")
            np.savetxt(file, matrix, fmt='%7d')
            file.write("\n")

# Save tables to .xlsx files
def save_table_to_xlsx(matrix, filename, iteration, overwrite=False, save_output='n'):
    if save_output == 'y':
        df = pd.DataFrame(matrix)
        with pd.ExcelWriter(filename, mode='a' if not overwrite else 'w', engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name=f"Iteration_{iteration}", index=False, header=False)

# Check if a column in a matrix is canonical
def is_canonical_column(matrix, col_index):
    column_values = matrix[:, col_index]
    return np.sum(column_values == 1) == 1 and np.sum(column_values == 0) == (column_values.size - 1)

# Transform the problem into canonical form
def convert_to_canonical_form(solution_matrix, cost_vector, constraint_matrix, rhs_vector, save_output='n'):
    total_cost = 0
    iteration = 0
    output_file = "Convert_to_Canonical_Form.txt"

    solution_vector = solution_matrix.flatten()
    basic_variable_indices = np.where(solution_vector > 0)[0]

    table = np.hstack((constraint_matrix, rhs_vector.reshape(-1, 1)))
    save_table_to_txt(table, output_file, iteration, overwrite=True, save_output=save_output)

    available_rows = np.arange(rhs_vector.size)

    for index in basic_variable_indices:
        if not is_canonical_column(constraint_matrix, index):
            iteration += 1
            row_indices_with_one = np.where(constraint_matrix[:, index] == 1)[0]
            valid_rows = np.intersect1d(row_indices_with_one, available_rows)

            selected_row = valid_rows[np.argmin(rhs_vector[valid_rows])]
            non_zero_indices = np.nonzero(constraint_matrix[:, index])[0]

            rows_to_zero = list(non_zero_indices)
            rows_to_zero.remove(selected_row)

            available_rows = available_rows[available_rows != selected_row]

            for row in rows_to_zero:
                factor = constraint_matrix[row, index]
                constraint_matrix[row, :] -= constraint_matrix[selected_row, :] * factor
                rhs_vector[row] -= rhs_vector[selected_row] * factor

            table = np.hstack((constraint_matrix, rhs_vector.reshape(-1, 1)))
            save_table_to_txt(table, output_file, iteration, overwrite=False, save_output=save_output)

    for index in basic_variable_indices:
        if cost_vector[index] != 0:
            factor = cost_vector[index]
            selected_row = np.argmax(constraint_matrix[:, index])
            cost_vector -= constraint_matrix[selected_row, :] * factor
            total_cost -= rhs_vector[selected_row] * factor

    return total_cost

# Prepare constraints for the simplex method
def prepare_constraints(cost_matrix, supply_vector, demand_vector):
    num_sources, num_destinations = cost_matrix.shape
    cost_vector = cost_matrix.flatten()

    constraint_matrix = []
    rhs_vector = []

    # Supply constraints
    for source in range(num_sources):
        constraint_row = [0] * (num_sources * num_destinations)
        for destination in range(num_destinations):
            constraint_row[source * num_destinations + destination] = 1
        constraint_matrix.append(constraint_row)
        rhs_vector.append(supply_vector[source])

    # Demand constraints
    for destination in range(num_destinations):
        constraint_row = [0] * (num_sources * num_destinations)
        for source in range(num_sources):
            constraint_row[source * num_destinations + destination] = 1
        constraint_matrix.append(constraint_row)
        rhs_vector.append(demand_vector[destination])

    return np.array(cost_vector), np.array(constraint_matrix), np.array(rhs_vector)

# Generate an initial solution using the Northwest Corner Rule
def generate_initial_solution_northwest(solution_matrix, supply_vector, demand_vector, save_output='n'):
    output_file = "Northwest_rule.txt"

    supply = supply_vector.copy()
    demand = demand_vector.copy()
    iteration = 0

    solutions_with_demand = np.vstack([solution_matrix, demand])
    solution_table = np.hstack([solutions_with_demand, np.append(supply, [0]).reshape(-1, 1)])

    save_table_to_txt(solution_table, output_file, iteration, overwrite=True, save_output=save_output)

    for source in range(supply.size):
        for destination in range(demand.size):
            previous_solution = solution_matrix.copy()
            if supply[source] >= demand[destination]:
                solution_matrix[source, destination] = demand[destination]
                supply[source] -= demand[destination]
                demand[destination] = 0
            else:
                solution_matrix[source, destination] = supply[source]
                demand[destination] -= supply[source]
                supply[source] = 0

            if not np.array_equal(solution_matrix, previous_solution):
                iteration += 1
                save_table_to_txt(solution_matrix, output_file, iteration, overwrite=False, save_output=save_output)
                previous_solution = solution_matrix.copy()

# Generate an initial solution using the Minimum Cost Method
def generate_initial_solution_minimum_cost(cost_matrix, solutions, supply_vector, demand_vector, save_output='n'):
    costs_output_file = "MC_costs.txt"
    solutions_output_file = "MC_solutions.txt"

    costs = cost_matrix.copy()
    supply = supply_vector.copy()
    demand = demand_vector.copy()

    iteration_solutions = 0
    iteration_costs = 0

    costs_with_demand = np.vstack([costs, demand])
    cost_table = np.hstack([costs_with_demand, np.append(supply, [0]).reshape(-1, 1)])

    solutions_with_demand = np.vstack([solutions, demand])
    solution_table = np.hstack([solutions_with_demand, np.append(supply, [0]).reshape(-1, 1)])

    save_table_to_txt(cost_table, costs_output_file, iteration_costs, overwrite=True, save_output=save_output)
    save_table_to_txt(solution_table, solutions_output_file, iteration_solutions, overwrite=True, save_output=save_output)

    while np.any(demand > 0) and np.any(supply > 0):
        cost_table_copy = cost_table.copy()
        solution_table_copy = solution_table.copy()

        min_index = np.argmin(costs)
        source, destination = np.unravel_index(min_index, costs.shape)

        if supply[source] >= demand[destination]:
            solutions[source, destination] = demand[destination]
            supply[source] -= demand[destination]
            demand[destination] = 0
        else:
            solutions[source, destination] = supply[source]
            demand[destination] -= supply[source]
            supply[source] = 0
        costs[source, destination] = INF

        costs_with_demand = np.vstack([costs, demand])
        cost_table = np.hstack([costs_with_demand, np.append(supply, [0]).reshape(-1, 1)])

        solutions_with_demand = np.vstack([solutions, demand])
        solution_table = np.hstack([solutions_with_demand, np.append(supply, [0]).reshape(-1, 1)])

        if not np.array_equal(solution_table_copy, solution_table):
            iteration_solutions += 1
            save_table_to_txt(solution_table, solutions_output_file, iteration_solutions, overwrite=False, save_output=save_output)

        if not np.array_equal(cost_table_copy, cost_table):
            iteration_costs += 1
            save_table_to_txt(cost_table, costs_output_file, iteration_costs, overwrite=False, save_output=save_output)


# Run the simplex algorithm
def run_simplex(total_cost, cost_vector, constraint_matrix, rhs_vector, save_output='n'):
    iteration = 0
    output_file = "simplex.txt"

    cost_vector_with_total_cost = np.append(cost_vector, total_cost)
    table = np.hstack((constraint_matrix, rhs_vector.reshape(-1, 1)))
    table = np.vstack((cost_vector_with_total_cost, table))

    save_table_to_txt(table, output_file, iteration, overwrite=True, save_output=save_output)
    while np.any(cost_vector > 0):
        iteration += 1

        entering_variable_index = np.argmax(cost_vector)
        valid_rows = np.where(constraint_matrix[:, entering_variable_index] == 1)[0]
        selected_row = valid_rows[np.argmin(rhs_vector[valid_rows])]

        non_zero_indices = np.nonzero(constraint_matrix[:, entering_variable_index])[0]
        rows_to_zero = non_zero_indices.tolist()
        rows_to_zero.remove(selected_row)

        factor = cost_vector[entering_variable_index]
        total_cost -= factor * rhs_vector[selected_row]
        cost_vector -= factor * constraint_matrix[selected_row, :]

        for row in rows_to_zero:
            factor = constraint_matrix[row, entering_variable_index]
            constraint_matrix[row, :] -= constraint_matrix[selected_row, :] * factor
            rhs_vector[row] -= rhs_vector[selected_row] * factor

        cost_vector_with_total_cost = np.append(cost_vector, total_cost)
        table = np.hstack((constraint_matrix, rhs_vector.reshape(-1, 1)))
        table = np.vstack((cost_vector_with_total_cost, table))
        save_table_to_txt(table, output_file, iteration, overwrite=False, save_output=save_output)

    return total_cost

# Main function
def main(cost_matrix, demand_vector, supply_vector, method='northwest', save_output='n'):
    solution_matrix = np.zeros_like(cost_matrix)

    if method == 'minimum_cost':
        generate_initial_solution_minimum_cost(cost_matrix, solution_matrix, supply_vector, demand_vector, save_output)
    elif method == 'northwest':
        generate_initial_solution_northwest(solution_matrix, supply_vector, demand_vector, save_output)

    cost_vector, constraint_matrix, rhs_vector = prepare_constraints(cost_matrix, supply_vector, demand_vector)
    cost_vector = -cost_vector

    total_cost = convert_to_canonical_form(solution_matrix, cost_vector, constraint_matrix, rhs_vector, save_output)
    result = run_simplex(total_cost, cost_vector, constraint_matrix, rhs_vector, save_output)

    print(f"Final result: {result}")
    return result


def launch_gui():
    root = tk.Tk()
    root.withdraw()  # Hide the main Tkinter window

    # Declare variables to hold user choices
    save_output = None
    method = None

    def get_save_output(choice):
        nonlocal save_output
        save_output = choice
        save_output_window.destroy()

    def get_method(choice):
        nonlocal method
        method = choice
        method_window.destroy()

    # Ask for the number of demands and supplies
    try:
        num_demands = simpledialog.askinteger("Demands", "Enter the number of demands:")
        if not num_demands or num_demands <= 0:
            print("Invalid number of demands. Exiting.")
            return

        num_supplies = simpledialog.askinteger("Supplies", "Enter the number of supplies:")
        if not num_supplies or num_supplies <= 0:
            print("Invalid number of supplies. Exiting.")
            return
    except ValueError:
        print("Invalid input for the number of demands or supplies. Exiting.")
        return

    # Input costs
    cost_input = simpledialog.askstring("Costs", 
                                        f"Enter costs matrix ({num_supplies} x {num_demands}) with columns separeted by space and rows by comma:\ne.g., for (2x3): '20 15 10, 12 8 16'")
    if not cost_input:
        print("No input provided for costs. Exiting.")
        return
    try:
        cost_matrix = np.array([list(map(int, row.split())) for row in cost_input.split(',')])
        if cost_matrix.shape != (num_supplies, num_demands):
            print(f"Invalid shape for costs matrix. Expected ({num_supplies}, {num_demands}). Exiting.")
            return
    except ValueError:
        print("Invalid input for costs. Please provide integers in the correct format.")
        return

    # Input demands
    demand_input = simpledialog.askstring("Demands", f"Enter demands vector ({num_demands}) separated by space:\ne.g., '20 40 60'")
    if not demand_input:
        print("No input provided for demands. Exiting.")
        return
    try:
        demand_vector = np.array(list(map(int, demand_input.split())))
        if demand_vector.shape[0] != num_demands:
            print(f"Invalid number of demands. Expected {num_demands}. Exiting.")
            return
    except ValueError:
        print("Invalid input for demands. Please provide integers in the correct format.")
        return

    # Input supplies
    supply_input = simpledialog.askstring("Supplies", f"Enter supplies vector ({num_supplies}) separeted by space:\ne.g., '50 70'")
    if not supply_input:
        print("No input provided for supplies. Exiting.")
        return
    try:
        supply_vector = np.array(list(map(int, supply_input.split())))
        if supply_vector.shape[0] != num_supplies:
            print(f"Invalid number of supplies. Expected {num_supplies}. Exiting.")
            return
    except ValueError:
        print("Invalid input for supplies. Please provide integers in the correct format.")
        return

    # Save output selection
    save_output_window = tk.Toplevel(root)
    save_output_window.title("Save Output")
    save_output_window.minsize(300, 100)  # Set minimum width and height
    tk.Label(save_output_window, text="Save output to files?").pack()
    tk.Button(save_output_window, text="Yes", command=lambda: get_save_output('y')).pack(side=tk.LEFT)
    tk.Button(save_output_window, text="No", command=lambda: get_save_output('n')).pack(side=tk.RIGHT)
    root.wait_window(save_output_window)  # Wait for the user to close the window

    # Method selection
    method_window = tk.Toplevel(root)
    method_window.title("Method Selection")
    method_window.minsize(300, 100)  # Set minimum width and height
    tk.Label(method_window, text="Choose method:").pack()
    tk.Button(method_window, text="Minimum Cost", command=lambda: get_method('minimum_cost')).pack(side=tk.LEFT)
    tk.Button(method_window, text="Northwest", command=lambda: get_method('northwest')).pack(side=tk.RIGHT)
    root.wait_window(method_window)  # Wait for the user to close the window

    # Validate choices
    if save_output is None or method is None:
        print("User did not complete the selection process. Exiting.")
        return

    # Call the main function
    main(cost_matrix, demand_vector, supply_vector, method, save_output)

if __name__ == "__main__":
    launch_gui()
