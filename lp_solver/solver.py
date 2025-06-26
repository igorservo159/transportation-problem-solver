import numpy as np

def is_canonical_column(matrix, col_index):
    """Verifica se uma coluna é um vetor base canônico (um 1, resto 0s)."""
    column = matrix[:, col_index]
    is_one = np.isclose(column, 1)
    is_zero = np.isclose(column, 0)
    return np.sum(is_one) == 1 and (np.sum(is_one) + np.sum(is_zero)) == len(column)

def solve(cost_matrix, supply_vector, demand_vector, initial_solution_matrix, callback=None):
    """
    Resolve o problema do transporte usando o método Simplex Geral.
    Baseado na sua implementação original e confiável.
    """
    num_sources, num_destinations = cost_matrix.shape
    num_vars = num_sources * num_destinations
    
    # 1. Preparar Restrições
    original_costs = cost_matrix.copy()
    cost_vector = -cost_matrix.flatten().astype(float)

    constraint_matrix = []
    # Restrições de oferta
    for i in range(num_sources):
        row = [0] * num_vars
        for j in range(num_destinations):
            row[i * num_destinations + j] = 1
        constraint_matrix.append(row)
    # Restrições de demanda
    for j in range(num_destinations):
        row = [0] * num_vars
        for i in range(num_sources):
            row[i * num_destinations + j] = 1
        constraint_matrix.append(row)
    
    constraint_matrix = np.array(constraint_matrix, dtype=float)
    rhs_vector = np.concatenate([supply_vector, demand_vector]).astype(float)
    
    # 2. Converter para Forma Canônica
    solution_vector = initial_solution_matrix.flatten()
    basic_indices = np.where(solution_vector > 1e-6)[0]
    available_rows = list(range(len(rhs_vector)))
    
    for index in basic_indices:
        possible_rows = [r for r in np.where(np.isclose(constraint_matrix[:, index], 1))[0] if r in available_rows]
        if not possible_rows: continue
        
        pivot_row = possible_rows[0]
        available_rows.remove(pivot_row)

        for r in range(len(rhs_vector)):
            if r != pivot_row and not np.isclose(constraint_matrix[r, index], 0):
                factor = constraint_matrix[r, index]
                constraint_matrix[r, :] -= factor * constraint_matrix[pivot_row, :]
                rhs_vector[r] -= factor * rhs_vector[pivot_row]

    for index in basic_indices:
        if not np.isclose(cost_vector[index], 0):
            factor = cost_vector[index]
            row_idx = np.argmax(constraint_matrix[:, index])
            cost_vector -= factor * constraint_matrix[row_idx, :]
    
    # 3. Rodar Algoritmo Simplex
    iteration = 0
    while np.any(cost_vector > 1e-6):
        iteration += 1
        
        entering_col = np.argmax(cost_vector)
        pivot_col = constraint_matrix[:, entering_col]
        ratios = np.divide(rhs_vector, pivot_col, out=np.full_like(rhs_vector, np.inf), where=(pivot_col > 1e-9))
        
        if np.all(np.isinf(ratios)):
            raise RuntimeError("Solução ilimitada.")

        leaving_row = np.argmin(ratios)

        pivot_element = constraint_matrix[leaving_row, entering_col]
        constraint_matrix[leaving_row, :] /= pivot_element
        rhs_vector[leaving_row] /= pivot_element
        
        for r in range(len(constraint_matrix)):
            if r != leaving_row:
                factor = constraint_matrix[r, entering_col]
                constraint_matrix[r, :] -= factor * constraint_matrix[leaving_row, :]
                rhs_vector[r] -= factor * rhs_vector[leaving_row]

        factor = cost_vector[entering_col]
        cost_vector -= factor * constraint_matrix[leaving_row, :]
        
        # --- LÓGICA DO CALLBACK ---
        # A cada iteração, reconstrói a matriz de solução e chama o callback
        if callback:
            current_solution = np.zeros(num_vars)
            for i in range(num_vars):
                if is_canonical_column(constraint_matrix, i):
                    row_idx = np.argmax(constraint_matrix[:, i])
                    current_solution[i] = rhs_vector[row_idx]
            callback(current_solution.reshape(num_sources, num_destinations), iteration)

    # 4. Reconstruir solução final
    final_solution = np.zeros(num_vars)
    for i in range(num_vars):
        if is_canonical_column(constraint_matrix, i):
            row_idx = np.argmax(constraint_matrix[:, i])
            final_solution[i] = rhs_vector[row_idx]
    
    final_solution = final_solution.reshape(num_sources, num_destinations)
    
    # 5. Calcular custo final de forma direta e confiável
    final_cost = np.sum(final_solution * original_costs)
    
    return final_solution, final_cost
