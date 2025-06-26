import streamlit as st
import numpy as np
import pandas as pd

from lp_solver import initial_solution, solver

# --- Configuração da Página ---
st.set_page_config(
    page_title="Transportation Solver",
    page_icon="🚚",
    layout="wide"
)

st.title("🚚 Transportation Problem Solver")
st.write("An interactive web application to find the minimum cost for transportation problems using the Simplex method.")

# --- Sidebar para Opções ---
with st.sidebar:
    st.header("⚙️ Options")
    solve_method = st.radio(
        "Select the Initial Solution Method:",
        ('Minimum Cost', 'Northwest Corner')
    )

# --- Área Principal para Inserção de Dados ---
st.header("1. Enter Data Manually")
st.info("The application loads with a default example. You can edit the values below or paste your own problem.")

# Problema de exemplo padrão
default_costs = "10 2 20 11\n12 7 9 20\n4 14 16 18"
default_supply = "15\n25\n10"
default_demand = "5 15 15 15"

col1, col2 = st.columns([3, 2])
with col1:
    costs_str = st.text_area("Cost Matrix", value=default_costs, height=150, help="Enter the cost matrix, with one row per source. Separate values with spaces.")
with col2:
    supply_str = st.text_area("Supply (Offers)", value=default_supply, height=100, help="Enter the supply for each source, one per line or separated by spaces.")
    demand_str = st.text_input("Demand", value=default_demand, help="Enter the demand for each destination, separated by spaces.")

# --- Lógica do Solver e Exibição dos Resultados ---
st.markdown("---")
if st.button("Solve Problem", type="primary", use_container_width=True):
    try:
        costs = np.array([list(map(float, row.split())) for row in costs_str.strip().split('\n')])
        supply = np.array(list(map(float, supply_str.strip().split())))
        demand = np.array(list(map(float, demand_str.strip().split())))

        # Validação das dimensões
        if costs.shape[0] != len(supply):
            st.error(f"Dimension mismatch: Cost matrix has {costs.shape[0]} rows, but {len(supply)} supply values were provided.")
            st.stop()
        if costs.shape[1] != len(demand):
            st.error(f"Dimension mismatch: Cost matrix has {costs.shape[1]} columns, but {len(demand)} demand values were provided.")
            st.stop()
        
        if not np.isclose(np.sum(supply), np.sum(demand)):
            st.warning("Warning: Total supply does not equal total demand. The problem is unbalanced.")
        
        with st.spinner(f"Generating initial solution using the {solve_method} method..."):
            if solve_method == 'Minimum Cost':
                initial_sol = initial_solution.generate_initial_solution_minimum_cost(costs, supply, demand)
            else:
                initial_sol = initial_solution.generate_initial_solution_northwest(supply, demand)

        with st.spinner("Optimizing the solution with the Simplex method..."):
            intermediate_steps = []
            def solution_callback(solution_matrix, iteration):
                # Esta função é chamada pelo solver a cada passo
                intermediate_steps.append((solution_matrix.copy(), iteration))

            # Chamada do solver, agora passando o callback
            final_solution, final_cost = solver.solve(costs, supply, demand, initial_sol, callback=solution_callback)
        
        st.success("✅ Optimal Solution Found!")
        
        st.header("📊 Results")
        st.metric(label="Minimum Total Cost", value=f"{final_cost:,.2f}")

        st.subheader("Final Allocation Matrix")
        solution_df = pd.DataFrame(
            final_solution,
            columns=[f"Destination {i+1}" for i in range(demand.shape[0])],
            index=[f"Source {i+1}" for i in range(supply.shape[0])]
        )
        st.dataframe(solution_df)
        
        # --- SEÇÃO PARA EXIBIR AS TABELAS INTERMEDIÁRIAS ---
        if intermediate_steps:
            with st.expander("View Simplex Optimization Steps"):
                # Exibe a solução inicial como passo 0
                initial_cost = np.sum(initial_sol * costs)
                st.write(f"**Initial Solution (Cost: {initial_cost:,.2f})**")
                st.dataframe(pd.DataFrame(initial_sol))
                
                # Exibe os passos da otimização Simplex
                for sol, i in intermediate_steps:
                    cost = np.sum(sol * costs)
                    st.write(f"**Simplex Iteration {i} (Cost: {cost:,.2f})**")
                    st.dataframe(pd.DataFrame(sol))

        st.balloons()

    except Exception as e:
        st.error(f"An error occurred during calculation. Please check your inputs are formatted correctly. Error: {e}")
