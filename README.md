# Transportation Problem Solver

> ## Federal University of Rio Grande do Norte  
> ## Technology Center  
> ### Department of Computer Engineering and Automation  
> #### Author: **João Igor Ramos de Lima :mortar_board:**
>
> Python-based solver for the **Transportation Problem**.
>
> ### Contact
> [igorservo159@gmail.com](mailto:igorservo159@gmail.com)
>
> Copyright (c) 2024, João Igor Ramos de Lima.  
> All rights reserved.   
> SPDX-License-Identifier: BSD-2-Clause

This repository contains a Python-based solver for the **Transportation Problem**, a classic optimization problem in linear programming. The solver supports two initialization methods for finding a basic feasible solution: **Minimum Cost Rule** and **Northwest Rule**. After initialization, the **Simplex Method** is applied to find the optimal solution.

The solution process also generates detailed logs of the iterations in **.txt** or **.xlxs** formats.

---

## Features

- **Initialization Methods**:
  - **Minimum Cost Rule**: Starts with the least-cost allocation.
  - **Northwest Rule**: Allocates from the top-left corner systematically.
  
- **Optimization**:
  - **Simplex Method**: Applied after initialization to find the optimal solution.

- **Output Options**:
  - Save iteration tables as **.txt** or **.xlxs** files for review.

---

## Usage

The solver is designed for flexibility and ease of use. You can run it as a Python script with GUI-based inputs or configure it for command-line usage if desired.

### Inputs

The program expects three main inputs, which are entered interactively via a graphical user interface (GUI):

1. **Cost Matrix**:
   - The transportation costs between each source and destination.
   - Input format: Each row separated by a comma (`,`), and values in a row separated by spaces.  
     For example, a `2x3` cost matrix would look like:  
     ```
     20 15 10, 12 8 16
     ```

2. **Demand Vector**:
   - The demand requirement of each destination.
   - Input format: Space-separated values.  
     For example:  
     ```
     30 40 40
     ```

3. **Supply Vector**:
   - The supply capacity of each source.
   - Input format: Space-separated values.  
     For example:  
     ```
     50 60
     ```

4. **Initialization Method**:
   - Choose between **Minimum Cost Rule** or **Northwest Rule** via a selection prompt.

5. **Output Format**:
   - Choose to save iteration tables in `.txt` or `.xlsx` format.

---

### Outputs

The program provides the following outputs:

1. **Optimal Solution**:
   - The transportation plan with minimized cost.

2. **Cost Summary**:
   - Displays the total transportation cost.

3. **Iteration Tables**:
   - Logs of the intermediate steps during the solution process.
   - Saved in `.txt` or `.xlsx` format, based on user preference.
   - Example file names:
     - `Minimal_cost_costs_table.txt`
     - `Minimal_costs_solutions_table.xlsx`
     - `Northwest_rule.txt`

4. **Console Outputs**:
   - If the GUI is disabled or used in debug mode, results and iterations are also printed to the console.

---
