# Transportation Problem Solver

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-name.streamlit.app)

[![License: BSD-2-Clause](https://img.shields.io/badge/License-BSD--2--Clause-blue.svg)](https://opensource.org/licenses/BSD-2-Clause)

An interactive web application built with Python and Streamlit to solve the classic **Transportation Problem**. This tool finds the minimum cost for shipping goods from a set of sources to a set of destinations.

The backend is powered by a robust implementation of the **General Simplex algorithm**, ensuring fast and accurate results for standard optimization problems.

---

## Key Features

-   **Interactive Web UI**: A clean, modern, and user-friendly web UI built with Streamlit.
-   **Dual Initial Solution Methods**: Choose between the **Minimum Cost Rule** or the **Northwest Corner Rule** to generate the initial feasible solution.
-   **Proven Simplex Solver**: Utilizes the robust General Simplex algorithm to find the optimal solution.
-   **In-App Step Visualization**: See the initial solution and each Simplex iteration displayed directly on the results page for easy analysis.
-   **Simple Data Input**: Enter data manually via straightforward text fields that are pre-populated with a default example.

---

## Installation & Setup

To run this application on your local machine, please follow these steps.

**Prerequisites:**
* Python 3.8+
* `pip` and `venv`

**1. Clone the Repository**
```bash
# If you haven't already, clone your project repository
git clone [https://github.com/your-username/transportation-problem-solver.git](https://github.com/your-username/transportation-problem-solver.git)
cd transportation-problem-solver
```

**2. Create and Activate a Virtual Environment**
It's highly recommended to use a virtual environment to manage project dependencies.

```bash
# Create the environment
python -m venv venv

# Activate it
# On Windows:
# .\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

**3. Install Dependencies**
Install all required libraries from the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

## How to Run the Application

With the setup complete, running the web application is simple:

1. Open your terminal in the project's root directory.
2. Run the following command:
   ```bash
   streamlit run app.py
   ```
3. The application will automatically open in a new tab in your default web browser.

## Usage

Once the application is running in your browser:

1. Select an **Initial Solution Method** from the options in the sidebar (`Minimum Cost` or `Northwest Corner`).

2. Enter Your Problem Data in the "Enter Data Manually" section. The app loads with a default example that you can solve immediately or edit.

3. Click the "Solve Problem" button.

4. View the Results: The optimal cost and final allocation matrix will be displayed. You can also click on the **"View Simplex Optimization Steps"** expander to see the initial solution and how the allocation changed with each iteration.

