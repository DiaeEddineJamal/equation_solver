# PINN Equation Solver - Data Format

## `examples/` Directory
This directory contains JSON files that define the input data for solving differential equations. Each file should define the following fields:

- **`equation_type`**: The type of the equation ("ODE" or "PDE").
- **`initial_conditions`**: A list of initial conditions for the equation. For ODEs, this is a single value. For PDEs, this could be 2D/3D data.
- **`boundary_conditions`**: Boundary conditions for the equation.
- **`number_of_points`**: Number of discretization points to use when solving the equation.
- **`domain`**: The range or domain for solving the equation (e.g., `[0, 5]` for ODEs, or `[0, 1, 0, 1]` for 2D PDEs).
- **`equation`**: A mathematical description of the equation.

## `results/` Directory
This directory contains the output of the solved equations, stored as JSON files. Each result contains:

- **`solution`**: A list of points with `x`, `y`, and optionally `t`, `u` values for ODEs or PDEs.
- **`equation`**: The equation for which this result was generated.

## Example Usage
1. Place your input data in the `examples/` folder.
2. Use the `solve_equation.py` script to train the PINN model and solve the equation.
3. The results will be saved in the `results/` folder.

## Notes
- Make sure to follow the JSON format strictly for proper training and results.
