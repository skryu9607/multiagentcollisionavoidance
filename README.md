# Buffered Voronoi-Based Multi-Agent Deadlock Solving Simulation

This repository contains a Python code for simulating Deadlock-solving dynamics based on Voronoi diagrams. The algorithm takes into account both the agent's preference to reach a goal and the presence of other agents and obstacles in the environment.

## Code Overview
 
The main functions are:

- `chk_cross()`: This function checks if two lines cross.

- `get_crosspt()`: This function gets the intersection point of two lines.

- `voronoi_finite_polygons_2d()`: This function computes the Voronoi diagram of a set of points.

The code creates N agents, each with a current position and a goal position, and places them in the environment. Then it iteratively computes the Voronoi diagram based on the current positions of the agents, determines the preferred velocity of each agent towards their goal, and adjusts this based on the proximity of other agents and obstacles.

## Requirements

The code uses the following Python libraries:
- `numpy`
- `numpy.linalg`
- `matplotlib.pyplot`
- `scipy.spatial`

Make sure these libraries are installed before running the code.

## Usage

To run the simulation, simply execute the script:

```bash
python voronoi_crowd_simulation.py
```

This will run the simulation for 200 iterations, with the initial and goal positions of the agents hard-coded in the script. You can easily change these positions, the number of iterations, or other parameters to suit your needs.

## Visualization

The script also includes code to visualize the positions of the agents and their Voronoi cells at each iteration. The agents' current positions are shown in black, and their goal positions are shown in blue. Each agent's Voronoi cell is shown in a different color.

## Contribute

Let's dive into it!