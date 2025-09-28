# Bird Murmuration - A Drone Mimicking Approach 

## Simple Simulation (`simulations/boids_simple.cpp`)

Relevant source files: 
+`boids_simple.cpp`
+`pipeline.sh`
+`analysis.py`
+`visualize.py`

We aimed to simulate the expected, multi-agent behavior we seek our drones to have. Our first simulation was a small scale simulation written in native cpp that aims to simulate communication over a network instead of writing each agent as omniscent to its environment. We simulate 8 agents in space where they must poll a ring-buffer to fetch the locations of other agents. This gates their ability to make an informed decision by being able to communicate with other agents in space. We filter distances over a certain radius to simulate communication fall-off (And because boids algorithm likewise benefits from considering neighborhoods of points). 

Python scripts to analyze the csv output of `boids_simple.cpp` as well as visualize the agents in 3D space were made. 

### Quantifiers for Boids Efficacy 

We want to show for layer of complexity added to simulation as well as improvement in design making for individual agents that they are acting effectively within the field. Therefore we have identified a few key identifiers for each of the three core rules that Boids aims to force. 

#### Separation 

(TODO Explain)

#### Alignment 

(TODO Explain)

#### Cohesion 

(TODO Explain)
