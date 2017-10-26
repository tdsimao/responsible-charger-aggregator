# CS4010 Project - Electric Vehicle Charging with Constrained MDPs

## Project Setup

1. Install [Gurobi](http://www.gurobi.com/downloads/gurobi-optimizer)
2. Install [gurobipy](http://www.gurobi.com/documentation/6.5/quickstart_mac/the_gurobi_python_interfac.html)
3. Install numpy



## Experiments
The file [experiments.py](./experiments.py) contains a set of experiments.

The section [Running experiments](#running-experiments) shows how to use this file.

### Experiment 1
Increase the probability of having low prices

To run and plot the results of this experiment use the following command: 
`python3 experiments.py --experiments 1 --plots 1`

**Results:**
- [Vehicles well distributed](out/experiment1_fleet1.pdf)
- [Vehicles in single node](out/experiment1_fleet2.pdf)

### Experiment 2
Increase the number of vehicles in a given grid

To run and plot the results of this experiment use the following command: 
`python3 experiments.py --experiments 2 --horizon 8 --vehicles_per_line_capacity 1`


**Results:**
- [Incresing number of vehicles](out/experiment2.pdf)

### Prices
To show `python3 experiments.py --render_prices`

This prints the tables with price transition probability at each time step. 
It also creates dot files with a graph representation 

Use the script [render_dot_files.sh](./render_dot_files.sh) to create the pdf files


**Graphs:**
- [History independent price](out/price_transition_probability_func.dot.pdf)
- [History dependent price](out/history_dependent_price_transition_probability_func.dot.pdf)

### Running experiments
With the command `python3 experiments.py -h` you can have access to the options available.
```
usage: experiments.py [-h] [--experiments [EXP_ID [EXP_ID ...]]]
                      [--plots [EXP_ID [EXP_ID ...]]] [--render_prices]
                      [--horizon HORIZON]
                      [--vehicles_per_line_capacity VEHICLES_PER_LINE_CAPACITY]

Run the experiments.

optional arguments:
  -h, --help            show this help message and exit
  --experiments [EXP_ID [EXP_ID ...]]
                        list of experiments to be executed
  --plots [EXP_ID [EXP_ID ...]]
                        list of experiments to be plotted
  --render_prices       render price transition
  --horizon HORIZON     horizon for the test
  --vehicles_per_line_capacity VEHICLES_PER_LINE_CAPACITY
                        number of vehicles that each line support
```


The option `--experiments`  defines which experiments will be executed.

The option `--plots` defines the plots that will be produce.

This options `--experiments` and `--plots` are separated so you can plot partial results while the experiments are running.

The option `--render_prices` show the prices used in the experiments.