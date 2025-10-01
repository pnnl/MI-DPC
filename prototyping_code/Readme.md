### Computationally efficient optimal control of chiller plant

directory:

- MIDPC.py # training mixed-integer neural policies
- MIDPC_experiment.py # run MIDPC.py with different hyperparameters

- MIMPC.py # implicit mixed-integer problem formulation
- rule_based_control.py # rule_based chiller staging implementation
- utils.py # defines system parameters and aux functions
- chiller_system.py # defines system dynamics in torch
- analysis.py # evaluate and compare the three control strategies

- run.sh

- [folder]results
    - [folder]MIDPC
        - torch policies
        - training data
        - simulation data
    - [folder]MIMPC
        - simulation data
    - [folder]rule_based
        - simulation data
    - analysis.log
- [folder]Plots