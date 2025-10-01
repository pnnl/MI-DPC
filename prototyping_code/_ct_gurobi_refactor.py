# %%
import numpy as np
import torch
import matplotlib.pyplot as plt

from pyomo.environ import (
    ConcreteModel, RangeSet, Var, Binary, Constraint, Objective, minimize, SolverFactory
)

import utils
from utils import *  # Needed to keep access to constants like T_return_min, etc.

# ----------------------------
# Helpers (no existing var names changed)
# ----------------------------
def build_mpc_model(load_forecast, Ts, M, T_return, T_supply):
    """Build and return a Pyomo model for one MPC step."""
    m = ConcreteModel()

    # Index sets
    nsteps = len(load_forecast)
    m.T = RangeSet(0, nsteps)          # state time grid
    m.t = RangeSet(0, nsteps - 1)      # decision time grid
    m.i = RangeSet(1, M)               # chiller index

    # ----------------------------
    # Variables
    # ----------------------------
    # States
    m.T_return = Var(m.T, bounds=(T_return_min, T_return_max + 10))
    m.T_supply = Var(m.T, m.i, bounds=(T_min, T_max))

    # Decisions
    m.flow = Var(m.t, m.i, bounds=(flow_min, flow_max))
    m.integer = Var(m.t, m.i, within=Binary)
    m.T_evap = Var(m.t, m.i, bounds=(T_evap_min, T_evap_max))

    # Stats
    m.T_out = Var(m.t, bounds=(T_min, T_max))
    m.Q_delivered = Var(m.t, bounds=(0.0, Q_delivered_max))
    m.P_chiller = Var(m.t)
    m.P_pump = Var(m.t)

    # Slack for McCormick (product of flow and integer)
    m.flow_active = Var(m.t, m.i, bounds=(0.0, flow_max))

    # ----------------------------
    # Initial conditions
    # ----------------------------
    m.T_return[0].fix(T_return.item())
    for i in m.i:
        m.T_supply[0, i].fix(T_supply[i - 1])

    # ----------------------------
    # Constraints
    # ----------------------------
    # McCormick envelope: flow_active = integer * flow
    m.flow_active_ub1 = Constraint(m.t, m.i, rule=lambda m, t, i: m.flow_active[t, i] <= flow_max * m.integer[t, i])
    m.flow_active_lb1 = Constraint(m.t, m.i, rule=lambda m, t, i: m.flow_active[t, i] >= flow_min * m.integer[t, i])
    m.flow_active_ub2 = Constraint(
        m.t, m.i, rule=lambda m, t, i: m.flow_active[t, i] <= m.flow[t, i] - flow_min * (1 - m.integer[t, i])
    )
    m.flow_active_lb2 = Constraint(
        m.t, m.i, rule=lambda m, t, i: m.flow_active[t, i] >= m.flow[t, i] - flow_max * (1 - m.integer[t, i])
    )

    # Supply dynamics (per chiller)
    def dynamics_supply_fn(m, t, i):
        return m.T_supply[t + 1, i] == m.T_supply[t, i] + Ts / C_i * (
            -m.flow_active[t, i] * c_p * (m.T_supply[t, i] - m.T_evap[t, i])
        )

    m.dynamics_supply_constr = Constraint(m.t, m.i, rule=dynamics_supply_fn)

    # Return-mix dynamics
    def dynamics_return_fn(m, t):
        return m.T_return[t + 1] == m.T_return[t] + Ts / C_r * (
            load_forecast[t] - sum(m.flow_active[t, i] * c_p * (m.T_return[t] - m.T_supply[t, i]) for i in m.i)
        )

    m.dynamics_return_constr = Constraint(m.t, rule=dynamics_return_fn)

    # Delivered cooling
    def Q_delivered_fn(m, t):
        return m.Q_delivered[t] == c_p * sum(
            m.flow_active[t, i] * (m.T_return[t] - m.T_supply[t, i]) for i in m.i
        )

    m.Q_delivered_constr = Constraint(m.t, rule=Q_delivered_fn)

    # Chiller and pump power
    m.P_chiller_contr = Constraint(
        m.t, rule=lambda m, t: m.P_chiller[t] == a + b * (m.Q_delivered[t] / Q_delivered_max) + c * (m.Q_delivered[t] / Q_delivered_max) ** 2
    )
    m.P_pump_constr = Constraint(m.t, rule=lambda m, t: m.P_pump[t] == sum(gamma * m.flow_active[t, i] ** 2 for i in m.i))

    switch_coeff=10.
    # Objective (energy only, as in original)
    m.obj = Objective(
        
        
        # expr=sum(m.P_chiller[t] + m.P_pump[t] for t in m.t),


           expr=(
        sum(m.P_chiller[t] + m.P_pump[t] for t in m.t)
        # penalize turning on/off compared to previous step
        + switch_coeff * sum(
            (m.integer[t, i] - (m.integer[t - 1, i] if t > 0 else 0))**2
            for t in m.t for i in m.i
        )),               
                      
                       sense=minimize)

    return m


def simulate_one_step(Ts, C_i, C_r, c_p, M,
                      T_return_solution, T_supply_solution, T_evap_solution, flow_solution, integer_solution,
                      T_return_solution_first, load_forecast_first):
    """Plant-side single-step update (keeps original algebra)."""
    total_heat = integer_solution[0] * flow_solution[0] * c_p * (T_supply_solution[0] - T_evap_solution[0])
    delta_supply = Ts / C_i * (-total_heat)
    T_supply_next = T_supply_solution[0] + delta_supply

    delivered = c_p * sum(
        integer_solution[0, j] * flow_solution[0, j] * (T_return_solution_first - T_supply_solution[0, j])
        for j in range(M)
    )
    T_return_next = T_return_solution[0] + Ts / C_r * (load_forecast_first - delivered)

    return T_supply_next, T_return_next


# ----------------------------
# Parameters & data
# ----------------------------
M = 2
Ts = 60.0
nsteps = 10

torch.manual_seed(utils.seed)

# Sampling + synthetic load
Ts = 60.0
n_days = 2
t, load_n = utils.generate_datacenter_load(
    number_of_days=2,
    sampling_time=Ts,
    night_baseline=300,
    day_baseline=800,
    ramp_hours=4,
)

load_n = load_n.reshape(1, -1, 1)
s_length = load_n.size(1) - nsteps
s_length = 1000  # override as in original

load = load_n[:, 0 : s_length + nsteps, :].reshape(-1).cpu().numpy()

# ----------------------------
# Solver setup
# ----------------------------
solver = SolverFactory("gurobi")
solver.options["FuncNonlinear"] = 1
solver.options["NonConvex"] = 2

# ----------------------------
# Initial conditions
# ----------------------------
T_return = np.full(1, 8.0)
T_supply = np.full(M, 8.0)

# ----------------------------
# History buffers
# ----------------------------
T_return_hist = []
T_supply_hist = []
T_out_hist, P_chiller_hist, P_pump_hist, Q_delivered_hist = [], [], [], []
T_evap_hist, flow_hist, integer_hist = [], [], []
solver_termination_hist, solver_time_hist = [], []

# ----------------------------
# Receding horizon loop
# ----------------------------
for k in range(s_length):
    print("Step", k)
    load_forecast = load[k : k + nsteps]

    # Build & solve model
    m = build_mpc_model(load_forecast=load_forecast, Ts=Ts, M=M, T_return=T_return, T_supply=T_supply)
    result = solver.solve(m, tee=False)

    print(f"Solution step {k} time: ", getattr(result.solver, "time", None))
    print(result.solver.termination_condition)

    # Read solution
    T_return_solution = np.array([[m.T_return[t].value] for t in m.T])
    T_supply_solution = np.array([[m.T_supply[t, i].value for i in m.i] for t in m.T])
    P_chiller_solution = np.array([[m.P_chiller[t].value] for t in m.t])
    P_pump_solution = np.array([[m.P_pump[t].value] for t in m.t])
    Q_delivered_solution = np.array([[m.Q_delivered[t].value] for t in m.t])
    T_evap_solution = np.array([[m.T_evap[t, i].value for i in m.i] for t in m.t])
    flow_solution = np.array([[m.flow[t, i].value for i in m.i] for t in m.t])
    integer_solution = np.array([[m.integer[t, i].value for i in m.i] for t in m.t])

    # One-step plant simulation (as in original)
    T_supply, T_return = simulate_one_step(
        Ts=Ts,
        C_i=C_i,
        C_r=C_r,
        c_p=c_p,
        M=M,
        T_return_solution=T_return_solution,
        T_supply_solution=T_supply_solution,
        T_evap_solution=T_evap_solution,
        flow_solution=flow_solution,
        integer_solution=integer_solution,
        T_return_solution_first=T_return_solution[0],
        load_forecast_first=load_forecast[0],
    )

    # Append step-0 data from the solved horizon (matches original behavior)
    T_return_hist.append(T_return_solution[0])
    T_supply_hist.append(T_supply_solution[0])
    P_chiller_hist.append(P_chiller_solution[0])
    P_pump_hist.append(P_pump_solution[0])
    Q_delivered_hist.append(Q_delivered_solution[0])
    T_evap_hist.append(T_evap_solution[0])
    flow_hist.append(flow_solution[0])
    integer_hist.append(integer_solution[0])

# ----------------------------
# Stack histories
# ----------------------------
T_return_array = np.vstack(T_return_hist)
T_supply_array = np.vstack(T_supply_hist)
P_chiller_array = np.vstack(P_chiller_hist)
P_pump_array = np.vstack(P_pump_hist)
Q_delivered_array = np.vstack(Q_delivered_hist)
T_evap_array = np.vstack(T_evap_hist)
flow_array = np.vstack(flow_hist)
integer_array = np.vstack(integer_hist)

# ----------------------------
# Metrics & plots
# ----------------------------
coeff = 0.0  # integer penalty
cost = P_chiller_array + P_pump_array + integer_array.sum(-1, keepdims=True) * coeff
control_RMSE = np.sqrt(np.mean((load[:s_length].reshape(-1, 1) - Q_delivered_array) ** 2))

fig, axes = plt.subplots(3, 2, figsize=(12, 10))
axes = axes.flatten()

# (1) Supply & evaporator temperatures
axes[0].plot(T_supply_array[:s_length, 0], label="T_supply1", c="indigo", linestyle="-", alpha=0.7)
if T_supply_array.shape[1] > 1:
    axes[0].plot(T_supply_array[:s_length, 1], label="T_supply2", c="green", linestyle="-", alpha=0.7)
axes[0].plot(np.ones(s_length) * T_max, "k--", label="Evap bounds")
axes[0].plot(np.ones(s_length) * T_min, "k--")
axes[0].plot(T_evap_array[:s_length, 0], color="indigo", linestyle=":", alpha=1, label="T_evap 1")
if T_evap_array.shape[1] > 1:
    axes[0].plot(T_evap_array[:s_length, 1], color="green", linestyle=":", alpha=1, label="T_evap 2")
axes[0].set_xlabel("Timestep")
axes[0].set_ylabel("Temperature [°C]")
axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.4), ncol=3)
axes[0].grid(True)

# (2) Cooling delivered vs. demand
axes[1].plot(load[:s_length], "k:", label="Q_demand")
axes[1].plot(Q_delivered_array, label="Q_delivered", alpha=0.6)
axes[1].set_xlabel("Timestep")
axes[1].set_ylabel("Cooling [kW]")
axes[1].legend()
axes[1].grid(True)

# (3) Return temperature and bounds
axes[2].plot(np.ones(s_length) * T_min, "g--", label="T_out bounds")
axes[2].plot(np.ones(s_length) * T_max, "g--")
axes[2].plot(T_return_array[:s_length], label="T_return", c="r")
axes[2].plot(np.ones(s_length) * T_return_min, "r:", label="T_return bounds")
axes[2].plot(np.ones(s_length) * T_return_max, "r:")
axes[2].set_xlabel("Timestep")
axes[2].set_ylabel("Temperature [°C]")
axes[2].legend()
axes[2].grid(True)

# (4) Power
axes[3].plot(P_chiller_array, label="P_chiller")
axes[3].plot(P_pump_array, label="P_pump")
axes[3].set_xlabel("Timestep")
axes[3].set_ylabel("Chiller [kW]")
axes[3].legend()
axes[3].grid(True)

# (5) Flow rates
axes[4].plot(flow_array[:, 0], label="Mass flow 1")
if flow_array.shape[1] > 1:
    axes[4].plot(flow_array[:, 1], "--", label="Mass flow 2")
axes[4].plot(np.ones(s_length) * flow_min, "k:")
axes[4].plot(np.ones(s_length) * flow_max, "k:", label="bounds")
axes[4].set_xlabel("Timestep")
axes[4].set_ylabel("Mass flowrates [kg/s]")
axes[4].legend()
axes[4].grid(True)

# (6) Integer status
axes[5].plot(integer_array[:, 0], label="Chiller 1 Status")
if integer_array.shape[1] > 1:
    axes[5].plot(integer_array[:, 1], "--", label="Chiller 2 Status")
axes[5].set_ylabel("Integer status [-]")
axes[5].set_xlabel("Timestep")
axes[5].set_yticks([-0.1, 1.1], labels=["OFF", "ON"])
axes[5].legend()
axes[5].grid(True)

axes[1].set_title(
    f"Total cost of operation:  {cost.sum().item():.1f} \n Tracking RMSE: {control_RMSE.item():.1f}"
)
print("Total cost of operation: ", cost.sum().item())

# Keep your original path and filename
plt.savefig(f"/home/bold914/chiller_staging/plots/chiller_control_MIMPC_bigM_{solver.name}.png")
# %%
