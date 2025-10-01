#%%
from chiller_staging.MIDPC import generate_load
import numpy as np
from pyomo.environ import *
import torch
import utils
from utils import *
import matplotlib.pyplot as plt

M = 2
Ts = 45.0
nsteps = 10
s_length = 300

loads_t_1d = generate_load(T=s_length+nsteps)

load = loads_t_1d[0:s_length+nsteps].cpu().numpy()

torch.manual_seed(202)
t, load_n = utils.generate_datacenter_load(sampling_time=45,
                                          number_of_days=1,
                                          ramp_hours=2,
                                          night_baseline=100,
                                          osc_night_amp=0,
                                          day_baseline=250,
                                          osc_day_amp=0,
                                          noise_scale=0,
                                          ramp_jitter=0,
                                          f_day=3,
                                          f_night=2)
load_n = load_n.reshape(1,-1, 1)
s_length = 1900
# s_length = load_n.size(1)
load=load_n[:,0:s_length+nsteps,:].reshape(-1).cpu().numpy()

#%%

# solver = SolverFactory("ipopt")  # use NLP solver for speed

# solver = SolverFactory('scip', executable='/home/bold914/miniconda3/envs/neuromancer/bin/scip')
# solver = SolverFactory("bonmin") # if you need binaries
# solver = SolverFactory("mindtpy") # if you need binaries

# solver.options['warm_start'] = 'yes' 
# 

# solver = SolverFactory("gurobi_persistent") # if you need binaries
solver = SolverFactory("gurobi") # if you need binaries
solver.options['FuncNonlinear'] = 1 
solver.options['NonConvex'] = 2

# Initial conditions
T_return = np.full(1, 10.0)
T_supply = np.full(M, 8.0)

# history lists
T_return_hist = []; 
# T_return_hist.append(T_return)
T_supply_hist = []; 
# T_supply_hist.append(T_supply.copy())

T_out_hist, P_chiller_hist, P_pump_hist, Q_delivered_hist = [], [], [], []

T_evap_hist, flow_hist, integer_hist = [], [], []

solver_termination_hist, solver_time_hist = [], []

for k in range(s_length):
    print('Step', k)
    load_forecast = load[k:k+nsteps]
    # Iterator ranges
    m = ConcreteModel()
    m.T = RangeSet(0,nsteps)
    m.t = RangeSet(0,nsteps-1)
    m.i = RangeSet(1,M) # Integers range

    ### Variables
    # States
    m.T_return = Var(m.T, bounds=(T_return_min, T_return_max)) # 1 dimensionl
    m.T_supply = Var(m.T,m.i, bounds=(T_min, T_max)) # i dimensional

    # Decisions
    m.flow = Var(m.t,m.i, bounds=(flow_min, flow_max))
    m.integer = Var(m.t,m.i, within=Binary) # original MINLP
    # m.integer = Var(m.t,m.i, bounds=(0,1))    # relaxed to continuous
    m.T_evap = Var(m.t, m.i, bounds=(T_evap_min, T_evap_max))

    # Stats
    m.T_out = Var(m.t, bounds=(T_min, T_max)) # i dimensional
    m.Q_delivered = Var(m.t, bounds=(0., None)) 
    m.P_chiller = Var(m.t)
    m.P_pump = Var(m.t)

    # big M
    # Add auxiliary variable
    m.flow_active = Var(m.t, m.i, bounds=(0., flow_max))

    # Big-M 
    # m.flow_active_upper_1_constr = Constraint(m.t, m.i, rule=lambda m,t,i: m.flow_active[t,i] <= flow_max * m.integer[t,i])
    # m.flow_active_upper_2_constr = Constraint(m.t, m.i, rule=lambda m,t,i: m.flow_active[t,i] <= m.flow[t,i])
    # m.flow_active_lower_1_constr = Constraint(m.t, m.i, rule=lambda m,t,i: m.flow_active[t,i] >= flow_min * m.integer[t,i])
    # m.flow_active_lower_2_constr = Constraint(m.t, m.i, rule=lambda m,t,i: m.flow_active[t,i] >= m.flow[t,i] - flow_max * (1 - m.integer[t,i]))

    #Mccormick envelope
    m.flow_active_ub1 = Constraint(m.t, m.i, rule=lambda m,t,i: m.flow_active[t,i] <= flow_max * m.integer[t,i])
    m.flow_active_lb1 = Constraint(m.t, m.i, rule=lambda m,t,i: m.flow_active[t,i] >= flow_min * m.integer[t,i])
    m.flow_active_ub2 = Constraint(m.t, m.i, rule=lambda m,t,i: m.flow_active[t,i] <= m.flow[t,i] - flow_min * (1 - m.integer[t,i]))
    m.flow_active_lb2 = Constraint(m.t, m.i, rule=lambda m,t,i: m.flow_active[t,i] >= m.flow[t,i] - flow_max * (1 - m.integer[t,i]))

    # Replace all integer * flow with flow_active in constraints:
    def dynamics_supply_fn(m,t,i):
        return m.T_supply[t+1,i] == m.T_supply[t,i] + Ts/C_i * (-m.flow_active[t,i] * c_p * (m.T_supply[t,i] - m.T_evap[t,i]))
    m.dynamics_supply_constr = Constraint(m.t,m.i, rule=dynamics_supply_fn)

    def dynamics_return_fn(m,t):
        return m.T_return[t+1] == m.T_return[t] + Ts/C_r * (load_forecast[t] - sum(m.flow_active[t,i] * c_p * (m.T_return[t] - m.T_supply[t,i]) for i in m.i))
    m.dynamics_return_constr = Constraint(m.t, rule=dynamics_return_fn)

    def Q_delivered_fn(m,t):
        return m.Q_delivered[t] == c_p * sum(m.flow_active[t,i] * (m.T_return[t] - m.T_supply[t,i]) for i in m.i)
    m.Q_delivered_constr = Constraint(m.t, rule=Q_delivered_fn)

    def T_out_fn(m,t):
        numerator = sum(m.flow_active[t,i] * m.T_supply[t,i] for i in m.i)
        denominator = sum(m.flow_active[t,i] for i in m.i) + 1e-10
        return m.T_out[t] == numerator / denominator
    # m.T_out_constr = Constraint(m.t, rule=T_out_fn)



    # Setting initial conditions
    m.T_return[0].fix(T_return.item())
    for i in m.i:
        m.T_supply[0,i].fix(T_supply[i-1])

    ### Constraints

    # Chiller power
    # m.P_chiller_contr = Constraint(m.t, rule=lambda m,t: m.P_chiller[t]==a0+a1*m.Q_delivered[t]+a2*m.Q_delivered[t]**2)
    m.P_chiller_contr = Constraint(m.t, rule=lambda m,t: m.P_chiller[t]==a+b*(m.Q_delivered[t]/Q_delivered_max)+c*(m.Q_delivered[t]/Q_delivered_max)**2)
    # Computing
    m.P_pump_constr = Constraint(m.t, rule=lambda m,t: m.P_pump[t]==sum(gamma*m.flow_active[t,i]**2 for i in m.i))

    m.demand = Constraint(m.t, rule=lambda m,t: m.Q_delivered[t] >= load_forecast[t])

    coeff = 0.0   # integer penalty
    # m.obj = Objective(
    #     expr=sum(m.P_chiller[t] + m.P_pump[t] + (m.Q_delivered[t] - load_forecast[t])**2 + sum(m.integer[t,i]*c for i in m.i) for t in m.t),
    #     sense=minimize
    # )
    m.obj = Objective(
        expr=sum(m.P_chiller[t] + m.P_pump[t] + sum(m.integer[t,i]*coeff for i in m.i) for t in m.t),
        sense=minimize
    )
    result = solver.solve(m, tee=False, 
                        #   warmstart=True, 
                        #   options={'TimeLimit': 100}
                          )
    print(f'Solution step {k} time: ', result.solver.time)
    print(result.solver.termination_condition)

    # Reading solution values
    T_return_solution = np.array([[m.T_return[t].value] for t in m.T])
    T_supply_solution = np.array([[m.T_supply[t, i].value for i in m.i] for t in m.T])

    T_out_solution = np.array([[m.T_out[t].value] for t in m.t])
    # After solution is found:
    T_out_solution = []
    for t in m.t:
        numerator = sum(m.integer[t,i].value * m.flow[t,i].value * m.T_supply[t,i].value for i in m.i)
        denominator = sum(m.integer[t,i].value * m.flow[t,i].value for i in m.i)
        T_out_solution.append(numerator / (1e-10 + denominator))
   
    P_chiller_solution = np.array([[m.P_chiller[t].value] for t in m.t])
    P_pump_solution = np.array([[m.P_pump[t].value] for t in m.t])
    Q_delivered_solution = np.array([[m.Q_delivered[t].value] for t in m.t])

    T_evap_solution = np.array([[m.T_evap[t, i].value for i in m.i] for t in m.t])
    flow_solution = np.array([[m.flow[t, i].value for i in m.i] for t in m.t])
    integer_solution = np.array([[m.integer[t, i].value for i in m.i] for t in m.t])

    # Simulate plant one step
    total_heat = integer_solution[0]*flow_solution[0]*c_p*(T_supply_solution[0] - T_evap_solution[0])
    delta_supply = Ts/C_i * (-total_heat)
    T_supply = T_supply_solution[0] + delta_supply
    delivered = c_p * sum(integer_solution[0,j]*flow_solution[0,j]*(T_return_solution[0] - T_supply_solution[0,j]) for j in range(M))
    T_return = T_return_solution[0] + Ts/C_r * (load_forecast[0] - delivered)

    # Appending
    T_return_hist.append(T_return_solution[0])
    T_supply_hist.append(T_supply_solution[0])
    
    T_out_hist.append(T_out_solution[0])
    P_chiller_hist.append(P_chiller_solution[0])
    P_pump_hist.append(P_pump_solution[0])
    Q_delivered_hist.append(Q_delivered_solution[0])

    T_evap_hist.append(T_evap_solution[0])
    flow_hist.append(flow_solution[0])
    integer_hist.append(integer_solution[0])

    # solver_termination_hist.append(result.solver.termination_condition)
    # solver_time_hist.append(result.solver.time)

    T_return_array = np.vstack(T_return_hist)
    T_supply_array = np.vstack(T_supply_hist)

    T_out_array = np.vstack(T_out_hist)
    P_chiller_array = np.vstack(P_chiller_hist)
    P_pump_array = np.vstack(P_pump_hist)
    Q_delivered_array = np.vstack(Q_delivered_hist)

    T_evap_array = np.vstack(T_evap_hist)
    flow_array = np.vstack(flow_hist)
    integer_array = np.vstack(integer_hist)

    # solver_termination_array = np.vstack(solver_termination_hist)
    # solver_time_array = np.vstack(solver_time_hist)


#%%

cost = P_chiller_array+P_pump_array+integer_array.sum(-1, keepdims=True)*coeff
control_RMSE = np.sqrt(np.mean((load[:s_length].reshape(-1,1) - Q_delivered_array)**2))


fig, axes = plt.subplots(3, 2, figsize=(12, 10))
axes = axes.flatten()
# 1) Return vs Out temperature
axes[0].plot(T_supply_array[:s_length,0], label="T_supply1", c='indigo', linestyle='-', alpha=0.7)
axes[0].plot(T_supply_array[:s_length,1], label="T_supply2", c='green', linestyle='-', alpha=0.7)
axes[0].plot(np.ones(s_length)*T_max, 'k--', label='Evap bounds'); axes[0].plot(np.ones(s_length)*T_min, 'k--')
axes[0].plot(T_evap_array[:s_length,0], color="indigo", linestyle=":", alpha=1, label="T_evap 1")
axes[0].plot(T_evap_array[:s_length,1], color="green", linestyle=":", alpha=1, label="T_evap 2")
axes[0].set_xlabel("Timestep"); axes[0].set_ylabel("Temperature [°C]")
axes[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.4), ncol=3); axes[0].grid(True)

# # # 2) Cooling delivered
axes[1].plot(load[:s_length], 'k:' , label="Q_demand",)
axes[1].plot(Q_delivered_array, label="Q_delivered", alpha=0.6)
axes[1].set_xlabel("Timestep")
axes[1].set_ylabel("Cooling [kW]")
axes[1].legend()
axes[1].grid(True)

# # # 3) Outlet vs retrun temperature
axes[2].plot(T_out_array, label="T_out", c='g')
axes[2].plot(np.ones(s_length)*T_min, 'g--' ,label="T_out bounds"); 
axes[2].plot(np.ones(s_length)*T_max, 'g--')

axes[2].plot(T_return_array[:s_length], label="T_return", c='r')
axes[2].plot(np.ones(s_length)*T_return_min, 'r:' ,label="T_return bounds"); 
axes[2].plot(np.ones(s_length)*T_return_max, 'r:'); 

axes[2].set_xlabel("Timestep")
axes[2].set_ylabel("Temperature [°C]")
axes[2].legend()
axes[2].grid(True)

# # # 4) Chiller energy consumption
axes[3].plot(P_chiller_array, label="P_chiller")
axes[3].plot(P_pump_array, label="P_pump")
axes[3].set_xlabel("Timestep")
axes[3].set_ylabel("Chiller [kW]")
axes[3].legend()
axes[3].grid(True)

# # # 5) Flow rates
axes[4].plot(flow_array[:,0], label="Mass flow 1")
axes[4].plot(flow_array[:,1], '--', label="Mass flow 2")
axes[4].plot(np.ones(s_length)*flow_min, 'k:'); axes[4].plot(np.ones(s_length)*flow_max,'k:', label='bounds')
axes[4].set_xlabel("Timestep")
axes[4].set_ylabel("Mass flowrates [kg/s]")
axes[4].legend()
axes[4].grid(True)

axes[5].plot(integer_array[:,0], label='Chiller 1 Status')
axes[5].plot(integer_array[:,1], '--',label='Chiller 2 Status')
axes[5].set_ylabel("Integer status [-]")
axes[5].set_xlabel("Timestep")
axes[5].set_yticks([-0.1,1.1], labels=['OFF','ON'])
# axes[5].set_ylims(-1.,2)
axes[5].legend()
axes[5].grid(True)

axes[1].set_title(f'Total cost of operation:  {cost.sum().item():.1f} \n Tracking RMSE: {control_RMSE.item():.1f}')
print('Total cost of operation: ', cost.sum().item())
plt.savefig(f'/home/bold914/chiller_staging/plots/chiller_control_MIMPC_bigM_{solver.name}.png')
# %%
