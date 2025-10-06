#%%
import utils; import init
import numpy as np
import torch
import matplotlib.pyplot as plt

from pyomo.environ import *
from pyomo.dae import ContinuousSet, DerivativeVar, Integral

class MIMPC_policy():
    def __init__(self, nsteps, measure_inference_time = True, M=init.M,
                  solver="gurobi", continous_time_formulation=False,
                  exponent=init.exponent):
        self.nsteps = nsteps
        self.measure_inference_time = measure_inference_time
        self.solver = SolverFactory(solver)
        self.solver.options['FuncNonlinear'] = 1
        self.solver.options['NonConvex'] = 2
        self.continous_time_formulation = continous_time_formulation
        self.M = M
        self.exponent = exponent

    def discrete_time_model(self, T_supply, T_return, load, Ts):
        m = ConcreteModel()
        m.T = RangeSet(0,self.nsteps)
        m.t = RangeSet(0,self.nsteps-1)
        m.i = RangeSet(1,self.M) # Integers range
        # # # Variables
        m.T_return = Var(m.T, bounds=(init.T_return_min, init.T_return_max)) # 1 dimensionl
        m.T_supply = Var(m.T,m.i, bounds=(init.T_min, init.T_max)) # i dimensional
        m.flow = Var(m.t,m.i, bounds=(init.flow_min, init.flow_max))
        m.integer = Var(m.t,m.i, within=Binary) # original MINLP
        m.flow_active = Var(m.t, m.i, bounds=(0., init.flow_max))
        m.T_evap = Var(m.t, m.i, bounds=(init.T_evap_min, init.T_evap_max))

        m.T_out = Var(m.t, bounds=(init.T_min, init.T_max)) # i dimensional
        m.Q_delivered = Var(m.t, bounds=(0., None)) 
        m.P_chiller = Var(m.t)
        m.P_pump = Var(m.t)
        
        # # # McCormick
        m.flow_active_ub1 = Constraint(m.t, m.i,
                            rule=lambda m,t,i: m.flow_active[t,i] <= init.flow_max * m.integer[t,i])
        m.flow_active_lb1 = Constraint(m.t, m.i, 
                            rule=lambda m,t,i: m.flow_active[t,i] >= init.flow_min * m.integer[t,i])
        m.flow_active_ub2 = Constraint(m.t, m.i, 
                            rule=lambda m,t,i: m.flow_active[t,i] <= m.flow[t,i] - init.flow_min * (1 - m.integer[t,i]))
        m.flow_active_lb2 = Constraint(m.t, m.i, 
                            rule=lambda m,t,i: m.flow_active[t,i] >= m.flow[t,i] - init.flow_max * (1 - m.integer[t,i]))

        # # # Constraints
        def dynamics_supply_fn(m,t,i):
            return m.T_supply[t+1,i] == m.T_supply[t,i] + Ts/init.C_i * (-m.flow_active[t,i] * init.c_p * (m.T_supply[t,i] - m.T_evap[t,i]))
        m.dynamics_supply_constr = Constraint(m.t,m.i, rule=dynamics_supply_fn)

        def dynamics_return_fn(m,t):
            return m.T_return[t+1] == m.T_return[t] + Ts/init.C_r * (load[t] - sum(m.flow_active[t,i] * init.c_p * (m.T_return[t] - m.T_supply[t,i]) for i in m.i))
        m.dynamics_return_constr = Constraint(m.t, rule=dynamics_return_fn)

        def Q_delivered_fn(m,t):
            return m.Q_delivered[t] == init.c_p * sum(m.flow_active[t,i] * (m.T_return[t] - m.T_supply[t,i]) for i in m.i)
        m.Q_delivered_constr = Constraint(m.t, rule=Q_delivered_fn)

        m.P_chiller_contr = Constraint(m.t, rule=lambda m,t: m.P_chiller[t]== \
                            init.a+init.b*(m.Q_delivered[t]/init.Q_delivered_max)+init.c*(m.Q_delivered[t]/init.Q_delivered_max)**2)

        m.P_pump_constr = Constraint(m.t, rule=lambda m,t: m.P_pump[t]== \
                                     sum(init.gamma*m.flow_active[t,i]**3 for i in m.i)) # TODO: PW approximation
        
        m.obj = Objective(
           expr=sum(m.P_chiller[t] + m.P_pump[t] + sum(m.integer[t,i]*init.delta_penalty for i in m.i) for t in m.t),
            sense=minimize
        )

        # # # Initial Conditions
        m.T_return[0].fix(T_return.item())
        for i in m.i:
            m.T_supply[0,i].fix(T_supply[i-1])

        return m
    
    def continous_time_model(self):
        pass
    
    def __call__(self, T_supply=None, T_return=None, load=None, Ts=300.):
        T_supply = T_supply.view(-1).numpy()
        T_return = T_return.view(-1).numpy()
        load = load.view(-1).numpy()
        
        if self.continous_time_formulation:
            model = self.continous_time_formulation()
        
        elif not self.continous_time_formulation:
            model = self.discrete_time_model(T_supply=T_supply, T_return=T_return, load=load, Ts=Ts)
        
        result = self.solver.solve(model, tee=False, symbolic_solver_labels=True)
        print(f'Solution time: ', result.solver.time)
        print(result.solver.termination_condition)
        output = {}
        if self.measure_inference_time:
            output['inference_time'] = torch.tensor(result.solver.time).view(1,1,1)


        return output

if __name__=="__main__":

    nsteps = 2
    T_supply = torch.tensor([9.0, 9.0])  # supply temperature
    T_return = torch.tensor([10.0])        # return temperature
    load = torch.tensor([100.0, 120.0])    # heat load
    
    policy = MIMPC_policy(
        nsteps=nsteps,
        continous_time_formulation=False
    )
    policy.solver.options['NonConvex'] = 2

    result = policy(
        T_supply=T_supply,
        T_return=T_return,
        load=load,
        Ts=300.0
    )
    