#%%
import utils; import init
import numpy as np
import torch
import matplotlib.pyplot as plt

from pyomo.environ import *
from pyomo.dae import ContinuousSet, DerivativeVar, Integral

class MIMPC_policy():
    def __init__(self, nsteps, measure_inference_time = True, M=init.M,
                  solver="gurobi", # solver type options ['scip', 'gurobi']
                  ocp_formulation=False, # options [0 - discret euler, 1 - discrete exact, 2 - continous time]
                  exponent=init.exponent, verbose=True):
        self.nsteps = nsteps
        self.measure_inference_time = measure_inference_time
        self.solver = SolverFactory(solver)
        self.solver.options['FuncNonlinear'] = 1
        self.solver.options['NonConvex'] = 2
        self.ocp_formulation = ocp_formulation
        self.M = M
        self.exponent = exponent
        self.verbose = verbose

    def discrete_time_model(self, T_supply, T_return, load, Ts):
        m = ConcreteModel()
        m.T = RangeSet(0,self.nsteps)
        m.t = RangeSet(0,self.nsteps-1)
        m.i = RangeSet(1,self.M) # Integers range
        # # # Variables
        m.T_return = Var(m.T, bounds=(init.T_return_min, init.T_return_max), within=NonNegativeReals) # 1 dimensionl
        m.T_supply = Var(m.T,m.i, bounds=(init.T_min, init.T_max), within=NonNegativeReals) # i dimensional
        m.flow = Var(m.t,m.i, bounds=(init.flow_min, init.flow_max), within=NonNegativeReals)
        m.integer = Var(m.t,m.i, within=Binary) # original MINLP
        m.flow_active = Var(m.t, m.i, bounds=(0., init.flow_max), within=NonNegativeReals)
        m.T_evap = Var(m.t, m.i, bounds=(init.T_evap_min, init.T_evap_max), within=NonNegativeReals)

        m.T_out = Var(m.t, bounds=(init.T_min, init.T_max), within=NonNegativeReals) # i dimensional
        m.Q_delivered = Var(m.t, m.i, bounds=(0., init.Q_delivered_max), within=NonNegativeReals) 
        m.P_chiller = Var(m.t, m.i, within=NonNegativeReals)
        m.P_pump = Var(m.t, m.i, within=NonNegativeReals)
        m.COP = Var(m.t, m.i, within=NonNegativeReals)

        # # # McCormick
        m.flow_active_ub1 = Constraint(m.t, m.i,
                            rule=lambda m,t,i: m.flow_active[t,i] <= init.flow_max * m.integer[t,i])
        m.flow_active_lb1 = Constraint(m.t, m.i, 
                            rule=lambda m,t,i: m.flow_active[t,i] >= init.flow_min * m.integer[t,i])
        m.flow_active_ub2 = Constraint(m.t, m.i, 
                            rule=lambda m,t,i: m.flow_active[t,i] <= m.flow[t,i] - init.flow_min * (1 - m.integer[t,i]))
        m.flow_active_lb2 = Constraint(m.t, m.i, 
                            rule=lambda m,t,i: m.flow_active[t,i] >= m.flow[t,i] - init.flow_max * (1 - m.integer[t,i]))
       
       # NOT SURE ABOUT THIS :(
        COP_min = 1.
        COP_max = 30.
        P_min = 0.
        P_max = 1000.
        m.Q_delivered = Var(m.t, m.i, bounds=(0, init.Q_delivered_max))
        m.P_chiller = Var(m.t, m.i, bounds=(P_min, P_max))
        m.COP = Var(m.t, m.i, bounds=(P_min, P_max))

        def Q_lb1_rule(m, t, i):
            return m.Q_delivered[t,i] >= COP_min * m.P_chiller[t,i]  + P_min * m.COP[t,i] - COP_min * P_min
        m.Q_lb1 = Constraint(m.t, m.i, rule=Q_lb1_rule)

        def Q_lb2_rule(m, t, i):
            return m.Q_delivered[t,i] >= COP_max * m.P_chiller[t,i]  + P_max * m.COP[t,i] - COP_max * P_max
        m.Q_lb2 = Constraint(m.t, m.i, rule=Q_lb2_rule)

        def Q_ub1_rule(m, t, i):
            return m.Q_delivered[t,i] <= COP_min * m.P_chiller[t,i]  + P_max * m.COP[t,i] - COP_min * P_max
        m.Q_ub1 = Constraint(m.t, m.i, rule=Q_ub1_rule)

        def Q_ub2_rule(m, t, i):
            return m.Q_delivered[t,i] <= COP_max * m.P_chiller[t,i]  + P_min * m.COP[t,i] - COP_max * P_min
        m.Q_ub2 = Constraint(m.t, m.i, rule=Q_ub2_rule)

       
       
        # New auxiliary variable for squares of flow
        m.flow_sq = Var(m.t, m.i, bounds=(0, init.flow_max**2), within=NonNegativeReals)
        m.flow_sq_constr = Constraint(m.t, m.i, rule=lambda m,t,i: m.flow_sq[t,i]== \
                                     m.flow[t, i]**2)
        
        # Slack variable for under-delivered cooling
        m.slack_under = Var(m.t, within=NonNegativeReals)

        # # # Constraints
        def dynamics_supply_fn(m,t,i):
            return m.T_supply[t+1,i] == m.T_supply[t,i] + Ts/init.C_i * init.eta_supply * (-m.flow_active[t,i] * init.c_p * (m.T_supply[t,i] - m.T_evap[t,i]))
        m.dynamics_supply_constr = Constraint(m.t,m.i, rule=dynamics_supply_fn)

        def dynamics_return_fn(m,t):
            return m.T_return[t+1] == m.T_return[t] + Ts/init.C_r * init.eta_return * (load[t] - sum(m.flow_active[t,i] * init.c_p * (m.T_return[t] - m.T_supply[t,i]) for i in m.i))
        m.dynamics_return_constr = Constraint(m.t, rule=dynamics_return_fn)

        def Q_delivered_fn(m,t,i):
            return m.Q_delivered[t,i] == init.c_p * init.eta_return * m.flow_active[t,i] * (m.T_return[t] - m.T_supply[t,i])
        m.Q_delivered_constr = Constraint(m.t, m.i, rule=Q_delivered_fn)

        # def PLR_fn(m,t,i):
        #     return m.PLR[t,i] == m.Q_delivered[t,i]/init.Q_delivered_max

        # m.PLR_constr = Constraint(m.t, m.i, rule=PLR_fn)
        def COP_fn(m,t,i):
            return m.COP[t,i] == init.a*m.integer[t,i]+init.b*(m.Q_delivered[t,i]/(init.Q_delivered_max)) + init.c*(m.Q_delivered[t,i]/init.Q_delivered_max)**2
        m.COP_constr = Constraint(m.t, m.i, rule=COP_fn)

        # def P_chiller_fn(m,t,i):
        #     return m.P_chiller[t,i] == m.Q_delivered[t,i]/m.COP[t,i]
        # m.P_chiller_constr = Constraint(m.t, m.i, rule=P_chiller_fn)


        # m.P_chiller_contr = Constraint(m.t, rule=lambda m,t: m.P_chiller[t]== \
                            # init.a+init.b*(m.Q_delivered[t]/(init.Q_delivered_max))+
                            # init.c*(m.Q_delivered[t]/init.Q_delivered_max)**2)

        # m.one_always_active = Constraint(m.t, rule=lambda m,t: sum(m.integer[t,i] for i in m.i) >= 1.)

        if self.exponent == 3:
            m.P_pump_constr = Constraint(m.t, m.i, rule=lambda m,t,i: m.P_pump[t,i]== \
                                     init.gamma * m.flow_active[t, i] * m.flow_sq[t, i]) # TODO: PW approximation
        elif self.exponent == 2:
            m.P_pump_constr = Constraint(m.t, m.i, rule=lambda m,t,i: m.P_pump[t,i]== \
                                     init.gamma * m.flow_active[t, i] **2 ) # TODO: PW approximation
        
        # m.Q_balance = Constraint(m.t, rule=lambda m, t: m.slack_under[t] >= load[t] - sum(m.Q_delivered[t,i] for i in m.i))


        # # # Control objective
        m.obj = Objective(
           expr=sum(m.P_chiller[t,i] + m.P_pump[t,i]+ m.integer[t,i]*100000. for t in m.t for i in m.i) + \
        #    sum(m.integer[t,i]*init.delta_penalty for i in m.i) +\
                # init.delta_penalty * sum((m.integer[t+1,i] - m.integer[t,i])**2 for t in m.t if t < self.nsteps-1 for i in m.i) +\
                    sum(load[t] - sum(m.Q_delivered[t,i] for i in m.i) for t in m.t)**2
                    # sum(m.slack_under[t] for t in m.t)
                    ,
            sense=minimize
        )

        # # # Initial Conditions
        m.T_return[0].fix(T_return.item())
        for i in m.i:
            m.T_supply[0,i].fix(T_supply[i-1])
        # # # One chiller always on
        for t in m.t:
            m.integer[t,1].fix(1.)

        # for t in m.t:
        #     for i in m.i:
        #        m.T_evap[t,i].fix(10.)
        return m
    
    def continous_time_model(self):
        pass # To be done
    
    def discrete_time_exact_model(self, T_supply, T_return, load, Ts):
        m = ConcreteModel()
        m.T = RangeSet(0,self.nsteps)
        m.t = RangeSet(0,self.nsteps-1)
        m.i = RangeSet(1,self.M) # Integers range
        # # # Variables
        m.T_return = Var(m.T, bounds=(init.T_return_min, init.T_return_max), within=NonNegativeReals) # 1 dimensionl
        m.T_supply = Var(m.T,m.i, bounds=(init.T_min, init.T_max), within=NonNegativeReals) # i dimensional
        m.flow = Var(m.t,m.i, bounds=(init.flow_min, init.flow_max), within=NonNegativeReals)
        m.integer = Var(m.t,m.i, within=Binary) # original MINLP
        m.flow_active = Var(m.t, m.i, bounds=(0., init.flow_max), within=NonNegativeReals)
        m.T_evap = Var(m.t, m.i, bounds=(init.T_evap_min, init.T_evap_max), within=NonNegativeReals)

        m.T_out = Var(m.t, bounds=(init.T_min, init.T_max), within=NonNegativeReals) # i dimensional
        m.Q_delivered = Var(m.t, bounds=(0., init.Q_delivered_max), within=NonNegativeReals) 
        m.P_chiller = Var(m.t, within=NonNegativeReals)
        m.P_pump = Var(m.t, within=NonNegativeReals)
        
        # # # McCormick
        m.flow_active_ub1 = Constraint(m.t, m.i,
                            rule=lambda m,t,i: m.flow_active[t,i] <= init.flow_max * m.integer[t,i])
        m.flow_active_lb1 = Constraint(m.t, m.i, 
                            rule=lambda m,t,i: m.flow_active[t,i] >= init.flow_min * m.integer[t,i])
        m.flow_active_ub2 = Constraint(m.t, m.i, 
                            rule=lambda m,t,i: m.flow_active[t,i] <= m.flow[t,i] - init.flow_min * (1 - m.integer[t,i]))
        m.flow_active_lb2 = Constraint(m.t, m.i, 
                            rule=lambda m,t,i: m.flow_active[t,i] >= m.flow[t,i] - init.flow_max * (1 - m.integer[t,i]))
        # New auxiliary variable for squares of flow
        m.flow_sq = Var(m.t, m.i, bounds=(0, init.flow_max**2), within=NonNegativeReals)

        # # # Constraints
        def dynamics_supply_fn(m,t,i):
            return m.T_supply[t+1,i] == \
            exp(-init.eta_supply* m.flow_active[t,i] * init.c_p * Ts/init.C_i) * m.T_supply[t,i] + \
            (1 - exp(-init.eta_supply* m.flow_active[t,i] * init.c_p * Ts/init.C_i)) * m.T_evap[t,i]
        m.dynamics_supply_constr = Constraint(m.t,m.i, rule=dynamics_supply_fn)

        def dynamics_return_fn(m,t):
            return m.T_return[t+1] == \
            exp(-sum(init.eta_return * m.flow_active[t,i] * init.c_p * Ts/init.C_r for i in m.i)) * m.T_return[t] + \
            (1 - exp(-sum(init.eta_return * m.flow_active[t,i] * init.c_p * Ts/init.C_r for i in m.i))) / \
            exp(sum(init.eta_return * m.flow_active[t,i] * init.c_p for i in m.i)) * \
            (load[t] + sum(init.eta_return * m.flow_active[t,i] * init.c_p * m.T_supply[t,i] for i in m.i)) 
        
        # m.T_return[t] + Ts/init.C_r * init.eta_return * (load[t] - sum(m.flow_active[t,i] * init.c_p * (m.T_return[t] - m.T_supply[t,i]) for i in m.i))
        m.dynamics_return_constr = Constraint(m.t, rule=dynamics_return_fn)

        def Q_delivered_fn(m,t):
            return m.Q_delivered[t] == init.c_p * init.eta_return * sum(m.flow_active[t,i] * (m.T_return[t] - m.T_supply[t,i]) for i in m.i)
        m.Q_delivered_constr = Constraint(m.t, rule=Q_delivered_fn)

        m.P_chiller_contr = Constraint(m.t, rule=lambda m,t: m.P_chiller[t]== \
                            init.a+init.b*(m.Q_delivered[t]/init.Q_delivered_max)+init.c*(m.Q_delivered[t]/init.Q_delivered_max)**2)

        # m.one_always_active = Constraint(m.t, rule=lambda m,t: sum(m.integer[t,i] for i in m.i) >= 1.)

        if self.exponent == 3:
            m.P_pump_constr = Constraint(m.t, rule=lambda m,t: m.P_pump[t]== \
                                     sum(init.gamma * m.flow_active[t, i] * m.flow_sq[t, i] for i in m.i)) # TODO: PW approximation
        elif self.exponent == 2:
            m.P_pump_constr = Constraint(m.t, rule=lambda m,t: m.P_pump[t]== \
                                     sum(init.gamma * m.flow_active[t, i] **2 for i in m.i)) # TODO: PW approximation
        
        # # # Control objective
        m.obj = Objective(
           expr=sum(m.P_chiller[t] + m.P_pump[t] + sum(m.integer[t,i]*init.delta_penalty for i in m.i) for t in m.t),
            sense=minimize
        )

        # # # Initial Conditions
        m.T_return[0].fix(T_return.item())
        for i in m.i:
            m.T_supply[0,i].fix(T_supply[i-1])
        # # # One chiller always on
        for t in m.t:
            m.integer[t,1].fix(1.)
        return m

    def get_variable_values(self, model, return_all=False):
        # # # States
        T_supply = torch.tensor([[model.T_supply[t, i].value for i in model.i] for t in model.T])
        T_return = torch.tensor([[model.T_return[t].value] for t in model.T])
        # # # Decisions
        flow = torch.tensor([[model.flow[t, i].value for i in model.i] for t in model.t ])
        integer = torch.tensor([[model.integer[t, i].value for i in model.i] for t in model.t ])
        T_evap = torch.tensor([[model.T_evap[t, i].value for i in model.i] for t in model.t])
        # # # Scores
        Q_delivered = torch.tensor([[model.Q_delivered[t, i].value for i in model.i] for t in model.t ])
        P_chiller = torch.tensor([[model.P_chiller[t,i].value for i in model.i] for t in model.t])
        P_pump = torch.tensor([[model.P_pump[t,i].value for i in model.i] for t in model.t])

        if return_all:
            return T_supply, T_return, flow, integer, T_evap, Q_delivered, P_chiller, P_pump
        else:
            return flow, integer, T_evap # Return decisions

    def __call__(self, T_supply=None, T_return=None, load=None, Ts=300.):
        T_supply = T_supply.view(-1).numpy()
        T_return = T_return.view(-1).numpy()
        load = load.view(-1).numpy()
        
        if self.ocp_formulation == 0:
            model = self.discrete_time_model(T_supply=T_supply, T_return=T_return, load=load, Ts=Ts)
        elif self.ocp_formulation == 1:
            model = self.discrete_time_exact_model(T_supply=T_supply, T_return=T_return, load=load, Ts=Ts)
        elif self.ocp_formulation == 2:
            model = self.ocp_formulation()
        
        # self.solver.set_instance(model)
        result = self.solver.solve(model, tee=False, symbolic_solver_labels=True)
    #     result = SolverFactory('mindtpy').solve(
    #     model,
    #     strategy='OA',            # or 'ECP'
    #     mip_solver='gurobi',
    #     nlp_solver='ipopt',
    #     tee=True
    # )
        # print(result.solver.status)
        # print(result.solver.termination_condition)

        if self.verbose:
            print(f'Solution time: ', result.solver.wall_time, 'Solution: ', result.solver.termination_condition)
        
        flow, integer, T_evap = self.get_variable_values(model)
        output = {}
        # if self.measure_inference_time:
            # output['inference_time'] = torch.tensor(result.solver.wall_time).view(1,1,1)
        output['termination_condition'] = result.solver.termination_condition
        output['integer'] = integer[0].view(1,1,-1)
        output['flow'] = flow[0].view(1,1,-1)
        output['T_evap'] = T_evap[0].view(1,1,-1)
        return output

if __name__=="__main__":

    nsteps = 6
    T_supply = torch.tensor([9.0, 9.0])  # supply temperature
    T_return = torch.tensor([15.0])        # return temperature
    load = torch.ones(1,nsteps,1)*400    # heat load
    
    policy = MIMPC_policy(
        nsteps=nsteps,
        ocp_formulation=0,
        verbose=False
    )

    result = policy(
        T_supply=T_supply,
        T_return=T_return,
        load=load,
        Ts=300.0
    )
    