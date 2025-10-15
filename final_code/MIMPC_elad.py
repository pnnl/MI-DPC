import utils; import init
import numpy as np
import torch
import matplotlib.pyplot as plt

from pyomo.environ import *

class MIMPC_policy():
    def __init__(self, nsteps, measure_inference_time = True, M=init.M,
                  solver="gurobi", # solver type options ['scip', 'gurobi']
                  ocp_formulation=False, # options [0 - discret euler, 1 - discrete exact, 2 - continous time]
                  exponent=init.exponent, verbose=True):
        self.nsteps = nsteps
        self.measure_inference_time = measure_inference_time
        self.solver = SolverFactory(solver)
        # self.solver.options['FuncNonlinear'] = 1
        # self.solver.options['NonConvex'] = 2
        self.ocp_formulation = ocp_formulation
        self.M = M
        self.exponent = exponent
        self.verbose = verbose


    def discrete_model(self, T_supply, T_return, load, Ts):

        # __      __        _       _     _           
        # \ \    / /       (_)     | |   | |          
        #  \ \  / /_ _ _ __ _  __ _| |__ | | ___  ___ 
        #   \ \/ / _` | '__| |/ _` | '_ \| |/ _ \/ __|
        #    \  / (_| | |  | | (_| | |_) | |  __/\__ \
        #     \/ \__,_|_|  |_|\__,_|_.__/|_|\___||___/
                                                
        m = ConcreteModel()
        m.T = RangeSet(0,self.nsteps)
        m.t = RangeSet(0,self.nsteps-1)
        m.tm1 = RangeSet(0,self.nsteps-2)
        m.i = RangeSet(1,self.M) # Integers range
        # # # Variables
        m.T_return = Var(m.T, bounds=(init.T_return_min, init.T_return_max), within=NonNegativeReals) # 1 dimensionl
        m.T_supply = Var(m.T,m.i, bounds=(init.T_min, init.T_max), within=NonNegativeReals) # i dimensional
        m.flow = Var(m.t,m.i, bounds=(init.flow_min, init.flow_max), within=NonNegativeReals)
        m.integer = Var(m.t,m.i, within=Binary) # original MINLP
        m.T_evap = Var(m.t, m.i, bounds=(init.T_evap_min, init.T_evap_max), within=NonNegativeReals)

        m.Q_delivered = Var(m.t, m.i, bounds=(0., init.Q_delivered_max), within=NonNegativeReals) 
        m.P_chiller = Var(m.t, m.i, within=NonNegativeReals)
        m.P_pump = Var(m.t, m.i, within=NonNegativeReals)
        m.COP = Var(m.t, m.i, within=NonNegativeReals)


        #   _____ _            _    
        #  / ____| |          | |   
        # | (___ | | __ _  ___| | __
        #  \___ \| |/ _` |/ __| |/ /
        #  ____) | | (_| | (__|   < 
        # |_____/|_|\__,_|\___|_|\_\
                               
                                   
        # Slack variable for under-delivered cooling and flow auxilliary variables for quadratic bullshit
        m.Q_slack = Var(m.t, within=NonNegativeReals)
        m.active_flow = Var(m.t, m.i, within=NonNegativeReals)
        m.flow_sq = Var(m.t, m.i, within=NonNegativeReals)
        m.integer_smooth = Var(m.t, m.i, bounds=(0,1) , within=NonNegativeReals)

        # # # # Constraints
        def active_flow_fn(m,t,i):
            return m.active_flow[t,i] == m.integer[t,i] * m.flow[t,i]
        m.active_flow_constr = Constraint(m.t,m.i, rule = active_flow_fn)
        
        def flow_sq_fn(m,t,i):
            return m.flow_sq[t,i] == m.flow[t,i]**2
        m.flow_sq_constr = Constraint(m.t,m.i, rule = flow_sq_fn)



        #  _____                              _          
        # |  __ \                            (_)         
        # | |  | |_   _ _ __   __ _ _ __ ___  _  ___ ___ 
        # | |  | | | | | '_ \ / _` | '_ ` _ \| |/ __/ __|
        # | |__| | |_| | | | | (_| | | | | | | | (__\__ \
        # |_____/ \__, |_| |_|\__,_|_| |_| |_|_|\___|___/
        #          __/ |                                 
        #         |___/                                  

        # # # FORWARD EULER DISCRETIZATION
        def euler_supply_dyn(m,t,i):
            return m.T_supply[t+1,i] == m.T_supply[t,i] - Ts/init.C_i * init.eta_supply * (
                m.active_flow[t,i] * init.c_p * (m.T_supply[t,i] - m.T_evap[t,i])
            )        

        def euler_return_dyn(m,t):
            return m.T_return[t+1] == m.T_return[t] + Ts/init.C_r * init.eta_return * (
                load[t] - quicksum(m.active_flow[t,i] * init.c_p * (m.T_return[t] - m.T_supply[t,i]) for i in m.i)
            )

        # # # EXACT DISCRETIZATION
        def exact_supply_dyn(m,t,i):
            return m.T_supply[t+1,i] == \
            exp(-init.eta_supply* m.active_flow[t,i] * init.c_p * Ts/init.C_i) * m.T_supply[t,i] + \
            (1 - exp(-init.eta_supply* m.active_flow[t,i] * init.c_p * Ts/init.C_i)) * m.T_evap[t,i]

        # I haven't checked this yet, but I suspect that the division by the decision variables will 
        # not be possible to optimize with gurobi, also chatgpt says the variable in the denom is wrong. Will check.
        def exact_return_dyn(m,t):
            return m.T_return[t+1] == \
            exp(-sum(init.eta_return * m.active_flow[t,i] * init.c_p * Ts/init.C_r for i in m.i)) * m.T_return[t] + \
            (1 - exp(-sum(init.eta_return * m.active_flow[t,i] * init.c_p * Ts/init.C_r for i in m.i))) / \
            exp(sum(init.eta_return * m.active_flow[t,i] * init.c_p for i in m.i)) * \
            (load[t] + sum(init.eta_return * m.active_flow[t,i] * init.c_p * m.T_supply[t,i] for i in m.i)) 

        if self.ocp_formulation == 0:
            m.dynamics_supply_constr = Constraint(m.t,m.i, rule=euler_supply_dyn)
            m.dynamics_return_constr = Constraint(m.t, rule=euler_return_dyn)
        else:
            m.dynamics_supply_constr = Constraint(m.t,m.i, rule=exact_supply_dyn)
            m.dynamics_return_constr = Constraint(m.t, rule=exact_return_dyn)            

        def Q_delivered_fn(m,t,i):
                return m.Q_delivered[t,i] == init.c_p * init.eta_return * m.active_flow[t,i] * (m.T_return[t] - m.T_supply[t,i])
        m.Q_delivered_constr = Constraint(m.t, m.i, rule=Q_delivered_fn)

        #  _____                       
        # |  __ \                      
        # | |__) |____      _____ _ __ 
        # |  ___/ _ \ \ /\ / / _ \ '__|
        # | |  | (_) \ V  V /  __/ |   
        # |_|   \___/ \_/\_/ \___|_|   
                                      

        def COP_fn(m,t,i):
            return m.COP[t,i] == init.a*m.integer[t,i]+init.b*(m.Q_delivered[t,i]/(init.Q_delivered_max)) + init.c*(m.Q_delivered[t,i]/init.Q_delivered_max)**2
        m.COP_constr = Constraint(m.t, m.i, rule=COP_fn)

        def P_chiller_fn(m,t,i):
            return m.COP[t,i] * m.P_chiller[t,i] - init.P0 * m.COP[t,i] * m.integer[t,i] == m.Q_delivered[t,i]
        m.P_chiller_constr = Constraint(m.t, m.i, rule=P_chiller_fn)

        def P_pump_fn(m,t,i):
            return m.P_pump[t,i] == init.gamma * m.flow_sq[t,i] * m.active_flow[t,i]

        m.P_pump_constr = Constraint(m.t, m.i, rule=P_pump_fn) # TODO: PW approximation
        m.flow_constr = Constraint(m.t, m.i, rule= lambda m,t,i: m.flow[t,i] <= init.flow_max*m.integer[t,i])
        
        def cooling_cnstr_fn(m,t):
            return m.Q_slack[t] >= load[t] - quicksum(m.Q_delivered[t,i] for i in m.i)
        m.Q_slack_constr = Constraint(m.t, rule=cooling_cnstr_fn)


        #   _____                       _   _     
        #  / ____|                     | | | |    
        # | (___  _ __ ___   ___   ___ | |_| |__  
        #  \___ \| '_ ` _ \ / _ \ / _ \| __| '_ \ 
        #  ____) | | | | | | (_) | (_) | |_| | | |
        # |_____/|_| |_| |_|\___/ \___/ \__|_| |_|
                                             
                                         
        def integer_smooth_1(m,t,i):
            return m.integer_smooth[t,i] >= m.integer[t+1,i] - m.integer[t,i]
        
        def integer_smooth_2(m,t,i):
            return m.integer_smooth[t,i] >= m.integer[t,i] - m.integer[t+1,i]

        m.integer_smooth_constr1 = Constraint(m.tm1, m.i, rule = integer_smooth_1)
        m.integer_smooth_constr2 = Constraint(m.tm1, m.i, rule = integer_smooth_2) 
        

        #   ____  _     _           _   _           
        #  / __ \| |   (_)         | | (_)          
        # | |  | | |__  _  ___  ___| |_ ___   _____ 
        # | |  | | '_ \| |/ _ \/ __| __| \ \ / / _ \
        # | |__| | |_) | |  __/ (__| |_| |\ V /  __/
        #  \____/|_.__/| |\___|\___|\__|_| \_/ \___|
        #             _/ |                          
        #            |__/                           

        under_delivery_gain = 10000.0
        # # # Control objective
        m.obj = Objective(
            expr=(
                quicksum(
                    m.P_chiller[t,i] + m.P_pump[t,i] + m.integer_smooth[t,i] for t in m.t for i in m.i
                ) 
                + under_delivery_gain * quicksum(m.Q_slack[t] for t in m.t)
            )
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

    def get_vals(self, model, keys=['flow','integer','T_evap']):
        # this assumes rectangular or scalar numerical variable values 
        # with continguous single indexing (iterating by 1)
        outputs = []
        for key in keys:

            # Catch scalars
            if not getattr(model,key).is_indexed():
                outputs.append(value(getattr(model,key)))
                
            var_idx = getattr(model,key).index_set()
            axes = [list(s) for s in var_idx.subsets()]
            idx_mins = [min(axis) for axis in axes]
            arr_shape = tuple(len(axis) for axis in axes)
            dims = len(idx_mins)
            
            arr = np.full(arr_shape, np.nan, dtype=float) #  fails loudly with nans
            vals = getattr(model,key).extract_values()
            
            for idx in vals:
                # if the indices in the pyomo model are not zero
                # indexed, this will shift them down to be zero indexed
                zero_idx = tuple(idx[i] - idx_mins[i] for i in range(dims))
                arr[zero_idx] = vals[idx]

            outputs.append(torch.from_numpy(arr))
            
        return outputs

    def __call__(self, T_supply=None, T_return=None, load=None, Ts=300.):
        T_supply = T_supply.view(-1).numpy()
        T_return = T_return.view(-1).numpy()
        load = load.view(-1).numpy()
        
        model = self.discrete_model(T_supply=T_supply, T_return=T_return, load=load, Ts=Ts)
        # elif self.ocp_formulation == 2:
        #     pass

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
        
        flow, integer, T_evap = self.get_vals(model)
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
    