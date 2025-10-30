# %%
import numpy as np
import torch

from pyomo.environ import (
    ConcreteModel, Var, Param, RangeSet, NonNegativeReals, Binary,
    Constraint, Objective, minimize, quicksum, value, SolverFactory
)

# Expecting your original 'init' and 'utils' modules to exist
from init import SystemParameters
init = SystemParameters()

class MIMPC_policy():
    """
    Optimized MPC with Pyomo:
      - Build the model once
      - Update mutable Params & initial conditions each call
      - Reuse variable .value for warm starts
    """
    def __init__(self,
                 nsteps,
                 measure_inference_time=True,
                 M=init.M,
                 solver="gurobi",                 # 'gurobi' | 'scip' | ...
                 ocp_formulation=0,               # 0: euler, 1: exact (nonlinear), 2: continuous (todo)
                 exponent=init.exponent,
                 verbose=True,
                 max_solver_time=300,
                 McCormick=False,
                 warmstart=False,
                 Ts=None):

        self.nsteps = int(nsteps)
        self.measure_inference_time = measure_inference_time
        self.ocp_formulation = ocp_formulation
        self.M = int(M)
        self.exponent = exponent
        self.verbose = verbose
        self.max_solver_time = max_solver_time
        self.McCormick = McCormick
        self.Ts_default = Ts
        self.warmstart = warmstart
        # Choose solver
        self.solver = SolverFactory(solver)
        if self.max_solver_time is not None:
            self.solver.options['TimeLimit'] = self.max_solver_time
        # Tight gap can be expensive; tune as needed
        self.solver.options['MIPGap'] = 1e-5

        # Build model once
        self.model = self._build_model()

        # A flag to know if we already solved once (for warm start defaults)
        self._has_solution = False


    # ---------------------------
    # Model construction (once)
    # ---------------------------
    def _build_model(self):
        m = ConcreteModel()

        # Index sets
        m.T = RangeSet(0, self.nsteps)          # 0..N
        m.t = RangeSet(0, self.nsteps - 1)      # 0..N-1
        # For smoothing constraints that use t and t+1
        m.tm1 = RangeSet(0, self.nsteps - 2)    # 0..N-2
        m.i = RangeSet(1, self.M)

        # -----------------
        # Mutable Params
        # -----------------
        # Time-varying inputs
        m.load_demand = Param(m.t, mutable=True, initialize=0.0)
        m.filtered_load = Param(m.t, mutable=True, initialize=0.0)

        # Sampling time (mutable to allow per-call change)
        m.Ts = Param(m.t, mutable=True, initialize=lambda m, t: self.Ts_default if self.Ts_default is not None else 180.)

        # -----------------
        # Variables
        # -----------------
        m.T_return = Var(m.T, bounds=(init.T_return_min, init.T_return_max), within=NonNegativeReals)
        m.T_supply = Var(m.T, m.i, bounds=(init.T_min, init.T_max), within=NonNegativeReals)

        m.flow = Var(m.t, m.i, bounds=(init.flow_min, init.flow_max), within=NonNegativeReals)
        m.integer = Var(m.t, m.i, within=Binary)
        m.T_evap = Var(m.t, m.i, bounds=(init.T_evap_min, init.T_evap_max), within=NonNegativeReals)

        m.Q_delivered = Var(m.t, m.i, bounds=(0., init.Q_delivered_max), within=NonNegativeReals)
        m.P_chiller = Var(m.t, m.i, within=NonNegativeReals)
        m.P_pump = Var(m.t, m.i, within=NonNegativeReals)
        m.COP = Var(m.t, m.i, within=NonNegativeReals)

        # Auxiliaries
        m.Q_slack = Var(m.t, within=NonNegativeReals)
        m.flow_sq = Var(m.t, m.i, within=NonNegativeReals)
        m.integer_smooth = Var(m.t, m.i, within=NonNegativeReals)
        m.flow_smooth = Var(m.t, m.i, within=NonNegativeReals)
        m.active_flow = Var(m.t, m.i, within=NonNegativeReals)

        # -----------------
        # Constraints
        # -----------------
        # Bilinear handling for active_flow
        if self.McCormick:
            def _ub1(m, t, i): return m.active_flow[t, i] <= init.flow_max * m.integer[t, i]
            def _lb1(m, t, i): return m.active_flow[t, i] >= init.flow_min * m.integer[t, i]
            def _ub2(m, t, i): return m.active_flow[t, i] <= m.flow[t, i] - init.flow_min * (1 - m.integer[t, i])
            def _lb2(m, t, i): return m.active_flow[t, i] >= m.flow[t, i] - init.flow_max * (1 - m.integer[t, i])
            m.active_flow_ub1 = Constraint(m.t, m.i, rule=_ub1)
            m.active_flow_lb1 = Constraint(m.t, m.i, rule=_lb1)
            m.active_flow_ub2 = Constraint(m.t, m.i, rule=_ub2)
            m.active_flow_lb2 = Constraint(m.t, m.i, rule=_lb2)
        else:
            def _active_flow_fn(m, t, i):
                return m.active_flow[t, i] == m.integer[t, i] * m.flow[t, i]
            m.active_flow_constr = Constraint(m.t, m.i, rule=_active_flow_fn)

        # flow_sq = flow^2
        def _flow_sq_fn(m, t, i):
            return m.flow_sq[t, i] == m.flow[t, i] ** 2
        m.flow_sq_constr = Constraint(m.t, m.i, rule=_flow_sq_fn)

        # Dynamics (Euler by default; exact uses exp() -> nonlinear)
        def _euler_supply_dyn(m, t, i):
            Ts = m.Ts[t]
            return m.T_supply[t + 1, i] == m.T_supply[t, i] - Ts / init.C_i * init.eta_supply * (
                m.active_flow[t, i] * init.c_p * (m.T_supply[t, i] - m.T_evap[t, i])
            )

        def _euler_return_dyn(m, t):
            Ts = m.Ts[t]
            return m.T_return[t + 1] == m.T_return[t] + Ts / init.C_r * init.eta_return * (
                m.load_demand[t] - quicksum(m.active_flow[t, i] * init.c_p * (m.T_return[t] - m.T_supply[t, i]) for i in m.i)
            )

        if self.ocp_formulation == 0:
            m.dynamics_supply_constr = Constraint(m.t, m.i, rule=_euler_supply_dyn)
            m.dynamics_return_constr = Constraint(m.t, rule=_euler_return_dyn)
        else:
            def _exact_supply_dyn(m, t, i):
                Ts = m.Ts[t]
                return m.T_supply[t + 1, i] == \
                    exp(-init.eta_supply * m.active_flow[t, i] * init.c_p * Ts / init.C_i) * m.T_supply[t, i] + \
                    (1 - exp(-init.eta_supply * m.active_flow[t, i] * init.c_p * Ts / init.C_i)) * m.T_evap[t, i]

            def _exact_return_dyn(m, t):
                Ts = m.Ts[t]
                return m.T_return[t + 1] == \
                    exp(-quicksum(init.eta_return * m.active_flow[t, i] * init.c_p * Ts / init.C_r for i in m.i)) * m.T_return[t] + \
                    (1 - exp(-quicksum(init.eta_return * m.active_flow[t, i] * init.c_p * Ts / init.C_r for i in m.i))) / \
                    exp(quicksum(init.eta_return * m.active_flow[t, i] * init.c_p for i in m.i)) * \
                    (m.load_demand[t] + quicksum(init.eta_return * m.active_flow[t, i] * init.c_p * m.T_supply[t, i] for i in m.i))

            m.dynamics_supply_constr = Constraint(m.t, m.i, rule=_exact_supply_dyn)
            m.dynamics_return_constr = Constraint(m.t, rule=_exact_return_dyn)

        # Delivered cooling
        def _Q_delivered_fn(m, t, i):
            return m.Q_delivered[t, i] == init.c_p * init.eta_return * m.active_flow[t, i] * (m.T_return[t] - m.T_supply[t, i])
        m.Q_delivered_constr = Constraint(m.t, m.i, rule=_Q_delivered_fn)

        # COP model (quadratic in Q_delivered)
        def _COP_fn(m, t, i):
            return m.COP[t, i] == (
                init.a
                + init.b * (m.Q_delivered[t, i] / init.Q_delivered_max)
                + init.c * (m.Q_delivered[t, i] / init.Q_delivered_max) ** 2
            )
        m.COP_constr = Constraint(m.t, m.i, rule=_COP_fn)

        # Chiller power (MIQP)
        def _P_chiller_fn(m, t, i):
            return m.COP[t, i] * m.P_chiller[t, i] - init.P0 * m.COP[t, i] * m.integer[t, i] == m.Q_delivered[t, i]
        m.P_chiller_constr = Constraint(m.t, m.i, rule=_P_chiller_fn)

        # Pump power (quadratic)
        def _P_pump_fn(m, t, i):
            # If you want linear MILP, replace with PW approximation
            return m.P_pump[t, i] == init.gamma * m.active_flow[t, i] * m.flow_sq[t,i]
        m.P_pump_constr = Constraint(m.t, m.i, rule=_P_pump_fn)

        # Smoothness terms (only when nsteps>=2)
        if self.nsteps >= 2:
            def _integer_smooth_1(m, t, i):
                return m.integer_smooth[t, i] >= m.integer[t + 1, i] - m.integer[t, i]

            def _integer_smooth_2(m, t, i):
                return m.integer_smooth[t, i] >= m.integer[t, i] - m.integer[t + 1, i]

            m.integer_smooth_constr1 = Constraint(m.tm1, m.i, rule=_integer_smooth_1)
            m.integer_smooth_constr2 = Constraint(m.tm1, m.i, rule=_integer_smooth_2)

            def _flow_smooth_1(m, t, i):
                return m.flow_smooth[t, i] >= m.flow[t + 1, i] - m.flow[t, i]

            def _flow_smooth_2(m, t, i):
                return m.flow_smooth[t, i] >= m.flow[t, i] - m.flow[t + 1, i]

            m.flow_smooth_constr1 = Constraint(m.tm1, m.i, rule=_flow_smooth_1)
            m.flow_smooth_constr2 = Constraint(m.tm1, m.i, rule=_flow_smooth_2)

        # Objective
        def _obj_expr(m):
            # Sum over time and units
            return quicksum(
                0.01 * (quicksum(m.Q_delivered[t, i] for i in m.i) - m.load_demand[t]) ** 2
                + quicksum(m.P_chiller[t, i] + m.P_pump[t, i] + 20.0 * m.integer_smooth[t, i] + 20.0 * m.flow_smooth[t, i] for i in m.i)
                for t in m.t
            )

        m.obj = Objective(expr=_obj_expr(m), sense=minimize)

        # Initial conditions (fixed later per call)
        # We'll fix/unfix at call time to avoid stale fixes
        return m


    # ---------------------------
    # Data update per call
    # ---------------------------
    def _update_data(self, T_supply, T_return, load, filtered_load, Ts):
        """Update mutable Params and initial conditions; keep structure intact."""
        m = self.model

        # Update time series Params
        for t in m.t:
            m.load_demand[t] = float(load[t])
            m.filtered_load[t] = float(filtered_load[t])
            m.Ts[t] = float(Ts)

        # Fix initial conditions
        # Unfix first in case they were fixed before
        if m.T_return[0].fixed: m.T_return[0].unfix()
        m.T_return[0].fix(float(T_return.item()))

        for i in m.i:
            if m.T_supply[0, i].fixed: m.T_supply[0, i].unfix()
            m.T_supply[0, i].fix(float(T_supply[i - 1]))

        # "One chiller always on"
        for t in m.t:
            if m.integer[t, 1].fixed: m.integer[t, 1].unfix()
            m.integer[t, 1].fix(1.0)


    # ---------------------------
    # Warm start helpers
    # ---------------------------
    def _warm_start_defaults(self):
        """
        If it's the first solve or some vars have None, put reasonable starts.
        This helps warmstart=True do their job.
        """
        m = self.model
        # Basic heuristic: if value is None, set to mid-bounds or zeros as appropriate
        for v in m.component_objects(Var, active=True):
            for idx in v:
                if v[idx].value is None:
                    lb, ub = v[idx].lb, v[idx].ub
                    if lb is not None and ub is not None:
                        v[idx].set_value(0.5 * (lb + ub))
                    else:
                        v[idx].set_value(0.0)


    def get_vals(self, model, keys=('flow', 'integer', 'T_evap')):
        """
        Same functionality: convert indexed Vars into torch tensors with the
        assumption of rectangular / contiguous indexing.
        """
        outputs = []
        for key in keys:
            comp = getattr(model, key)

            # Scalar
            if not comp.is_indexed():
                outputs.append(torch.tensor([value(comp)]))
                continue

            var_idx = comp.index_set()
            axes = [list(s) for s in var_idx.subsets()]
            idx_mins = [min(axis) for axis in axes]
            arr_shape = tuple(len(axis) for axis in axes)
            dims = len(idx_mins)

            arr = np.full(arr_shape, np.nan, dtype=float)
            vals = comp.extract_values()

            for idx, val in vals.items():
                zero_idx = tuple(idx[i] - idx_mins[i] for i in range(dims))
                arr[zero_idx] = val

            outputs.append(torch.from_numpy(arr))

        return outputs

    def __call__(self, T_supply=None, T_return=None, load=None, filtered_load=None, Ts=None):
        """
        Solve the *existing* model with updated Params and warm start.
        """
        m = self.model

        # Default Ts
        Ts = self.Ts_default if Ts is None else Ts

        # Convert tensors -> numpy flat
        T_supply = T_supply.view(-1).cpu().numpy()
        T_return = T_return.view(-1).cpu().numpy()
        load = load.view(-1).cpu().numpy()
        if filtered_load is None:
            filtered_load = load.copy()
        else:
            filtered_load = filtered_load.view(-1).cpu().numpy()

        # Update data & initial conditions
        self._update_data(T_supply=T_supply, T_return=T_return, load=load, filtered_load=filtered_load, Ts=Ts)

        # Provide default starts if needed (first call)
        if not self._has_solution:
            self._warm_start_defaults()

        # Solve
        results = self.solver.solve(m, tee=False, symbolic_solver_labels=False, warmstart=self.warmstart)

        if self.verbose:
            try:
                print('Solution time:', results.solver.wall_time, '— status:', results.solver.termination_condition)
            except Exception:
                print('Status:', results.solver.termination_condition)

        # Mark that we have a solution for the next warm start
        self._has_solution = True

        # Extract outputs
        flow, integer, T_evap = self.get_vals(m, keys=('flow', 'integer', 'T_evap'))
        out = {
            'termination_condition': results.solver.termination_condition,
            'integer': integer[0].view(1, 1, -1),
            'flow': flow[0].view(1, 1, -1),
            'T_evap': T_evap[0].view(1, 1, -1),
        }
        if self.measure_inference_time:
            out['inference_time'] = torch.tensor([results.solver.wall_time]).view(1,1,1)
        return out


if __name__ == "__main__":
    nsteps = 6
    T_supply = torch.tensor([9.0, 9.0])      # supply temperature
    T_return = torch.tensor([15.0])          # return temperature
    load = torch.ones(1, nsteps, 1) * 400    # heat load

    policy = MIMPC_policy(
        nsteps=nsteps,
        ocp_formulation=0,        # Euler
        verbose=False,
        solver="gurobi",
        McCormick=False,          # set True for MILP linearization of active_flow
        Ts=300.0
    )

    result = policy(
        T_supply=T_supply,
        T_return=T_return,
        load=load,
        Ts=300.0
    )

    # Subsequent calls will warm-start automatically:
    result2 = policy(
        T_supply=T_supply,
        T_return=T_return,
        load=load * 1.05,
        Ts=300.0
    )
