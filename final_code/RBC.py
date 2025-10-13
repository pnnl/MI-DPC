#%%
import init; import utils; import torch
from chiller_system import ChillerSystem
from neuromancer.dynamics import integrators
import matplotlib.pyplot as plt
torch.manual_seed(init.seed)
# add argparse for all script level arguments

system = ChillerSystem(M=init.M, Ts=init.Ts, C_r=init.C_r, C_i=init.C_i, 
                       a=init.a, b=init.b ,c=init.c , c_p=init.c_p, 
                       gamma=init.gamma, Q_rated=init.Q_delivered_max)

integrator = integrators.RK4(system, h=torch.tensor(init.Ts))


class RBC_policy():
  def __init__(self, PLR_on=0.6, PLR_off=0.2, n_active_chillers=init.M, 
               T_evap_const=8., mass_flow_const=15.):
    self.PLR_on=PLR_on
    self.PLR_off=PLR_off
    self.n_active_chillers = n_active_chillers
    
    self.T_evap_const = T_evap_const
    self.T_evap = torch.ones(1,1,init.M)*self.T_evap_const
    
    self.mass_flow_const = mass_flow_const
    self.mass_flow = torch.ones(1,1,init.M)*self.mass_flow_const
  
  def __call__(self, *, T_supply=None, T_return=None, load=None):
    del load
    integer = torch.zeros(1,1,init.M) # initialize
    integer[:,:,:self.n_active_chillers] = 1. # overwrite number of active integer
    PLR = system.get_cooling_delivered_per_chiller(integer, self.mass_flow,
                                        T_return, T_supply).sum(-1, keepdim=True) \
            /(init.Q_delivered_max*self.n_active_chillers)
    if PLR > self.PLR_on:
      self.n_active_chillers += 1
    if PLR < self.PLR_off:
      self.n_active_chillers -= 1
    
    self.n_active_chillers = max(1, min(self.n_active_chillers, init.M)) # at least one on
    
    output={'integer': integer, 'flow': self.mass_flow, 'T_evap': self.T_evap}
    return output