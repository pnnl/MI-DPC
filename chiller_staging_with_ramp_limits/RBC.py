#%%
from init import SystemParameters; 
import utils; import torch
from chiller_system import ChillerSystem
from neuromancer.dynamics import integrators
import matplotlib.pyplot as plt

init = SystemParameters()
torch.manual_seed(init.seed)
system = ChillerSystem( init=init)
                        # M=init.M, Ts=init.Ts, C_r=init.C_r, C_i=init.C_i, 
                      #  a=init.a, b=init.b ,c=init.c , c_p=init.c_p, 
                      #  gamma=init.gamma, Q_rated=init.Q_delivered_max)

class RBC_policy():
  def __init__(self, PLR_on=0.6, PLR_off=0.2, n_active_chillers=2, Q_delivered_max = 500, M=2,
               T_evap_const=8., mass_flow_const=15., system=None):
    self.PLR_on=PLR_on
    self.PLR_off=PLR_off
    self.n_active_chillers = n_active_chillers
    self.Q_delivered_max = Q_delivered_max
    self.M = M
    self.T_evap_const = T_evap_const
    self.T_evap = torch.ones(1,1,self.M)*self.T_evap_const
    
    self.mass_flow_const = mass_flow_const
    self.mass_flow = torch.ones(1,1,self.M)*self.mass_flow_const
    self.system=system
  def __call__(self, *, T_supply=None, T_return=None, load=None, filtered_load=None):
    del load, filtered_load
    integer = torch.zeros(1,1,self.M) # initialize
    integer[:,:,:self.n_active_chillers] = 1. # overwrite number of active integer
    PLR = self.system.get_cooling_delivered_per_chiller(integer, self.mass_flow,
                                        torch.cat((T_supply,T_return), dim=-1),
                                        ramp_bounds=True, update_memory=False
                                        ).sum(-1, keepdim=True) \
            /(self.Q_delivered_max*self.n_active_chillers)
    # print(PLR)
    if PLR > self.PLR_on:
      self.n_active_chillers += 1
    if PLR < self.PLR_off:
      self.n_active_chillers -= 1
    
    self.n_active_chillers = max(1, min(self.n_active_chillers, self.M)) # at least one on
    
    output={'integer': integer, 'flow': self.mass_flow, 'T_evap': self.T_evap}
    return output