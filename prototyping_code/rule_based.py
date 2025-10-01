#%%
import utils
from utils import *
from chiller_control_DPC import generate_load
from chiller_system import ChillerSystem, kelvin2celsius, celsius2kelvin
import torch
import matplotlib.pyplot as plt
import numpy as np

torch.manual_seed(202)
t, load_n = utils.generate_datacenter_load(sampling_time=45,
                                          number_of_days=1,
                                          ramp_hours=2,
                                          night_baseline=100,
                                          osc_night_amp=100,
                                          day_baseline=450,
                                          osc_day_amp=50,
                                          noise_scale=10,
                                          ramp_jitter=0,
                                          f_day=3,
                                          f_night=2)
load_n = load_n.reshape(1,-1, 1)

# a = 2.114130 # kW
# b = 52.185990 #kW
# c = 35.171498 #kW
Ts = 60.
system = ChillerSystem(M=M, Ts=Ts, C_r=C_r, C_i=C_i, a0=a0, a1=a1, a2=a2, c_p=c_p, gamma=gamma)

s_length = 50
# s_length = load_n.size(1)

with torch.no_grad():

  loads_t_1d = generate_load(T=s_length+nsteps)

  load = loads_t_1d[0:s_length].cpu().reshape(1,-1,1)
  load=load_n[:,0:s_length,:]
  flow_const = 8.
  T_evap_const= 7.0
  
  mass_flow = torch.ones(1,s_length,M)*flow_const
  T_evap = torch.ones(1,s_length,M)*T_evap_const
  integer_status = torch.ones(1,1,M)


  T_return = torch.ones(1,1,1)*10.0
  T_supply = torch.ones(1,1,M)*8.0


  # Q_delivered_max = (T_return_max - T_evap_const) * c_p * flow_const

  Q_delivered_max = (T_return_max - T_evap_min) * c_p * flow_max


  # history lists
  T_return_hist = []; 
  T_supply_hist = []; 
  T_out_hist, P_chiller_hist, P_pump_hist, Q_delivered_hist = [], [], [], []
  T_evap_hist, flow_hist, integer_hist = [], [], []

  # Switching Rules
  PLRon = 0.4
  PLRoff = 0.1

  T_out = system.get_outlet_temperature(integer_status=integer_status, mass_flow=mass_flow, T_supply=T_supply)
  T_out_hist.append(T_out)
  T_return_hist.append(T_return)
  T_supply_hist.append(T_supply)


  for k in range(s_length):
      T_return_hist.append(T_return); T_supply_hist.append(T_supply); T_out_hist.append(T_out); integer_hist.append(integer_status)
      PLR = system.get_cooling_delivered(integer_status, mass_flow[:,k,:], T_return, T_supply)/(Q_delivered_max*integer_status.sum().item())
      # print(PLR)
          
      T_return_next, T_supply_next = system.forward_euler(integer_status, mass_flow[:,k,:],
                                            T_evap[:,k,:], T_return, 
                                            T_supply, load[:,k,:],
                                              Ts=Ts)
      T_return, T_supply = T_return_next, T_supply_next
      T_out = system.get_outlet_temperature(mass_flow=mass_flow,T_supply=T_supply,integer_status=integer_status)
      if PLR > PLRon:
        integer_status = torch.ones(1,1,M)
      if PLR < PLRoff:
        integer_status = torch.ones(1,1,M)
        integer_status[:,:,1] = 0.0

  T_return_tensor = torch.vstack(T_return_hist).swapaxes(0,1)
  T_supply_tensor = torch.vstack(T_supply_hist).swapaxes(0,1)
  T_out_tensor = torch.vstack(T_out_hist).swapaxes(0,1)
  integer_tensor = torch.vstack(integer_hist).swapaxes(0,1)
  

  cooling_in_simulation =  system.get_cooling_delivered(integer_status=integer_tensor,
                                                                mass_flow=mass_flow,
                                                                T_return=T_return_tensor[0,:s_length,:],
                                                                T_supply=T_supply_tensor[0,:s_length,:])
  chiller_power_simulation = system.get_chiller_consumption(integer_status=integer_tensor,
                                                    mass_flow=mass_flow,
                                                    T_return=T_return_tensor[:,:s_length,:],
                                                    T_supply=T_supply_tensor[:,:s_length,:])
 
  chiller_power_PLR = system.get_chiller_power_PLR(integer_status=integer_tensor, mass_flow=mass_flow, T_return=T_return_tensor[:,:s_length,:],
                                                   T_supply=T_supply_tensor[:,:s_length,:], a=a,b=b,c=c, Q_rated=Q_delivered_max)

  pump_power_simulation = system.get_pump_consumption(mass_flow=mass_flow, integer=integer_tensor, exponent=2)
  const = 0.
  # cost = pump_power_simulation + chiller_power_simulation + const * integer_tensor.sum(-1,keepdim=True)
  cost = pump_power_simulation + chiller_power_PLR + const * integer_tensor.sum(-1,keepdim=True)
  control_RMSE = torch.sqrt(torch.mean((load - cooling_in_simulation)**2))
  PLR_sim = cooling_in_simulation[0,:,:].cpu()/(integer_tensor[0,:,:].sum(-1,keepdim=True)*Q_delivered_max)
  COP_sim = PLR_sim*Q_delivered_max/(a+b*PLR_sim+ c*torch.square(PLR_sim))

  n_violations = 0
  tolerance = 2 # tolerance for violation [kW]
  for i in range(s_length):
     if cooling_in_simulation[0,i,0] + tolerance < load[0,i,0]:
      n_violations += 1


  fig, axes = plt.subplots(4, 2, figsize=(12, 14))
  axes = axes.flatten()
  # 1) Return vs Out temperature
  axes[0].plot(T_supply_tensor[0,:s_length,0].cpu(), label="T_supply1", c='indigo', linestyle='-', alpha=0.7)
  axes[0].plot(T_supply_tensor[0,:s_length,1].cpu(), label="T_supply2", c='green', linestyle='-', alpha=0.7)
  axes[0].plot(T_evap[0,:,0].cpu(), color="indigo", linestyle=":", alpha=1, label="T_evap 1")
  axes[0].plot(T_evap[0,:,1].cpu(), color="green", linestyle=":", alpha=1, label="T_evap 2")
  axes[0].plot(np.ones(s_length)*T_supply_max, 'k--', label='Evap bounds'); axes[0].plot(np.ones(s_length)*T_supply_min, 'k--')
  axes[0].set_xlabel("Timestep"); axes[0].set_ylabel("Temperature [°C]")
  axes[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.4), ncol=3); axes[0].grid(True)

  # # 2) Cooling delivered
  axes[1].plot(load[0,:s_length,:].cpu(), 'k' , label="Q_demand")
  axes[1].plot(cooling_in_simulation[0,:,:].cpu(), label="Q_delivered")
  axes[1].set_xlabel("Timestep")
  axes[1].set_ylabel("Cooling [kW]")
  axes[1].legend()
  axes[1].grid(True)

  # # 3) Outlet vs retrun temperature
  axes[2].plot(T_out_tensor[0,:,:].cpu(), label="T_out", c='g')
  axes[2].plot(np.ones(s_length)*T_min, 'g--' ,label="T_out bounds"); 
  axes[2].plot(np.ones(s_length)*T_max, 'g--')

  axes[2].plot(T_return_tensor[0,:s_length,0].cpu(), label="T_return", c='r')
  axes[2].plot(np.ones(s_length)*T_return_min, 'r:' ,label="T_return bounds"); 
  axes[2].plot(np.ones(s_length)*T_return_max, 'r:'); 

  axes[2].set_xlabel("Timestep")
  axes[2].set_ylabel("Temperature [°C]")
  axes[2].legend()
  axes[2].grid(True)

  # # 4) Chiller energy consumption
  axes[3].plot(chiller_power_PLR[0,:,:].cpu(), label="P_chiller")
  axes[3].plot(pump_power_simulation[0,:,:].cpu(), label="P_pump")
  axes[3].set_xlabel("Timestep")
  axes[3].set_ylabel("Chiller [kW]")
  axes[3].legend()
  axes[3].grid(True)

  # # 5) Flow rates
  axes[4].plot(integer_tensor[0,:,0]*mass_flow[0,:,0].cpu(), label="Mass flow 1")
  axes[4].plot(integer_tensor[0,:,1]*mass_flow[0,:,1].cpu(), '--', label="Mass flow 2")
  # axes[4].plot(np.ones(s_length)*flow_min, 'k:'); axes[4].plot(np.ones(s_length)*flow_max,'k:', label='bounds')
  axes[4].set_xlabel("Timestep")
  axes[4].set_ylabel("Mass flowrates [kg/s]")
  axes[4].legend()
  axes[4].grid(True)

  axes[5].plot(integer_tensor[0,:,0].cpu(), label='Chiller 1 Status')
  axes[5].plot(integer_tensor[0,:,1].cpu(), '--',label='Chiller 2 Status')
  axes[5].set_ylabel("Integer status [-]")
  axes[5].set_xlabel("Timestep")
  axes[5].set_yticks([0,1], labels=['OFF','ON'])
  axes[5].legend()
  axes[5].grid(True)

  axes[6].plot(COP_sim, label="COP")
  axes[6].set_xlabel("Timestep")
  axes[6].set_ylabel("COP [-]")
  axes[6].legend()
  axes[6].grid(True)
  
  axes[7].plot(PLR_sim, label="PLR")
  axes[7].set_xlabel("Timestep")
  axes[7].set_ylabel("PLR [-]")
  axes[7].legend()
  axes[7].grid(True)
  
  axes[1].set_title(f'Total cost of operation:  {cost.sum().item():.1f} \n Tracking RMSE: {control_RMSE.item():.1f}')
  print('Total cost of operation: ', cost.sum().item())

  plt.savefig('/home/bold914/chiller_staging/plots/chiller_control_rule_based.png')
  plt.show()


  print("Mean COP", COP_sim.mean().item())
  print("Number of violations (Q_delivered > load): ", n_violations)
# %%

