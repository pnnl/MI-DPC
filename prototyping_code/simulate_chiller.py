#%%
import utils; from utils import * # aaron doesnt like the *
from chiller_system import ChillerSystem
from neuromancer.dynamics import integrators
import matplotlib.pyplot as plt
from argparse import ArgumentParser





if __name__=='__main__':
    torch.manual_seed(utils.seed)
    parser = ArgumentParser()
    parser.add_argument('-policy', choices=['DPC', 'MPC', 'RBC'], 
                        help='Choice of control strategy can be MI-DPC, implicit MI-MPC or Rule-based controller.')
    parser.add_argument('-s_length')
    args = parser.parse_args()

    # add argparse for all script level arguments



    Ts = 60. # Sampling time in seconds
    n_days = 2
    t, load_n = utils.generate_datacenter_load(number_of_days=2, sampling_time=Ts,  # load_test
                                                    night_baseline=300,
                                                    day_baseline=800,
                                                    ramp_hours=4)
    s_hours = 24*n_days
    s_length = int(s_hours*3600//Ts) # to utils

    load_n = load_n.reshape(1,-1, 1)
    load=load_n[:,0:s_length,:] # to utils

    system = ChillerSystem(M=M, Ts=Ts, C_r=C_r*1, C_i=C_i, a=a, b=b ,c=c , c_p=c_p, gamma=gamma)
    integrator = integrators.RK4(system, h=torch.tensor(Ts))
    class RBC():
        def init(self, flow_const=18., T_evap_const=T_evap_min, chiller_status=1, Q_delivered_max=utils.Q_delivered_max):
            self.M = len(chiller_status)
            self.mass_flow = torch.ones(1,s_length,M)*flow_const
            self.Q_delivered_max = Q_delivered_max

        def __call__(self, integer_status, T_return, T_supply, load_signal=None):
            PLR = system.get_cooling_delivered(integer_status, self.mass_flow[:,k,:], T_return, T_supply)/(Q_delivered_max*integer_status.sum().item())
            if PLR > PLRon: # Works only for 2 integers, TODO: make it scalable for more integers
                chiller_status += 1
            if PLR < PLRoff:
                chiller_status -= 1
            chiller_status_tensor = torch.zeros(1,1,M)
            chiller_status_tensor[:,:,:chiller_status]+1.0

    with torch.no_grad():
        flow_const = 18.
        T_evap_const= T_evap_min
        
        mass_flow = torch.ones(1,s_length,M)*flow_const # Constant mass flow
        T_evap = torch.ones(1,s_length,M)*T_evap_const # Constant evaporation temperature
        integer_status = torch.ones(1,1,M) # Initial integer status

        T_return = torch.ones(1,1,1)*8. # Initial condition
        T_supply = torch.ones(1,1,M)*8. # Initial condition

        # HISTORY LIST
        T_return_hist, T_supply_hist = [], [] 
        T_out_hist, P_chiller_hist, P_pump_hist, Q_delivered_hist = [], [], [], []
        T_evap_hist, flow_hist, integer_hist = [], [], []

        T_out = system.get_outlet_temperature(integer_status=integer_status, mass_flow=mass_flow, T_supply=T_supply)
        T_out_hist.append(T_out); T_return_hist.append(T_return); T_supply_hist.append(T_supply)

        # SWITCHING RULES
        PLRon = 0.6
        PLRoff = 0.2

        for k in range(s_length):
            T_return_hist.append(T_return); T_supply_hist.append(T_supply); T_out_hist.append(T_out); integer_hist.append(integer_status)
            PLR = system.get_cooling_delivered(integer_status, mass_flow[:,k,:], T_return, T_supply)/(Q_delivered_max*integer_status.sum().item())
                
            T_supply_and_return = integrator(torch.cat([T_supply, T_return], dim=-1), integer_status, mass_flow[:,[k],:],
                                                    T_evap[:,[k],:], load[:,[k],:],
                                                    )
            T_supply, T_return = T_supply_and_return[:,:,:system.M], T_supply_and_return[:,:,system.M:]
            T_out = system.get_outlet_temperature(mass_flow=mass_flow,T_supply=T_supply,integer_status=integer_status)
            
            if PLR > PLRon: # Works only for 2 integers, TODO: make it scalable for more integers
                integer_status = torch.ones(1,1,M)
            if PLR < PLRoff:
                integer_status = torch.ones(1,1,M)
                integer_status[:,:,1] = 0.0

        T_return_tensor = torch.vstack(T_return_hist).swapaxes(0,1)
        T_supply_tensor = torch.vstack(T_supply_hist).swapaxes(0,1)
        T_out_tensor = torch.vstack(T_out_hist).swapaxes(0,1)
        integer_tensor = torch.vstack(integer_hist).swapaxes(0,1)

        cooling_in_simulation =  system.get_cooling_delivered_per_chiller(integer_status=integer_tensor,
                                                                        mass_flow=mass_flow,
                                                                        T_return=T_return_tensor[0,:s_length,:],
                                                                        T_supply=T_supply_tensor[0,:s_length,:])
        
        chiller_power_PLR = system.get_chiller_power_PLR(integer_status=integer_tensor, 
                                                        mass_flow=mass_flow, 
                                                        T_return=T_return_tensor[:,:s_length,:],
                                                        T_supply=T_supply_tensor[:,:s_length,:], 
                                                        Q_rated=Q_delivered_max)

        pump_power_simulation = system.get_pump_consumption(mass_flow=mass_flow, integer_status=integer_tensor, exponent=exponent)
        
        const = 0.
        cost = torch.sum((pump_power_simulation.sum(dim=-1, keepdim=True) + chiller_power_PLR.sum(dim=-1, keepdim=True) + \
                            const * integer_tensor.sum(-1,keepdim=True)))*(Ts/3600)
        
        control_RMSE = torch.sqrt(torch.mean((load - cooling_in_simulation.sum(dim=-1, keepdim=True))**2))
        PLR_sim = cooling_in_simulation/(Q_delivered_max)
        COP_sim = a+b*PLR_sim+ c*torch.square(PLR_sim)

        n_violations = 0
        tolerance = 5 # tolerance for violation [kW]
        for i in range(s_length):
            if not cooling_in_simulation[0,i,0] + tolerance >= load[0,i,0]:
             n_violations += 1

    fig, axes = plt.subplots(4, 2, figsize=(12, 14))
    axes = axes.flatten()
    # 1) Return vs Out temperature
    axes[0].plot(t[:s_length],T_supply_tensor[0,:s_length,0].cpu(), label="T_supply1", c='indigo', linestyle='-', alpha=0.7)
    axes[0].plot(t[:s_length],T_supply_tensor[0,:s_length,1].cpu(), label="T_supply2", c='green', linestyle='-', alpha=0.7)
    axes[0].plot(t[:s_length],T_evap[0,:s_length,0].cpu(), color="indigo", linestyle=":", alpha=1, label="T_evap 1")
    axes[0].plot(t[:s_length],T_evap[0,:s_length,1].cpu(), color="green", linestyle=":", alpha=1, label="T_evap 2")
    axes[0].plot(t[:s_length],torch.ones(s_length)*T_supply_max, 'k--', label='Evap bounds'); axes[0].plot(t[:s_length],torch.ones(s_length)*T_supply_min, 'k--')
    axes[0].set_xlabel("Time [h]"); axes[0].set_ylabel("Temperature [°C]")
    axes[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.4), ncol=3); axes[0].grid(True)

    # # 2) Cooling delivered
    axes[1].plot(t[:s_length],load[0,:s_length,:].cpu(), 'k' , label="Q_demand")
    axes[1].plot(t[:s_length],cooling_in_simulation[0,:s_length,:].sum(-1,keepdim=True).cpu(), label="Q_delivered")
    axes[1].set_xlabel("Time [h]")
    axes[1].set_ylabel("Cooling [kW]")
    axes[1].legend()
    axes[1].grid(True)

    # # 3) Outlet vs retrun temperature
    axes[2].plot(t[:s_length],T_out_tensor[0,:s_length,:].cpu(), label="T_out", c='g')
    axes[2].plot(t[:s_length],torch.ones(s_length)*T_min, 'g--' ,label="T_out bounds"); 
    axes[2].plot(t[:s_length],torch.ones(s_length)*T_max, 'g--')

    axes[2].plot(t[:s_length],T_return_tensor[0,:s_length,0].cpu(), label="T_return", c='r')
    axes[2].plot(t[:s_length],torch.ones(s_length)*T_return_min, 'r:' ,label="T_return bounds"); 
    axes[2].plot(t[:s_length],torch.ones(s_length)*T_return_max, 'r:'); 

    axes[2].set_xlabel("Time [h]")
    axes[2].set_ylabel("Temperature [°C]")
    axes[2].legend()
    axes[2].grid(True)

    # # 4) Chiller energy consumption
    axes[3].plot(t[:s_length],chiller_power_PLR[0,:s_length,:].cpu(), label="P_chiller")
    axes[3].plot(t[:s_length],pump_power_simulation[0,:s_length,:].cpu(), label="P_pump")
    axes[3].set_xlabel("Time [h]")
    axes[3].set_ylabel("Chiller [kW]")
    axes[3].legend()
    axes[3].grid(True)

    # # 5) Flow rates
    axes[4].plot(t[:s_length],integer_tensor[0,:s_length,0]*mass_flow[0,:s_length,0].cpu(), label="Mass flow 1")
    axes[4].plot(t[:s_length],integer_tensor[0,:s_length,1]*mass_flow[0,:s_length,1].cpu(), '--', label="Mass flow 2")
    axes[4].set_xlabel("Time [h]")
    axes[4].set_ylabel("Mass flowrates [kg/s]")
    axes[4].legend()
    axes[4].grid(True)

    axes[5].plot(t[:s_length],integer_tensor[0,:s_length,0].cpu(), label='Chiller 1 Status')
    axes[5].plot(t[:s_length],integer_tensor[0,:s_length,1].cpu(), '--',label='Chiller 2 Status')
    axes[5].set_ylabel("Integer status [-]")
    axes[5].set_xlabel("Time [h]")
    axes[5].set_yticks([0,1], labels=['OFF','ON'])
    axes[5].legend()
    axes[5].grid(True)

    
    axes[6].plot(t[:s_length],PLR_sim[0,:s_length,:], label="PLR")
    axes[6].set_xlabel("Time [h]")
    axes[6].set_ylabel("PLR [-]")
    axes[6].legend()
    axes[6].grid(True)
    
    axes[7].plot(t[:s_length],COP_sim[0,:s_length,:], label="COP")
    axes[7].set_xlabel("Time [h]")
    axes[7].set_ylabel("COP [-]")
    axes[7].legend()
    axes[7].grid(True)
    axes[1].set_title(f'Total cost of operation:  {cost.item():.1f} [kWh] \n \
                        Tracking RMSE: {control_RMSE.item():.1f} [kW] \n \
                        Number of violations: {n_violations}')

    plt.savefig(f'/home/bold914/chiller_staging/plots/chiller_control_rule_based_{int(Ts)}.png')
    plt.show()

    print('Total cost of operation: ', cost.sum().item(), ' [kWh]')
    print("Mean COP", COP_sim.mean().item())
    print("Number of violations (Q_delivered < load): ", n_violations)
    # %%

