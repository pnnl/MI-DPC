#%%
from argparse import ArgumentParser
import torch; from init import SystemParameters
from neuromancer.dynamics import integrators
from chiller_system import ChillerSystem
from utils import generate_datacenter_load, plot_chiller_data
from utils import customMPL;
import time
torch.set_default_device('cpu')
def simulate(
        T_supply_0, T_return_0, load_signal, 
        dynamics_forward, policy, nsteps=10, #time_limit=3600
        verbose=False, system=None, n_days=1, Ts=180, s_length=None, time_limit=3600):
    # # # History Lists
    T_supply_hist, T_return_hist, T_evap_hist, mass_flow_hist, chiller_status_hist, \
    relaxed_integer_hist, inference_time_hist = \
    [], [], [], [], [], [], []      
 
    T_supply, T_return = T_supply_0, T_return_0 # Initial conditions
    T_supply_hist.append(T_supply_0); T_return_hist.append(T_return_0) # Save initial condition
    s_length = int((n_days*24*60*60)/(Ts)) if s_length is None else s_length # Simulation length
    filtered = []
    for k in range(load_test.size(1)):
        filtered.append(chiller_system.apply_load_filter(load_test[0,k]))
    filtered_load = torch.vstack(filtered).view(1,-1,1)
    start_time = time.time()
    for k in range(s_length): # Simulation Loop
        print("Timestep: ", k) if verbose else None
        decisions = policy(T_supply=T_supply, T_return=T_return, load=load_signal[:,k:k+nsteps,:], filtered_load=filtered_load[:,k:k+nsteps,:]) # Compute decisions
        # # # Read data
        relaxed_integer, inference_time = decisions.get('relaxed_integer'), decisions.get('inference_time')
        integer, mass_flow, T_evap = decisions['integer'], decisions['flow'], decisions['T_evap']
        # # # Dynamics
        x = dynamics_forward(torch.cat((T_supply,T_return), dim=-1),
                            integer, mass_flow, T_evap, system.apply_load_filter(load_signal[:,[k],:])) # Forward dynamics
        # # # Decouple
        T_supply = x[:,:,:-1]
        T_return = x[:,:,[-1]] # Last state is T_return
        # # # Histroy
        T_supply_hist.append(T_supply); T_return_hist.append(T_return); # Save states
        chiller_status_hist.append(integer); mass_flow_hist.append(mass_flow); T_evap_hist.append(T_evap) # Save decision
        relaxed_integer_hist.append(relaxed_integer) if relaxed_integer is not None else None # Optional argument
        inference_time_hist.append(inference_time) if inference_time is not None else None # Optional argument
        if time.time() - start_time > time_limit: # Exceeding time limit
            print("Time limit exceeded")
            break

    # # # Output Dictionary
    output = {}
    if relaxed_integer_hist: # Relaxed integer for MIDPC
        output['relaxed_integer'] = torch.vstack(relaxed_integer_hist).swapaxes(0, 1)
    if inference_time_hist: # Inference time for [MIDPC, MIMPC]
        output['inference_time'] = torch.vstack(inference_time_hist).swapaxes(0, 1)
    output['T_supply'] = torch.vstack(T_supply_hist).swapaxes(0,1)
    output['T_return'] = torch.vstack(T_return_hist).swapaxes(0,1)
    output['chiller_status'] = torch.vstack(chiller_status_hist).swapaxes(0,1)
    output['mass_flow'] = torch.vstack(mass_flow_hist).swapaxes(0,1)
    output['T_evap'] = torch.vstack(T_evap_hist).swapaxes(0,1)
    # # # Compute scores
    output['P_chiller'] = system.get_chiller_power_PLR(
        integer_status=output['chiller_status'], mass_flow=output['mass_flow'],
        T_return=output['T_return'][:,:-1,:], T_supply=output['T_supply'][:,:-1,:],
    )
    output['P_pump'] = system.get_pump_consumption(
        integer_status=output['chiller_status'], mass_flow=output['mass_flow']
        )
    output['Q_delivered'] = system.get_cooling_delivered_per_chiller(
        integer_status=output['chiller_status'], mass_flow=output['mass_flow'],
        T_return=output['T_return'][:,:-1,:], T_supply=output['T_supply'][:,:-1,:],
    )
    output['T_out'] = system.get_outlet_temperature(
        integer_status=output['chiller_status'], mass_flow=output['mass_flow'],
        T_supply=output['T_supply'][:,:-1,:]
    )
    output['load'] = load_signal[:,:s_length,:]
    return output

if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument('-policy', choices=['MIDPC', 'MIMPC', 'RBC'], default='MIMPC',
        help='Choice of control strategy can be MI-DPC, implicit MI-MPC or Rule-based controller.')
    parser.add_argument('-nsteps', default=2, type=int, help='Prediction horizon length.')
    parser.add_argument('-Ts', default=180, type=int, help='Sampling time.')
    parser.add_argument('-M', default=2, type=int, help='Number of chillers.')
    parser.add_argument('-n_days', default=7, type=int, help='Number of days of simulation.')
    parser.add_argument('-plotting', default=True, type=bool, help='Plot or not.')
    parser.add_argument('-s_length', default=None, type=int, help='Overrides n_days if defined.')
    
    # args = parser.parse_args()
    args, unknown = parser.parse_known_args()
    init = SystemParameters(Ts=args.Ts, M=args.M)
    chiller_system = ChillerSystem(init=init)
    s_length = args.s_length
    # # # Initialize the policy
    if args.policy == 'RBC':
        from RBC import RBC_policy
        policy = RBC_policy(
            PLR_on=0.6,
            PLR_off=0.15,
            n_active_chillers=init.M,
            M = init.M,
            Q_delivered_max=init.Q_delivered_max,
            T_evap_const=10.,
            mass_flow_const=10.,
        system = chiller_system
            )
   
    elif args.policy == 'MIDPC':
        from MIDPC import MIDPC_policy, round_fn, load_filter
        policy = MIDPC_policy(
            load_path=f'results/MIDPC/policies/N_{args.nsteps}_Ts_{args.Ts}_M_{init.M}.pt',
            nsteps=args.nsteps,
            measure_inference_time=True,
            )
        
    elif args.policy == 'MIMPC':
        from MIMPC import MIMPC_policy
        # if args.s_length is None:
        #     s_length = 20
        policy = MIMPC_policy(
            nsteps=args.nsteps,
            M = args.M,
            Ts = args.Ts,
            measure_inference_time=True,
            ocp_formulation=0,
            exponent=init.exponent,
            solver='gurobi',
            verbose=True,
            max_solver_time=180,
            McCormick=True,
            warmstart=False
        )
    
    integrator = integrators.RK4(chiller_system, h=torch.tensor(args.Ts))
    
    # # # Load test
    seed = init.seed
    load_time, load_test = generate_datacenter_load(number_of_days=args.n_days+1,
                                                    sampling_time=args.Ts, 
                                                    signal_seed=seed,
                                                    ramp_hours=init.ramp_hours,
                                                    f_day=5, f_night=6, 
                                                    day_baseline=init.day_baseline, 
                                                    night_baseline=init.night_baseline,
                                                    osc_night_amp=20, osc_day_amp=20,
                                                    noise_scale=5,
                                                    )
    load_test = load_test.reshape(1,-1,1)
    # # # Initial conditions
    T_supply_0 = torch.ones(1,1,init.M) * 8.
    T_return_0 = torch.ones(1,1,1) * 8.
    print(f'Simulating chiller with {args.policy}, N={args.nsteps}, M={init.M}')
    outputs = simulate(
                        T_supply_0=T_supply_0, # IC
                        T_return_0=T_return_0, # IC
                        load_signal=load_test, # Disturbance
                        dynamics_forward=integrator, # Dynamics model [integrator or chiller_system.forward]
                        policy=policy, # Control strategy
                        nsteps=args.nsteps, # Prediction horizon for [MIDPC, MIMPC]
                        verbose=True, # Print current timestep
                        system=chiller_system, # For computing score variables
                        n_days=args.n_days,
                        s_length=s_length
                       ) # Returns dictionary
    # # # Save outputs for analysis
    torch.save(outputs, f'results/{args.policy}/data_N{args.nsteps}_Ts_{args.Ts}_M_{init.M}.pt')
    
    if args.plotting:
        plot_chiller_data(outputs, Ts=args.Ts, time_unit='h',save_path=f'plots/{args.policy}/data_N{args.nsteps}_Ts_{args.Ts}_M_{init.M}.pdf')
# if __name__ == '__main__':
#     import matplotlib.pyplot as plt
#     plt.show()
#     PLR = outputs['Q_delivered'][0,:,:].sum(-1,keepdim=True)/ \
#              (init.Q_delivered_max*outputs['chiller_status'][0,:,:].sum(-1,keepdim=True))
#     plt.plot(outputs['Q_delivered'][0,:,:].sum(-1,keepdim=True)/
#              (init.Q_delivered_max*outputs['chiller_status'][0,:,:].sum(-1,keepdim=True)))
#     plt.plot(outputs['chiller_status'][0,:,:])
#     print(PLR[1400:1600])
#%%
    # import matplotlib.pyplot as plt
    # plt.plot(load_test[0,:50])
    # # plt.plot(chiller_system.apply_load_filter(load_test[0]).view(1,-1,1)[0])
    # filtered = []
    # for k in range(load_test.size(1)):
    #     filtered.append(chiller_system.apply_load_filter(load_test[0,k]))
    # filtered_tensor = torch.vstack(filtered)
    # plt.plot(filtered_tensor[:50])