#%%
from argparse import ArgumentParser
import init; import torch
from neuromancer.dynamics import integrators
from chiller_system import ChillerSystem
from utils import generate_datacenter_load
from RBC import RBC_policy
from utils import customMPL;
from MIDPC import round_fn, MIDPC_policy
torch.set_default_device('cpu')

def simulate(T_supply_0, T_return_0, load_signal, dynamics_forward, policy, nsteps=10):
    T_supply_hist, T_return_hist, T_evap_hist, mass_flow_hist, chiller_status_hist, \
    relaxed_integer_hist, inference_time_hist = \
    [], [], [], [], [], [], []      # History lists
 
    T_supply, T_return = T_supply_0, T_return_0 # Initial conditions
    T_supply_hist.append(T_supply_0); T_return_hist.append(T_return_0) # Save initial condition``
    s_length = load_signal.size(1)-nsteps # Simulation length
    
    for k in range(s_length): # Simulation Loop
        decisions = policy(T_supply=T_supply, T_return=T_return, load=load_signal[:,k:k+nsteps,:]) # Compute decisions
       
        relaxed_integer = decisions.get('relaxed_integer')
        inference_time = decisions.get('inference_time')
        integer, mass_flow, T_evap = decisions['integer'], decisions['flow'], decisions['T_evap']
    
        x = dynamics_forward(torch.cat((T_supply,T_return), dim=-1),
                            integer, mass_flow, T_evap, load_signal[:,[k],:]) # Forward dynamics
        T_supply = x[:,:,:init.M]
        T_return = x[:,:,[-1]] # Last state is T_return
        T_supply_hist.append(T_supply); T_return_hist.append(T_return); # Save states
        chiller_status_hist.append(integer); mass_flow_hist.append(mass_flow); T_evap_hist.append(T_evap) # Save decision
        relaxed_integer_hist.append(relaxed_integer) if relaxed_integer is not None else None # Optional argument
        inference_time_hist.append(inference_time) if inference_time is not None else None # Optional argument
    # # # Torch tensors
    output = {}
    if relaxed_integer_hist:
        output['relaxed_integer'] = torch.vstack(relaxed_integer_hist).swapaxes(0, 1)
    if inference_time_hist:
        output['inference_time'] = torch.vstack(inference_time_hist).swapaxes(0, 1)
    output['T_supply'] = torch.vstack(T_supply_hist).swapaxes(0,1)
    output['T_return'] = torch.vstack(T_return_hist).swapaxes(0,1)
    output['chiller_status'] = torch.vstack(chiller_status_hist).swapaxes(0,1)
    output['mass_flow'] = torch.vstack(mass_flow_hist).swapaxes(0,1)
    output['T_evap'] = torch.vstack(T_evap_hist).swapaxes(0,1)
    return output

if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument('-policy', choices=['MIDPC', 'MIMPC', 'RBC'], default='MIDPC',
        help='Choice of control strategy can be MI-DPC, implicit MI-MPC or Rule-based controller.')
    parser.add_argument('-nsteps', default=100, type=int)
    parser.add_argument('-n_days', default=2, type=int)
    # args = parser.parse_args()
    args, unknown = parser.parse_known_args()
    
    # # # Initialize policy
    if args.policy == 'RBC':
        policy = RBC_policy(
                        PLR_on=0.6, PLR_off=0.2,
                        n_active_chillers=init.M,
                        T_evap_const=8., 
                        mass_flow_const=15.
                        )
   
    elif args.policy == 'MIDPC':
        policy = MIDPC_policy(
            load_path=f'results/MIDPC/policies/N_{args.nsteps}.pt',
            nsteps=args.nsteps,
            measure_inference_time=True,
            )
        
    elif args.policy == 'MIMPC':
        # policy = MPC_policy()
        pass

    # # # System init
    Ts = 300
    chiller_system = ChillerSystem(a=init.a, b=init.b, c=init.c,
                                    C_r=init.C_r, C_i=init.C_i, c_p=init.c_p,
                                    gamma=init.gamma, exponent=init.exponent, M=init.M, Ts=Ts)
    integrator = integrators.RK4(chiller_system, h=torch.tensor(Ts))
    
    # # # Load test
    seed = init.seed
    load_time, load_test = generate_datacenter_load(number_of_days=args.n_days,
                                                    sampling_time=Ts, signal_seed=seed,
                                                    ramp_hours=4, night_baseline=300, day_baseline=600)
    load_test = load_test.reshape(1,-1,1)

    T_supply_0 = torch.ones(1,1,init.M) * 8.
    T_return_0 = torch.ones(1,1,1) * 8.

    outputs = simulate(T_supply_0=T_supply_0, T_return_0=T_return_0, 
                       load_signal=load_test, dynamics_forward=integrator, 
                       policy=policy, nsteps=args.nsteps)
    torch.save(outputs, f'results/{args.policy}/data_N{args.nsteps}.pt')
# %%
    # cl_system = torch.load('results/MIDPC/policies/N_100.pt', weights_only=False)

