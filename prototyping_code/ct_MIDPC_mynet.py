#%%
import utils; from utils import *
import matplotlib.pyplot as plt
from chiller_system import ChillerSystem
from neuromancer.system import Node, SystemPreview
from neuromancer.modules import blocks
from neuromancer.modules.activations import activations
from neuromancer.constraint import variable
from neuromancer.loss import PenaltyLoss
from neuromancer.problem import Problem
from neuromancer.dataset import DictDataset
from neuromancer.dynamics import integrators
from neuromancer.trainer import Trainer
torch.manual_seed(202)

# flow_min = flow_min - flow_min
# flow_max = flow_max - flow_min
# flow_min = 5
# flow_max = 20

# T_return_max = 30
if __name__=='__main__':
    unit_list = ['kW', 'MW']
    units = unit_list[1]
    unit_coeff = 1e-3 if units == 'MW' else 1.
    layer_norm = True
    affine_norm = False
    exponent = 2
    load_min = 0
    load_max = 800*unit_coeff
    nsteps = 80
    Ts = 60.
    mins = [T_supply_min] * M + [T_return_min] * 1 + [load_min] * (nsteps + 1) 
    maxs = [T_supply_max] * M + [T_return_max] * 1 + [load_max] * (nsteps + 1) 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_default_device(device=device)
    system = ChillerSystem(M=M, Ts=Ts, C_r=C_r*unit_coeff, C_i=C_i*unit_coeff, c_p=c_p*unit_coeff,
                            a=a, b=b, c=c, gamma=gamma*unit_coeff)
    integrator = integrators.RK4(system, h=torch.tensor(Ts))

    dynamics_node = Node(integrator,
                        input_keys=['T_supply_and_return', 'integer', 'flow', 'T_evap', 'load'],
                        output_keys=['T_supply_and_return'],
                        name='system_dynamics')
    layer_width = 140
    policy = utils.policy(layer_width=140, input_features=1*M+1+1*(nsteps+1))
    
   
    mynet = Node(policy, input_keys=['T_supply_and_return','load'],
                 output_keys=['flow', 'T_evap', 'integer'], name='policy')
    
    # round_fn = lambda x: torch.clip(relaxed_round(x),0,1)
    round_fn = lambda x: torch.cat((torch.clip(relaxed_round(x), 0., 1.), torch.ones((x.size(0), 1), requires_grad=True)), dim=-1)
    rounding_node = Node(round_fn, input_keys=['relaxed_integer'], output_keys=['integer'], name='soft_rounding')
    decat_node = Node(lambda x: (x[:,:M], x[:,M:]), input_keys=['T_supply_and_return'], output_keys=['T_supply', 'T_return'])


    # NEUROMANCER SYSTEM
    cl_system = SystemPreview([ mynet, dynamics_node, decat_node],
                                nsteps=nsteps, name='cl_system', pad_mode='reflect',
                                preview_keys_map={'load': ['policy']})

    # test_output = cl_system({ 'T_supply': torch.rand(1,1,M), 'T_return': torch.rand(1,1,1), 'load': torch.rand(1,nsteps+1,1),})
    test_output = cl_system({ 'T_supply_and_return': torch.rand(1,1,M+1), 'load': torch.rand(1,nsteps+1,1),})
    
    #%%
    """ Variables
    States: T_return, T_supply
    Decisions: integer, flow, cooling_delivered
    External: load
    Scores: T_out, cooling_delivered
    """
    # relaxed_integer_variable = variable('relaxed_integer') # Decision
    integer_variable = variable('integer') # Decisions
    flow_variable = variable('flow') # Decision
    T_evap_variable = variable('T_evap') # Decision
    load_variable = variable('load') # External
    T_return_variable = variable('T_return')[:,:nsteps,:] # No terminal state
    T_supply_variable = variable('T_supply')[:,:nsteps,:] # No terminal state
    
    T_out_variable = system.get_outlet_temperature(integer_status=integer_variable, 
                                                mass_flow=flow_variable,
                                                    T_supply=T_supply_variable) # State

    cooling_delivered_variable = system.get_cooling_delivered_per_chiller(integer_status=integer_variable,
                                                                mass_flow=flow_variable,
                                                                T_return=T_return_variable,
                                                                T_supply=T_supply_variable) # Decisions
    
    #%% CONTROL OBJECTIVES
    chiller_loss = 1. * ((system.get_chiller_power_PLR(integer_status=integer_variable, 
                                                    Q_rated=Q_delivered_max*unit_coeff,
                                                    mass_flow=flow_variable,
                                                    T_return=T_return_variable,
                                                    T_supply=T_supply_variable) == 0.))
    
    pump_loss = 1. * ((system.get_pump_consumption(mass_flow=flow_variable, 
                        integer_status=integer_variable, exponent=exponent) == 0.))
    
    c = 20. # Switching cost coefficient
    switching_loss = c*((integer_variable[:, 1:, :] == integer_variable[:, :-1, :])^2)
    # switching_loss = c*((relaxed_integer_variable[:, 1:, :] == relaxed_integer_variable[:, :-1, :])^2)
    # switching_loss = c*(integer_variable == 0)

    # TRACKING
    cooling_loss = 0.1*(cooling_delivered_variable == load_variable[:,:nsteps,:])^2.

    COP_loss = -10. * (system.a*integer_variable + 
                system.b * cooling_delivered_variable/(Q_delivered_max*unit_coeff) + 
                system.c * torch.pow(cooling_delivered_variable/(Q_delivered_max*unit_coeff),2)
                == 0.)
    COP_loss.name = 'COP_loss'

    chiller_loss.name = 'chiller_loss'; pump_loss.name = 'pump_loss'
    switching_loss.name = 'switching_loss'; cooling_loss.name = 'cooling_loss'
    
    flowin_loss = 1.*((flow_variable == 18.)^2.)

    loss_list = [
                chiller_loss,  
                pump_loss, 
                # COP_loss,
                # switching_loss, 
                # cooling_loss 
                # flowin_loss
                ]
    #%% CONSTRAINTS
    T_out_lb, T_out_ub = 50.*(T_out_variable >= T_min), 50.*(T_out_variable <= T_max)
    T_return_lb, T_return_ub = 100.*(T_return_variable >= T_return_min), 100.*(T_return_variable <= T_return_max)
    T_supply_lb, T_supply_ub = 50.*(T_supply_variable >= T_supply_min), 50.*(T_supply_variable <= T_supply_max)
    
    cooling_bound = 0.01* (cooling_delivered_variable[:,:nsteps,:] >= load_variable[:,:nsteps,:]); cooling_bound.name='cooling_bound'

    flow_lb = 50.*(flow_variable >= flow_min); flow_ub = 50.* (flow_variable <= flow_max)
    T_evap_lb = 50.*(T_evap_variable >= T_evap_min); T_evap_ub = 50.* (T_evap_variable <= T_evap_max)
    
    # relaxed_integer_variable_lb = 20.*(relaxed_integer_variable >= -.49)
    # relaxed_integer_variable_ub = 20.*(relaxed_integer_variable <= 1.49)

    PLR_bound = 200.*(cooling_delivered_variable/(Q_delivered_max*unit_coeff) == 0. or \
                     cooling_delivered_variable/(Q_delivered_max*unit_coeff) >= 0.2) 
    PLR_bound.name = 'PLR_bound'

    T_out_lb.name, T_out_ub.name = 'T_out_lb','T_out_ub'
    T_return_lb.name, T_return_ub.name = 'T_return_lb','T_return_ub'
    T_supply_lb.name, T_supply_ub.name = 'T_supply_lb','T_supply_ub'
    flow_lb.name, flow_ub.name = 'flow_lb','flow_ub'
    T_evap_lb.name, T_evap_ub.name = 'T_evap_lb', 'T_evap_ub`'

    constraints = [
            # T_out_lb, 
            # T_out_ub, 
            T_return_lb,
            T_return_ub,
            T_supply_lb,
            T_supply_ub,
            # cooling_bound,
            # PLR_bound,
            flow_lb, flow_ub,
            T_evap_lb, T_evap_ub,
            # relaxed_integer_variable_lb, relaxed_integer_variable_ub
            ]
    
    # PROBLEM DEFINITION
    loss = PenaltyLoss(loss_list, constraints)
    problem = Problem([cl_system], loss)
    #%% Dataloaders
    num_data = 30000; num_train_data = 24000; batch_size = 10000
    # num_data = 10000; num_train_data = 8000; batch_size = 2000
    T_supply_t = torch.rand(num_data, 1, M).uniform_(T_supply_min, T_supply_max)
    # T_supply_t = T_supply_t.expand(-1, -1 , M)
    # T_return_t = T_supply_t.mean(-1, keepdim=True).clone()+6
    T_return_t = T_supply_t.mean(-1, keepdim=True).uniform_(T_supply_max, T_return_max)
    # T_return_t = torch.rand(num_data, 1, 1).uniform_(T_return_min, T_return_max)

    _, loads_t_1d = utils.generate_datacenter_load(number_of_days=4000, sampling_time=Ts, day_baseline=800)
    # loads_t = loads_t_1d.unfold(0,nsteps+1,1)
    # loads_t = loads_t[:num_data,:].unsqueeze(-1)
    loads_t = loads_t_1d[:num_data*(nsteps+1)].reshape(num_data,nsteps+1,1)*unit_coeff
    # train_data = DictDataset({'T_return': T_return_t[:num_train_data].to(device),
    #                         'T_supply': T_supply_t[:num_train_data].to(device),
    #                         #    'T_out': T_out_t[:num_train_data],
    #                         'load': loads_t[:num_train_data].to(device)}, name='train')  # Split conditions into train and dev
    # dev_data = DictDataset({'T_return': T_return_t[num_train_data:].to(device),
    #                         'T_supply': T_supply_t[num_train_data:].to(device),
    #                         #    'T_out': T_out_t[num_train_data:],
    #                         'load': loads_t[num_train_data:].to(device)}, name='dev')
    train_data = DictDataset({'T_supply_and_return':torch.cat((T_supply_t[:num_train_data].to(device),
                                                               T_return_t[:num_train_data].to(device)),dim=-1),
                            #    'T_out': T_out_t[:num_train_data],
                            'load': loads_t[:num_train_data].to(device)}, name='train')  # Split conditions into train and dev
    dev_data = DictDataset({'T_supply_and_return': torch.cat((T_supply_t[num_train_data:].to(device),
                                                              T_return_t[num_train_data:].to(device)),dim=-1),
                            'load': loads_t[num_train_data:].to(device)}, name='dev')
    # instantiate data loaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, collate_fn=train_data.collate_fn)
    dev_loader = torch.utils.data.DataLoader(dev_data, batch_size=batch_size, collate_fn=dev_data.collate_fn)

    #%% Optimizer
    optimizer = torch.optim.AdamW(cl_system.parameters(), lr=0.001, weight_decay=0.0)
    trainer = Trainer(
        problem.to(device),
        train_loader, dev_loader,
        optimizer=optimizer,
        epochs=200,
        train_metric='train_loss',
        dev_metric='dev_loss',
        eval_metric='dev_loss',
        warmup=20,
        patience=80,
        clip=10.,
        device=device
    )
    best_model = trainer.train()    # start optimization
    trainer.model.load_state_dict(best_model) # load best 

    #%% TEST MODEL
    # s_length = 300 # number of simulation steps
    # load_test = loads_t_1d[0:s_length+nsteps].unsqueeze(0).unsqueeze(-1).clone()
    torch.set_default_device('cpu')
    cl_system.to('cpu')
    cl_system.nodes[-2].callable.h = torch.tensor(Ts, device='cpu')
    torch.manual_seed(seed)
    t, load_test = utils.generate_datacenter_load(number_of_days=2, sampling_time=Ts, 
                                                  night_baseline=300,
                                                  day_baseline=450,
                                                  ramp_hours=4)
    load_test = load_test.reshape(1,-1,1)*unit_coeff
    s_length = load_test.size(1)
    # sim_data = {
    #         'T_supply': torch.ones(1,1,M)*8.,
    #         'T_return': torch.ones(1,1,1)*8.,
    #         'load': load_test}
    sim_data = {
            'T_supply_and_return': torch.ones(1,1,M+1)*8.,
            # 'T_return': torch.ones(1,1,1)*8.,
            'load': load_test}

    cl_system.nsteps = s_length
    cl_system.eval()
    with torch.no_grad():
        trajectories_cuda = cl_system(sim_data) # simulate
    trajectories = {k: v.cpu() for k, v in trajectories_cuda.items()}

    cooling_in_simulation =  system.get_cooling_delivered_per_chiller(integer_status=trajectories['integer'],
                                                                mass_flow=trajectories['flow'],
                                                                T_return=trajectories['T_return'][0,:,:],
                                                                T_supply=trajectories['T_supply'][0,:,:])
    
    outlet_temp_in_simulation = system.get_outlet_temperature(integer_status=trajectories['integer'], 
                                                    mass_flow=trajectories['flow'], T_supply=trajectories['T_supply'][0,:,:])

    chiller_power_simulation = system.get_chiller_power_PLR(integer_status=trajectories['integer'],
                                                    mass_flow=trajectories['flow'],
                                                    T_return=trajectories['T_return'][:,:,:],
                                                    T_supply=trajectories['T_supply'][:,:,:],
                                                    Q_rated=Q_delivered_max*unit_coeff)

    pump_power_simulation = system.get_pump_consumption(mass_flow=trajectories['flow'], 
                                                        integer_status=trajectories['integer'], 
                                                        exponent=exponent)

    cost = torch.sum(pump_power_simulation + chiller_power_simulation + c * trajectories['integer'].sum(-1,keepdim=True)) *(Ts/3600)
    control_RMSE = torch.sqrt(torch.mean((trajectories['load'][:,:s_length,:] - cooling_in_simulation)**2))

#%%
    fig, axes = plt.subplots(4, 2, figsize=(12, 10))
    axes = axes.flatten()
    # 1) Return vs Out temperature
    axes[0].plot(trajectories['T_supply'][0,:s_length,0].cpu(), label="T_supply1", c='indigo', linestyle='-', alpha=0.7)
    axes[0].plot(trajectories['T_supply'][0,:s_length,1].cpu(), label="T_supply2", c='green', linestyle='-', alpha=0.7)
    axes[0].plot(trajectories['T_evap'][0,:,0].cpu(), color="indigo", linestyle=":", alpha=1, label="T_evap 1")
    axes[0].plot(trajectories['T_evap'][0,:,1].cpu(), color="green", linestyle=":", alpha=1, label="T_evap 2")
    axes[0].plot(torch.ones(s_length).cpu()*T_supply_max, 'k--', label='Evap bounds'); axes[0].plot(torch.ones(s_length).cpu()*T_supply_min, 'k--')
    axes[0].set_xlabel("Timestep"); axes[0].set_ylabel("Temperature [°C]")
    axes[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.4), ncol=3); axes[0].grid(True)

    # # 2) Cooling delivered
    axes[1].plot(trajectories['load'][0,:s_length,:].cpu(), 'k--' , label="Q_demand")
    axes[1].plot(cooling_in_simulation[0,:,:].sum(-1, keepdim=True).cpu(), label="Q_delivered")
    axes[1].set_xlabel("Timestep")
    axes[1].set_ylabel(f"Cooling [{units}]")
    axes[1].legend()
    axes[1].grid(True)

    # # 3) Outlet vs retrun temperature
    axes[2].plot(outlet_temp_in_simulation[0,:,:].cpu(), label="T_out", c='g')
    axes[2].plot(torch.ones(s_length).cpu()*T_min, 'g--' ,label="T_out bounds"); 
    axes[2].plot(torch.ones(s_length).cpu()*T_max, 'g--')

    axes[2].plot(trajectories['T_return'][0,:s_length,0].cpu(), label="T_return", c='r')
    axes[2].plot(torch.ones(s_length).cpu()*T_return_min, 'r:' ,label="T_return bounds"); 
    axes[2].plot(torch.ones(s_length).cpu()*T_return_max, 'r:'); 

    axes[2].set_xlabel("Timestep")
    axes[2].set_ylabel("Temperature [°C]")
    axes[2].legend()
    axes[2].grid(True)

    # # 4) Chiller energy consumption
    axes[3].plot(chiller_power_simulation[0,:,:].cpu(), label="P_chiller")
    axes[3].plot(pump_power_simulation[0,:,:].cpu(), label="P_pump")
    axes[3].set_xlabel("Timestep")
    axes[3].set_ylabel(f"Chiller [{units}]")
    axes[3].legend()
    axes[3].grid(True)

    # # 5) Flow rates
    axes[4].plot(trajectories['flow'][0,:,0].cpu(), label="Mass flow 1")
    axes[4].plot(trajectories['flow'][0,:,1].cpu(), '--', label="Mass flow 2")
    axes[4].plot(torch.ones(s_length).cpu()*flow_min, 'k:'); axes[4].plot(torch.ones(s_length).cpu()*flow_max,'k:', label='bounds')
    axes[4].set_xlabel("Timestep")
    axes[4].set_ylabel("Mass flowrates [kg/s]")
    axes[4].legend()
    axes[4].grid(True)

    axes[5].plot(trajectories['integer'][0,:,0].cpu(), label='Chiller 1 Status')
    axes[5].plot(trajectories['integer'][0,:,1].cpu(), '--',label='Chiller 2 Status')
    axes[5].plot(trajectories['relaxed_integer'][0,:,0].cpu(), '--',label='Relaxed Chiller 1 Status')
    # axes[5].plot(trajectories['relaxed_integer'][0,:,1].cpu(), '--',label='Relaxed Chiller 2 Status')
    axes[5].set_ylabel("Integer status [-]")
    axes[5].set_xlabel("Timestep")
    # axes[5].set_yticks([0,1], labels=['OFF','ON'])
    axes[5].legend()
    axes[5].grid(True)
    PLR = cooling_in_simulation.cpu()/(Q_delivered_max*unit_coeff)
    COP = system.a+system.b*PLR+system.c*PLR**2
    axes[6].plot(PLR[0,:,0].cpu(), label='PLR Chiller 1')
    axes[6].plot(PLR[0,:,1].cpu(), label='PLR Chiller 1')
    axes[6].set_ylabel("PLR [-]")
    axes[6].set_xlabel("Timestep")
    # axes[6].set_yticks([0,1], labels=['OFF','ON'])
    axes[6].legend()
    axes[6].grid(True)
    
    axes[7].plot(COP[0,:,0].cpu(), label='COP Chiller 1')
    axes[7].plot(COP[0,:,1].cpu(), label='COP Chiller 1')
    axes[7].set_ylabel("COP [-]")
    axes[7].set_xlabel("Timestep")
    # axes[7].set_yticks([0,1], labels=['OFF','ON'])
    axes[7].legend()
    axes[7].grid(True)

    axes[1].set_title(f'Total cost of operation:  {cost.item():.1f} \n Tracking RMSE: {control_RMSE.item():.1f}')
    print('Total cost of operation: ', cost.item())

    plt.savefig('/home/bold914/chiller_staging/plots/ct_chiller_control_MIDPC.png')

    n_violations = 0
    tolerance = 2 # tolerance for violation [kW]
    for i in range(s_length):
        if not cooling_in_simulation[0,i,0] + tolerance*unit_coeff >= trajectories['load'][0,i,0]:
            n_violations += 1
    print("Number of violations (Q_delivered < load): ", n_violations)

# %%
