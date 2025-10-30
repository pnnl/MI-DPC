#%%
import utils; import torch
from init import SystemParameters
from chiller_system import ChillerSystem
from neuromancer.system import Node, SystemPreview
from neuromancer.constraint import variable
from neuromancer.loss import PenaltyLoss
from neuromancer.problem import Problem
from neuromancer.dataset import DictDataset
from neuromancer.dynamics import integrators
from neuromancer.trainer import Trainer
from neuromancer.loggers import BasicLogger
from argparse import ArgumentParser
import time
LAMBDAS=[0, 1, 10, 100]
cl_systems = {}
if __name__=='__main__':
        for LAMBDA in LAMBDAS:
            init = SystemParameters()
            def relaxed_binary(x, slope=1.0, threshold=0.5):
                    logits = slope * (x - threshold)
                    sig = torch.sigmoid(logits)
                    return (x > threshold).float() + (sig - sig.detach())
            def round_fn(x):
                    return torch.cat((relaxed_binary(x), torch.ones((x.size(0), 1), requires_grad=True)), dim=-1)
            system_filter = ChillerSystem(init = init)
            def load_filter(x):
                    return system_filter.apply_load_filter(x[:,[0]])

            torch.manual_seed(202)
            nsteps = 20
            Ts = 180
            init = SystemParameters(M=2)
            layer_norm = False; affine_norm = False; spectral_norm = False
            # exponent = 2
            load_min = 0; load_max = (init.Q_delivered_max*init.M)*0.75
            mins = [init.T_supply_min] * init.M + [init.T_return_min] * 1 + [load_min] * (nsteps+1) 
            maxs = [init.T_supply_max] * init.M + [init.T_return_max] * 1 + [load_max] * (nsteps+1) 
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            torch.set_default_device(device=device)
            system = ChillerSystem(init=init)
        

            integrator = integrators.RK4(system, h=torch.tensor(Ts))
                            # In Size: T_supply + T_return + load + filtered_load+ relaxed_integer
            net_flow = utils.customMPL(
                                            insize=1*init.M+1+1*(nsteps)+1, outsize=init.M, hsizes=[20, 20, 20],
                                            nonlin=torch.nn.ReLU(), layer_norm=layer_norm, affine=affine_norm, dropout_prob=0.0,
                                            mins=mins, maxs=maxs, u_min=init.flow_min, u_max=init.flow_max, 
                                            clipping=False, spectral_norm=spectral_norm)

            net_evap = utils.customMPL(
                                    insize=1*init.M+1+1*(nsteps)+1, outsize=init.M, hsizes=[20, 200, 20], 
                                    nonlin=torch.nn.ReLU(), layer_norm=layer_norm, affine=affine_norm, dropout_prob=0.0,
                                    mins=mins, maxs=maxs, u_min=init.T_evap_min, u_max=init.T_evap_max, 
                                    clipping=False, spectral_norm=spectral_norm)

            net_integer = utils.customMPL(insize=1*init.M+1+1*(nsteps)+1+0, outsize=init.M-1, hsizes=[200, 200, 200],
                                            nonlin=torch.nn.ReLU(), layer_norm=layer_norm, affine=affine_norm, dropout_prob=0.,
                                            mins=mins, maxs=maxs, u_min=0., u_max=1., 
                                            clipping=False, spectral_norm=spectral_norm)

            # NEUROMANCER NODES
            load_filter_node = Node(load_filter, input_keys=['load'], output_keys=['filtered_load'])
            load_filter_node({'load': torch.zeros(1,1, device=device)})

            dynamics_node = Node(integrator,
                                    input_keys=['T_supply_and_return', 'integer', 'flow', 'T_evap', 'filtered_load'],
                                    output_keys=['T_supply_and_return'],
                                    name='system_dynamics')
            
            policy_integer_node = Node(net_integer,
                            input_keys=['T_supply_and_return','load', 'filtered_load'],
                            output_keys=['relaxed_integer'],
                            name='policy_integer')
            

            rounding_node = Node(round_fn, input_keys=['relaxed_integer'], output_keys=['integer'], name='soft_rounding')

            policy_flow_node = Node(net_flow,
                            input_keys=['T_supply_and_return','load', 'filtered_load'],
                            output_keys=['flow'],
                            name='policy_flow')

            policy_evap_node = Node(net_evap,
                            input_keys=['T_supply_and_return','load', 'filtered_load'],
                            output_keys=['T_evap'],
                            name='policy_evap')

            # NEUROMANCER SYSTEM
            cl_system = SystemPreview([load_filter_node, policy_integer_node, rounding_node, policy_evap_node, policy_flow_node , dynamics_node],
                                            nsteps=nsteps, name='cl_system', pad_mode='circular', pad_constant = 300,
                                            preview_keys_map={'load': ['policy_flow', 'policy_integer', 'policy_evap']},
                                            preview_length={'load': nsteps-1})

            test_output = cl_system({ 'T_supply_and_return': torch.rand(1,1,init.M+1),
                                    'load': torch.rand(1,nsteps,1),})
            
            
            """ Variables
            States: T_return, T_supply
            Decisions: integer, flow, cooling_delivered
            External: load
            Scores: T_out, cooling_delivered
            """
            relaxed_integer_variable = variable('relaxed_integer') # Decision
            integer_variable = variable('integer') # Decisions
            flow_variable = variable('flow') # Decision
            T_evap_variable = variable('T_evap') # Decision
            load_variable = variable('load') # External
            filtered_load_variable = variable('filtered_load')
            T_supply_and_return_variable = variable('T_supply_and_return')
            T_return_variable = variable('T_supply_and_return')[:,:nsteps,init.M:] # No terminal state
            T_supply_variable = variable('T_supply_and_return')[:,:nsteps,:init.M] # No terminal state

            T_out_variable = system.get_outlet_temperature(integer_status=integer_variable, 
                                                            mass_flow=flow_variable,
                                                            T_supply=T_supply_variable) # State

            cooling_delivered_variable = system.get_cooling_delivered_per_chiller(integer_status=integer_variable,
                                                                            mass_flow=flow_variable,
                                                                            T_return=T_return_variable,
                                                                            T_supply=T_supply_variable) # Decisions
            
            chiller_loss =  ((system.get_chiller_power_PLR(integer_status=integer_variable, 
                                                            mass_flow=flow_variable,
                                                            T_return=T_return_variable,
                                                            T_supply=T_supply_variable) == 0.))
            
            pump_loss =  ((system.get_pump_consumption(mass_flow=flow_variable, 
                                    integer_status=integer_variable) == 0.))
            
            cooling_loss = 0.001*((torch.sum(cooling_delivered_variable,dim=-1,keepdim=True) == load_variable)^2.)
            c = 10.
            switching_loss = c*((integer_variable[:, 1:, :] == integer_variable[:, :-1, :])^2)

            binary_regularization = LAMBDA*((relaxed_integer_variable * (1-relaxed_integer_variable) == 0.)^2)

            chiller_loss.name = 'chiller_loss'; pump_loss.name = 'pump_loss'; switching_loss.name = 'switching_loss'
            cooling_loss.name = 'cooling_loss'
            loss_list = [
                            chiller_loss,
                            pump_loss,
                            cooling_loss,
                            switching_loss,
                            binary_regularization
                            ]
            # T_out_lb, T_out_ub = 10.*(T_out_variable >= init.T_min), 10.*(T_out_variable <= init.T_max)
            T_return_lb  = 10.*(T_supply_and_return_variable[:,:,init.M:] >= init.T_return_min) # States
            T_return_ub = 10.*(T_supply_and_return_variable[:,:,init.M:] <= init.T_return_max)
            T_supply_lb = 10.*(T_supply_and_return_variable[:,:,:init.M] >= init.T_supply_min) 
            T_supply_ub = 10.*(T_supply_and_return_variable[:,:,:init.M] <= init.T_supply_max)
            
            cooling_bound = 0.5 * (torch.sum(cooling_delivered_variable[:,:nsteps,:],dim=-1,keepdim=True) + init.tolerance >= load_variable[:,:nsteps,:]) # Cooling constr
            cooling_bound.name='cooling_bound'

            flow_lb = 10.*(flow_variable >= init.flow_min); flow_ub = 10.* (flow_variable <= init.flow_max) # Decisions
            T_evap_lb = 10.*(T_evap_variable >= init.T_evap_min); T_evap_ub = 10.* (T_evap_variable <= init.T_evap_max)
            
            relaxed_integer_variable_lb = 5.*(relaxed_integer_variable >= 0.)
            relaxed_integer_variable_ub = 5.*(relaxed_integer_variable <= 1.)

            # T_out_lb.name, T_out_ub.name = 'T_out_lb','T_out_ub'
            T_return_lb.name, T_return_ub.name = 'T_return_lb','T_return_ub'
            T_supply_lb.name, T_supply_ub.name = 'T_supply_lb','T_supply_ub'
            flow_lb.name, flow_ub.name = 'flow_lb','flow_ub'
            T_evap_lb.name, T_evap_ub.name = 'T_evap_lb', 'T_evap_ub`'

            
            relaxed_integer_variable_lb = 15.*(relaxed_integer_variable >= 0.)
            relaxed_integer_variable_ub = 15.*(relaxed_integer_variable <= 1.)

            constraints = [
                    T_return_lb, T_return_ub,
                    T_supply_lb, T_supply_ub,
                    flow_lb, flow_ub,
                    T_evap_lb, T_evap_ub,
                    relaxed_integer_variable_lb, relaxed_integer_variable_ub,
                    ]
            
            # PROBLEM DEFINITION
            loss = PenaltyLoss(loss_list, constraints)
            problem = Problem([cl_system], loss)
            
            num_data = 4000; num_train_data = 3000; batch_size = 1000
            T_supply_t = torch.rand(num_data, 1, init.M).uniform_(init.T_supply_min, init.T_supply_max)
            T_return_t = T_supply_t.mean(-1, keepdim=True).uniform_(init.T_supply_max, init.T_return_max)

            _, loads_t_1d = utils.generate_datacenter_load(
                                                            number_of_days=14000, 
                                                            sampling_time=Ts, 
                                                            ramp_hours=init.ramp_hours,
                                                            f_day=5, f_night=6, 
                                                            day_baseline=init.day_baseline, 
                                                            night_baseline=init.night_baseline,
                                                            osc_night_amp=20, osc_day_amp=20,
                                                            noise_scale=5
                                                            )
            
            loads_t = loads_t_1d[:num_data*(nsteps)].reshape(num_data,nsteps,1)
            
            train_data = DictDataset({'T_supply_and_return':torch.cat((T_supply_t[:num_train_data].to(device),
                                                                    T_return_t[:num_train_data].to(device)),dim=-1),
                                    'load': loads_t[:num_train_data].to(device)}, name='train')  # Split conditions into train and dev
            
            dev_data = DictDataset({'T_supply_and_return': torch.cat((T_supply_t[num_train_data:].to(device),
                                                                    T_return_t[num_train_data:].to(device)),dim=-1),
                                    'load': loads_t[num_train_data:].to(device)}, name='dev')
            # instantiate data loaders
            train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, collate_fn=train_data.collate_fn)
            dev_loader = torch.utils.data.DataLoader(dev_data, batch_size=batch_size, collate_fn=dev_data.collate_fn)
            print(f'Training MIDPC policy for LAMBDA={LAMBDA}')
            optimizer = torch.optim.Adam(cl_system.parameters(), lr=0.001, weight_decay=0.00)
            trainer = Trainer(
                    problem.to(device),
                    train_loader, dev_loader,
                    optimizer=optimizer,
                    epochs=20,
                    train_metric='train_loss',
                    dev_metric='dev_loss',
                    # eval_metric='dev_loss',
                    warmup=20,
                    patience=100,
                    clip=100., 
                    lr_scheduler=False,
                    device=device,
                    epoch_verbose=1,
            )
            best_model = trainer.train()    # start optimization
            trainer.model.load_state_dict(best_model) # load best 

            cl_systems[f'{LAMBDA}'] = cl_system
        print('DONE')
#%%
import matplotlib.pyplot as plt
import matplotlib
plt.style.use('tableau-colorblind10')
# plt.set_cmap("viridis") 

matplotlib.use("pgf")
plt.rcParams.update({
        "pgf.texsystem": "pdflatex",
        "text.usetex": True,
        "font.family": "serif",
        "font.size": 10,
        "pgf.rcfonts": False,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8
    })
x = torch.empty(40)
x[:x.size(0)//2] = 150
x[x.size(0)//2:] = 500
until = 25
test_data = {'T_supply_and_return': torch.ones(1,1,init.M+1)* 8.,
             'load': x.view(1,-1,1)}

fig1, ax = plt.subplots(1,1, figsize=(3.5,2.),sharex=False)

for LAMBDA in LAMBDAS:
    cl_systems[f'{LAMBDA}'].eval()
    cl_systems[f'{LAMBDA}'].nsteps = x.size(0)
    cl_systems[f'{LAMBDA}'].pad_mode = 'replicate'
    with torch.no_grad():
        trajectories = cl_systems[f'{LAMBDA}'].forward(test_data)
    print(trajectories['relaxed_integer'].shape)
    ax.plot(trajectories['relaxed_integer'][0,:until,:].cpu(), alpha=.9,
     label=f"$\Lambda = {LAMBDA}$")

ax.plot(0.5*torch.ones_like(x)[:until].cpu(), 'k:')

fig1.tight_layout(pad=0.0)
ax.set_yticks([0, 0.5, 1])
ax.set_xlim([0, until-1])
ax.set_ylim([0, 1])
ax.legend(framealpha=1.0, edgecolor='gray',fancybox=False)
ax.set_xlabel('Time step --- $k$')
ax.set_ylabel('$\\tilde\delta(k)$')
ax.grid()
fig1.show()
fig1.savefig(f'experiment_plot.pdf', bbox_inches='tight',pad_inches=0.05,transparent=True)
fig1.savefig(f'experiment_plot.pgf', bbox_inches='tight', pad_inches=0.05,transparent=True)
