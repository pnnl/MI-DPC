#%%
import utils; import torch; import init
from chiller_system import ChillerSystem
from neuromancer.system import Node, SystemPreview
from neuromancer.constraint import variable
from neuromancer.loss import PenaltyLoss
from neuromancer.problem import Problem
from neuromancer.dataset import DictDataset
from neuromancer.dynamics import integrators
from neuromancer.trainer import Trainer
from neuromancer.loggers import BasicLogger
import time
torch.manual_seed(202)

class MIDPC_policy():
        def __init__(self, nsteps, load_path, measure_inference_time = False):
            self.nsteps = nsteps
            self.load_path = load_path
            self.cl_system = torch.load(load_path, weights_only=False, map_location=torch.device('cpu'))
            self.integer_relaxed_node = self.cl_system.nodes[0]
            self.integer_node = self.cl_system.nodes[1]
            self.T_evap_node = self.cl_system.nodes[2]
            self.flow_node = self.cl_system.nodes[3]
            self.measure_inference_time = measure_inference_time
            print(self.cl_system)
        def __call__(self, T_supply=None, T_return=None, load=None):
                input_dict = {
                       'T_supply_and_return': torch.cat((T_supply,T_return), dim=-1).reshape(1,-1),
                       'load': load.reshape(1,-1)
                              }
                
                with torch.no_grad():
                        if not self.measure_inference_time:
                                relaxed_integer = self.integer_relaxed_node(input_dict) # dict - key: 'relaxed_integer'
                                integer = self.integer_node(relaxed_integer) # dict - key: 'integer'
                                T_evap = self.T_evap_node(input_dict) # dict - key: 'T_evap'
                                mass_flow = self.flow_node(input_dict) # dict - key: 'flow'
                     
                        elif self.measure_inference_time:
                                [self.integer_relaxed_node(input_dict) for warmup in range(3)] # Warmup
                                start_time = time.perf_counter()
                                relaxed_integer = self.integer_relaxed_node(input_dict) # dict - key: 'relaxed_integer'
                                integer = self.integer_node(relaxed_integer) # dict - key: 'integer'
                                T_evap = self.T_evap_node(input_dict) # dict - key: 'T_evap'
                                mass_flow = self.flow_node(input_dict) # dict - key: 'flow'
                                inference_time = (time.perf_counter() - start_time)  # unit [seconds]
                output = {}

                if self.measure_inference_time:
                        output['inference_time'] = torch.tensor(inference_time).view(1,1,1)
                output['integer'] = integer['integer'].unsqueeze(0)
                output['relaxed_integer'] = relaxed_integer['relaxed_integer'].unsqueeze(0)
                output['flow'] = mass_flow['flow'].unsqueeze(0)
                output['T_evap'] = T_evap['T_evap'].unsqueeze(0)
                return output

def relaxed_binary(x, slope=5.0, threshold=0.5):
        logits = slope * (x - threshold)
        sig = torch.sigmoid(logits)
        return (x > threshold).float() + (sig - sig.detach())
def round_fn(x):
        return torch.cat((relaxed_binary(x), torch.ones((x.size(0), 1), requires_grad=False)), dim=-1)



if __name__=='__main__':
    layer_norm = False
    affine_norm = False
    spectral_norm = False
    # exponent = 2
    load_min = 0
    load_max = 1000
    nsteps = 100
    Ts = init.Ts
    mins = [init.T_supply_min] * init.M + [init.T_return_min] * 1 + [load_min] * (nsteps) 
    maxs = [init.T_supply_max] * init.M + [init.T_return_max] * 1 + [load_max] * (nsteps) 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_default_device(device=device)
    system = ChillerSystem(M=init.M, Ts=Ts, C_r=init.C_r, C_i=init.C_i, c_p=init.c_p,
                            a=init.a, b=init.b, c=init.c, gamma=init.gamma, exponent=init.exponent)
    integrator = integrators.RK4(system, h=torch.tensor(Ts))

    net_flow = utils.customMPL(insize=1*init.M+1+1*(nsteps)+0, outsize=init.M, hsizes=[120, 120, 120, 120],
                                nonlin=torch.nn.Mish(), layer_norm=layer_norm, affine=affine_norm, dropout_prob=0.0,
                                mins=mins, maxs=maxs, u_min=init.flow_min, u_max=init.flow_max, 
                                clipping=False, spectral_norm=spectral_norm)

    net_evap = utils.customMPL(insize=1*init.M+1+1*(nsteps)+0, outsize=init.M, hsizes=[120, 120, 120, 120], 
                               nonlin=torch.nn.Mish(), layer_norm=layer_norm, affine=affine_norm, dropout_prob=0.0,
                               mins=mins, maxs=maxs, u_min=init.T_evap_min, u_max=init.T_evap_max, 
                               clipping=False, spectral_norm=spectral_norm)

    net_integer = utils.customMPL(insize=1*init.M+1+1*(nsteps)+0, outsize=init.M-1, hsizes=[120, 120, 120, 120],
                                   nonlin=torch.nn.Tanh(), layer_norm=layer_norm, affine=affine_norm, dropout_prob=0.,
                                   mins=mins, maxs=maxs, u_min=-0.49, u_max=1.49, 
                                   clipping=False, spectral_norm=spectral_norm)

    # NEUROMANCER NODES
    dynamics_node = Node(integrator,
                        input_keys=['T_supply_and_return', 'integer', 'flow', 'T_evap', 'load'],
                        output_keys=['T_supply_and_return'],
                        name='system_dynamics')

    policy_integer_node = Node(net_integer,
                    input_keys=['T_supply_and_return','load'],
                    output_keys=['relaxed_integer'],
                    name='policy_integer')
    

    # round_fn = lambda x: torch.cat((relaxed_binary(x), torch.ones((x.size(0), 1), requires_grad=False)), dim=-1)# bound integers to avoid training instability
    rounding_node = Node(round_fn, input_keys=['relaxed_integer'], output_keys=['integer'], name='soft_rounding')

    policy_flow_node = Node(net_flow,
                    input_keys=['T_supply_and_return','load'],
                    output_keys=['flow'],
                    name='policy_flow')

    policy_evap_node = Node(net_evap,
                    input_keys=['T_supply_and_return','load'],
                    output_keys=['T_evap'],
                    name='policy_evap')

    # NEUROMANCER SYSTEM
    cl_system = SystemPreview([policy_integer_node, rounding_node, policy_evap_node, policy_flow_node, dynamics_node],
                                nsteps=nsteps, name='cl_system', pad_mode='reflect', pad_constant = 300,
                                preview_keys_map={'load': ['policy_flow', 'policy_integer', 'policy_evap']},
                                preview_length={'load': nsteps-1})

    test_output = cl_system({ 'T_supply_and_return': torch.rand(1,1,init.M+1),
                              'load': torch.rand(1,nsteps+1,1),})
    
    #%%
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
    
    #%% CONTROL OBJECTIVES
    chiller_loss = 0.01 * ((system.get_chiller_power_PLR(integer_status=integer_variable, 
                                                    Q_rated=init.Q_delivered_max,
                                                    mass_flow=flow_variable,
                                                    T_return=T_return_variable,
                                                    T_supply=T_supply_variable) == 0.))
    
    pump_loss = 0.01 * ((system.get_pump_consumption(mass_flow=flow_variable, 
                        integer_status=integer_variable) == 0.))
    
    c = init.delta_penalty # Switching cost coefficient
    switching_loss = c*((integer_variable[:, 1:, :] == integer_variable[:, :-1, :])^2)
    # switching_loss = c*((relaxed_integer_variable * (1-relaxed_integer_variable) == 0.)^2)
    # switching_loss = c*((relaxed_integer_variable[:, 1:, :] == relaxed_integer_variable[:, :-1, :])^2)
    # switching_loss = c*(integer_variable == 0)

    chiller_loss.name = 'chiller_loss'; pump_loss.name = 'pump_loss'; switching_loss.name = 'switching_loss'
    loss_list = [
                chiller_loss,
                pump_loss,
                switching_loss, 
                ]
    #%% CONSTRAINTS
    # T_out_lb, T_out_ub = 10.*(T_out_variable >= init.T_min), 10.*(T_out_variable <= init.T_max)
    T_return_lb  = 10.*(T_supply_and_return_variable[:,:,init.M:] >= init.T_return_min) # States
    T_return_ub = 10.*(T_supply_and_return_variable[:,:,init.M:] <= init.T_return_max)
    T_supply_lb = 10.*(T_supply_and_return_variable[:,:,:init.M] >= init.T_supply_min) 
    T_supply_ub = 10.*(T_supply_and_return_variable[:,:,:init.M] <= init.T_supply_max)
    
    cooling_bound = 0.1* (cooling_delivered_variable[:,:nsteps,:] >= load_variable[:,:nsteps,:]) # Cooling constr
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

    constraints = [
            T_return_lb, T_return_ub,
            T_supply_lb, T_supply_ub,
            flow_lb, flow_ub,
            T_evap_lb, T_evap_ub,
            relaxed_integer_variable_lb, relaxed_integer_variable_ub,
            # cooling_bound,
            ]
    
    # PROBLEM DEFINITION
    loss = PenaltyLoss(loss_list, constraints)
    problem = Problem([cl_system], loss)
    #%% Dataloaders
    num_data = 30000; num_train_data = 20000; batch_size = 10000
    T_supply_t = torch.rand(num_data, 1, init.M).uniform_(init.T_supply_min, init.T_supply_max)
    T_return_t = T_supply_t.mean(-1, keepdim=True).uniform_(init.T_supply_max, init.T_return_max)

    _, loads_t_1d = utils.generate_datacenter_load(number_of_days=14000, sampling_time=Ts, 
                        ramp_hours=init.ramp_hours, day_baseline=init.day_baseline, 
                        night_baseline=init.night_baseline)
    
    # loads_t = loads_t_1d.unfold(0,nsteps+1,1)
    # loads_t = loads_t[:num_data,:].unsqueeze(-1)
    loads_t = loads_t_1d[:num_data*(nsteps+1)].reshape(num_data,nsteps+1,1)
    
    train_data = DictDataset({'T_supply_and_return':torch.cat((T_supply_t[:num_train_data].to(device),
                                                               T_return_t[:num_train_data].to(device)),dim=-1),
                            'load': loads_t[:num_train_data].to(device)}, name='train')  # Split conditions into train and dev
    
    dev_data = DictDataset({'T_supply_and_return': torch.cat((T_supply_t[num_train_data:].to(device),
                                                              T_return_t[num_train_data:].to(device)),dim=-1),
                            'load': loads_t[num_train_data:].to(device)}, name='dev')
    # instantiate data loaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, collate_fn=train_data.collate_fn)
    dev_loader = torch.utils.data.DataLoader(dev_data, batch_size=batch_size, collate_fn=dev_data.collate_fn)
    # logger = BasicLogger(stdout=['train_loss','dev_loss'],verbosity=1)
    #%% Optimizer
    optimizer = torch.optim.AdamW(cl_system.parameters(), lr=0.001, weight_decay=0.0)
    trainer = Trainer(
        problem.to(device),
        train_loader, dev_loader,
        optimizer=optimizer,
        epochs=1000,
        train_metric='train_loss',
        dev_metric='dev_loss',
        eval_metric='dev_loss',
        warmup=20,
        patience=500,
        clip=torch.inf, 
        lr_scheduler=False,
        device=device,
        # logger=logger,
    )

    best_model = trainer.train()    # start optimization
    trainer.model.load_state_dict(best_model) # load best 
    torch.save(cl_system, f'results/MIDPC/policies/N_{nsteps}.pt')
   