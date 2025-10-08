#%%
import init; import torch
import matplotlib.pyplot as plt

def plot_chiller_data(data, save_path=None):
    ls = ['-','--']
    clrs=['tab:blue', 'tab:red']
    s_length = data['chiller_status'].size(1)
    rng = range(data['T_evap'].size(-1))
    
    # # # 1) T_evap vs T_supply
    fig, axes = plt.subplots(5, 2, figsize=(12, 10))
    axes = axes.flatten()
    axes[0].plot(torch.ones(s_length)*init.T_supply_max, 'k--', label='Bounds')
    axes[0].plot(torch.ones(s_length)*init.T_supply_min, 'k--') 
    [axes[0].plot(data['T_supply'][0,:-1,i], 
                    label=[f"T_supply{i+1}"],
                    linestyle=ls[i],
                    alpha=0.7) for i in rng]

    [axes[0].plot(data['T_evap'][0,:,i], label=[f"T_evap{i+1}"], linestyle=ls[i],
                   alpha=0.7) for i in rng]
    axes[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.5), ncol=3); axes[0].grid(True)
    axes[0].set_xlabel("Timestep"); axes[0].set_ylabel("Temperature [°C]")

    # # # 2) Load vs Q_delivered
    axes[1].plot(data['load'][0,:s_length,:].cpu(), 'k--' , label="Q_demand")
    axes[1].plot(data['Q_delivered'][0,:,:].sum(-1, keepdim=True).cpu(), label="Q_delivered")
    axes[1].set_xlabel("Timestep"); axes[1].set_ylabel(f"Cooling [kW]")
    axes[1].legend(); axes[1].grid(True)

    # # # 3) Outlet vs retrun temperature
    axes[2].plot(data['T_out'][0,:,:].cpu(), label="T_out", c='b')
    axes[2].plot(torch.ones(s_length).cpu()*init.T_min, 'b:' ,label="T_out bounds"); 
    axes[2].plot(torch.ones(s_length).cpu()*init.T_max, 'b:')
    axes[2].set_xlabel("Timestep"); axes[2].set_ylabel("Temperature [°C]")
    axes[2].grid(True);    axes[2].legend()

    axes[3].plot(data['P_chiller'][0,:,:].cpu(), label=[f'P_chiller{i+1}' for i in rng])
    axes[3].set_xlabel("Timestep")
    axes[3].set_ylabel(f"Chiller [kW]")
    axes[3].legend()
    axes[3].grid(True)

    axes[4].plot(data['T_return'][0,:s_length,0].cpu(), label="T_return", c='r')
    axes[4].plot(torch.ones(s_length).cpu()*init.T_return_min, 'r:' ,label="T_return bounds"); 
    axes[4].plot(torch.ones(s_length).cpu()*init.T_return_max, 'r:'); 
    axes[4].set_xlabel("Timestep"); axes[4].set_ylabel("Temperature [°C]")
    axes[4].grid(True);    axes[4].legend()

    axes[5].plot(data['P_pump'][0,:,:].cpu(), label=[f'P_pump{i+1}' for i in rng])
    axes[5].grid(True);    axes[5].legend()
    axes[5].set_xlabel("Timestep")
    axes[5].set_ylabel(f"Pump [kW]")

    axes[6].plot(data['mass_flow'][0,:,:], label=[f'Chiller{i+1}' for i in rng])
    axes[6].plot(torch.ones(s_length)*init.flow_min, 'k:'); axes[6].plot(torch.ones(s_length)*init.flow_max,'k:', label='bounds')
    axes[6].set_xlabel("Timestep")
    axes[6].set_ylabel("Mass flowrates [kg/s]")
    axes[6].legend(); axes[6].grid(True)

    axes[7].plot(data['chiller_status'][0,:,:], label=[f'Chiller{i+1}' for i in rng])
    try:
        axes[7].plot(data['relaxed_integer'][0,:,:].cpu(), '--',label=[f'Chiller{i+1} relaxed' for i in range(data['relaxed_integer'].size(-1))])
    except:
        pass
    axes[7].set_ylabel("Chiller status [-]"); axes[7].set_xlabel("Timestep")
    axes[7].legend(); axes[7].grid(True)
    
    PLR = data['Q_delivered']/(init.Q_delivered_max)
    COP = init.a+init.b*PLR+init.c*PLR**2
    axes[8].plot(PLR[0,:,:].sum(-1,keepdim=True)/data['chiller_status'][0,:,:].sum(-1,keepdim=True), 
    # label=[f'Chiller{i+1}' for i in rng]
    )
    axes[8].set_ylabel("PLR [-]"); axes[8].set_xlabel("Timestep")
    axes[8].legend(); axes[8].grid(True)
    
    axes[9].plot(COP[0,:,:], label=[f'Chiller{i+1}' for i in rng])
    axes[9].set_ylabel("COP [-]"); axes[9].set_xlabel("Timestep")
    axes[9].legend(); axes[9].grid(True)

    n_violations = 0; tolerance = 2 # [kW]
    for i in range(s_length):
        if not data['Q_delivered'][0,i,:].sum(dim=-1, keepdim=True) + tolerance >= data['load'][0,i,0]:
            n_violations += 1
    cost = torch.sum(data['P_pump'].sum(dim=-1,keepdim=True) + data['P_chiller'].sum(dim=-1, keepdim=True) + \
                     0. *  data['chiller_status'].sum(-1,keepdim=True)) *(init.Ts/3600)
    
    control_RMSE = torch.sqrt(torch.mean((data['load'][:,:s_length,:] - data['Q_delivered'].sum(dim=-1, keepdim=True))**2))


    axes[1].set_title(f'Total cost of operation:  {cost.item():.1f} [kWh] \n  \
                        Tracking RMSE: {control_RMSE.item():.1f} [kW] \n \
                        Number of violations {n_violations} [-]') 
    # print('Total cost of operation: ', cost.item(), 'kWh')
    if save_path is not None:
        plt.savefig(save_path)

if __name__=='__main__':
    data = torch.load('results/RBC/data_N100.pt')
    plot_chiller_data(data=data)