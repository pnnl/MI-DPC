#%%
from init import SystemParameters;
import torch
import matplotlib.pyplot as plt
from utils import plot_chiller_data_nice, plot_chiller_data

from tabulate import tabulate
init = SystemParameters()

def get_kilowatthours(pump_power,chiller_power, Ts=180):
    return torch.sum(pump_power.sum(dim=-1,keepdim=True) + chiller_power.sum(dim=-1, keepdim=True)) *(Ts/3600)

def get_mean_COP(cooling, chiller_status):
    PLR = cooling.sum(-1,keepdim=True)/(init.Q_delivered_max*chiller_status.sum(-1,keepdim=True))
    return torch.mean(init.a+init.b*PLR+init.c*PLR**2)

def get_control_rmse(load, cooling):
    return torch.sqrt(torch.mean((load[:,:cooling.size(1),:] - cooling.sum(dim=-1, keepdim=True))**2))

if __name__=='__main__':
    Ts = 180
    
#     headers = ["Approach", "Metric", "10", "15", "20", "25", "30", "40"]

#     rows = [
#         ["MI-DPC (Sigmoid STE)", r"$\ell_\mathrm{mean}$", 6.7101, 4.7318, 4.0879, 3.9581, 3.8702, 3.8436],
#         ["", "RSM (\\%)", "15.13\\%", "7.25\\%", "1.81\\%", "1.77\\%", "0.60\\%", "--"],
#         ["", "MIT (s)\\tnote{1}", 0.0004, 0.0004, 0.0004, 0.0004, 0.0004, 0.0004],
#         ["", "NTP (-)", 82603, 84003, 85403, 86803, 88203, 91003],
#         ["", "TT (s)\\tnote{2}", 585.4, 664.9, 721.9, 789.2, 862.4, 868.6],
#         ["MI-DPC (Softmax STE)", r"$\ell_\mathrm{mean}$", 6.7930, 4.7851, 4.1327, 3.9483, 3.8721, 3.8656],
#         ["", "RSM (\\%)", "16.55\\%", "8.45\\%", "2.93\\%", "1.53\\%", "0.65\\%", "--"],
#         ["", "MIT (s)\\tnote{1}", 0.0004, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005],
#         ["", "NTP (-)", 83026, 84426, 85826, 87226, 88626, 91426],
#         ["", "TT (s)\\tnote{2}", 419.3, 557.8, 734.8, 589.9, 628.4, 621.5],
#     ]

# # Generate LaTeX table
#     # latex_table = tabulate(rows, headers, tablefmt="latex")
#     latex_table = tabulate(rows, headers)

#     print(latex_table)

    RBC_data = {}
    for M in [2, 3, 4]:
        RBC_data[f'M={M}'] = torch.load(f'results/RBC/data_N20_Ts_180_M_{M}.pt')
        mean_cop = get_mean_COP(
                                cooling=RBC_data[f'M={M}']['Q_delivered'], 
                                chiller_status=RBC_data[f'M={M}']['chiller_status']
                                )
        energy_cons = get_kilowatthours(pump_power=RBC_data[f'M={M}']['P_pump'],
                                        chiller_power=RBC_data[f'M={M}']['P_chiller'],)
        cooling_rmse = get_control_rmse(load=RBC_data[f'M={M}']['load'],
                                        cooling=RBC_data[f'M={M}']['Q_delivered'])
        print(f"RBC policy, M={M} Mean_COP: {mean_cop:.2f}, Energy_cons: {energy_cons:.2f} kWh, RMSE: {cooling_rmse:.2f}")
    print('='*80)
    print('='*80)
    DPC_data = {}
    for M in [2, 3, 4]:
        for nsteps in [20, 40, 60 ,80]:
            DPC_data[f'M={M}, N={nsteps}'] = torch.load(f'results/MIDPC/data_N{nsteps}_Ts_180_M_{M}.pt')
            mean_cop = get_mean_COP(
                                    cooling=DPC_data[f'M={M}, N={nsteps}']['Q_delivered'], 
                                    chiller_status=DPC_data[f'M={M}, N={nsteps}']['chiller_status']
                                    )
            energy_cons = get_kilowatthours(pump_power=DPC_data[f'M={M}, N={nsteps}']['P_pump'],
                                            chiller_power=DPC_data[f'M={M}, N={nsteps}']['P_chiller'],)
            cooling_rmse = get_control_rmse(load=DPC_data[f'M={M}, N={nsteps}']['load'],
                                            cooling=DPC_data[f'M={M}, N={nsteps}']['Q_delivered'])
            print(f"DPC policy, M={M}, N={nsteps} Mean_COP: {mean_cop:.2f}, Energy_cons: {energy_cons:.2f} kWh, RMSE: {cooling_rmse:.2f}")
        print('-'*80)

    nsteps_plot = 20
    M_plot = 2
    plot_chiller_data(DPC_data[f'M={M_plot}, N={nsteps_plot}'],
                            time_unit='h', save_path='control_plot.pdf')
    plot_chiller_data(RBC_data[f'M={M_plot}'],
                            time_unit='h',)
    plot_chiller_data_nice(DPC_data[f'M={M_plot}, N={nsteps_plot}'], labels=[''],
                            # RBC_data[f'M={M_plot}'],
                            time_unit='h', save_path='control_plot')

# %%
