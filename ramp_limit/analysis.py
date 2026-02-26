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
    M_list = [2, 3, 4, 5]
    N_list = [30, 40, 50 ,60]
    RBC_data = {}
    for M in M_list:
        RBC_data[f'M={M}'] = torch.load(f'results/RBC/data_N20_Ts_180_M_{M}.pt')
        mean_cop = get_mean_COP(
                                cooling=RBC_data[f'M={M}']['Q_delivered'], 
                                chiller_status=RBC_data[f'M={M}']['chiller_status']
                                )
        energy_cons = get_kilowatthours(pump_power=RBC_data[f'M={M}']['P_pump'],
                                        chiller_power=RBC_data[f'M={M}']['P_chiller'],)
        cooling_rmse = get_control_rmse(load=RBC_data[f'M={M}']['load'],
                                        cooling=RBC_data[f'M={M}']['Q_delivered'])
        RBC_data[f'M={M}']['Mean_COP']=mean_cop
        RBC_data[f'M={M}']['Savings']=0.
        RBC_data[f'M={M}']['Energy']=energy_cons
        RBC_data[f'M={M}']['Cooling_RMSE']=cooling_rmse
        print(f"RBC policy, M={M} Mean_COP: {mean_cop:.2f}, Energy_cons: {energy_cons:.2f} kWh, RMSE: {cooling_rmse:.2f}")

    print('='*80)
    print('='*80)
    DPC_data = {}
    for M in M_list:
        for nsteps in N_list:
            DPC_data[f'M={M}, N={nsteps}'] = torch.load(f'results/MIDPC/data_N{nsteps}_Ts_180_M_{M}.pt')
            mean_cop = get_mean_COP(
                                    cooling=DPC_data[f'M={M}, N={nsteps}']['Q_delivered'], 
                                    chiller_status=DPC_data[f'M={M}, N={nsteps}']['chiller_status']
                                    )
            energy_cons = get_kilowatthours(pump_power=DPC_data[f'M={M}, N={nsteps}']['P_pump'],
                                            chiller_power=DPC_data[f'M={M}, N={nsteps}']['P_chiller'],)
            cooling_rmse = get_control_rmse(load=DPC_data[f'M={M}, N={nsteps}']['load'],
                                            cooling=DPC_data[f'M={M}, N={nsteps}']['Q_delivered'])
            DPC_data[f'M={M}, N={nsteps}']['Mean_COP']=mean_cop
            DPC_data[f'M={M}, N={nsteps}']['Energy']=energy_cons
            DPC_data[f'M={M}, N={nsteps}']['Savings']=((RBC_data[f'M={M}']['Energy']-energy_cons)/RBC_data[f'M={M}']['Energy'])*100
            DPC_data[f'M={M}, N={nsteps}']['Cooling_RMSE']=cooling_rmse

            print(f"DPC policy, M={M}, N={nsteps} Mean_COP: {mean_cop:.2f}, Energy_cons: {energy_cons:.2f} kWh, RMSE: {cooling_rmse:.2f}")
        print('-'*80)
    
    import pandas as pd

    # Define table structure
    methods = ["RBC", "MIDPC"]
    rows = []

    RBC_rows = len(M_list)
    DPC_rows = len(M_list)*len(N_list)
    for M in M_list:
        rows.append([f"\\multirow{{{RBC_rows}}}{{*}}{{RBC}}" if M == 2 else "",
                     
                      "-", M,
                      f"{RBC_data[f'M={M}']['Energy']:.2f}", 
                      f"{RBC_data[f'M={M}']['Savings']:.2f}",
                      f"{RBC_data[f'M={M}']['Mean_COP']:.2f}",
                      f"{RBC_data[f'M={M}']['Cooling_RMSE']:.2f}",
                      ])
    # MIDPC rows (N = 20, 40, 60, 80; M = 2,4)
    for N in N_list:
        for M in M_list:
            method = f"\\multirow{{{DPC_rows}}}{{*}}{{MIDPC}}" if (N == N_list[0] and M == M_list[0]) else ""
            rows.append([method, f" \\multirow{{{len(M_list)}}}{{*}}{{{N}}}" if M == 2 else "", M,
                         f"{DPC_data[f'M={M}, N={N}']['Energy']:.2f}",
                          f"{DPC_data[f'M={M}, N={N}']['Savings']:.2f}",
                          f"{DPC_data[f'M={M}, N={N}']['Mean_COP']:.2f}",
                          f"{DPC_data[f'M={M}, N={N}']['Cooling_RMSE']:.2f}"
                          ])

    # Create DataFrame
    df = pd.DataFrame(rows, columns=["Method", "N", "M", "Energy [kWh]", "Savings [\%]", "Mean COP [-]", "RMSE [kW]"])

    # Export to LaTeX
    latex_code = df.to_latex(
        index=False,
        escape=False,  # Allow \multirow to render
        caption="Energy and COP comparison between RBC and MIDPC methods.",
        label="tab:energy_cop",
        column_format="@{}llccccc@{}"  # Align columns
    )

    print(latex_code)

#%%

#%%
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

    import matplotlib
    from cycler import cycler
    import matplotlib.cm as cm
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import torch
    from init import SystemParameters;
    init = SystemParameters()


    # colors = cm.get_cmap('Set2', 8).colors
    # plt.rcParams['axes.prop_cycle'] = cycler(color=colors)
    matplotlib.use("pgf")
    plt.rcParams.update({
            "pgf.texsystem": "pdflatex",
            "text.usetex": True,
            "font.family": "serif",
            "font.size": 10,
            "pgf.rcfonts": False,
            "legend.fontsize": 6,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8
        })
    TT2_list, TT3_list, TT4_list, TT5_list = [], [], [], []
    MIT2_list, MIT3_list, MIT4_list, MIT5_list = [], [], [], []
    # inference_N_list = [5, 10, 15, 20, 40, 60]
    inference_data_DPC =  torch.load(f'results/MIDPC/data_N{15}_Ts_180_M_{3}.pt')
    inference_data_RBC =  torch.load(f'results/RBC/data_N{20}_Ts_180_M_{3}.pt')

    Ts = 180
    t_1_day = int(24*60*60/Ts)
    time = torch.arange(0, inference_data_DPC["load"].size(1)) * (Ts / 3600)
    time = time[:t_1_day]
    
    fig1, ax = plt.subplots(3,1, figsize=(3.5,3),sharex=True)
    ax = ax.flatten()
    ax[0].plot(time,
               inference_data_DPC["load"][0,:int(t_1_day),0], 'k--',
                alpha=.95, 
        label="$Q_\mathrm{load}$") 
    ax[0].plot(time,
               inference_data_RBC["Q_delivered"][0,:int(t_1_day),:].sum(-1),
                alpha=.95, 
        label="RBC")
    ax[0].plot(time,
               inference_data_DPC["Q_delivered"][0,:int(t_1_day),:].sum(-1), '--',
                alpha=.95, color='crimson',
        label="MI-DPC")
    
    ax[1].plot(time,
               inference_data_RBC["T_return"][0,:int(t_1_day),:].sum(-1),
                alpha=.95, 
        label="_nolegend_")
    ax[1].plot(time,
               inference_data_DPC["T_return"][0,:int(t_1_day),:].sum(-1), '--',
                alpha=.95, color='crimson',
        label="_nolegend_")
    
    ax[1].plot(time,
               torch.ones_like(time)*init.T_return_max, 'k:',
                alpha=.95, label='$T_\mathrm{r}^{\mathrm{min}}, T_\mathrm{r}^{\mathrm{min}}$')
    ax[1].plot(time,
               torch.ones_like(time)*init.T_return_min, 'k:',
                alpha=.95)
    ax[2].plot(time,
               inference_data_RBC["chiller_status"][0,:int(t_1_day),:].sum(-1),
                alpha=.95, 
        label="RBC")
    ax[2].plot(time,
               inference_data_DPC["chiller_status"][0,:int(t_1_day),:].sum(-1), '--',
                alpha=.95, color='crimson',
        label="MI-DPC")




    fig1.tight_layout(pad=0.0)
    fig1.show()
    ax[0].legend(framealpha=1.0, edgecolor='gray',fancybox=False)
    ax[1].legend(framealpha=1.0, edgecolor='gray',fancybox=False)
    # ax[1].legend(framealpha=1.0, edgecolor='gray',fancybox=False)
    ax[0].set_ylabel("$Q,Q_\mathrm{load}$ [kW]")
    ax[1].set_ylabel('$T_\mathrm{r}$ [°C]')
    ax[2].set_ylabel('$s$ [-]')
    ax[-1].set_xlabel('Time [h]')
    # ax[1].set_yticks([init.T_return_min, (init.T_return_max+init.T_return_min)/2, init.T_return_max])
    ax[0].grid()
    ax[1].grid()
    ax[2].grid()
    fig1.savefig(f'violation_plot.pdf', bbox_inches='tight',pad_inches=0.05,transparent=True)
    fig1.savefig(f'violation_plot.pgf', bbox_inches='tight', pad_inches=0.05,transparent=True)

# %%
