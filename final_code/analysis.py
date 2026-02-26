#%%
from init import SystemParameters;
import torch
import matplotlib.pyplot as plt
from utils import plot_chiller_data_nice, plot_chiller_data, plot_chiller_data_paper
import re
from tabulate import tabulate
from IPython.display import display, Latex

init = SystemParameters()

def get_kilowatthours(pump_power,chiller_power, Ts=180, megawatt=False):
    if megawatt:
        return torch.sum(pump_power.sum(dim=-1,keepdim=True) + chiller_power.sum(dim=-1, keepdim=True)) *(Ts/3600) /1000
    else:
        return torch.sum(pump_power.sum(dim=-1,keepdim=True) + chiller_power.sum(dim=-1, keepdim=True)) *(Ts/3600)

def get_kilowatthours_pump(pump_power, Ts=180, megawatt=False):
    if megawatt:
        return torch.sum(pump_power.sum(dim=-1,keepdim=True)) *(Ts/3600) /1000
    else:
        return torch.sum(pump_power.sum(dim=-1,keepdim=True)) *(Ts/3600)

def get_kilowatthours_chiller(chiller_power, Ts=180, megawatt=False):
    if megawatt:
        return torch.sum(chiller_power.sum(dim=-1,keepdim=True)) *(Ts/3600) /1000
    else:
        return torch.sum(chiller_power.sum(dim=-1,keepdim=True)) *(Ts/3600)

def get_mean_COP(cooling, chiller_power):
    COP = cooling.sum(-1,keepdim=True)/chiller_power.sum(-1, keepdim=True)
    return COP.mean()

def get_control_rmse(load, cooling):
    return torch.sqrt(torch.mean((load[:,:cooling.size(1),:] - cooling.sum(dim=-1, keepdim=True))**2))

def get_mean_RCE(load, cooling): # Mean Relative Control Error
    return torch.mean(
        (load[:,:cooling.size(1),:] - cooling.sum(dim=-1, keepdim=True)).abs()/load[:,:cooling.size(1),:].abs()
        )*100.

def get_median_RCE(load, cooling): # Mean Relative Control Error
    return torch.median(
        (load[:,:cooling.size(1),:] - cooling.sum(dim=-1, keepdim=True)).abs()/load[:,:cooling.size(1),:].abs()
        )*100.
        


if __name__=='__main__':
    Ts = 180
    M_list = [2, 3]
    # N_list = [20, 40, 60]
    N_list = [5, 10, 15, 20, 40, 60]
    N_list_tab = [5, 10, 15]
    RBC_data = {}
    for M in M_list:
        RBC_data[f'M={M}'] = torch.load(f'results/RBC/data_N20_Ts_180_M_{M}.pt')
        mean_cop = get_mean_COP(
                                cooling=RBC_data[f'M={M}']['Q_delivered'], 
                                chiller_power=RBC_data[f'M={M}']['P_chiller']
                                )
        energy_cons = get_kilowatthours(pump_power=RBC_data[f'M={M}']['P_pump'],
                                        chiller_power=RBC_data[f'M={M}']['P_chiller'], megawatt=True)
        energy_cons_chiller = get_kilowatthours_chiller(chiller_power=RBC_data[f'M={M}']['P_chiller'], megawatt=True)
        energy_cons_pump = get_kilowatthours_pump(pump_power=RBC_data[f'M={M}']['P_pump'], megawatt=True)
        mean_RCE = get_mean_RCE(load=RBC_data[f'M={M}']['load'],
                                        cooling=RBC_data[f'M={M}']['Q_delivered'])
        median_RCE = get_median_RCE(load=RBC_data[f'M={M}']['load'],
                                        cooling=RBC_data[f'M={M}']['Q_delivered'])
        num_switches = torch.sum(RBC_data[f'M={M}']['chiller_status'][:,1:,:] != \
                                RBC_data[f'M={M}']['chiller_status'][:,:-1,:])

        RBC_data[f'M={M}']['Mean_COP']=mean_cop
        RBC_data[f'M={M}']['Savings']=0.
        RBC_data[f'M={M}']['Energy']=energy_cons 
        RBC_data[f'M={M}']['Energy_Pump']=energy_cons_pump 
        RBC_data[f'M={M}']['Energy_Chiller']=energy_cons_chiller 
        RBC_data[f'M={M}']['Mean_RCE']=mean_RCE
        RBC_data[f'M={M}']['Median_RCE']=median_RCE
        RBC_data[f'M={M}']['Num_Switches']=int(num_switches)
        print(f"RBC policy, M={M} Mean_COP: {mean_cop:.2f}, Energy_cons: {energy_cons:.2f} kWh, Mean_RCE: {mean_RCE:.2f}")

    print('='*80)
    print('='*80)
    DPC_data = {}; training_data = {}
    for M in M_list:
        for nsteps in N_list:
            DPC_data[f'M={M}, N={nsteps}'] = torch.load(f'results/MIDPC/data_N{nsteps}_Ts_180_M_{M}.pt')
            training_data[f'M={M}, N={nsteps}'] = torch.load(f'results/MIDPC/policies/training_data_N_{nsteps}_Ts_{Ts}_M_{M}.pt')
            mean_cop = get_mean_COP(
                                    cooling=DPC_data[f'M={M}, N={nsteps}']['Q_delivered'], 
                                    chiller_power=DPC_data[f'M={M}, N={nsteps}']['P_chiller']
                                    )
            energy_cons = get_kilowatthours(pump_power=DPC_data[f'M={M}, N={nsteps}']['P_pump'],
                                            chiller_power=DPC_data[f'M={M}, N={nsteps}']['P_chiller'], megawatt=True)
            energy_cons_chiller = get_kilowatthours_chiller(chiller_power=DPC_data[f'M={M}, N={nsteps}']['P_chiller'], megawatt=True)
            energy_cons_pump = get_kilowatthours_pump(pump_power=DPC_data[f'M={M}, N={nsteps}']['P_pump'], megawatt=True)
            mean_RCE = get_mean_RCE(load=DPC_data[f'M={M}, N={nsteps}']['load'],
                                            cooling=DPC_data[f'M={M}, N={nsteps}']['Q_delivered'])
            median_RCE = get_median_RCE(load=DPC_data[f'M={M}, N={nsteps}']['load'],
                                            cooling=DPC_data[f'M={M}, N={nsteps}']['Q_delivered'])
            num_switches = torch.sum(DPC_data[f'M={M}, N={nsteps}']['chiller_status'][:,1:,:] != \
                                DPC_data[f'M={M}, N={nsteps}']['chiller_status'][:,:-1,:])
            
            DPC_data[f'M={M}, N={nsteps}']['Mean_COP']=mean_cop
            DPC_data[f'M={M}, N={nsteps}']['Energy']=energy_cons
            DPC_data[f'M={M}, N={nsteps}']['Energy_Pump']=energy_cons_pump
            DPC_data[f'M={M}, N={nsteps}']['Energy_Chiller']=energy_cons_chiller
            DPC_data[f'M={M}, N={nsteps}']['Savings']=((RBC_data[f'M={M}']['Energy']-energy_cons)/RBC_data[f'M={M}']['Energy'])*100
            DPC_data[f'M={M}, N={nsteps}']['Mean_RCE']=mean_RCE
            DPC_data[f'M={M}, N={nsteps}']['Median_RCE']=median_RCE
            DPC_data[f'M={M}, N={nsteps}']['Inference_Time'] = DPC_data[f'M={M}, N={nsteps}']['inference_time'].mean()
            DPC_data[f'M={M}, N={nsteps}']['Num_Switches'] = int(num_switches)
            
            DPC_data[f'M={M}, N={nsteps}']['Training_Time'] = training_data[f'M={M}, N={nsteps}']['eltime']
            DPC_data[f'M={M}, N={nsteps}']['N_Parameters'] = training_data[f'M={M}, N={nsteps}']['n_parameters']
            
            print(f"DPC policy, M={M}, N={nsteps} Mean_COP: {mean_cop:.2f}, Energy_cons: {energy_cons:.2f} kWh, Mean RCE: {mean_RCE:.2f}")
        print('-'*80)
    
    MIMPC_data = {}
    for M in M_list:
        for nsteps in N_list_tab:
            MIMPC_data[f'M={M}, N={nsteps}'] = torch.load(f'results/MIMPC/data_N{nsteps}_Ts_180_M_{M}.pt')
            MIMPC_data[f'M={M}, N={nsteps}']['Inference_Time'] = MIMPC_data[f'M={M}, N={nsteps}']['inference_time'].mean()
   
    import pandas as pd
    metrics_rbc = [
        ("EC [MWh]", "Energy"),
        ("EC Chillers [MWh]", "Energy_Chiller"),
        ("EC Pumps [MWh]", "Energy_Pump"),
        ("COP [-]", "Mean_COP"),
        ("Num. of switches [-]", "Num_Switches"),
        ("Mean RCE [\%]", "Mean_RCE"),
        ("Median RCE [\%]", "Median_RCE")
    ]

    metrics_midpc = [
        ("EC [MWh]", "Energy"),
        ("EC Chillers [MWh]", "Energy_Chiller"),
        ("EC Pumps [MWh]", "Energy_Pump"),
        ("COP [-]", "Mean_COP"),
        ("Num. of switches [-]", "Num_Switches"),
        ("Mean RCE [\%]", "Mean_RCE"),
        ("Median RCE [\%]", "Median_RCE"),
        ("Savings [\\%]", "Savings"),
        ("MIT [s]", "Inference_Time"),
        ("TT [s]", "Training_Time"),
        ("NTP [-]", "N_Parameters"),
    ]

    metrics_mimpc = [
        ("Inference Time", "Inference_Time")
    ]

    rows = []

    # RBC block
    for i, (label, key) in enumerate(metrics_rbc):
        rows.append([
            "\\multirow{3}{*}{RBC}" if i == 0 else "",
            label
        ] + [f"{RBC_data[f'M={M}'][key]:.2f}" if N == N_list[0] else "-"
        for M in M_list for N in N_list_tab])

    # MIDPC block
    for i, (label, key) in enumerate(metrics_midpc):
        rows.append([
            "\\midrule\\multirow{5}{*}{MIDPC}" if i == 0 else "",
            label
        ] + [
           f"{int(DPC_data[f'M={M}, N={N}'][key])}" if key == "N_Parameters"
        else f"{DPC_data[f'M={M}, N={N}'][key]:.1e}" if key == "Inference_Time"
        else f"{DPC_data[f'M={M}, N={N}'][key]:.2f}"
        for M in M_list for N in N_list_tab
        ])

    # MIMPC block
    for i, (label, key) in enumerate(metrics_mimpc):
        rows.append([
            "\\midrule MIMPC",
            label
        ] + [
            f"{MIMPC_data[f'M={M}, N={N}'][key]:.2f}" if M in M_list
            else '-'
            for M in M_list for N in N_list_tab
        ])

        # Build DataFrame
    df = pd.DataFrame(rows, columns=["Method", "Metric"] +
                    ["" for _ in range(len(M_list) * len(N_list_tab))])

    # Create dynamic header row for N values
    n_headers = " & ".join([f"$N\\!=\\!{N}$" for _ in M_list for N in N_list_tab])

    # Top-level M headers
    m_headers = " & ".join([
        f"\\multicolumn{{{len(N_list_tab)}}}{{c}}{{$M\\!=\\!{M}$}}"
        for M in M_list
    ])

    # CMIDRULES for each M block
    cmidrules = " ".join([
        f"\\cmidrule(lr){{{3 + i*len(N_list_tab)}-{2 + (i+1)*len(N_list_tab)}}}"
        for i in range(len(M_list))
    ])

    # Custom table header with correct alignment
    custom_header = rf"""
    \toprule
    \multirow{{2}}{{*}}{{Method}} & \multirow{{2}}{{*}}{{Metric}} & {m_headers} \\
    {cmidrules}
    &  & {n_headers} \\

    """

    # Generate the table body (no headers from pandas)
    body = df.to_latex(
        index=False,
        header=False,
        escape=False,
        column_format=f"@{{}}ll{ 'c'*len(M_list)*len(N_list_tab) }@{{}}"
    )

    # Merge header + body
    latex_code = body.replace("\\toprule", custom_header)

    # Ensure bottomrule exists once
    if "\\bottomrule" not in latex_code:
        latex_code += "\\bottomrule\n"

    print(latex_code)

    df.columns = ["Method", "Metric"] + [f"M={M}, N={N}" for M in M_list for N in N_list_tab]
    print(df.to_markdown(index=False))


# %%
    nsteps_plot = 15
    M_plot = 2
    # # # Control Plot
    plot_chiller_data_paper(DPC_data[f'M={M_plot}, N={nsteps_plot}'], plot_h=3.5,
                            time_unit='h', save_path=f'control_plot_N{nsteps_plot}')



# %%
    import matplotlib
    from cycler import cycler
    import matplotlib.cm as cm
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    colors = cm.get_cmap('Set2', 8).colors
    plt.rcParams['axes.prop_cycle'] = cycler(color=colors)
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
    inference_N_list = [5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    for N in inference_N_list:
        training_data_M2 = torch.load(f'results/MIDPC/policies/training_data_N_{N}_Ts_{Ts}_M_{2}.pt')
        training_data_M3 = torch.load(f'results/MIDPC/policies/training_data_N_{N}_Ts_{Ts}_M_{3}.pt')
        training_data_M4 = torch.load(f'results/MIDPC/policies/training_data_N_{N}_Ts_{Ts}_M_{4}.pt')
        training_data_M5 = torch.load(f'results/MIDPC/policies/training_data_N_{N}_Ts_{Ts}_M_{5}.pt')
        inference_data_M2 =  torch.load(f'results/MIDPC/data_N{N}_Ts_180_M_{2}.pt')
        inference_data_M3 =  torch.load(f'results/MIDPC/data_N{N}_Ts_180_M_{3}.pt')
        inference_data_M4 =  torch.load(f'results/MIDPC/data_N{N}_Ts_180_M_{4}.pt')
        inference_data_M5 =  torch.load(f'results/MIDPC/data_N{N}_Ts_180_M_{5}.pt')
        
        TT2_list.append(training_data_M2['eltime']);TT3_list.append(training_data_M3['eltime'])
        TT4_list.append(training_data_M4['eltime']);TT5_list.append(training_data_M5['eltime'])
        MIT2_list.append(inference_data_M2['inference_time'].mean());MIT3_list.append(inference_data_M3['inference_time'].mean())
        MIT4_list.append(inference_data_M4['inference_time'].mean());MIT5_list.append(inference_data_M5['inference_time'].mean())


    TT2 = torch.tensor(TT2_list).unsqueeze(1); TT3 = torch.tensor(TT3_list).unsqueeze(1)
    TT4 = torch.tensor(TT4_list).unsqueeze(1); TT5 = torch.tensor(TT5_list).unsqueeze(1)
    MIT2 = torch.tensor(MIT2_list).unsqueeze(1); MIT3 = torch.tensor(MIT3_list).unsqueeze(1)
    MIT4 = torch.tensor(MIT4_list).unsqueeze(1); MIT5 = torch.tensor(MIT5_list).unsqueeze(1)
    # MIT = torch.tensor(MIT_list).unsqueeze(1)
    
    fig1, ax = plt.subplots(2,1, figsize=(3.5,2.),sharex=True)
    ax = ax.flatten()
    x =torch.vstack([torch.tensor([n]) for n in inference_N_list])
    ax[0].plot(
                x,
                TT2[:,0],
                 alpha=.95,
        label="$M\!=\!2$")
    
    ax[0].plot(
                x,
                TT3[:,0],
                 alpha=.95,
        label="$M\!=\!3$")
    ax[0].plot(
                x,
                TT4[:,0],
                 alpha=.95,
        label="$M\!=\!4$")
    ax[0].plot(
                x,
                TT5[:,0],
                 alpha=.95,
        label="$M\!=\!5$")
    
    fig1.tight_layout(pad=0.0)
    ax[0].set_xticks(inference_N_list)
    ax[0].legend(framealpha=1.0, edgecolor='gray',fancybox=False)
    # ax[0].set_xlabel('$N$ [-]')
    ax[0].set_ylabel('TT [s]')
    ax[0].grid()
    
    ax[1].plot(
                x,
                MIT2[:,0],
                 alpha=.95,
        label="$M\!=\!2$")
    
    ax[1].plot(
                x,
                MIT3[:,0],
                 alpha=.95,
        label="$M\!=\!3$")
    ax[1].plot(
                x,
                MIT4[:,0],
                 alpha=.95,
        label="$M\!=\!4$")
    ax[1].plot(
                x,
                MIT5[:,0],
                 alpha=.95,
        label="$M\!=\!5$")
    
    fig1.tight_layout(pad=0.0, h_pad=1.0)
    ax[1].set_xticks(inference_N_list)
    # ax[1].legend(framealpha=1.0, edgecolor='gray',fancybox=False)
    ax[1].set_xlabel('Prediction horizon length --- $N$')
    ax[1].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    ax[1].set_ylabel('MIT [s]')
    ax[1].set_yticks([0.00018,0.00019, 0.00020])
    ax[1].grid()
    fig1.show()
    fig1.savefig(f'MIT_plot.pdf', bbox_inches='tight',pad_inches=0.05,transparent=True)
    fig1.savefig(f'MIT_plot.pgf', bbox_inches='tight', pad_inches=0.05,transparent=True)

# %%
