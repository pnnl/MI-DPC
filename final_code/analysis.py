#%%
from init import SystemParameters;
import torch
import matplotlib.pyplot as plt
from utils import plot_chiller_data_nice, plot_chiller_data
import re
from tabulate import tabulate
init = SystemParameters()

def get_kilowatthours(pump_power,chiller_power, Ts=180):
    return torch.sum(pump_power.sum(dim=-1,keepdim=True) + chiller_power.sum(dim=-1, keepdim=True)) *(Ts/3600)

def get_mean_COP(cooling, chiller_power):
    COP = cooling.sum(-1,keepdim=True)/chiller_power.sum(-1, keepdim=True)
    return COP.mean()

def get_control_rmse(load, cooling):
    return torch.sqrt(torch.mean((load[:,:cooling.size(1),:] - cooling.sum(dim=-1, keepdim=True))**2))

if __name__=='__main__':
    Ts = 180
    M_list = [2, 3]
    N_list = [20, 40, 60]
    RBC_data = {}
    for M in M_list:
        RBC_data[f'M={M}'] = torch.load(f'results/RBC/data_N20_Ts_180_M_{M}.pt')
        mean_cop = get_mean_COP(
                                cooling=RBC_data[f'M={M}']['Q_delivered'], 
                                chiller_power=RBC_data[f'M={M}']['P_chiller']
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
                                            chiller_power=DPC_data[f'M={M}, N={nsteps}']['P_chiller'],)
            cooling_rmse = get_control_rmse(load=DPC_data[f'M={M}, N={nsteps}']['load'],
                                            cooling=DPC_data[f'M={M}, N={nsteps}']['Q_delivered'])
            DPC_data[f'M={M}, N={nsteps}']['Mean_COP']=mean_cop
            DPC_data[f'M={M}, N={nsteps}']['Energy']=energy_cons
            DPC_data[f'M={M}, N={nsteps}']['Savings']=((RBC_data[f'M={M}']['Energy']-energy_cons)/RBC_data[f'M={M}']['Energy'])*100
            DPC_data[f'M={M}, N={nsteps}']['Cooling_RMSE']=cooling_rmse
            DPC_data[f'M={M}, N={nsteps}']['Inference_Time'] = DPC_data[f'M={M}, N={nsteps}']['inference_time'].mean()
            DPC_data[f'M={M}, N={nsteps}']['Training_Time'] = training_data[f'M={M}, N={nsteps}']['eltime']
            DPC_data[f'M={M}, N={nsteps}']['N_Parameters'] = training_data[f'M={M}, N={nsteps}']['n_parameters']
            
            print(f"DPC policy, M={M}, N={nsteps} Mean_COP: {mean_cop:.2f}, Energy_cons: {energy_cons:.2f} kWh, RMSE: {cooling_rmse:.2f}")
        print('-'*80)
    
    MIMPC_data = {}
    for M in M_list:
        for nsteps in N_list:
            MIMPC_data[f'M={M}, N={nsteps}'] = torch.load(f'results/MIMPC/data_N{nsteps}_Ts_180_M_{M}.pt')
            MIMPC_data[f'M={M}, N={nsteps}']['Inference_Time'] = MIMPC_data[f'M={M}, N={nsteps}']['inference_time'].mean()
   
    import pandas as pd
    metrics_rbc = [
        ("Energy [MWh]", "Energy"),
        ("COP [-]", "Mean_COP"),
        ("RMSE [kW]", "Cooling_RMSE")
    ]

    metrics_midpc = [
        ("Energy [MWh]", "Energy"),
        ("COP [-]", "Mean_COP"),
        ("RMSE [kW]", "Cooling_RMSE"),
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
        ] + [f"{RBC_data[f'M={M}'][key]:.2f}" for M in M_list for _ in N_list])

    # MIDPC block
    for i, (label, key) in enumerate(metrics_midpc):
        rows.append([
            "\\midrule\\multirow{5}{*}{MIDPC}" if i == 0 else "",
            label
        ] + [
            f"{DPC_data[f'M={M}, N={N}'][key]:.2f}"
            for M in M_list for N in N_list
        ])

    # MIMPC block
    for i, (label, key) in enumerate(metrics_mimpc):
        rows.append([
            "\\midrule MIMPC",
            label
        ] + [
            f"{MIMPC_data[f'M={M}, N={N}'][key]:.2f}"
            for M in M_list for N in N_list
        ])

        # Build DataFrame
    df = pd.DataFrame(rows, columns=["Method", "Metric"] +
                    ["" for _ in range(len(M_list) * len(N_list))])

    # Create dynamic header row for N values
    n_headers = " & ".join([f"$N\\!=\\!{N}$" for _ in M_list for N in N_list])

    # Top-level M headers
    m_headers = " & ".join([
        f"\\multicolumn{{{len(N_list)}}}{{c}}{{$M\\!=\\!{M}$}}"
        for M in M_list
    ])

    # CMIDRULES for each M block
    cmidrules = " ".join([
        f"\\cmidrule(lr){{{3 + i*len(N_list)}-{2 + (i+1)*len(N_list)}}}"
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
        column_format=f"@{{}}ll{ 'c'*len(M_list)*len(N_list) }@{{}}"
    )

    # Merge header + body
    latex_code = body.replace("\\toprule", custom_header)

    # Ensure bottomrule exists once
    if "\\bottomrule" not in latex_code:
        latex_code += "\\bottomrule\n"

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
