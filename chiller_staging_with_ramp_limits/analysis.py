
# %%
if __name__ == '__main__':
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
    inference_data_RBC =  torch.load(f'results/RBC/data_N{15}_Ts_180_M_{3}.pt')

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
                alpha=.95, label='$T_\mathrm{r}^{\mathrm{min}}, T_\mathrm{r}^{\mathrm{max}}$ bounds')
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




    fig1.tight_layout(pad=0.1)
    fig1.show()
    ax[0].legend(framealpha=1.0, edgecolor='gray', fancybox=False, bbox_to_anchor=(0.55, 0.85))
    ax[1].legend(framealpha=1.0, edgecolor='gray', fancybox=False, bbox_to_anchor=(1.0, 1.01))
    # ax[1].legend(framealpha=1.0, edgecolor='gray',fancybox=False)
    ax[0].set_ylabel("$Q,Q_\mathrm{load}$ [kW]")
    ax[1].set_ylabel('$T_\mathrm{r}$ [°C]')
    ax[2].set_ylabel('$s$ [-]')
    ax[-1].set_xlabel('Time [h]')
    ax[0].grid()
    ax[1].grid()
    ax[2].grid()
    fig1.savefig(f'violation_plot.pdf', bbox_inches='tight',pad_inches=0.01,transparent=True)
    fig1.savefig(f'violation_plot.pgf', bbox_inches='tight', pad_inches=0.01,transparent=True)
    fig1.savefig(f'violation_plot.svg',bbox_inches='tight', pad_inches=0.01, transparent=True)
    fig1.savefig(f'violation_plot.eps', bbox_inches='tight', pad_inches=0.01,transparent=True)

# %%
